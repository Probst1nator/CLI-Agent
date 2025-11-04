"""
Autocomplete API endpoints for CLI-Agent.

Provides intelligent autocomplete suggestions using hybrid fuzzy matching (rapidfuzz + BM25):
- Commands (e.g., /exit, /regenerate)
- Tools (e.g., bash, readfile)
- File paths (recent + all cached)
"""

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aiohttp import web
    from aiohttp.web import Application, Request, Response

try:
    from aiohttp import web
    from aiohttp.web import Application, Request, Response

    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None  # type: ignore
    Request = object  # type: ignore
    Response = object  # type: ignore
    Application = object  # type: ignore
    AIOHTTP_AVAILABLE = False

try:
    from rapidfuzz import fuzz

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

logger = logging.getLogger(__name__)

# Cache for dynamic path suggestions (path_prefix -> (suggestions, timestamp))
_path_suggestion_cache: dict[str, tuple[list[dict], float]] = {}
_CACHE_TTL_SECONDS = 1.0  # Cache results for 1 second during typing


def tokenize(text: str) -> list[str]:
    """Simple tokenization for BM25"""
    return text.lower().replace("/", " ").replace("_", " ").replace("-", " ").split()


def extract_path_from_input(input_text: str) -> Optional[str]:
    """
    Extract a filesystem path from user input.

    Looks for path patterns like:
    - /absolute/path/to/something
    - ~/home/relative/path
    - ./relative/path

    Args:
        input_text: The full user input string

    Returns:
        The extracted path string, or None if no path pattern found

    Examples:
        "yes /home/prob/OneD" -> "/home/prob/OneD"
        "read ~/Documents/file" -> "~/Documents/file"
        "hello world" -> None
    """
    # Pattern matches:
    # - Paths starting with / (absolute)
    # - Paths starting with ~ (home directory)
    # - Paths starting with ./ or ../ (relative)
    # Captures until whitespace or end of string
    path_pattern = r'([~/](?:[^\s]+)?|\.\.?/(?:[^\s]+)?)'

    matches = re.findall(path_pattern, input_text)
    if matches:
        # Return the rightmost (last) path found
        return matches[-1]

    return None


def get_dynamic_path_suggestions(
    path_prefix: str, max_depth: int = 4, max_results: int = 50
) -> list[dict[str, Any]]:
    """
    Dynamically scan filesystem for path suggestions based on user input.

    Scans directories up to max_depth levels deep from the given path prefix.
    Results are cached for 1 second to avoid redundant filesystem scans during typing.

    Args:
        path_prefix: Partial or complete path to scan (e.g., "/home/prob/OneD")
        max_depth: Maximum directory depth to scan (default: 4)
        max_results: Maximum number of suggestions to return (default: 50)

    Returns:
        List of suggestion dicts with 'text' and 'score' keys, sorted by relevance

    Examples:
        "/home/prob/OneD" -> [
            {"text": "/home/prob/OneDrive", "score": 95},
            {"text": "/home/prob/OneDrive/Documents", "score": 90},
            ...
        ]
    """
    import os
    import time
    from pathlib import Path

    # Check cache first
    cache_key = f"{path_prefix}:{max_depth}"
    if cache_key in _path_suggestion_cache:
        cached_suggestions, timestamp = _path_suggestion_cache[cache_key]
        if time.time() - timestamp < _CACHE_TTL_SECONDS:
            logger.debug(
                f"[PathAutocomplete] Cache hit for '{path_prefix}' ({len(cached_suggestions)} results)"
            )
            return cached_suggestions

    suggestions = []
    start_time = time.time()

    try:
        # Expand ~ and resolve relative paths
        expanded_path = os.path.expanduser(path_prefix)
        expanded_path = os.path.abspath(expanded_path)

        # Determine base directory and partial match
        if os.path.isdir(expanded_path):
            base_dir = expanded_path
            partial_name = ""
        else:
            base_dir = os.path.dirname(expanded_path)
            partial_name = os.path.basename(expanded_path)

        # Check if base directory exists
        if not os.path.isdir(base_dir):
            logger.debug(
                f"[PathAutocomplete] Base directory does not exist: {base_dir}"
            )
            return []

        # Optimization: If we have a partial_name, first check immediate children
        # This avoids deep scanning when matches are likely in the current directory
        if partial_name:
            try:
                immediate_entries = os.listdir(base_dir)
                for entry in immediate_entries:
                    if entry.startswith("."):
                        continue  # Skip hidden
                    if not entry.startswith(partial_name):
                        continue  # Only matches

                    full_path = os.path.join(base_dir, entry)
                    if os.path.isdir(full_path):
                        suggestions.append({"text": full_path, "score": 100})
                    elif os.path.isfile(full_path):
                        suggestions.append({"text": full_path, "score": 95})

                    if len(suggestions) >= max_results:
                        break
            except PermissionError:
                pass  # Will try os.walk below

        # If we found matches at depth 0, return them (optimization)
        if suggestions and partial_name:
            logger.info(
                f"[PathAutocomplete] Found {len(suggestions)} immediate matches for '{path_prefix}'"
            )
            _path_suggestion_cache[cache_key] = (suggestions, time.time())
            return suggestions

        # Otherwise, walk directory tree with depth limit
        for root, dirs, files in os.walk(base_dir):
            # Check timeout (100ms max)
            if time.time() - start_time > 0.1:
                logger.warning(
                    f"[PathAutocomplete] Timeout after {len(suggestions)} results"
                )
                break

            # Calculate current depth relative to base_dir
            depth = root[len(base_dir) :].count(os.sep)
            if depth >= max_depth:
                # Don't descend deeper
                dirs[:] = []
                continue

            # Filter out hidden directories from traversal
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            # If we have a partial_name and we're at depth > 0, skip non-matching directories
            # This prunes the search tree early
            if partial_name and depth == 0:
                dirs[:] = [d for d in dirs if d.startswith(partial_name)]

            # Add matching directories
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                # Calculate score based on depth (shallower = higher score)
                score = 100 - (depth * 5)
                suggestions.append({"text": full_path, "score": max(score, 80)})

                if len(suggestions) >= max_results:
                    break

            # Add matching files (only at depth 0 when no partial_name)
            if depth == 0 and not partial_name:
                for file_name in files:
                    if file_name.startswith("."):
                        continue
                    full_path = os.path.join(root, file_name)
                    suggestions.append({"text": full_path, "score": 95})

                    if len(suggestions) >= max_results:
                        break

            if len(suggestions) >= max_results:
                break

        # Sort by score descending
        suggestions.sort(key=lambda x: x["score"], reverse=True)

        # Cache the results
        _path_suggestion_cache[cache_key] = (suggestions, time.time())

        logger.info(
            f"[PathAutocomplete] Found {len(suggestions)} paths for '{path_prefix}' in {(time.time() - start_time)*1000:.1f}ms"
        )

    except PermissionError as e:
        logger.warning(f"[PathAutocomplete] Permission denied: {e}")
    except Exception as e:
        logger.error(f"[PathAutocomplete] Error scanning paths: {e}", exc_info=True)

    return suggestions


def extract_relevant_content(content: str) -> str:
    """
    Extract text from specific XML tags for cleaner autocomplete context.

    Extracts content from:
    - <think>...</think>
    - <todos>...</todos>
    - <response>...</response>
    - <visualresponse>...</visualresponse> (only if type is markdown, not html)
    - Any raw text outside of XML tags

    Args:
        content: Raw message content that may contain XML tags

    Returns:
        Extracted text concatenated with newlines, or original content if empty
    """
    extracted_parts = []

    # Extract from <think> tags
    think_matches = re.findall(
        r"<think\s*>(.*?)</think>", content, re.DOTALL | re.IGNORECASE
    )
    for match in think_matches:
        extracted_parts.append(match.strip())

    # Extract from <todos> tags
    todos_matches = re.findall(
        r"<todos\s*>(.*?)</todos>", content, re.DOTALL | re.IGNORECASE
    )
    for match in todos_matches:
        extracted_parts.append(match.strip())

    # Extract from <response> tags
    response_matches = re.findall(
        r"<response\s*>(.*?)</response>", content, re.DOTALL | re.IGNORECASE
    )
    for match in response_matches:
        extracted_parts.append(match.strip())

    # Extract from <visualresponse> tags - only if type is markdown
    visualresponse_matches = re.findall(
        r"<visualresponse\s*>(.*?)</visualresponse>", content, re.DOTALL | re.IGNORECASE
    )
    for vr_content in visualresponse_matches:
        # Check if this visualresponse has type=markdown (not html)
        type_match = re.search(
            r"<type\s*>(.*?)</type>", vr_content, re.DOTALL | re.IGNORECASE
        )
        if type_match and type_match.group(1).strip().lower() == "markdown":
            # Extract the content from this markdown visualresponse
            content_match = re.search(
                r"<content\s*>(.*?)</content>", vr_content, re.DOTALL | re.IGNORECASE
            )
            if content_match:
                extracted_parts.append(content_match.group(1).strip())

    # Also extract any raw text that's outside XML tags
    # Remove all XML blocks (tags + content) to get only the raw text
    text_without_xml = content
    # Remove all complete XML blocks (opening tag + content + closing tag)
    text_without_xml = re.sub(
        r"<\w+[^>]*>.*?</\w+>", "", text_without_xml, flags=re.DOTALL
    )
    # Remove any remaining standalone tags
    text_without_xml = re.sub(r"<[^>]+>", "", text_without_xml)
    text_without_xml = text_without_xml.strip()

    if text_without_xml:
        # Insert raw text at the beginning if it exists
        extracted_parts.insert(0, text_without_xml)

    # Return extracted content if any was found, otherwise return original
    if extracted_parts:
        return "\n\n".join(extracted_parts)
    else:
        return content


def hybrid_score_suggestion(
    input_text: str,
    candidate: str,
    _candidate_tokens: list[str],
    bm25_score: float = 0.0,
    is_recent_file: bool = False,
    is_cwd_file: bool = False,
) -> float:
    """
    Hybrid scoring combining rapidfuzz and BM25.

    Args:
        input_text: User's input query
        candidate: Suggestion text to score
        candidate_tokens: Tokenized candidate for BM25
        bm25_score: Pre-computed BM25 score (0-100 scale)
        is_recent_file: True if file is in recent paths
        is_cwd_file: True if file is in current working directory

    Returns:
        Combined score (0-100+, higher is better)
    """
    if not input_text:
        return 0

    input_lower = input_text.lower()
    candidate_lower = candidate.lower()

    # === Rapidfuzz Scoring (50% weight) ===
    rapidfuzz_score = 0
    if RAPIDFUZZ_AVAILABLE:
        # Overall similarity
        ratio_score = fuzz.ratio(input_lower, candidate_lower)

        # Partial matching (substring)
        partial_score = fuzz.partial_ratio(input_lower, candidate_lower)

        # Take best of the two
        rapidfuzz_score = max(ratio_score, partial_score)

        # Bonus for prefix match
        if candidate_lower.startswith(input_lower):
            rapidfuzz_score = min(100, rapidfuzz_score + 30)

        # Bonus for start-of-word match
        words = candidate_lower.split("/")
        for word in words:
            if word.startswith(input_lower):
                rapidfuzz_score = min(100, rapidfuzz_score + 20)
                break
    else:
        # Fallback to simple scoring if rapidfuzz not available
        if candidate_lower.startswith(input_lower):
            rapidfuzz_score = 100
        elif input_lower in candidate_lower:
            rapidfuzz_score = 50

    # === BM25 Scoring (50% weight) ===
    # BM25 score is passed in (pre-computed)

    # === Combine Scores ===
    combined_score = (rapidfuzz_score * 0.5) + (bm25_score * 0.5)

    # === Apply Boosts ===
    if is_recent_file:
        combined_score *= 1.2  # 20% boost for recent files

    if is_cwd_file:
        combined_score *= 1.1  # 10% boost for cwd files

    return combined_score


def get_command_suggestions(input_text: str) -> list[dict[str, str]]:
    """Get command suggestions using hybrid scoring."""
    try:
        from agent.core.command_registry import get_global_command_registry

        registry = get_global_command_registry()
        descriptions = registry.get_command_descriptions_for_completion()

        if not descriptions:
            return []

        # Build BM25 corpus
        candidates = []
        corpus = []
        for alias, description in descriptions.items():
            candidates.append((alias, description))
            # Combine alias and description for BM25
            text = f"{alias} {description}"
            corpus.append(tokenize(text))

        # Compute BM25 scores
        bm25_scores = {}
        if BM25_AVAILABLE and corpus:
            try:
                bm25 = BM25Okapi(corpus)
                query_tokens = tokenize(input_text)
                scores = bm25.get_scores(query_tokens)

                # Normalize BM25 scores to 0-100 range
                max_score = float(scores.max()) if scores.max() > 0 else 1.0
                bm25_scores = {
                    alias: (float(score) / max_score) * 100
                    for (alias, _), score in zip(candidates, scores, strict=False)
                }
            except Exception as e:
                logger.debug(f"BM25 scoring failed: {e}")

        # Score all candidates with hybrid approach
        suggestions = []
        for alias, description in candidates:
            bm25_score = bm25_scores.get(alias, 0)
            candidate_tokens = tokenize(alias)

            score = hybrid_score_suggestion(
                input_text, alias, candidate_tokens, bm25_score=bm25_score
            )

            # Filter low scores
            if score > 10:
                suggestions.append(
                    {
                        "text": alias,
                        "description": description,
                        "type": "command",
                        "score": score,
                    }
                )

        return suggestions

    except Exception as e:
        logger.error(f"Error getting command suggestions: {e}", exc_info=True)
        return []


def get_tool_suggestions(input_text: str) -> list[dict[str, str]]:
    """Get tool suggestions using hybrid scoring."""
    try:
        from core.globals import g

        if not hasattr(g, "tool_classes") or not g.tool_classes:  # type: ignore
            return []

        # Build candidates list
        candidates = []
        corpus = []
        for tool_name, tool_class in g.tool_classes.items():  # type: ignore
            try:
                tool_info = tool_class.get_tool_info()
                description = tool_info.get("description", "No description")
                # Truncate long descriptions
                if len(description) > 80:
                    description = description[:77] + "..."

                candidates.append((tool_name, description))
                # Combine name and description for BM25
                text = f"{tool_name} {description}"
                corpus.append(tokenize(text))
            except Exception as e:
                logger.debug(f"Error getting tool info for {tool_name}: {e}")

        # Compute BM25 scores
        bm25_scores = {}
        if BM25_AVAILABLE and corpus:
            try:
                bm25 = BM25Okapi(corpus)
                query_tokens = tokenize(input_text)
                scores = bm25.get_scores(query_tokens)

                # Normalize BM25 scores to 0-100 range
                max_score = float(scores.max()) if scores.max() > 0 else 1.0
                bm25_scores = {
                    tool_name: (float(score) / max_score) * 100
                    for (tool_name, _), score in zip(candidates, scores, strict=False)
                }
            except Exception as e:
                logger.debug(f"BM25 scoring failed: {e}")

        # Score all candidates
        suggestions = []
        for tool_name, description in candidates:
            bm25_score = bm25_scores.get(tool_name, 0)
            candidate_tokens = tokenize(tool_name)

            score = hybrid_score_suggestion(
                input_text, tool_name, candidate_tokens, bm25_score=bm25_score
            )

            # Filter low scores
            if score > 10:
                suggestions.append(
                    {
                        "text": tool_name,
                        "description": description,
                        "type": "tool",
                        "score": score,
                    }
                )

        return suggestions

    except Exception as e:
        logger.error(f"Error getting tool suggestions: {e}", exc_info=True)
        return []


def get_file_path_suggestions(input_text: str) -> list[dict[str, str]]:
    """Get file path suggestions from recent + cached paths using hybrid scoring."""
    try:
        from core.globals import g

        # Collect all candidate paths (deduplicated)
        all_paths = set()
        recent_paths = set()
        cwd = os.getcwd()

        # Add recent paths (with tracking for boost)
        if hasattr(g, "RECENT_FILE_PATHS") and g.RECENT_FILE_PATHS:
            recent_paths = set(g.RECENT_FILE_PATHS[:50])  # Top 50 recent
            all_paths.update(recent_paths)

        # Add cached paths
        if hasattr(g, "CACHED_FILE_PATHS") and g.CACHED_FILE_PATHS:
            all_paths.update(g.CACHED_FILE_PATHS)

        if not all_paths:
            return []

        # Build corpus for BM25
        candidates = []
        corpus = []
        for path in all_paths:
            basename = os.path.basename(path)
            # Index both full path and basename for better matching
            text = f"{path} {basename}"
            candidates.append(path)
            corpus.append(tokenize(text))

        # Compute BM25 scores
        bm25_scores = {}
        if BM25_AVAILABLE and corpus:
            try:
                bm25 = BM25Okapi(corpus)
                query_tokens = tokenize(input_text)
                scores = bm25.get_scores(query_tokens)

                # Normalize to 0-100
                max_score = float(scores.max()) if scores.max() > 0 else 1.0
                bm25_scores = {
                    path: (float(score) / max_score) * 100
                    for path, score in zip(candidates, scores, strict=False)
                }
            except Exception as e:
                logger.debug(f"BM25 scoring failed: {e}")

        # Score all candidates
        suggestions = []
        for path in candidates:
            basename = os.path.basename(path)

            # Score against both full path and basename, take max
            bm25_score = bm25_scores.get(path, 0)

            # Try matching full path
            score_full = hybrid_score_suggestion(
                input_text,
                path,
                tokenize(path),
                bm25_score=bm25_score,
                is_recent_file=path in recent_paths,
                is_cwd_file=path.startswith(cwd),
            )

            # Try matching basename
            score_basename = hybrid_score_suggestion(
                input_text,
                basename,
                tokenize(basename),
                bm25_score=bm25_score,
                is_recent_file=path in recent_paths,
                is_cwd_file=path.startswith(cwd),
            )

            # Take best score
            score = max(score_full, score_basename)

            # Filter low scores
            if score > 10:
                # Shorten path for display
                display_path = path
                if len(display_path) > 60:
                    display_path = "..." + display_path[-57:]

                # Mark recent files in description
                desc_prefix = "Recent: " if path in recent_paths else ""

                suggestions.append(
                    {
                        "text": path,
                        "description": f"{desc_prefix}{display_path}",
                        "type": "file",
                        "score": score,
                    }
                )

        return suggestions

    except Exception as e:
        logger.error(f"Error getting file path suggestions: {e}", exc_info=True)
        return []


async def get_autocomplete_suggestions(request: Request) -> Response:  # type: ignore[valid-type]
    """POST /api/autocomplete

    Get autocomplete suggestions based on current input.

    Request body:
        {
            "input": "current input text",
            "max_suggestions": 5  // optional, default 5
        }

    Response:
        {
            "status": "success",
            "suggestions": [
                {
                    "text": "suggestion text",
                    "description": "what this suggestion does",
                    "type": "command|tool|file"
                }
            ]
        }
    """
    if not AIOHTTP_AVAILABLE or web is None:
        raise RuntimeError("aiohttp not available")

    try:
        # Parse request body
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {
                    "status": "error",
                    "error": "Invalid JSON in request body",
                    "code": "INVALID_REQUEST",
                },
                status=400,
            )

        input_text = data.get("input", "").strip()
        max_suggestions = data.get("max_suggestions", 5)

        # Return empty list if no input provided
        if len(input_text) < 1:
            return web.json_response({"status": "success", "suggestions": []})

        # Collect all suggestions
        all_suggestions = []

        # Commands (if input starts with /)
        if input_text.startswith("/"):
            all_suggestions.extend(get_command_suggestions(input_text))
        else:
            # Tools and file paths for regular input
            all_suggestions.extend(get_tool_suggestions(input_text))
            all_suggestions.extend(get_file_path_suggestions(input_text))

            # Dynamic path suggestions (depth-4 scanning for partial paths)
            extracted_path = extract_path_from_input(input_text)
            if extracted_path:
                logger.debug(
                    f"[PathAutocomplete] Detected path in input: '{extracted_path}'"
                )
                dynamic_path_suggestions = get_dynamic_path_suggestions(
                    extracted_path, max_depth=4, max_results=50
                )
                all_suggestions.extend(dynamic_path_suggestions)

        # Sort by score (descending) and filter by minimum threshold
        all_suggestions.sort(key=lambda s: s["score"], reverse=True)

        # Only return suggestions with score >= 50
        filtered_suggestions = [s for s in all_suggestions if s.get("score", 0) >= 50]
        top_suggestions = filtered_suggestions[:max_suggestions]

        # Round scores to 1 decimal place for cleaner display
        for suggestion in top_suggestions:
            if "score" in suggestion:
                suggestion["score"] = round(suggestion["score"], 1)

        return web.json_response({"status": "success", "suggestions": top_suggestions})

    except Exception as e:
        logger.error(f"Error getting autocomplete suggestions: {e}", exc_info=True)
        return web.json_response(
            {"status": "error", "error": str(e), "code": "INTERNAL_ERROR"}, status=500
        )


async def get_custom_autocomplete_suggestions(request: Request) -> Response:  # type: ignore[valid-type]
    """POST /api/autocomplete/custom

    Get custom LLM-powered autocomplete suggestions using context_chat.
    Triggered after 1 second of user idle time.

    Request body:
        {
            "input": "current input text",
            "max_suggestions": 5  // optional, default 5
        }

    Response:
        {
            "status": "success",
            "suggestions": [
                {
                    "text": "suggestion text",
                    "description": "suggestion description",
                    "type": "custom",
                    "score": 95.0
                }
            ]
        }
    """
    if not AIOHTTP_AVAILABLE or web is None:
        raise RuntimeError("aiohttp not available")

    try:
        # Parse request body
        try:
            data = await request.json()
        except Exception:
            return web.json_response(
                {
                    "status": "error",
                    "error": "Invalid JSON in request body",
                    "code": "INVALID_REQUEST",
                },
                status=400,
            )

        input_text: str = data.get("input", "").strip()
        max_suggestions = data.get("max_suggestions", 5)

        suggestions = []

        if input_text.endswith((".", "!", "?", ";", '"', "'", "*")):
            logger.info(
                f"Custom autocomplete skipped because of detected stop character: '{input_text[-1]}'"
            )
            return web.json_response({"status": "success", "suggestions": []})
        logger.info(f"Custom autocomplete triggered for input: '{input_text}'")

        # Example structure (you'll implement this):

        from core.chat import Chat, Role
        from core.globals import g
        from core.llm_router import LlmRouter, LLMRole

        # Get context from main.py
        # Note: main.py sets g.chat (not g.context_chat)
        context_chat: Optional[Chat] = getattr(g, "chat", None)
        logger.info(
            f"[Autocomplete] context_chat available: {context_chat is not None}"
        )

        # Build your custom chat
        if not context_chat:
            logger.warning(
                "[Autocomplete] No context_chat available, returning empty suggestions"
            )
            return web.json_response({"status": "success", "suggestions": []})

        # Check if autocomplete model is configured
        autocomplete_models = g.autocomplete_llms_list
        if not autocomplete_models or len(autocomplete_models) == 0:
            logger.info(
                "[Autocomplete] No autocomplete model configured, returning empty suggestions"
            )
            return web.json_response({"status": "success", "suggestions": []})

        # TODO: We are for now not including the whole context_history for faster response times, more context could be used. This should be reconsidered after further testing
        # custom_chat = Chat(context_chat.messages[0].content)
        logger.info(
            f"[Autocomplete] Creating custom chat (context has {len(context_chat.messages)} messages)"
        )
        custom_chat = Chat()

        logger.info("[Autocomplete] Added system message to custom chat")

        # Add context messages (last 5 for brevity) while switching roles so the llm will generate an *assistant* tokenized block
        context_messages = context_chat.messages[-3:]
        logger.info(f"[Autocomplete] Adding {len(context_messages)} context messages")
        for i, msg in enumerate(context_messages):
            # Extract only relevant content from XML tags (think, todos, response, markdown visualresponse)
            extracted_text = extract_relevant_content(msg.content)
            custom_chat.add_message(
                Role.USER if msg.role == Role.ASSISTANT else Role.ASSISTANT,
                extracted_text,
            )
        custom_chat.add_message(
            Role.ASSISTANT,
            input_text,
        )

        # Get LLM response using autocomplete role
        llm_router = LlmRouter()
        logger.info(
            f"[Autocomplete] Calling LLM with role=LLMRole.AUTOCOMPLETE for input: '{input_text}'"
        )

        response = await llm_router.generate_completion(
            chat=custom_chat, role=LLMRole.AUTOCOMPLETE, thinking_budget=0
        )

        suggestions = [
            {
                "text": input_text + response,
                "description": "Autocomplete",
                "type": "custom",
                "score": 95.0,
            }
        ]

        return web.json_response({"status": "success", "suggestions": suggestions})

    except Exception as e:
        logger.error(
            f"Error getting custom autocomplete suggestions: {e}", exc_info=True
        )
        return web.json_response(
            {"status": "error", "error": str(e), "code": "INTERNAL_ERROR"}, status=500
        )


def setup_routes(app: Application) -> None:  # type: ignore[valid-type]
    """Register all autocomplete API routes"""
    app.router.add_post("/api/autocomplete", get_autocomplete_suggestions)
    app.router.add_post("/api/autocomplete/custom", get_custom_autocomplete_suggestions)
    logger.info("Autocomplete API routes registered")
