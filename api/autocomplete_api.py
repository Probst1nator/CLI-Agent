"""
Autocomplete API endpoints for CLI-Agent.

Provides intelligent autocomplete suggestions using hybrid fuzzy matching (rapidfuzz + BM25):
- Commands (e.g., /exit, /regenerate)
- Tools (e.g., bash, readfile)
- File paths (recent + all cached)
"""

import logging
import os
from typing import Any, List, Dict

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None
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


def tokenize(text: str) -> List[str]:
    """Simple tokenization for BM25"""
    return text.lower().replace('/', ' ').replace('_', ' ').replace('-', ' ').split()


def hybrid_score_suggestion(
    input_text: str,
    candidate: str,
    candidate_tokens: List[str],
    bm25_score: float = 0.0,
    is_recent_file: bool = False,
    is_cwd_file: bool = False
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
        words = candidate_lower.split('/')
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


def get_command_suggestions(input_text: str) -> List[Dict[str, str]]:
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
                max_score = max(scores) if max(scores) > 0 else 1
                bm25_scores = {
                    alias: (score / max_score) * 100
                    for (alias, _), score in zip(candidates, scores)
                }
            except Exception as e:
                logger.debug(f"BM25 scoring failed: {e}")

        # Score all candidates with hybrid approach
        suggestions = []
        for alias, description in candidates:
            bm25_score = bm25_scores.get(alias, 0)
            candidate_tokens = tokenize(alias)

            score = hybrid_score_suggestion(
                input_text,
                alias,
                candidate_tokens,
                bm25_score=bm25_score
            )

            # Filter low scores
            if score > 10:
                suggestions.append({
                    'text': alias,
                    'description': description,
                    'type': 'command',
                    'score': score
                })

        return suggestions

    except Exception as e:
        logger.error(f"Error getting command suggestions: {e}", exc_info=True)
        return []


def get_tool_suggestions(input_text: str) -> List[Dict[str, str]]:
    """Get tool suggestions using hybrid scoring."""
    try:
        from core.globals import g

        if not hasattr(g, 'tool_classes') or not g.tool_classes:
            return []

        # Build candidates list
        candidates = []
        corpus = []
        for tool_name, tool_class in g.tool_classes.items():
            try:
                tool_info = tool_class.get_tool_info()
                description = tool_info.get('description', 'No description')
                # Truncate long descriptions
                if len(description) > 80:
                    description = description[:77] + '...'

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
                max_score = max(scores) if max(scores) > 0 else 1
                bm25_scores = {
                    tool_name: (score / max_score) * 100
                    for (tool_name, _), score in zip(candidates, scores)
                }
            except Exception as e:
                logger.debug(f"BM25 scoring failed: {e}")

        # Score all candidates
        suggestions = []
        for tool_name, description in candidates:
            bm25_score = bm25_scores.get(tool_name, 0)
            candidate_tokens = tokenize(tool_name)

            score = hybrid_score_suggestion(
                input_text,
                tool_name,
                candidate_tokens,
                bm25_score=bm25_score
            )

            # Filter low scores
            if score > 10:
                suggestions.append({
                    'text': tool_name,
                    'description': description,
                    'type': 'tool',
                    'score': score
                })

        return suggestions

    except Exception as e:
        logger.error(f"Error getting tool suggestions: {e}", exc_info=True)
        return []


def get_file_path_suggestions(input_text: str) -> List[Dict[str, str]]:
    """Get file path suggestions from recent + cached paths using hybrid scoring."""
    try:
        from core.globals import g

        # Collect all candidate paths (deduplicated)
        all_paths = set()
        recent_paths = set()
        cwd = os.getcwd()

        # Add recent paths (with tracking for boost)
        if hasattr(g, 'RECENT_FILE_PATHS') and g.RECENT_FILE_PATHS:
            recent_paths = set(g.RECENT_FILE_PATHS[:50])  # Top 50 recent
            all_paths.update(recent_paths)

        # Add cached paths
        if hasattr(g, 'CACHED_FILE_PATHS') and g.CACHED_FILE_PATHS:
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
                max_score = max(scores) if max(scores) > 0 else 1
                bm25_scores = {
                    path: (score / max_score) * 100
                    for path, score in zip(candidates, scores)
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
                is_cwd_file=path.startswith(cwd)
            )

            # Try matching basename
            score_basename = hybrid_score_suggestion(
                input_text,
                basename,
                tokenize(basename),
                bm25_score=bm25_score,
                is_recent_file=path in recent_paths,
                is_cwd_file=path.startswith(cwd)
            )

            # Take best score
            score = max(score_full, score_basename)

            # Filter low scores
            if score > 10:
                # Shorten path for display
                display_path = path
                if len(display_path) > 60:
                    display_path = '...' + display_path[-57:]

                # Mark recent files in description
                desc_prefix = 'Recent: ' if path in recent_paths else ''

                suggestions.append({
                    'text': path,
                    'description': f'{desc_prefix}{display_path}',
                    'type': 'file',
                    'score': score
                })

        return suggestions

    except Exception as e:
        logger.error(f"Error getting file path suggestions: {e}", exc_info=True)
        return []


async def get_autocomplete_suggestions(request: Any) -> Any:
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
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not available")

    try:
        # Parse request body
        try:
            data = await request.json()
        except Exception:
            return web.json_response({
                'status': 'error',
                'error': 'Invalid JSON in request body',
                'code': 'INVALID_REQUEST'
            }, status=400)

        input_text = data.get('input', '').strip()
        max_suggestions = data.get('max_suggestions', 5)

        # Return empty list if no input provided
        if len(input_text) < 1:
            return web.json_response({
                'status': 'success',
                'suggestions': []
            })

        # Collect all suggestions
        all_suggestions = []

        # Commands (if input starts with /)
        if input_text.startswith('/'):
            all_suggestions.extend(get_command_suggestions(input_text))
        else:
            # Tools and file paths for regular input
            all_suggestions.extend(get_tool_suggestions(input_text))
            all_suggestions.extend(get_file_path_suggestions(input_text))

        # Sort by score (descending) and filter by minimum threshold
        all_suggestions.sort(key=lambda s: s['score'], reverse=True)

        # Only return suggestions with score >= 50
        filtered_suggestions = [s for s in all_suggestions if s.get('score', 0) >= 50]
        top_suggestions = filtered_suggestions[:max_suggestions]

        # Round scores to 1 decimal place for cleaner display
        for suggestion in top_suggestions:
            if 'score' in suggestion:
                suggestion['score'] = round(suggestion['score'], 1)

        return web.json_response({
            'status': 'success',
            'suggestions': top_suggestions
        })

    except Exception as e:
        logger.error(f"Error getting autocomplete suggestions: {e}", exc_info=True)
        return web.json_response({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }, status=500)


def setup_routes(app: Any):
    """Register all autocomplete API routes"""
    app.router.add_post('/api/autocomplete', get_autocomplete_suggestions)
    logger.info("Autocomplete API routes registered")
