"""
searchconversations tool - Search through archived chat conversations using BM25 ranking.

This tool allows the agent to search through past conversations to find relevant
context from previous discussions. Results are returned with relevance scores,
snippets, and metadata.

The search uses BM25 ranking algorithm with fuzzy matching and stemming for
robust retrieval across diverse query types.
"""

import re
from typing import Any

from shared.utils.tool_logging import get_logger_for_tool_module

logger = get_logger_for_tool_module(__name__)


class searchconversations:
    """
    Tool for searching archived conversations using BM25 ranking.

    Parameters (via XML):
        query (required): Search query string
        top_k (optional): Maximum number of results to return (default: 10)
        order_by (optional): Sort order - 'best_match' (default), 'newest', or 'oldest'
    """

    @staticmethod
    def get_delim() -> str:
        """Returns the XML tag name for this tool."""
        return "searchconversations"

    @staticmethod
    def get_tool_info() -> dict:
        """Returns tool metadata for injection into system prompt."""
        return {
            "name": "searchconversations",
            "description": (
                "Search archived conversations with BM25 ranking. Returns snippets, scores, "
                "and metadata. Supports relevance/date sorting. "
                "Use <help>searchconversations</help> for query optimization tips."
            ),
            "example": (
                "<searchconversations><query>docker deployment</query><top_k>5</top_k>"
                "<order_by>best_match</order_by></searchconversations>"
            ),
        }

    @staticmethod
    def get_detailed_tool_info() -> str:
        """Provides comprehensive help documentation for this tool."""
        return """SEARCHCONVERSATIONS - EXTENDED HELP

USAGE:
  <searchconversations><query>authentication bug</query></searchconversations>

PURPOSE:
Search through archived conversation history. Returns matching conversations with
message previews and timestamps.

PARAMETERS:
  <query> - Search query (required, case-insensitive substring match)

EXAMPLES:
  Basic: <searchconversations><query>login</query></searchconversations>
  Specific: <searchconversations><query>TypeError in parse_json</query></searchconversations>

SEARCH BEHAVIOR:
  • Case-insensitive substring matching
  • Searches message content only
  • Returns up to 10 most recent matches
  • Shows message preview (first 200 chars)
  • Includes timestamp and conversation ID

RESULT FORMAT:
  [2024-01-15 14:30] Conversation ID: abc123
  Preview: "Working on authentication bug fix..."
  Messages: 15

WORKFLOW:
  1. Search: <searchconversations><query>bug fix</query></searchconversations>
  2. Review results
  3. Load specific conversation using UI or chat history

TIPS:
  • Use specific terms for better results
  • Include error messages or function names
  • Try synonyms if no results
  • Recent conversations ranked higher

LIMITATIONS:
  • Max 10 results returned
  • Substring match only (no regex)
  • Searches content, not filenames
  • Requires conversation archiving enabled

COMMON ERRORS:
  "No conversations found" → Check query spelling or try broader term
  "Search failed" → Conversation storage may not be configured

RELATED: <help>searchfile</help> <help>grepsearch</help>
"""

    @staticmethod
    def _parse_parameters(content: str) -> dict[str, Any]:
        """
        Parse XML parameters from tool input.

        Args:
            content: XML content between opening/closing tags

        Returns:
            Dictionary with parsed parameters:
                - query: Search query (required)
                - top_k: Max results (optional, default 10)
                - order_by: Sort order (optional, default 'best_match')

        Raises:
            ValueError: If required parameters missing or invalid
        """
        # Extract query (required)
        query_match = re.search(r"<query>(.*?)</query>", content, re.DOTALL)
        if not query_match:
            raise ValueError("Missing required <query> parameter")

        query = query_match.group(1).strip()
        if not query:
            raise ValueError("<query> cannot be empty")

        # Extract top_k (optional, default 10)
        top_k = 10
        top_k_match = re.search(r"<top_k>(-?\d+)</top_k>", content, re.DOTALL)
        if top_k_match:
            top_k = int(top_k_match.group(1))
            if top_k <= 0:
                raise ValueError("<top_k> must be positive")
            if top_k > 100:
                logger.warning(f"Capping top_k from {top_k} to 100 to avoid overwhelming results")
                top_k = 100

        # Extract order_by (optional, default 'best_match')
        order_by = "best_match"
        order_by_match = re.search(r"<order_by>(.*?)</order_by>", content, re.DOTALL)
        if order_by_match:
            order_by = order_by_match.group(1).strip().lower()
            valid_orders = ["best_match", "newest", "oldest"]
            if order_by not in valid_orders:
                raise ValueError(
                    f"Invalid <order_by> value: '{order_by}'. "
                    f"Must be one of: {', '.join(valid_orders)}"
                )

        logger.debug(
            f"Parsed parameters: query='{query[:50]}...', top_k={top_k}, order_by={order_by}"
        )

        return {"query": query, "top_k": top_k, "order_by": order_by}

    @staticmethod
    def _search_conversations(
        query: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """
        Search archived conversations using Chat.search_chats() method.

        Args:
            query: Search query string
            top_k: Maximum number of results to return

        Returns:
            List of search results with metadata and scores
        """
        try:
            from core.chat import Chat

            logger.info(f"Searching conversations with query: '{query}' (top_k={top_k})")
            results = Chat.search_chats(query, top_k=top_k)
            logger.info(f"Found {len(results)} matching conversations")
            return results

        except Exception as e:
            logger.error(f"Error searching conversations: {e}", exc_info=True)
            raise

    @staticmethod
    def _apply_sorting(results: list[dict[str, Any]], order_by: str) -> list[dict[str, Any]]:
        """
        Apply sorting to search results.

        Args:
            results: List of search results
            order_by: Sort order ('best_match', 'newest', or 'oldest')

        Returns:
            Sorted list of results
        """
        if order_by == "newest":
            sorted_results = sorted(results, key=lambda x: x.get("timestamp", 0), reverse=True)
            logger.debug("Sorted results by newest (descending timestamp)")
        elif order_by == "oldest":
            sorted_results = sorted(results, key=lambda x: x.get("timestamp", 0))
            logger.debug("Sorted results by oldest (ascending timestamp)")
        else:  # best_match (default)
            sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            logger.debug("Sorted results by best_match (descending score)")

        return sorted_results

    @staticmethod
    def _format_result(result: dict[str, Any], index: int) -> str:
        """
        Format a single search result for display.

        Args:
            result: Search result dictionary
            index: Result number (1-based)

        Returns:
            Formatted string representation of the result
        """
        from datetime import datetime

        chat_id = result.get("id", "unknown")
        timestamp = result.get("timestamp", 0)
        preview = result.get("preview", "No preview available")
        message_count = result.get("message_count", 0)
        score = result.get("score", 0.0)
        snippet = result.get("search_snippet", "")

        # Format timestamp
        if timestamp:
            dt = datetime.fromtimestamp(timestamp)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_str = "Unknown date"

        # Truncate preview to avoid overwhelming output
        if len(preview) > 80:
            preview = preview[:77] + "..."

        # Build formatted output
        lines = [
            f"\n[{index}] Conversation #{chat_id[:8]}... (ID: {chat_id})",
            f"    Date: {time_str}",
            f"    Messages: {message_count}",
            f"    Relevance Score: {score:.2f}",
            f"    Preview: {preview}",
        ]

        if snippet:
            lines.append(f"    Best Match: \"{snippet}\"")

        return "\n".join(lines)

    @staticmethod
    def _format_output(results: list[dict[str, Any]], query: str, order_by: str) -> str:
        """
        Format search results for LLM consumption.

        Args:
            results: List of search results
            query: Original search query
            order_by: Sort order applied

        Returns:
            Formatted string for output to agent
        """
        if not results:
            return f"No archived conversations found matching '{query}'"

        # Build output
        lines = [
            f"Found {len(results)} archived conversation(s) matching '{query}' "
            f"(sorted by {order_by}):",
            "━" * 75,
        ]

        for idx, result in enumerate(results, 1):
            lines.append(searchconversations._format_result(result, idx))

        lines.append("━" * 75)
        lines.append(
            "\nTo load and continue from a specific conversation, "
            "use the conversation ID from the result above."
        )

        return "\n".join(lines)

    @staticmethod
    def run(content: str) -> str:
        """
        Execute the search conversations tool.

        Parses XML parameters, searches archived conversations, applies sorting,
        and returns formatted results.

        Args:
            content: XML content between opening/closing tags

        Returns:
            Formatted search results or error message
        """
        try:
            logger.debug("searchconversations tool invoked")

            # Parse parameters
            params = searchconversations._parse_parameters(content)
            query = params["query"]
            top_k = params["top_k"]
            order_by = params["order_by"]

            # Perform search
            results = searchconversations._search_conversations(query, top_k)

            # Apply sorting
            sorted_results = searchconversations._apply_sorting(results, order_by)

            # Format and return output
            output = searchconversations._format_output(sorted_results, query, order_by)
            logger.info(f"Returning {len(sorted_results)} results to agent")
            return output

        except ValueError as e:
            logger.warning(f"Invalid parameters: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in searchconversations: {e}", exc_info=True)
            return f"Error: Failed to search conversations: {str(e)}"
