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
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                   SEARCHCONVERSATIONS TOOL - EXTENDED HELP                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

BASIC USAGE:
  <searchconversations>
    <query>your search query</query>
    <top_k>10</top_k>
    <order_by>best_match</order_by>
  </searchconversations>

PURPOSE:
Search through archived chat conversations to find relevant context from previous
discussions, past solutions, troubleshooting sessions, or related conversations.
Uses BM25 ranking algorithm for intelligent relevance scoring.

PARAMETERS:

  <query> (REQUIRED)
    - Your search query string
    - Natural language or keywords
    - Multi-word queries supported
    - Case-insensitive matching
    - Supports fuzzy matching and stemming

    Examples:
    <query>docker deployment issues</query>
    <query>how to implement authentication</query>
    <query>error handling best practices</query>

  <top_k> (OPTIONAL, default: 10)
    - Maximum number of results to return
    - Must be positive integer
    - Automatically capped at 100 to avoid overwhelming output
    - Smaller values for focused results
    - Larger values for comprehensive search

    Examples:
    <top_k>5</top_k>   # Quick overview
    <top_k>20</top_k>  # Thorough search

  <order_by> (OPTIONAL, default: 'best_match')
    - Sort order for results
    - Three modes available:

    1. 'best_match' - Sort by relevance score (highest first)
       Best for: Finding most relevant conversations

    2. 'newest' - Sort by date (most recent first)
       Best for: Finding recent discussions

    3. 'oldest' - Sort by date (oldest first)
       Best for: Historical context or chronological review

SEARCH ALGORITHM:

BM25 (Best Matching 25):
  - Industry-standard ranking algorithm
  - Considers term frequency and document length
  - Applies fuzzy matching for typo tolerance
  - Uses stemming for linguistic variations
  - Weights rare terms higher than common words

This means:
  - "deployment" matches "deploying", "deployed", "deploy"
  - "authentication" matches "authenticate", "auth"
  - Typos like "docekr" still match "docker"
  - Specific technical terms score higher than common words

RESULT FORMAT:

Each result includes:
  - Conversation ID: Unique identifier for the chat
  - Date: When the conversation occurred
  - Messages: Total message count in conversation
  - Relevance Score: BM25 score (higher = more relevant)
  - Preview: First 80 characters of conversation
  - Best Match: Snippet showing why this conversation matched

Example output:
[1] Conversation #a1b2c3d4... (ID: a1b2c3d4-5678-90ef-ghij-klmnopqrstuv)
    Date: 2024-01-15 14:23:45
    Messages: 28
    Relevance Score: 12.45
    Preview: User asked about Docker deployment strategies for microservices...
    Best Match: "We solved the Docker networking issue by configuring bridge mode"

SEARCH STRATEGIES:

1. SPECIFIC QUERIES (Best for targeted search):
   <query>Redis connection timeout error</query>
   <query>React component lifecycle methods</query>
   <query>PostgreSQL indexing performance</query>

2. BROAD QUERIES (Best for exploration):
   <query>authentication</query>
   <query>database optimization</query>
   <query>testing strategies</query>

3. MULTI-TERM QUERIES (Best for context):
   <query>async await error handling Python</query>
   <query>JWT token refresh implementation</query>
   <query>Docker compose production setup</query>

4. NATURAL LANGUAGE (Supported):
   <query>how did we solve the memory leak last time</query>
   <query>what was that bug with the authentication flow</query>
   <query>where did we discuss API rate limiting</query>

COMMON USE CASES:

1. Find past solutions:
   <searchconversations>
     <query>how we fixed CORS issues</query>
     <top_k>3</top_k>
     <order_by>best_match</order_by>
   </searchconversations>

2. Review recent work:
   <searchconversations>
     <query>API development</query>
     <top_k>10</top_k>
     <order_by>newest</order_by>
   </searchconversations>

3. Find related discussions:
   <searchconversations>
     <query>WebSocket implementation</query>
     <top_k>15</top_k>
     <order_by>best_match</order_by>
   </searchconversations>

4. Troubleshooting history:
   <searchconversations>
     <query>database migration errors</query>
     <top_k>5</top_k>
     <order_by>newest</order_by>
   </searchconversations>

5. Learning from past:
   <searchconversations>
     <query>security best practices</query>
     <top_k>20</top_k>
     <order_by>best_match</order_by>
   </searchconversations>

TIPS AND BEST PRACTICES:

1. USE SPECIFIC TECHNICAL TERMS:
   ✓ "Redis cache invalidation"
   ✗ "that thing with the cache"

   More specific terms yield better relevance scores.

2. COMBINE KEYWORDS:
   ✓ "React hooks useState useEffect"
   ✗ "React"

   Multiple relevant keywords improve precision.

3. ADJUST TOP_K BASED ON NEED:
   - Quick reference: top_k=3-5
   - Thorough research: top_k=15-20
   - Comprehensive review: top_k=50+

4. CHOOSE APPROPRIATE SORTING:
   - Known relevant info exists: 'best_match'
   - Recent discussion recall: 'newest'
   - Chronological review: 'oldest'

5. ITERATE YOUR QUERIES:
   If first search doesn't yield results, try:
   - Different terminology (e.g., "auth" vs "authentication")
   - Broader terms (e.g., "database" vs "PostgreSQL")
   - Different phrasing (e.g., "error" vs "issue" vs "problem")

6. USE PREVIEW AND SNIPPETS:
   - Preview gives context of conversation topic
   - Best Match snippet shows exact relevance
   - Use these to decide which conversation to load

LOADING CONVERSATIONS:

After finding relevant conversations, use the conversation ID to load:
  "To load and continue from conversation a1b2c3d4-5678-90ef-ghij-klmnopqrstuv,
   use the conversation ID from the search results."

(Note: The exact method to load conversations depends on your CLI-Agent commands)

PERFORMANCE NOTES:

- Search is fast even with many archived conversations
- BM25 indexing ensures O(log n) complexity
- Results typically return in <100ms for most queries
- Large result sets (top_k>50) may take slightly longer

LIMITATIONS:

- Only searches archived conversations (not current session)
- Cannot search by date range (use order_by + top_k instead)
- Cannot filter by specific metadata (e.g., project, tags)
- No boolean operators (AND/OR/NOT) in queries
- Maximum 100 results (automatically capped)

ERROR MESSAGES:

"Missing required <query> parameter"
  → Must include <query> tag with search text

"<query> cannot be empty"
  → Query must contain at least some text

"<top_k> must be positive"
  → Use positive integers for top_k

"Invalid <order_by> value"
  → Must be 'best_match', 'newest', or 'oldest'

EXAMPLES IN CONTEXT:

1. Debugging a recurring issue:
   "I remember we had this error before..."
   <searchconversations>
     <query>TypeError undefined property</query>
     <top_k>10</top_k>
     <order_by>best_match</order_by>
   </searchconversations>

2. Building on previous work:
   "What did we discuss about API design?"
   <searchconversations>
     <query>RESTful API design patterns</query>
     <top_k>5</top_k>
     <order_by>newest</order_by>
   </searchconversations>

3. Learning from past decisions:
   "Why did we choose that architecture?"
   <searchconversations>
     <query>microservices architecture decision</query>
     <top_k>3</top_k>
     <order_by>best_match</order_by>
   </searchconversations>

RELATED CONCEPTS:

- BM25: Probabilistic ranking algorithm for text retrieval
- Stemming: Reducing words to root form (e.g., "running" → "run")
- Fuzzy matching: Tolerating minor spelling variations
- Relevance score: Numeric measure of query-document similarity

╚══════════════════════════════════════════════════════════════════════════════╝
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
