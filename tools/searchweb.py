# tools/main_cli_agent/tools/searchweb.py
"""
This file implements the 'searchweb' tool. It allows the agent to perform
a web search using the Brave Search API. It supports different search types
like text, images, and cited results.
"""
import asyncio
import datetime
import html
import os
import re
import sys

from dotenv import load_dotenv

from core.chat import Chat, Role
from core.llm_router import LlmRouter

# Load environment variables
load_dotenv()

# Ensure proper imports when running as module or script
if __name__ == '__main__':
    # Add project root to path for direct execution
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from shared.utils.memory_utils import save_to_memory_file
from shared.utils.search.brave_search import BraveSearchAPI
from shared.utils.search.search_vector_db import SearchResultsVectorDB


class searchweb:
    """
    A tool to perform a web search with different content types.
    Maintains a persistent knowledge base of all searches with smart freshness checking.
    """
    MAX_RESULTS = 10
    DEFAULT_FRESHNESS_HOURS = 12  # Prefer fresh API results if cached results are older than this

    # Initialize vector database for caching and semantic search
    _vector_db = None

    @classmethod
    def _get_vector_db(cls):
        """Get or create the vector database instance"""
        if cls._vector_db is None:
            cls._vector_db = SearchResultsVectorDB()
        return cls._vector_db


    @staticmethod
    def get_delim() -> str:
        return 'searchweb'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool."""
        return {
            "name": "searchweb",
            "description": (
                f"Performs a web search using a specified query and type. "
                f"Supported types: 'text' (default), 'images', 'files', 'any'. "
                f"Returns up to {searchweb.MAX_RESULTS} results. "
                f"Maintains a persistent knowledge base with smart freshness checking. "
                f"Use <help>searchweb</help> for advanced options."
            ),
            "example": (
                "<searchweb><query>How to use curl</query></searchweb>\n"
                "<searchweb><query>golden retriever puppy</query><type>images</type></searchweb>"
            )
        }

    @staticmethod
    def get_detailed_tool_info() -> str:
        """Provides comprehensive help documentation for this tool."""
        return f"""SEARCHWEB - EXTENDED HELP

USAGE:
  <searchweb><query>search terms</query><type>text</type></searchweb>

PURPOSE:
  Performs web searches using Brave Search API with intelligent caching and persistent knowledge base.
  All searches are permanently stored with timestamps for future retrieval and analysis.

PARAMETERS:
  <query> - Search query (required)
  <type> - Search type (optional, default "text")
  <freshness_hours> - Cache freshness threshold in hours (optional, default {searchweb.DEFAULT_FRESHNESS_HOURS})

SEARCH TYPES:
  text (default) - Web search with LLM-generated summary
  images - Image search with direct URLs and thumbnails
  files - File search targeting PDF, DOC, TXT formats
  any - General web results with structured citations

PERSISTENT KNOWLEDGE BASE:
  • All search results stored permanently in local vector database
  • Builds searchable knowledge base over time (never expires)
  • Each result includes timestamp: "Source: URL | Title: ... | Accessed: YYYY-MM-DD HH:MM:SS"
  • Agent can see dates and judge if information is current enough

FRESHNESS THRESHOLD (Default: {searchweb.DEFAULT_FRESHNESS_HOURS} hours):
  • Cached results older than threshold trigger fresh API fetch
  • Fresh results combined with relevant cached results for comprehensive coverage
  • Newer cached results (within threshold) used without API call
  • Override threshold: <freshness_hours>24</freshness_hours>

EXAMPLES:
  Basic search:
    <searchweb><query>Python async best practices</query></searchweb>

  Image search:
    <searchweb><query>sunset photography</query><type>images</type></searchweb>

  Time-sensitive (short freshness window):
    <searchweb><query>Bitcoin price analysis</query><freshness_hours>1</freshness_hours></searchweb>

  Long freshness window (more caching):
    <searchweb><query>Python documentation</query><freshness_hours>168</freshness_hours></searchweb>

  Maximum results:
    <searchweb><query>machine learning tutorials</query><max_results>15</max_results></searchweb>

INTELLIGENT CACHING:
  • Hybrid search: Vector similarity (semantic) + BM25 (keyword)
  • Reciprocal Rank Fusion combines both ranking methods
  • Distance threshold ≤ 0.5 for semantic matching
  • Fresh results automatically prioritized in ranking
  • Duplicate URLs filtered across cached + fresh results

RESULT COMPOSITION:
  • Always checks cache first for relevant results
  • If cached results < MAX_RESULTS OR too old → API call
  • Combines fresh + cached results, sorted by relevance and freshness
  • Final set limited to MAX_RESULTS ({searchweb.MAX_RESULTS} default)

TEXT SEARCH SUMMARIES:
  • LLM generates concise summary from top results
  • Summaries cached to avoid regeneration
  • Full search results saved to memory file if substantial (>1000 chars)
  • Memory file path provided for detailed exploration

WORKFLOW EXAMPLES:
  Quick search:
    <searchweb><query>React hooks tutorial</query></searchweb>

  Refine search:
    <searchweb><query>React useEffect cleanup pattern</query></searchweb>

  Breaking news (1-hour freshness):
    <searchweb><query>latest tech announcements</query><freshness_hours>1</freshness_hours></searchweb>

  Documentation (7-day freshness):
    <searchweb><query>TypeScript generics guide</query><freshness_hours>168</freshness_hours></searchweb>

ADVANCED TIPS:
  • Set freshness_hours=0 to force fresh API results only (ignores old cache)
  • Set freshness_hours=999999 to prefer cache even if very old
  • Agent can read timestamps in results to assess information currency
  • Knowledge base grows with every search, improving future searches
  • Semantic search finds conceptually similar past searches

FRESHNESS GUIDELINES:
  • News/prices/events: 1-6 hours
  • Technical trends: 24-72 hours
  • Documentation: 168+ hours (1 week)
  • Historical info: 999999 hours (always use cache)

RELATED TOOLS:
  <help>help</help> - List all available tools
  <readfile> - Read memory files containing full search results
"""

    @staticmethod
    def _remove_tool_calls(text: str) -> str:
        decoded_text = html.unescape(text)
        clean_text = re.sub(r'<.*?>', '', decoded_text)
        return clean_text.strip()

    @staticmethod
    def result_to_source(result: dict, date: str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')) -> str:
        snippets = result.get('extra_snippets', [])
        title = result.get('title', [])
        url = result.get('url', [])
        if not snippets and result.get('description'):
            snippets = [result.get('description')]
        if snippets:
            clean_snippet = searchweb._remove_tool_calls("\n".join(snippets))
            return f"Source: {url} | Title: {title} | Accessed: {date}\n{clean_snippet}"
        return ""

    @staticmethod
    async def _format_text_summary(results: list[dict], query: str) -> str:
        """ The original functionality: returns a single condensed string. """
        if not results:
            return "No web search results found."

        # Check for existing summary to avoid duplicates
        vector_db = searchweb._get_vector_db()
        existing_summary = vector_db.get_summary_by_query(query, 'text')
        if existing_summary:
            print("✅ Using cached LLM summary (skipping generation)")
            return f"Web search summary: {existing_summary}"
        else:
            print("🤖 No cached summary found, generating new LLM summary")


        sourced_snippets = []

        for i, result in enumerate(results):
            source_content = searchweb.result_to_source(result)
            sourced_snippets.append(source_content)

        if not sourced_snippets:
            return "No useful information found."

        joined_text = " ".join(sourced_snippets)

        chat = Chat()
        chat.add_message(Role.USER, f"The search query is: {query}\nJudge and compare these different web sources and summarize: {joined_text}")

        # Use await for async LLM completion (force_preferred_model=False to allow any free model)
        summary = await LlmRouter.generate_completion(chat, force_preferred_model=False)

        # Store the generated summary in the vector database with enhanced metadata
        try:
            vector_db = searchweb._get_vector_db()

            # Collect source IDs for audit trail
            source_ids = []
            for result in results:
                # Create the same content hash as used in store_search_results
                title = result.get('title', '')
                description = result.get('description', '')
                snippets = result.get('extra_snippets', [])
                snippet_text = ' '.join(snippets) if snippets else ''
                content = f"{title} {description} {snippet_text}".strip()
                if content:
                    content_hash = vector_db._create_content_hash(content)
                    source_ids.append(f"result_{content_hash}")

            # Prepare generation info with actual model used
            generation_info = {
                'method': 'llm',
                'model': LlmRouter.get_last_used_model()
            }

            if vector_db.store_summary(query, 'text', summary, source_ids, generation_info):
                print(f"💾 Stored LLM summary in vector database (linked to {len(source_ids)} source documents)")
        except Exception as e:
            print(f"⚠️ Failed to store summary in vector database: {e}")

        # Save full search results to memory file for detailed reference
        memory_file_path = None
        if len(joined_text) > 1000:  # Save to memory if substantial content
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
            memory_file_path = save_to_memory_file(
                joined_text,
                f"searchweb_{query_hash}"
            )

        # Build the response
        response = f"Web search summary: {summary}"

        # Add memory file reference if created
        if memory_file_path:
            response += f"\n\n📁 Full search results saved to: {memory_file_path}"
            response += f"\nUse <readfile>{memory_file_path}</readfile> to read all source details."

        return response

    @staticmethod
    def _format_cited_text(results: list) -> str:
        """ Returns a numbered list with titles, URLs, and snippets. """
        if not results:
            return "No web search results found."

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = searchweb._remove_tool_calls(result.get('title', 'No Title'))
            url = result.get('url', 'No URL')
            snippet = searchweb._remove_tool_calls(result.get('description', 'No snippet available.'))
            formatted_results.append(f"Result {i}:\n  Title: {title}\n  URL: {url}\n  Snippet: {snippet}")

        return "\n\n".join(formatted_results)

    @staticmethod
    def _format_image_results(results: list) -> str:
        """ Returns a numbered list of images with titles and direct URLs. """
        if not results:
            return "No image results found."

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = searchweb._remove_tool_calls(result.get('title', 'No Title'))

            # Get the actual image URL from properties.url (direct image)
            # The top-level 'url' field contains search page URLs, not image URLs
            properties = result.get('properties', {})
            image_url = properties.get('url', 'No direct image URL')

            # Get thumbnail URL as fallback
            thumbnail = result.get('thumbnail', {})
            thumbnail_url = thumbnail.get('src', 'No thumbnail')

            # Get source page from meta_url or top-level url
            meta_url = result.get('meta_url', {})
            source_page = meta_url.get('netloc', result.get('url', 'No source page'))

            # Format the result with both direct image URL and thumbnail
            formatted_results.append(
                f"Image {i}:\n"
                f"  Title: {title}\n"
                f"  Direct Image URL: {image_url}\n"
                f"  Thumbnail URL: {thumbnail_url}\n"
                f"  Source: {source_page}"
            )

        return "\n\n".join(formatted_results)

    @staticmethod
    async def run(content: str) -> str:
        """
        Executes a web search based on the provided query and type.
        """
        try:
            query_match = re.search(r'<query>(.*?)</query>', content, re.DOTALL)
            type_match = re.search(r'<type>(.*?)</type>', content, re.DOTALL)
            freshness_match = re.search(r'<freshness_hours>(.*?)</freshness_hours>', content, re.DOTALL)
            max_results_match = re.search(r'<max_results>(.*?)</max_results>', content, re.DOTALL)

            if not query_match:
                # Fallback for old format
                query = content.strip()
                search_type = 'text'
            else:
                query = query_match.group(1).strip()
                search_type = type_match.group(1).strip().lower() if type_match else 'text'

            # Parse freshness threshold (default: 12 hours)
            freshness_hours = searchweb.DEFAULT_FRESHNESS_HOURS
            if freshness_match:
                try:
                    freshness_hours = float(freshness_match.group(1).strip())
                    print(f"🕐 Using custom freshness threshold: {freshness_hours} hours")
                except ValueError:
                    print(f"⚠️ Invalid freshness_hours value, using default: {searchweb.DEFAULT_FRESHNESS_HOURS} hours")

            # Parse max_results (default: MAX_RESULTS, capped at 20)
            max_results = searchweb.MAX_RESULTS
            if max_results_match:
                try:
                    max_results = min(int(max_results_match.group(1).strip()), 20)
                    print(f"📊 Requesting {max_results} results (capped at 20)")
                except ValueError:
                    print(f"⚠️ Invalid max_results value, using default: {searchweb.MAX_RESULTS}")

            if not query:
                return "Error: No search query provided in the <query> tag."

            print(f"🔎 Performing '{search_type}' search for: '{query}'...")

            # Get vector database instance
            vector_db = searchweb._get_vector_db()

            results_list = []
            need_brave_api = True

            # Search in vector database for similar results (using distance threshold)
            cached_results_with_distances = vector_db.search_similar_results(query, search_type, max_results, 0.5)

            if cached_results_with_distances:
                # Filter by freshness threshold
                from datetime import datetime, timezone, timedelta
                freshness_cutoff = datetime.now(timezone.utc) - timedelta(hours=freshness_hours)

                fresh_cached = []
                stale_cached = []

                for metadata, distance in cached_results_with_distances:
                    # Parse timestamp from metadata
                    timestamp_str = metadata.get('timestamp', '')
                    try:
                        if timestamp_str:
                            # Handle both ISO format and numeric timestamp
                            if isinstance(timestamp_str, str):
                                result_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            else:
                                result_time = datetime.fromtimestamp(timestamp_str, tz=timezone.utc)

                            if result_time >= freshness_cutoff:
                                fresh_cached.append((metadata, distance))
                            else:
                                stale_cached.append((metadata, distance))
                        else:
                            # No timestamp, treat as stale
                            stale_cached.append((metadata, distance))
                    except (ValueError, TypeError):
                        # Invalid timestamp, treat as stale
                        stale_cached.append((metadata, distance))

                # Calculate distance details (lower = better)
                distances = [distance for _, distance in cached_results_with_distances]
                avg_distance = sum(distances) / len(distances)
                min_distance = min(distances)  # Best match
                max_distance = max(distances)  # Worst match

                print(f"📊 Found {len(cached_results_with_distances)} cached results (distance ≤ 0.5)")
                print(f"   └─ Distance scores: avg={avg_distance:.3f}, best={min_distance:.3f}, worst={max_distance:.3f}")
                print(f"   └─ Freshness: {len(fresh_cached)} within {freshness_hours}h, {len(stale_cached)} older")

                # Add fresh cached results to results list
                for metadata, distance in fresh_cached:
                    metadata['is_cached'] = True
                    metadata['similarity_score'] = 1 - distance  # Convert to similarity for compatibility
                    results_list.append(metadata)

                # Skip API call only if we have enough fresh cached results
                if len(fresh_cached) >= max_results:
                    need_brave_api = False
                    print(f"✅ Using {len(fresh_cached)} fresh cached results only (no API call needed)")
                elif len(fresh_cached) > 0:
                    print(f"🔄 Using {len(fresh_cached)} fresh cached + fetching more from API")
                else:
                    print(f"⏰ All {len(stale_cached)} cached results are stale (>{freshness_hours}h old), fetching fresh from API")
            else:
                print("📊 No cached results found with distance ≤ 0.5")

            # If we need more results or have no cached results, call Brave API
            if need_brave_api:
                api_key = os.getenv("BRAVE_API_KEY")
                if not api_key:
                    if results_list:
                        print("⚠️ No BRAVE_API_KEY found, using cached results only")
                        need_brave_api = False
                    else:
                        return "Error: BRAVE_API_KEY environment variable is not set."

                if need_brave_api:
                    print("🌐 Fetching additional results from Brave API...")

                    api_endpoint = "web"
                    brave_query = query

                    if search_type == 'images':
                        api_endpoint = "images"
                    elif search_type == 'files':
                        brave_query = f"filetype:pdf OR filetype:doc OR filetype:txt {query}"

                    params = {"q": brave_query}
                    # Run blocking API call in a thread to avoid blocking
                    results_json = await asyncio.to_thread(
                        BraveSearchAPI.search, api_key, api_endpoint, params
                    )

                    if results_json:
                        # Parse results based on endpoint type
                        brave_results = []
                        if api_endpoint == "web":
                            brave_results = results_json.get('web', {}).get('results', [])
                        elif api_endpoint == "images":
                            brave_results = results_json.get('results', [])
                        else:
                            # For other endpoints, try both locations
                            brave_results = results_json.get('results', []) or results_json.get('web', {}).get('results', [])

                        if brave_results:
                            print(f"🌐 Retrieved {len(brave_results)} fresh results from Brave API")

                            # Store new results in vector database for future searches
                            if vector_db.store_search_results(query, search_type, brave_results, api_endpoint):
                                print("💾 Stored new results in vector database for future use")

                            # Combine cached and fresh results, avoiding duplicates
                            seen_urls = {result.get('url', '') for result in results_list}
                            fresh_added = 0
                            duplicates_skipped = 0

                            for brave_result in brave_results:
                                url = brave_result.get('url', '')
                                if url not in seen_urls:
                                    # Add flag to identify fresh results and assign high score
                                    brave_result['is_cached'] = False
                                    brave_result['similarity_score'] = 1.0  # Fresh results get max score
                                    results_list.append(brave_result)
                                    seen_urls.add(url)
                                    fresh_added += 1
                                else:
                                    duplicates_skipped += 1

                            if duplicates_skipped > 0:
                                print(f"   └─ Added {fresh_added} fresh results, skipped {duplicates_skipped} duplicates")

                            # Debug: Log the structure of the first result for images
                            if search_type == 'images' and brave_results:
                                print("DEBUG: First image result structure:")
                                first_result = brave_results[0]
                                print(f"  Top-level keys: {list(first_result.keys())}")
                                if 'properties' in first_result:
                                    print(f"  Properties keys: {list(first_result['properties'].keys())}")
                                if 'thumbnail' in first_result:
                                    print(f"  Thumbnail keys: {list(first_result['thumbnail'].keys())}")
                                if 'meta_url' in first_result:
                                    print(f"  Meta URL keys: {list(first_result['meta_url'].keys())}")
                                print("  Sample URLs:")
                                print(f"    url: {first_result.get('url', 'None')}")
                                print(f"    properties.url: {first_result.get('properties', {}).get('url', 'None')}")
                                print(f"    thumbnail.src: {first_result.get('thumbnail', {}).get('src', 'None')}")

            # Intelligent ranking: prioritize fresh results, then by similarity score
            results_list.sort(key=lambda r: (
                not r.get('is_cached', False),  # Fresh results first (not cached = True = higher priority)
                r.get('similarity_score', 0)   # Then by similarity score descending
            ), reverse=True)

            # Limit total results to MAX_RESULTS AFTER ranking
            results_list = results_list[:searchweb.MAX_RESULTS]

            if not results_list:
                return f"No results found for '{query}' (type: {search_type})."

            # Final summary of result composition
            cached_count = sum(1 for r in results_list if r.get('is_cached', False))
            fresh_count = len(results_list) - cached_count
            print(f"🎯 Final result set: {len(results_list)} total ({fresh_count} fresh + {cached_count} cached)")

            if search_type == 'text':
                return await searchweb._format_text_summary(results_list, query)
            elif search_type == 'images':
                # Check for existing image summary to avoid duplicates
                existing_summary = vector_db.get_summary_by_query(query, search_type)
                if existing_summary:
                    print("✅ Found and using existing image summary from vector database")
                    return existing_summary

                result = searchweb._format_image_results(results_list)
                # Store image list summary in vector database with enhanced metadata
                try:
                    # Collect source IDs for audit trail
                    source_ids = []
                    for res in results_list:
                        title = res.get('title', '')
                        description = res.get('description', '')
                        content = f"{title} {description}".strip()
                        if content:
                            content_hash = vector_db._create_content_hash(content)
                            source_ids.append(f"result_{content_hash}")

                    generation_info = {
                        'method': 'image_list',
                        'model': 'N/A'  # Image lists are formatted, not generated by LLM
                    }

                    if vector_db.store_summary(query, search_type, result, source_ids, generation_info):
                        print("💾 Stored image list summary in vector database")
                except Exception as e:
                    print(f"⚠️ Failed to store image summary in vector database: {e}")
                return result
            elif search_type in ['files', 'any']:
                # Check for existing summary to avoid duplicates
                existing_summary = vector_db.get_summary_by_query(query, search_type)
                if existing_summary:
                    print("✅ Found and using existing summary from vector database")
                    return existing_summary

                result = searchweb._format_cited_text(results_list)
                # Store summary in vector database with enhanced metadata
                try:
                    # Collect source IDs for audit trail
                    source_ids = []
                    for res in results_list:
                        title = res.get('title', '')
                        description = res.get('description', '')
                        content = f"{title} {description}".strip()
                        if content:
                            content_hash = vector_db._create_content_hash(content)
                            source_ids.append(f"result_{content_hash}")

                    generation_info = {
                        'method': 'structured_list',
                        'model': 'N/A'  # Structured lists are formatted, not generated by LLM
                    }

                    if vector_db.store_summary(query, search_type, result, source_ids, generation_info):
                        print("💾 Stored summary in vector database")
                except Exception as e:
                    print(f"⚠️ Failed to store summary in vector database: {e}")
                return result
            else:
                return f"Error: Unknown search type '{search_type}'. Use 'text', 'images', 'files', or 'any'."

        except Exception as e:
            return f"An unexpected error occurred in the searchweb tool: {str(e)}"