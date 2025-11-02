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

from shared.utils.search.brave_search import BraveSearchAPI
from shared.utils.search.search_vector_db import SearchResultsVectorDB
from shared.utils.text_resource_manager import TextResourceManager


class searchweb:
    """
    A tool to perform a web search with different content types.
    """
    MAX_RESULTS = 10

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
                f"Requires <query> and an optional <type> tag. Supported types are 'text' (default), "
                f"'images', 'files', and 'any'. Returns up to {searchweb.MAX_RESULTS} results. "
                f"Disable the locally cached results for time-sensitive queries via <cache>false</cache>."
            ),
            "example": (
                "<searchweb><query>How to use curl</query><type>text</type></searchweb>\n"
                "<searchweb><query>golden retriever puppy</query><type>images</type></searchweb>\n"
                "<searchweb><query>latest Mathematical discoveries</query><type>text</type><cache>false</cache></searchweb>"
            )
        }

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
    async def _format_text_summary(results: list[dict], query: str, use_cache: bool = True) -> str:
        """ The original functionality: returns a single condensed string. """
        if not results:
            return "No web search results found."

        # Check for existing summary to avoid duplicates (only if caching enabled)
        if use_cache:
            vector_db = searchweb._get_vector_db()
            existing_summary = vector_db.get_summary_by_query(query, 'text')
            if existing_summary:
                print("✅ Using cached LLM summary (skipping generation)")
                return f"Web search summary: {existing_summary}"
            else:
                print("🤖 No cached summary found, generating new LLM summary")
        else:
            print("🤖 Cache disabled: generating fresh LLM summary")


        sourced_snippets = []
        created_resources = []  # Track created text resources

        for i, result in enumerate(results):
            source_content = searchweb.result_to_source(result)
            sourced_snippets.append(source_content)

            # Create individual text resources (only if caching enabled)
            if use_cache and source_content.strip():
                try:
                    resource_manager = TextResourceManager._get_manager()

                    # Create a hash-based resource ID for deduplication
                    import hashlib
                    content_hash = hashlib.md5(source_content.encode()).hexdigest()[:8]
                    resource_id = f"search_result_{content_hash}"

                    # Check if resource already exists
                    existing_resource = resource_manager.get_resource(resource_id)
                    if existing_resource:
                        # Update access time for existing resource
                        existing_resource.update_access_time()
                        created_resources.append({
                            'resource_id': resource_id,
                            'title': result.get('title', 'No Title'),
                            'url': result.get('url', 'No URL'),
                            'is_new': False
                        })
                    else:
                        # Create new resource
                        resource_manager.create_resource(
                            content=source_content,
                            source=result.get('url', f'search_result_{i}'),
                            source_type='web_search_result',
                            resource_id=resource_id
                        )
                        created_resources.append({
                            'resource_id': resource_id,
                            'title': result.get('title', 'No Title'),
                            'url': result.get('url', 'No URL'),
                            'is_new': True
                        })
                except Exception as e:
                    print(f"⚠️ Failed to create text resource for result {i}: {e}")

        if not sourced_snippets:
            return "No useful information found."

        joined_text = " ".join(sourced_snippets)

        chat = Chat()
        chat.add_message(Role.USER, f"The search query is: {query}\nJudge and compare these different web sources and summarize: {joined_text}")

        # Use await for async LLM completion (force_preferred_model=False to allow any free model)
        summary = await LlmRouter.generate_completion(chat, force_preferred_model=False)

        # Store the generated summary in the vector database with enhanced metadata (only if caching enabled)
        if use_cache:
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

        # Build the enhanced response with resource information
        response = f"Web search summary: {summary}"

        # Add individual source access information (only if resources were created)
        if use_cache and created_resources:
            response += "\n\nIndividual Sources Available:"
            for resource_info in created_resources:
                status_indicator = "📄" if resource_info['is_new'] else "🔄"
                response += f"\n- {status_indicator} Resource ID: {resource_info['resource_id']} | Title: \"{resource_info['title']}\" | URL: {resource_info['url']}"

            response += "\n\nAccess individual sources using:"
            response += "\n• Read full source: <readfile>{resource_id}</readfile>"
            response += "\n• Search within source: <searchresource><resource_id>{resource_id}</resource_id><search>your search term</search></searchresource>"

            new_count = sum(1 for r in created_resources if r['is_new'])
            existing_count = len(created_resources) - new_count
            if new_count > 0:
                response += f"\n\n📄 Created {new_count} new text resources"
            if existing_count > 0:
                response += f" | 🔄 Updated access time for {existing_count} existing resources"

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
            cache_match = re.search(r'<cache>(.*?)</cache>', content, re.DOTALL)

            if not query_match:
                # Fallback for old format
                query = content.strip()
                search_type = 'text'
            else:
                query = query_match.group(1).strip()
                search_type = type_match.group(1).strip().lower() if type_match else 'text'

            # Parse cache flag (default: true)
            use_cache = True
            if cache_match and cache_match.group(1).strip().lower() == 'false':
                use_cache = False
                print("⚠️ Caching is explicitly disabled for this query.")

            if not query:
                return "Error: No search query provided in the <query> tag."

            print(f"🔎 Performing '{search_type}' search for: '{query}'...")

            # Get vector database instance
            vector_db = searchweb._get_vector_db()

            results_list = []
            need_brave_api = True

            # Only search cache if caching is enabled
            if use_cache:
                # Search in vector database for similar results (using distance threshold)
                cached_results_with_distances = vector_db.search_similar_results(query, search_type, searchweb.MAX_RESULTS, 0.5)

                if cached_results_with_distances:
                    # Calculate distance details (lower = better)
                    distances = [distance for _, distance in cached_results_with_distances]
                    avg_distance = sum(distances) / len(distances)
                    min_distance = min(distances)  # Best match
                    max_distance = max(distances)  # Worst match

                    print(f"📊 Found {len(cached_results_with_distances)} cached results (distance ≤ 0.5)")
                    print(f"   └─ Distance scores: avg={avg_distance:.3f}, best={min_distance:.3f}, worst={max_distance:.3f}")

                    # Add cached results with their distance scores (all are already below threshold)
                    for metadata, distance in cached_results_with_distances:
                        metadata['is_cached'] = True
                        metadata['similarity_score'] = 1 - distance  # Convert to similarity for compatibility
                        results_list.append(metadata)

                    # Skip API call if we have enough high-quality cached results
                    if len(cached_results_with_distances) >= searchweb.MAX_RESULTS:
                        need_brave_api = False
                        print(f"✅ Using {len(cached_results_with_distances)} cached results only (no API call needed)")
                    else:
                        print(f"🔄 Using {len(cached_results_with_distances)} cached + fetching {searchweb.MAX_RESULTS - len(cached_results_with_distances)} more from API")
                else:
                    print("📊 No cached results found with distance ≤ 0.5")
            else:
                print("🚫 Cache disabled: bypassing all cached results")

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

                            # Store new results in vector database for future searches (only if caching enabled)
                            if use_cache and vector_db.store_search_results(query, search_type, brave_results, api_endpoint):
                                print("💾 Stored new results in vector database for future use")
                            elif not use_cache:
                                print("🚫 Not storing results (cache disabled)")

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
                return await searchweb._format_text_summary(results_list, query, use_cache)
            elif search_type == 'images':
                # Check for existing image summary to avoid duplicates (only if caching enabled)
                if use_cache:
                    existing_summary = vector_db.get_summary_by_query(query, search_type)
                    if existing_summary:
                        print("✅ Found and using existing image summary from vector database")
                        return existing_summary

                result = searchweb._format_image_results(results_list)
                # Store image list summary in vector database with enhanced metadata (only if caching enabled)
                if use_cache:
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
                # Check for existing summary to avoid duplicates (only if caching enabled)
                if use_cache:
                    existing_summary = vector_db.get_summary_by_query(query, search_type)
                    if existing_summary:
                        print("✅ Found and using existing summary from vector database")
                        return existing_summary

                result = searchweb._format_cited_text(results_list)
                # Store summary in vector database with enhanced metadata (only if caching enabled)
                if use_cache:
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