# tools/main_cli_agent/tools/searchweb.py
"""
This file implements the 'searchweb' tool. It allows the agent to perform
a web search using the Brave Search API. It supports different search types
like text, images, and cited results.
"""
import os
import sys
import re
import html
from dotenv import load_dotenv
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

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

class searchweb:
    """
    A tool to perform a web search with different content types.
    """
    MAX_RESULTS = 10
    
    VECTOR_DB_PATH = os.path.expanduser("~/.cli-agent/searchweb_vectordb")
    COLLECTION_NAME = "web_search_results"

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
                f"'images', 'files', 'cited text', and 'any'. Returns up to {searchweb.MAX_RESULTS} results."
            ),
            "example": (
                "<searchweb><query>How to use curl</query><type>text</type></searchweb>\n"
                "<searchweb><query>golden retriever puppy</query><type>images</type></searchweb>"
            )
        }

    @staticmethod
    def _clean_snippet(text: str) -> str:
        decoded_text = html.unescape(text)
        clean_text = re.sub(r'<.*?>', '', decoded_text)
        return clean_text.strip()

    @staticmethod
    def _format_text_summary(results: list) -> str:
        """ The original functionality: returns a single condensed string. """
        if not results:
            return "No web search results found."
        
        condensed_snippets = []
        for result in results:
            snippets = result.get('extra_snippets', [])
            if not snippets and result.get('description'):
                snippets = [result.get('description')]

            if snippets:
                clean_snippet = searchweb._clean_snippet("\n".join(snippets))
                condensed_snippets.append(clean_snippet)
        
        if not condensed_snippets:
            return "No useful information found."

        summary = " ".join(condensed_snippets)
        # LlmRouter # TODO: test out using a llm generated summary (how do we get the needed context directive)

        return f"Web search summary: {summary}"

    @staticmethod
    def _format_cited_text(results: list) -> str:
        """ Returns a numbered list with titles, URLs, and snippets. """
        if not results:
            return "No web search results found."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = searchweb._clean_snippet(result.get('title', 'No Title'))
            url = result.get('url', 'No URL')
            snippet = searchweb._clean_snippet(result.get('description', 'No snippet available.'))
            formatted_results.append(f"Result {i}:\n  Title: {title}\n  URL: {url}\n  Snippet: {snippet}")
        
        return "\n\n".join(formatted_results)

    @staticmethod
    def _format_image_results(results: list) -> str:
        """ Returns a numbered list of images with titles and direct URLs. """
        if not results:
            return "No image results found."

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = searchweb._clean_snippet(result.get('title', 'No Title'))

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
    def run(content: str) -> str:
        """
        Executes a web search based on the provided query and type.
        """
        try:
            query_match = re.search(r'<query>(.*?)</query>', content, re.DOTALL)
            type_match = re.search(r'<type>(.*?)</type>', content, re.DOTALL)

            if not query_match:
                # Fallback for old format
                query = content.strip()
                search_type = 'text'
            else:
                query = query_match.group(1).strip()
                search_type = type_match.group(1).strip().lower() if type_match else 'any'

            if not query:
                return "Error: No search query provided in the <query> tag."
            
            api_key = os.getenv("BRAVE_API_KEY")
            if not api_key:
                return "Error: BRAVE_API_KEY environment variable is not set."

            print(f"🔎 Performing '{search_type}' search for: '{query}'...")

            api_endpoint = "web"
            original_query = query
            
            if search_type == 'images':
                api_endpoint = "images"
            elif search_type == 'files':
                query = f"filetype:pdf OR filetype:doc OR filetype:txt {query}"

            params = {"q": query}
            results_json = BraveSearchAPI.search(api_key, api_endpoint, params)

            if not results_json:
                return "Error: Web search failed. The API returned no response."

            # Parse results based on endpoint type
            if api_endpoint == "web":
                results_list = results_json.get('web', {}).get('results', [])
            elif api_endpoint == "images":
                results_list = results_json.get('results', [])
            else:
                # For other endpoints, try both locations
                results_list = results_json.get('results', []) or results_json.get('web', {}).get('results', [])

            if not results_list:
                return f"No results found for '{original_query}' (type: {search_type})."

            # Debug: Log the structure of the first result for images
            if search_type == 'images' and results_list:
                print("DEBUG: First image result structure:")
                first_result = results_list[0]
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

            results_list = results_list[:searchweb.MAX_RESULTS]

            if search_type == 'text':
                return searchweb._format_text_summary(results_list)
            elif search_type == 'images':
                return searchweb._format_image_results(results_list)
            elif search_type in ['files', 'cited text', 'any']:
                return searchweb._format_cited_text(results_list)
            else:
                return f"Error: Unknown search type '{search_type}'. Use 'text', 'images', 'files', 'cited text', or 'any'."

        except Exception as e:
            return f"An unexpected error occurred in the searchweb tool: {str(e)}"