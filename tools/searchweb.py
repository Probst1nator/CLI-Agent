# tools/main_cli_agent/tools/searchweb.py
"""
This file implements the 'searchweb' tool. It allows the agent to perform
a web search using the Brave Search API and receive a condensed summary
of the top results as a single string.
"""
import os
import sys
import re
import html

# Ensure proper imports when running as module or script
if __name__ == '__main__':
    # Add project root to path for direct execution
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

# This import assumes the script is run from a context where the project root
# is in the Python path. The main guard below handles this for direct execution.
from shared.utils.search.brave_search import BraveSearchAPI
from core.providers.cls_ollama_interface import OllamaClient

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

class searchweb:
    """
    A tool to perform a web search and get a concise, clean summary of results.
    Enhanced version that stores results in a vector database for future retrieval.
    """
    # The number of search result snippets to combine into the summary.
    MAX_RESULTS = 20
    
    # Vector database settings
    VECTOR_DB_PATH = os.path.expanduser("~/.cli-agent/searchweb_vectordb")
    COLLECTION_NAME = "web_search_results"

    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool."""
        return 'searchweb'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool."""
        return {
            "name": "searchweb",
            "description": f"Performs a web search and returns a condensed, single-string summary of the top {searchweb.MAX_RESULTS} search results, with URLs and HTML formatting removed. Results are stored in a vector database for future semantic retrieval.",
            "example": "<searchweb>latest advancements in large language models</searchweb>"
        }

    @staticmethod
    def _clean_snippet(text: str) -> str:
        """
        Cleans a string by decoding HTML entities and removing HTML tags.
        """
        # Decode HTML entities (e.g., &#x27; -> ', &amp; -> &)
        decoded_text = html.unescape(text)
        # Remove HTML tags (e.g., <strong>, <em>)
        clean_text = re.sub(r'<.*?>', '', decoded_text)
        return clean_text.strip()
    
    @staticmethod
    def _get_vector_collection():
        """
        Initialize and return the ChromaDB collection for storing search results.
        """
        if not CHROMADB_AVAILABLE:
            return None
        
        try:
            # Ensure the vector database directory exists
            os.makedirs(searchweb.VECTOR_DB_PATH, exist_ok=True)
            
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(
                path=searchweb.VECTOR_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            collection = client.get_or_create_collection(
                name=searchweb.COLLECTION_NAME,
                metadata={"description": "Web search results with embeddings"}
            )
            
            return collection
        except Exception as e:
            print(f"Warning: Could not initialize vector database: {e}")
            return None
    
    @staticmethod
    def _store_in_vector_db(collection, results: list, query: str):
        """
        Store search results in the vector database with embeddings.
        """
        if not collection or not results:
            return
        
        try:
            documents = []
            metadatas = []
            ids = []
            embeddings = []
            
            for i, result in enumerate(results):
                title = result.get('title', '')
                description = result.get('description', '')
                url = result.get('url', '')
                extra_snippets = result.get('extra_snippets', [])
                
                if not title:  # Skip if no title
                    continue
                
                # Generate embedding for the title
                try:
                    title_embedding = OllamaClient.generate_embedding(
                        title
                    )
                    if not title_embedding:
                        continue
                except Exception as e:
                    print(f"Warning: Could not generate embedding for title '{title}': {e}")
                    continue
                
                # Create document content
                document = f"Title: {title}\nDescription: {description}\nSnippets: {' '.join(extra_snippets)}"
                
                # Create metadata
                import time
                metadata = {
                    'title': title,
                    'url': url,
                    'search_query': query,
                    'result_index': i,
                    'timestamp': str(int(time.time()))
                }
                
                documents.append(document)
                metadatas.append(metadata)
                ids.append(f"search_{query[:20]}_{i}_{hash(url)}")
                embeddings.append(title_embedding)
            
            if documents:
                # Add to collection
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                print(f"📚 Stored {len(documents)} results in vector database")
            
        except Exception as e:
            print(f"Warning: Could not store results in vector database: {e}")

    @staticmethod
    def run(content: str) -> str:
        """
        Executes a web search and returns a single, condensed string of information.

        Args:
            content: A string containing the search query.

        Returns:
            A single string summarizing the search results, or an error message.
        """
        try:
            query = content.strip()
            if not query:
                return "Error: No search query provided."

            api_key = os.getenv("BRAVE_API_KEY")
            if not api_key:
                return "Error: BRAVE_API_KEY environment variable is not set. Cannot perform web search."

            print(f"🔎 Condensing web search for: '{query}'...")

            params = {"q": query}
            results_json = BraveSearchAPI.search(api_key, "web", params)

            if not results_json:
                return "Error: Web search failed. The API returned no response or an error."

            web_results = results_json.get('web', {}).get('results', [])

            if not web_results:
                return f"No web search results found for '{query}'."

            # Store results in vector database first
            collection = searchweb._get_vector_collection()
            if collection is not None:
                searchweb._store_in_vector_db(collection, web_results[:searchweb.MAX_RESULTS], query)
            
            # Collect cleaned descriptions from the results
            condensed_snippets = []
            for result in web_results[:searchweb.MAX_RESULTS]:
                title = result.get('title', '')
                extra_snippets = "\n".join(result.get('extra_snippets', []))
                # description = result.get('description', '')
                if extra_snippets:
                    clean_snippet = searchweb._clean_snippet(extra_snippets)
                    condensed_snippets.append(clean_snippet)
            
            if not condensed_snippets:
                 return f"No useful information found for '{query}'."

            # Join the cleaned snippets into a single, space-separated string
            summary = " ".join(condensed_snippets)
            
            vector_status = "(stored in vector DB)" if collection is not None else "(vector DB unavailable)"
            return f"Web search summary: {summary} {vector_status}"

        except Exception as e:
            return f"An unexpected error occurred in the searchweb tool: {str(e)}"

# --- Test Guard ---
if __name__ == '__main__':
    import sys
    from dotenv import load_dotenv

    try:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        sys.path.insert(0, PROJECT_ROOT)
        ENV_FILE_PATH = os.path.join(PROJECT_ROOT, '.env')
    except NameError:
        PROJECT_ROOT = os.getcwd()
        ENV_FILE_PATH = os.path.join(PROJECT_ROOT, '.env')

    print("--- Testing searchweb Tool (Condensed Output Version) ---")

    if not os.path.exists(ENV_FILE_PATH):
        print(f"Error: .env file not found at the expected path: {ENV_FILE_PATH}")
    else:
        load_dotenv(ENV_FILE_PATH)
        if not os.getenv("BRAVE_API_KEY"):
            print("Error: BRAVE_API_KEY not found in the .env file. Aborting test.")
        else:
            test_query = "What is the requests library in Python?"
            print(f"\nTest Query: \"{test_query}\"\n")
            
            result = searchweb.run(test_query)
            
            print("--- Result ---")
            print(result)