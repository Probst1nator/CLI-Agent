# utils/web_fetch.py
import json
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any

from agent.utils_manager.util_base import UtilBase

class WebFetchUtil(UtilBase):
    """
    A utility for fetching and parsing the content of a web page.
    This tool retrieves the raw HTML and provides a clean, text-only version.
    """

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["get url", "fetch page", "read website", "scrape content", "download html", "get text from url", "api endpoint", "json data", "model info", "huggingface api", "github releases", "download metadata", "model card", "repository info", "extract data", "parse response"],
            "use_cases": [
                "Fetch the text content of the article at https://example.com/news/story.",
                "Download the raw HTML of the homepage.",
                "Scrape the data from this API endpoint.",
                "Get model information from a HuggingFace model card.",
                "Fetch the latest release data from a GitHub repository.",
                "Download metadata about available models from an API."
            ],
            "arguments": {
                "url": "The full URL of the web page to fetch.",
                "get_text_only": "If true, returns cleaned text. If false, returns the raw HTML source."
            }
        }

    @staticmethod
    def _run_logic(url: str, get_text_only: bool = True) -> str:
        """
        Fetches the content from a specific URL.

        Args:
            url (str): The full URL of the web page to fetch.
            get_text_only (bool): If True, returns clean text content. If False, returns raw HTML.

        Returns:
            str: A JSON string with a 'result' key on success, or an 'error' key on failure.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

            content_type = response.headers.get('Content-Type', '').lower()
            
            if 'html' in content_type:
                if get_text_only:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    # Remove script and style elements
                    for script_or_style in soup(["script", "style"]):
                        script_or_style.decompose()
                    # Get text and clean up whitespace
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    content = '\n'.join(chunk for chunk in chunks if chunk)
                else:
                    content = response.text
            else:
                # For non-HTML content like JSON, XML, or plain text
                content = response.text

            result = {
                "result": {
                    "status": "Success",
                    "message": f"Successfully fetched content from {url}",
                    "url": url,
                    "content_type": content_type,
                    "content_length": len(content),
                    "content": content
                }
            }
            return json.dumps(result, indent=2)

        except requests.exceptions.HTTPError as e:
            return json.dumps({"error": f"HTTP Error fetching URL {url}: {e}"}, indent=2)
        except requests.exceptions.RequestException as e:
            return json.dumps({"error": f"Could not fetch URL {url}. Reason: {e}"}, indent=2)
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"}, indent=2)


# Module-level run function for CLI-Agent compatibility
def run(url: str, get_text_only: bool = True) -> str:
    """
    Module-level wrapper for WebFetchUtil._run_logic() to maintain compatibility with CLI-Agent.
    
    Args:
        url (str): The full URL of the web page to fetch
        get_text_only (bool): If True, returns clean text content. If False, returns raw HTML.
        
    Returns:
        str: JSON string with result or error
    """
    return WebFetchUtil._run_logic(url=url, get_text_only=get_text_only)