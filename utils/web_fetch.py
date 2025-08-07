# utils/web_fetch.py
import json
import requests
from bs4 import BeautifulSoup

from py_classes.cls_util_base import UtilBase

class WebFetchUtil(UtilBase):
    """
    A utility for fetching and parsing the content of a web page.
    This tool retrieves the raw HTML and provides a clean, text-only version.
    """

    @staticmethod
    def run(url: str, get_text_only: bool = True) -> str:
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