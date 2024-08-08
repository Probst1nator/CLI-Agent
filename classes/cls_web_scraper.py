import base64
import logging
import re
from typing import List, Optional
import requests
from googlesearch import search

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebScraper:
    """
    A class for performing web searches and extracting text from web pages.
    """

    def __init__(self):
        """
        Initialize the WebSearcher with default settings.
        """
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def fetch_url_content(self, url: str) -> Optional[str]:
        """
        Fetch the content of a given URL.
        
        :param url: The URL to fetch content from
        :return: The content of the URL as a string, or None if fetching fails
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logging.error(f"Failed to fetch URL content: {e} for URL: {url}")
            return None

    def extract_text(self, page_content: str) -> str:
        """
        Extract text content from a web page.
        
        :param page_content: The HTML content of the web page
        :return: Extracted text content
        """
        # For simplicity, we're just returning the raw content here.
        # You might want to use BeautifulSoup or another HTML parser for more precise extraction.
        return page_content

    def search_and_extract_texts(self, keyword: str, num_results: int) -> List[str]:
        """
        Perform a web search and extract text from the resulting web pages.
        
        :param keyword: The search keyword
        :param num_results: The number of results to process
        :return: A list of extracted texts from the search results
        """
        texts = []
        try:
            # Perform the search
            search_results = search(keyword, num_results=num_results)
            
            for url in search_results:
                page_content = self.fetch_url_content(url)
                if page_content:
                    extracted_text = self.extract_text(page_content)
                    if extracted_text:
                        texts.append(extracted_text)
                        logging.info(f"Extracted text from: {url}")
                
                if len(texts) >= num_results:
                    break
            
        except Exception as e:
            logging.error(f"An error occurred during the search: {e}")
        
        return texts


def get_github_readme(repo_url: str) -> str:
    """
    Retrieve and read a GitHub repository's README file into a string.

    :param repo_url: The URL of the GitHub repository
    :return: The content of the README file as a string
    """
    # Extract owner and repo name from the URL
    match = re.search(r'github\.com/([^/]+)/([^/]+)', repo_url)
    if not match:
        raise ValueError("Invalid GitHub repository URL")
    
    owner, repo = match.groups()

    # GitHub API endpoint for repository contents
    api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"

    # Send a GET request to the GitHub API
    response = requests.get(api_url)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Parse the JSON response
    data = response.json()

    # Get the content of the README file
    if data['encoding'] == 'base64':
        # If the content is base64 encoded, decode it
        readme_content = base64.b64decode(data['content']).decode('utf-8')
    else:
        # If not encoded, just get the content directly
        readme_content = data['content']

    return readme_content