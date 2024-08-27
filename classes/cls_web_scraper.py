import base64
import os
import re
from typing import Optional
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from termcolor import colored
from dotenv import load_dotenv
from classes.cls_few_shot_factory import FewShotProvider

from brave import Brave

def scrape_text_from_url(url: str) -> str:
    """
    Scrape the human-readable text from a given URL.
    :param url: The URL to scrape
    :return: The scraped text content
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        # Get text
        text = soup.get_text()
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return ""

def search_brave(query: str, num_results: int = 2, summarization_llm: str = "") -> list[str]:
    """
    Search the web using the Brave browser and return the scraped readable texts from each result.
    :param query: The search query
    :param num_results: The number of results to return
    :return: A list of scraped text contents from the found sites
    """
    brave = Brave(os.getenv('BRAVE_API_KEY'))
    search_results = brave.search(q=query, count=num_results)
    scraped_texts = []
    for web_result in search_results.web_results:
        # print(web_result.title)
        # print(web_result.url)
        # print(web_result.description)
        # print(web_result.age)
        scraped_content = scrape_text_from_url(web_result['url'])
        scraped_texts.append(scraped_content)
    
    if summarization_llm:
        summarized_content = FewShotProvider.few_shot_distilText(query, scraped_texts, [summarization_llm])
        return [summarized_content]
    return scraped_texts

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