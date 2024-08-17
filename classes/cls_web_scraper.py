import base64
import re
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from termcolor import colored

def search_and_scrape(query: str, num_sites: int = 1) -> list[str]:
    """
    Perform a Google search using the given query and scrape the content of the specified number of sites.
    
    Args:
        query (str): The search query.
        num_sites (int): The number of sites to scrape. Default is 1.
        
    Returns:
        list: A list of strings, each containing the scraped text from the respective search results.
        
    Raises:
        requests.RequestException: If there is an error fetching a webpage.
    """
    # Perform a Google search and get the top results
    print (colored(f"Searching {num_sites} sites on the internet, please give me a second...", "green"))
    results = search(query, num_results=num_sites*4)
    
    scraped_contents = []
    
    for result in results:
        print(f"Result URL: {result}")
        
        # Fetch the webpage content
        try:
            page_response = requests.get(result, headers={'User-Agent': 'Mozilla/5.0'})
            page_response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching the webpage: {e}")
            continue
        
        # Parse the HTML content
        soup = BeautifulSoup(page_response.content, 'html.parser')
        
        # Extract and print the text
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        scraped_contents.append(text)
        
        if (len(scraped_contents) == num_sites):
            return scraped_contents
    
    return scraped_contents

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