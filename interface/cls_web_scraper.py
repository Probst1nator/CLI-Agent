import base64
import logging
import random
from io import BytesIO
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_base64_image(base64_string: str) -> bool:
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        image.verify()
        return image.format != "GIF" and min(image.size) >= 250
    except Exception as e:
        logging.error(f"Image validation failed: {e}")
        return False

def correct_url_scheme(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        return f"https:{url}" if url.startswith("//") else f"https://{url}"
    return url

class WebScraper:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        ]
        self.max_retries = 100
        self.retry_sleep = 1

    def fetch_url_content(self, url: str) -> Optional[str]:
        corrected_url = correct_url_scheme(url)
        headers = {"User-Agent": random.choice(self.user_agents)}
        try:
            response = requests.get(corrected_url, headers=headers, timeout=5)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logging.error(f"Failed to fetch URL content: {e} for URL: {corrected_url}")
            return None

    def extract_text(self, page_content: str) -> Optional[str]:
        soup = BeautifulSoup(page_content, "html.parser")
        paragraphs = soup.find_all('p')
        text_content = ' '.join(p.get_text() for p in paragraphs if p.get_text())
        if text_content:
            return text_content
        return None

    def search_and_extract_texts(self, keyword: str, num_results: int) -> List[str]:
        texts = []
        try:
            urls = DDGS().text(keyword, max_results=num_results)
            urls = [result["href"] for result in urls]
            for url in urls:
                page_content = self.fetch_url_content(url)
                if page_content:
                    extracted_text = self.extract_text(page_content)
                    if extracted_text:
                        texts.append(extracted_text)
        except Exception as e:
            logging.error(f"Failed to search or extract texts: {e}")
        return texts
