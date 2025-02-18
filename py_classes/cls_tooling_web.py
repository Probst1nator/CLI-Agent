import base64
import os
import re
from typing import Optional, List, Dict, Any, Tuple
import requests
from bs4 import BeautifulSoup
from termcolor import colored
import chromadb
import hashlib
from datetime import datetime
from chromadb.config import Settings
from brave import Brave
from py_classes.ai_providers.cls_ollama_interface import OllamaClient
from py_classes.globals import g

class WebTools:
    def __init__(self):
        """
        Initialize the WebTools class with persistent storage.
        
        Args:
            persistent_dir (str): Directory for persistent ChromaDB storage
        """
        self.brave_api_key = os.getenv("BRAVE_API_KEY")
        self.persistent_dir = g.PROJ_PERSISTENT_STORAGE_PATH + "/web_search_db"
        
        # Initialize persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.persistent_dir,
            settings=Settings(
                allow_reset=True,
                is_persistent=True,
                anonymized_telemetry=False
            )
        )

        
    def _chunk_text(self, text: str, chunk_size: int = 4000, overlap_size: int = 200) -> List[str]:
        """
        Split text into chunks of approximately equal size with overlapping content.
        
        Args:
            text (str): Text to split
            chunk_size (int): Maximum size of each chunk
            overlap_size (int): Number of characters to overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        # Split text into sentences (basic implementation)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        last_sentences = []  # Keep track of the last few sentences
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) >= chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Calculate how many previous sentences we need for overlap
                    overlap_text = ""
                    overlap_length = 0
                    for prev_sentence in reversed(last_sentences):
                        if overlap_length + len(prev_sentence) > overlap_size:
                            break
                        overlap_text = prev_sentence + " " + overlap_text
                        overlap_length += len(prev_sentence) + 1
                    
                    # Start new chunk with overlap
                    current_chunk = overlap_text.strip() + " " + sentence
                    # Reset last_sentences but keep current sentence
                    last_sentences = [sentence]
                else:
                    current_chunk = sentence
                    last_sentences = [sentence]
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                last_sentences.append(sentence)
                
                # Keep only enough sentences for overlap
                while sum(len(s) + 1 for s in last_sentences) > overlap_size * 2:
                    last_sentences.pop(0)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def scrape_text_from_url(self, url: str) -> str:
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

    def search_brave(self, query: str, num_results: int = 2, top_k: int = 3) -> List[Tuple[str, Dict]]:
        """
        Search the web using Brave browser and return the most relevant chunks.
        
        Args:
            query (str): The search query
            num_results (int): Number of web results to fetch
            top_k (int): Number of most relevant chunks to return
            
        Returns:
            List[Tuple[str, str]]: List of (document, url) tuples for most relevant chunks
        """
        print(colored(f"BRAVE: Searching {num_results} websites for: {query}", "green"))
        

        # Create a new collection for this search
        collection_name = f"brave_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={
                "description": f"Brave search results for query: {query}",
                "created_at": datetime.now().isoformat(),
                "query": query
            }
        )
        
        # Perform Brave search
        brave = Brave(self.brave_api_key)
        search_results = brave.search(q=query, count=num_results, safesearch="off")
        
        all_chunks = []
        chunk_ids = []
        source_urls = []
        
        # First gather all chunks and metadata
        for web_result in search_results.web_results:
            url = web_result['url']
            scraped_content = self.scrape_text_from_url(url)
            
            if not scraped_content:
                continue
                
            # Split content into chunks
            chunks = self._chunk_text(scraped_content)
            
            # Process chunks
            for i, chunk in enumerate(chunks):
                # Generate unique ID for the chunk
                chunk_id = hashlib.md5(f"{url}_{i}_{chunk[:100]}".encode()).hexdigest()
                
                all_chunks.append(chunk)
                chunk_ids.append(chunk_id)
                source_urls.append(url)
        

        if not all_chunks:
            return []
            
        # Generate all embeddings at once
        chunk_embeddings = OllamaClient.generate_embedding(all_chunks, model="bge-m3")
        if not chunk_embeddings:
            return []
        
        collection.add(
            ids=chunk_ids,
            embeddings=chunk_embeddings,
            metadatas=[{"source_url": str(source_url)} for source_url in source_urls],
            documents=all_chunks
        )


        
        # Generate embedding for query
        query_embedding = OllamaClient.generate_embedding(query)
        if not query_embedding:
            return []
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Combine documents and metadata
        if results["documents"] and results["metadatas"]:
            urls = [metadata["source_url"] for metadata in results["metadatas"][0]]
            return list(zip(results["documents"][0], urls))
        return []


# Example usage:
if __name__ == "__main__":
    # Initialize with persistent storage
    web_tools = WebTools()
    
    # Search and get results
    query = "Latest developments in quantum computing"
    results = web_tools.search_brave(query, num_results=2)
    
    # Print results
    for doc, url in results:
        print(f"URL: {url}")
        print(f"Chunk:")
        print(doc)
        print("-" * 80)