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

class WebTools:
    def __init__(self, persistent_dir: str = "web_search_db"):
        """
        Initialize the WebTools class with persistent storage.
        
        Args:
            persistent_dir (str): Directory for persistent ChromaDB storage
        """
        self.brave_api_key = os.getenv("BRAVE_API_KEY")
        self.persistent_dir = os.path.abspath(persistent_dir)
        
        # Initialize persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.persistent_dir,
            settings=Settings(
                allow_reset=True,
                is_persistent=True
            )
        )
        
    def _get_collection(self, date: str = None) -> chromadb.Collection:
        """
        Get or create a collection for storing search results. Collections are created per day.
        
        Args:
            date (str, optional): Date in YYYY-MM-DD format. If None, uses today's date.
            
        Returns:
            chromadb.Collection: ChromaDB collection
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
            
        collection_name = f"brave_search_{date}"
        
        try:
            # Try to get existing collection for the date
            collection = self.chroma_client.get_collection(name=collection_name)
            print(colored(f"Using existing collection for {date}", "green"))
        except ValueError:
            # Create new collection if it doesn't exist for the date
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "description": f"Brave search results for {date}",
                    "created_at": datetime.now().isoformat(),
                    "date": date
                }
            )
            print(colored(f"Created new collection for {date}: {collection_name}", "green"))
            
        return collection
        
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

    def search_brave(self, query: str, num_results: int = 2, force_refresh: bool = False, top_k: int = 3) -> List[Tuple[str, Dict]]:
        """
        Search the web using Brave browser, store results in persistent ChromaDB, and return the most relevant chunks.
        
        Args:
            query (str): The search query
            num_results (int): Number of web results to fetch
            force_refresh (bool): If True, force new search even if results exist
            top_k (int): Number of most relevant chunks to return
            
        Returns:
            List[Tuple[str, Dict]]: List of (document, metadata) tuples for most relevant chunks
        """
        print(colored(f"BRAVE: Searching {num_results} websites for: {query}", "green"))
        
        # Get or create collection for today
        collection = self._get_collection()
        
        # Check if we already have results for this query and force_refresh is False
        if not force_refresh:
            existing_results = collection.get(
                where={"query": query}
            )
            if existing_results["ids"]:
                print(colored(f"Using existing results for query: {query}", "green"))
                return self.query_stored_results(query, top_k=top_k)
        
        # Perform Brave search
        brave = Brave(self.brave_api_key)
        search_results = brave.search(q=query, count=num_results, safesearch="off")
        
        all_chunks = []
        chunk_embeddings = []
        chunk_ids = []
        chunk_metadata = []
        
        for web_result in search_results.web_results:
            url = web_result['url']
            scraped_content = self.scrape_text_from_url(url)
            
            if not scraped_content:
                continue
                
            # Split content into chunks
            chunks = self._chunk_text(scraped_content)
            
            # Process chunks in batches
            for i, chunk in enumerate(chunks):
                # Generate unique ID for the chunk
                chunk_id = hashlib.md5(f"{url}_{i}_{chunk[:100]}".encode()).hexdigest()
                
                # Create metadata
                metadata = {
                    "url": str(url),  # Ensure URL is converted to string
                    "chunk_index": i,
                    "timestamp": datetime.now().isoformat(),
                    "query": str(query)  # Ensure query is converted to string
                }
                
                try:
                    # Generate embedding using OllamaClient
                    embedding = OllamaClient.generate_embedding(chunk)
                    
                    # Collect chunks for batch insertion
                    all_chunks.append(chunk)
                    chunk_embeddings.append(embedding)
                    chunk_ids.append(chunk_id)
                    chunk_metadata.append(metadata)
                    
                except Exception as e:
                    print(colored(f"Error processing chunk {i} from {url}: {str(e)}", "red"))
                    continue
        
        # Batch add all chunks to collection
        if chunk_ids:
            collection.add(
                ids=chunk_ids,
                embeddings=chunk_embeddings,
                metadatas=chunk_metadata,
                documents=all_chunks
            )
        
        # Return relevant chunks using embedding-based search
        return self.query_stored_results(query, top_k=top_k)
    
    def query_stored_results(self, query: str, top_k: int = 3, date: str = None) -> List[Tuple[str, Dict]]:
        """
        Query the stored results in ChromaDB for a given search query.
        
        Args:
            query (str): Query string
            top_k (int): Number of top results to return
            date (str, optional): Specific date to query in YYYY-MM-DD format. If None, uses today's collection.
            
        Returns:
            List[Tuple[str, Dict]]: List of (document, metadata) tuples
        """
        collection = self._get_collection(date)
        
        # Generate embedding for query using OllamaClient
        query_embedding = OllamaClient.generate_embedding(query)
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"query": query}  # Only get results for this specific query
        )
        
        # Combine documents and metadata
        if results["documents"] and results["metadatas"]:
            return list(zip(results["documents"][0], results["metadatas"][0]))
        return []

    def list_collections(self) -> List[str]:
        """
        List all available search collections.
        
        Returns:
            List[str]: List of collection names
        """
        return [col.name for col in self.chroma_client.list_collections()]

    def clear_collection(self, date: str = None):
        """
        Clear a specific day's search collection.
        
        Args:
            date (str, optional): Date in YYYY-MM-DD format. If None, clears today's collection.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        collection_name = f"brave_search_{date}"
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            collection.delete(where={})
            print(colored(f"Cleared collection for {date}: {collection_name}", "green"))
        except ValueError:
            print(colored(f"Collection not found for {date}: {collection_name}", "yellow"))

# Example usage:
if __name__ == "__main__":
    # Initialize with persistent storage
    web_tools = WebTools(persistent_dir="./web_search_db")
    
    # Search and store results
    query = "Latest developments in quantum computing"
    results = web_tools.search_brave(query, num_results=2)
    
    # Query stored results
    stored_results = web_tools.query_stored_results(query)
    for doc, metadata in stored_results:
        print(f"URL: {metadata['url']}")
        print(f"Chunk {metadata['chunk_index']}:")
        print(doc)
        print("-" * 80)
        
    # List available collections
    print("Available collections:")
    print(web_tools.list_collections())