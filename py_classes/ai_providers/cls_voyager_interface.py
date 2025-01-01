import os
from typing import List, Dict, Any, Optional
import voyageai
import numpy as np
from termcolor import colored

class VoyagerApi:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the VoyageAI client.
        
        Args:
            api_key (str, optional): VoyageAI API key. If not provided, will try to get from environment variable.
        """
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VoyageAI API key must be provided or set in VOYAGE_API_KEY environment variable")
            
        self.client = voyageai.Client(api_key=self.api_key)
        self.default_model = "voyage-3"
        
    @staticmethod
    def _validate_input(text: str | List[str]) -> List[str]:
        """
        Validate and normalize input text.
        
        Args:
            text (str | List[str]): Input text or list of texts
            
        Returns:
            List[str]: Normalized list of texts
        """
        if isinstance(text, str):
            return [text]
        elif isinstance(text, list):
            if not all(isinstance(t, str) for t in text):
                raise ValueError("All elements in the list must be strings")
            return text
        else:
            raise ValueError("Input must be a string or list of strings")
    
    def generate_embedding(self, text: str | List[str], model: Optional[str] = None, input_type: str = "document") -> List[List[float]]:
        """
        Generate embeddings for the input text using VoyageAI.
        
        Args:
            text (str | List[str]): Input text or list of texts to embed
            model (str, optional): Model name to use. Defaults to voyage-3.
            input_type (str): Type of input - either "document" or "query". Defaults to "document".
            
        Returns:
            List[List[float]]: List of embeddings
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            texts = self._validate_input(text)
            model = model or self.default_model
            
            print(colored(f"Generating embeddings using {model} for {len(texts)} texts...", "cyan"))
            
            # Generate embeddings
            response = self.client.embed(
                texts,
                model=model,
                input_type=input_type
            )
            
            print(colored(f"Successfully generated embeddings", "green"))
            return response.embeddings
            
        except Exception as e:
            print(colored(f"Error generating embeddings: {str(e)}", "red"))
            return []
    
    def rerank(self, query: str, documents: List[str], model: str = "rerank-2", top_k: int = 3) -> Dict[str, Any]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query (str): The search query
            documents (List[str]): List of documents to rerank
            model (str): Reranking model to use. Defaults to rerank-2.
            top_k (int): Number of top results to return. Defaults to 3.
            
        Returns:
            Dict[str, Any]: Dictionary containing reranked results with scores
        """
        try:
            print(colored(f"Reranking {len(documents)} documents...", "cyan"))
            
            # Perform reranking
            results = self.client.rerank(
                query,
                documents,
                model=model,
                top_k=top_k
            )
            
            print(colored(f"Successfully reranked documents", "green"))
            return results
            
        except Exception as e:
            print(colored(f"Error during reranking: {str(e)}", "red"))
            return {"results": []}
    
    def semantic_search(self, query: str, documents: List[str], top_k: int = 3, use_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        Perform semantic search on documents using embeddings and optional reranking.
        
        Args:
            query (str): Search query
            documents (List[str]): List of documents to search through
            top_k (int): Number of results to return
            use_reranking (bool): Whether to use reranking after embedding search
            
        Returns:
            List[Dict[str, Any]]: List of search results with scores and document content
        """
        try:
            # Generate embeddings for query and documents
            query_embedding = self.generate_embedding(query, input_type="query")[0]
            doc_embeddings = self.generate_embedding(documents, input_type="document")
            
            # Compute similarities (Voyage embeddings are normalized, so dot product = cosine similarity)
            similarities = np.dot(doc_embeddings, query_embedding)
            
            # Get top-k indices
            if use_reranking:
                # Get more candidates for reranking
                top_indices = np.argsort(similarities)[-top_k*2:][::-1]
                candidate_docs = [documents[i] for i in top_indices]
                
                # Rerank candidates
                reranked = self.rerank(query, candidate_docs, top_k=top_k)
                
                results = []
                for r in reranked.results:
                    results.append({
                        "document": r.document,
                        "score": r.relevance_score,
                        "original_index": top_indices[r.index]
                    })
                
            else:
                # Just use embedding similarities
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                results = [
                    {
                        "document": documents[i],
                        "score": float(similarities[i]),
                        "original_index": i
                    }
                    for i in top_indices
                ]
            
            return results
            
        except Exception as e:
            print(colored(f"Error during semantic search: {str(e)}", "red"))
            return []

# Example usage:
if __name__ == "__main__":
    # Initialize client
    voyage_client = VoyageAIClient()
    
    # Example documents
    documents = [
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables.",
        "Photosynthesis in plants converts light energy into glucose.",
        "20th-century innovations centered on electronic advancements."
    ]
    
    # Generate embeddings
    embeddings = voyage_client.generate_embedding(documents)
    
    # Perform semantic search
    query = "What is the Mediterranean diet?"
    results = voyage_client.semantic_search(query, documents, top_k=2)
    
    for result in results:
        print(f"Score: {result['score']:.4f}")
        print(f"Document: {result['document']}")
        print("-" * 80)