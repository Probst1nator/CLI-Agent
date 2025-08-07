# py_classes/cls_vector_db.py
import numpy as np
from typing import List, Dict, Any, Type
import os
import warnings
import json
import hashlib
from pathlib import Path

# Suppress various warnings that occur during vector DB initialization
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore', message='.*encoder_attention_mask.*')

# It's good practice to handle optional imports
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    import torch # Often a dependency of sentence-transformers
    _SENTENCE_TRANSFORMERS_INSTALLED = True
except ImportError:
    _SENTENCE_TRANSFORMERS_INSTALLED = False
    # Define dummy classes if the library isn't installed
    class SentenceTransformer:
        def __init__(self, model_name_or_path):
            print("\033[93mWarning: sentence-transformers not installed. Vector DB will not function.\033[0m")
            print("\033[93mPlease run: pip install sentence-transformers torch\033[0m")
        def encode(self, text, convert_to_tensor=False):
            return np.zeros((1, 384)) # Return a dummy vector
    def cos_sim(a, b):
        # Proper fallback cosine similarity implementation
        if len(a.shape) == 1:
            a = a.reshape(1, -1)
        if len(b.shape) == 1:
            b = b.reshape(1, -1)
        
        # Normalize vectors
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        
        # Avoid division by zero
        a_norm = np.where(a_norm == 0, 1, a_norm)
        b_norm = np.where(b_norm == 0, 1, b_norm)
        
        a_normalized = a / a_norm
        b_normalized = b / b_norm
        
        # Compute cosine similarity
        return np.dot(a_normalized, b_normalized.T)


from py_classes.cls_util_base import UtilBase

class ToolVectorDB:
    """A simple, in-memory vector database for tool retrieval."""
    _instance = None

    # Minimal startup hints - system will self-evolve through adaptive learning
    HARDCODED_HINTS = {
        # Essential bootstrapping hints only
        "python": ["Remember that you can execute python code on the current system by providing it like this:\n```python\nprint('Hello world')\n```", "writefile", "editfile"],
        "ip address": ["To display the current IP address, use the following Python command:\n```python\nimport socket\nhostname = socket.gethostname()\nip_address = socket.gethostbyname(hostname)\nprint(f'IP Address: {ip_address}')\n```", "writefile", "editfile"],
        "weather": ["For weather information, you'll need to use an API service or create a weather utility tool.", "writefile", "editfile"],
        "location": ["For location information, you can use geolocation services or IP-based location APIs.", "writefile", "editfile"],
        "file": ["writefile", "editfile", "viewfiles"],
        "code": ["editfile", "writefile", "viewfiles", "Remember that you can run any bash code instantly by providing it like this:\n```bash\nawk -v START=3 -v END=6 'NR >= START && NR <= END { print NR \"s: \" $0 }' sample.txt"],
        "write": ["writefile", "editfile"],
        "help": ["architectnewutil"],
        "unable": ["editfile", "architectnewutil"]
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ToolVectorDB, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "bge-m3", use_ollama: bool = True):
        if self._initialized:
            return
        
        self.use_ollama = use_ollama
        self.model_name = model_name
        self.embedding_model = None
        self.embedding_dim = 1024  # BGE-M3 default dimension
        
        if use_ollama:
            # Try Ollama BGE-M3 first
            try:
                from py_classes.ai_providers.cls_ollama_interface import OllamaClient
                # Test if the model is available
                test_embedding = OllamaClient.generate_embedding("test", model=model_name, silent=True)
                if test_embedding and len(test_embedding) > 0:
                    self.is_ready = True
                    self.embedding_dim = len(test_embedding)
                    import os
                    if os.environ.get('CLAUDE_CODE_DEBUG') == '1':
                        print(f"Initializing ToolVectorDB with Ollama model: {model_name} (dim: {self.embedding_dim})")
                else:
                    raise Exception(f"Model {model_name} not available or returned empty embedding")
            except Exception as e:
                print(f"Ollama {model_name} not available, falling back to sentence-transformers: {e}")
                self.use_ollama = False
        
        if not self.use_ollama:
            # Fallback to sentence-transformers
            if not _SENTENCE_TRANSFORMERS_INSTALLED:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # This will print a warning
                self.is_ready = False
                self.embedding_dim = 384
            else:
                # Only print initialization message if this is truly the first time
                import os
                if os.environ.get('CLAUDE_CODE_DEBUG') == '1':
                    print("Initializing ToolVectorDB with sentence-transformers model: all-MiniLM-L6-v2")
                # Use singleton pattern to avoid reloading the model
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                self.is_ready = True
                self.embedding_dim = 384

        self.tools: Dict[str, Dict[str, Any]] = {}
        self.guidance_hints: Dict[str, Dict[str, Any]] = {}
        self.vectors: np.ndarray = None
        self.tool_names: List[str] = []
        
        # Initialize embedding cache
        self._init_embedding_cache()
        
        self._initialized = True
        
        # Initialize guidance hints from hardcoded hints
        self._initialize_guidance_hints()

    def _init_embedding_cache(self):
        """Initialize the embedding cache system."""
        try:
            from py_classes.globals import g
            cache_dir = Path(g.CLIAGENT_PERSISTENT_STORAGE_PATH)
            cache_dir.mkdir(exist_ok=True)
            self.embedding_cache_path = cache_dir / "embedding_cache.json"
            
            # Load existing cache
            if self.embedding_cache_path.exists():
                with open(self.embedding_cache_path, 'r') as f:
                    self.embedding_cache = json.load(f)
            else:
                self.embedding_cache = {}
        except Exception:
            # Fallback to memory-only cache if file operations fail
            self.embedding_cache = {}
            self.embedding_cache_path = None

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text."""
        return hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()

    def _save_embedding_cache(self):
        """Save the embedding cache to disk."""
        if self.embedding_cache_path:
            try:
                with open(self.embedding_cache_path, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_cache = {}
                    for key, value in self.embedding_cache.items():
                        if isinstance(value, np.ndarray):
                            serializable_cache[key] = value.tolist()
                        else:
                            serializable_cache[key] = value
                    json.dump(serializable_cache, f)
            except Exception:
                pass  # Silent fail for cache saves

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates a vector embedding for the given text using Ollama BGE-M3 or sentence-transformers fallback."""
        if not self.is_ready:
            return np.zeros(self.embedding_dim)
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.embedding_cache:
            cached_embedding = self.embedding_cache[cache_key]
            if isinstance(cached_embedding, list):
                return np.array(cached_embedding, dtype=np.float32)
            return cached_embedding
        
        # Generate new embedding
        embedding = None
        if self.use_ollama:
            try:
                from py_classes.ai_providers.cls_ollama_interface import OllamaClient
                # Generate embedding without debug output for caching
                embedding = OllamaClient.generate_embedding(text, model=self.model_name, silent=True)
                
                if embedding is None or len(embedding) == 0:
                    embedding = np.zeros(self.embedding_dim)
                else:
                    embedding = np.array(embedding, dtype=np.float32)
            except Exception as e:
                if os.environ.get('CLAUDE_CODE_DEBUG') == '1':
                    print(f"Error getting Ollama embedding, falling back to zeros: {e}")
                embedding = np.zeros(self.embedding_dim)
        else:
            # Use sentence-transformers fallback
            embedding = self.embedding_model.encode(text, show_progress_bar=False)
        
        # Cache the result
        if embedding is not None:
            self.embedding_cache[cache_key] = embedding
            self._save_embedding_cache()
        
        return embedding

    def add_tool(self, tool_class: Type[UtilBase]):
        """Adds a tool to the database by creating and storing its embedding."""
        if not self.is_ready:
            return
            
        name = UtilBase.get_name(tool_class)
        description = UtilBase.get_description(tool_class)
        
        # Get both hardcoded metadata and adaptive keywords
        get_metadata_func = getattr(tool_class, 'get_metadata', lambda: {})
        metadata = get_metadata_func()
        
        # Try to get adaptive keywords, fall back to hardcoded
        keywords = metadata.get('keywords', [])
        use_cases = metadata.get('use_cases', [])
        
        try:
            from py_classes.cls_adaptive_keywords import adaptive_keywords
            adaptive_keywords_list = adaptive_keywords.get_keywords_for_tool(tool_class.__name__)
            if adaptive_keywords_list:
                keywords = adaptive_keywords_list
        except Exception:
            pass  # Use hardcoded keywords if adaptive system fails
        
        text_parts = [f"Tool name: {name}", f"Description: {description}"]
        if keywords:
            text_parts.append(f"Keywords: {', '.join(keywords)}")
        if use_cases:
            text_parts.append(f"Use cases: {', '.join(use_cases)}")
        
        text_to_embed = ". ".join(text_parts)
        embedding = self._get_embedding(text_to_embed)
        
        self.tools[name] = {
            "class": tool_class,
            "description": description,
            "embedding": embedding,
            "metadata": metadata,
            "adaptive_keywords": keywords  # Store adaptive keywords separately
        }
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuilds the numpy array of vectors for efficient searching."""
        if not self.tools:
            return
        
        self.tool_names = list(self.tools.keys())
        # Stack all embeddings into a single numpy array, ensuring consistent float32 dtype
        embeddings = [self.tools[name]["embedding"] for name in self.tool_names]
        self.vectors = np.vstack(embeddings).astype(np.float32)

    def _get_hardcoded_hints(self, query: str, top_k: int = 3) -> List[str]:
        """
        Check if the query matches any hardcoded hints and return prioritized tool names.
        
        Args:
            query: The search query
            top_k: Maximum number of hints to return
            
        Returns:
            List of tool names that match hardcoded hints, empty if no matches
        """
        query_lower = query.lower()
        matched_tools = set()
        
        # Only use vector search - no hardcoded hints
        
        # Convert to list and limit total results
        result_tools = list(matched_tools)[:top_k]
        
        # Ensure tools actually exist in our database
        available_tools = [tool for tool in result_tools if tool in self.tools]
        
        return available_tools

    def _initialize_guidance_hints(self):
        """Initialize guidance hints from hardcoded hints into persistent vector storage."""
        if not self.is_ready:
            return
            
        for keyword, hint_data in self.HARDCODED_HINTS.items():
            # Extract guidance text from hint_data
            guidance_text = None
            if isinstance(hint_data, list) and hint_data:
                # Find the first string that looks like guidance (contains code blocks or detailed instructions)
                for item in hint_data:
                    if isinstance(item, str) and (len(item) > 20 or '```' in item):
                        guidance_text = item
                        break
            
            if guidance_text:
                # Create embedding for the keyword to match against queries
                embedding = self._get_embedding(f"guidance hint for {keyword}")
                
                self.guidance_hints[keyword] = {
                    "keyword": keyword,
                    "guidance_text": guidance_text,
                    "embedding": embedding
                }

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches for the most relevant tools based on a query using vector search.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
            
        Returns:
            A list of dictionaries, each representing a relevant tool.
        """
        if not self.is_ready or self.vectors is None or len(self.vectors) == 0:
            return []

        query_embedding = self._get_embedding(query)
        
        # Ensure consistent float32 dtype for both query and stored vectors
        query_embedding = query_embedding.astype(np.float32)
        
        # Calculate cosine similarity between query and all tool vectors
        # Reshape for single query against multiple vectors
        similarities = cos_sim(query_embedding.reshape(1, -1), self.vectors)[0]
        
        # Get the indices of the top_k most similar vectors
        # Ensure we don't ask for more items than exist
        k = min(top_k, len(similarities))
        if k == 0:
            return []
        
        # Get indices sorted by similarity (descending order)
        sorted_indices = np.argsort(similarities)
        top_indices = sorted_indices[-k:].tolist()
        top_indices.reverse()  # Reverse to get descending order

        results = []
        for index in top_indices:
            tool_name = self.tool_names[index]
            tool_info = self.tools[tool_name]
            results.append({
                "name": tool_name,
                "score": float(similarities[index]),
                "class": tool_info["class"],
                "metadata": tool_info["metadata"]
            })
        
        # Let adaptive learning handle prioritization - no hardcoded biases
        
        return results
    
    def get_relevant_guidance(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Search for relevant guidance hints based on query.
        
        Args:
            query: The search query
            top_k: Number of guidance hints to return
            
        Returns:
            List of relevant guidance hints
        """
        if not self.is_ready or not self.guidance_hints:
            return []
            
        query_embedding = self._get_embedding(query)
        query_embedding = query_embedding.astype(np.float32)
        
        # Collect all guidance embeddings
        hint_keys = list(self.guidance_hints.keys())
        if not hint_keys:
            return []
            
        hint_embeddings = np.vstack([
            self.guidance_hints[key]["embedding"] for key in hint_keys
        ]).astype(np.float32)
        
        # Calculate similarities
        similarities = cos_sim(query_embedding.reshape(1, -1), hint_embeddings)[0]
        
        # Get ALL hints sorted by similarity (we'll filter for uniqueness below)
        sorted_indices = np.argsort(similarities)
        all_indices = sorted_indices.tolist()
        all_indices.reverse()  # Get descending order (highest similarity first)
        
        results = []
        seen_guidance_texts = set()  # Track unique guidance texts to avoid duplicates
        
        # Iterate through ALL sorted results until we have top_k unique guidance texts
        for index in all_indices:
            # Stop if we have enough unique results
            if len(results) >= top_k:
                break
                
            hint_key = hint_keys[index]
            hint_info = self.guidance_hints[hint_key]
            guidance_text = hint_info["guidance_text"]
            
            # Skip if we've already seen this exact guidance text
            if guidance_text not in seen_guidance_texts:
                seen_guidance_texts.add(guidance_text)
                results.append({
                    "keyword": hint_info["keyword"],
                    "guidance_text": guidance_text,
                    "score": float(similarities[index])
                })
            
        return results
    
    def record_tool_selection(self, tool_name: str, query: str):
        """
        Record that a tool was selected for a specific query to improve future matching.
        This is the core self-evolution mechanism - every selection teaches the system.
        
        Args:
            tool_name: Name of the selected tool
            query: The original search query that led to tool selection
        """
        try:
            from py_classes.cls_adaptive_keywords import adaptive_keywords
            
            # Find the actual class name from the tool name
            actual_class_name = None
            for name, tool_info in self.tools.items():
                if name == tool_name:
                    actual_class_name = tool_info["class"].__name__
                    break
            
            if actual_class_name:
                # Record the selection for adaptive learning
                adaptive_keywords.record_tool_selection(actual_class_name, query)
                
                # Immediately refresh the tool's embedding with updated keywords
                self._refresh_tool_embedding(actual_class_name)
                
                # Debug output for evolution tracking
                import os
                if os.environ.get('CLAUDE_CODE_DEBUG') == '1':
                    print(f"Evolved: '{query}' -> {tool_name}")
                
        except Exception as e:
            # Don't let adaptive keyword errors break tool selection
            import os
            if os.environ.get('CLAUDE_CODE_DEBUG') == '1':
                print(f"Warning: Failed to record tool selection for evolution: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the self-evolution progress."""
        try:
            from py_classes.cls_adaptive_keywords import adaptive_keywords
            return adaptive_keywords.get_statistics()
        except Exception as e:
            return {"error": f"Could not get learning stats: {e}"}
    
    def _refresh_tool_embedding(self, class_name: str):
        """Refresh a tool's embedding after its keywords have been updated."""
        try:
            from py_classes.cls_adaptive_keywords import adaptive_keywords
            
            # Find the tool by class name
            for name, tool_info in self.tools.items():
                if tool_info["class"].__name__ == class_name:
                    # Get updated keywords
                    updated_keywords = adaptive_keywords.get_keywords_for_tool(class_name)
                    
                    if updated_keywords:
                        # Rebuild embedding with new keywords
                        description = tool_info["description"]
                        metadata = tool_info["metadata"]
                        use_cases = metadata.get('use_cases', [])
                        
                        text_parts = [f"Tool name: {name}", f"Description: {description}"]
                        text_parts.append(f"Keywords: {', '.join(updated_keywords)}")
                        if use_cases:
                            text_parts.append(f"Use cases: {', '.join(use_cases)}")
                        
                        text_to_embed = ". ".join(text_parts)
                        new_embedding = self._get_embedding(text_to_embed)
                        
                        # Update the tool's embedding and keywords
                        self.tools[name]["embedding"] = new_embedding
                        self.tools[name]["adaptive_keywords"] = updated_keywords
                        
                        # Rebuild the entire index
                        self._rebuild_index()
                        break
                        
        except Exception as e:
            # Don't let embedding refresh errors break the system
            import os
            if os.environ.get('CLAUDE_CODE_DEBUG') == '1':
                print(f"Warning: Failed to refresh tool embedding: {e}")