# tools/memory.py
"""
A self-contained vector database memory tool for the CLI-Agent.
Handles its own database connection, querying, and memory compaction logic.
"""
import json
import re
import uuid
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path to ensure 'core' can be imported.
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from termcolor import colored
from core.globals import g

class memory:
    """
    Manages and searches the agent's long-term memory using a persistent vector database.
    """
    _client = None
    _collection = None
    _initialized = False

    @classmethod
    def _initialize_db(cls):
        """Initializes the ChromaDB client and collection as a singleton."""
        if cls._initialized:
            return

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            cache_dir = Path(g.CLIAGENT_PERSISTENT_STORAGE_PATH)
            cache_dir.mkdir(exist_ok=True, parents=True)
            db_path = cache_dir / "memory_db"
            
            cls._client = chromadb.PersistentClient(path=str(db_path))

            try:
                embedding_function = embedding_functions.OllamaEmbeddingFunction(
                    url=f"{g.DEFAULT_OLLAMA_HOSTS[0]}/api/embeddings",
                    model_name="bge-m3"
                )
                embedding_function(["test"])
                logging.info(colored("  - Memory: Using Ollama embedding function (bge-m3).", "cyan"))
            except Exception:
                logging.warning(colored("  - Memory: Ollama not available. Falling back to default sentence-transformer.", "yellow"))
                embedding_function = embedding_functions.DefaultEmbeddingFunction()

            cls._collection = cls._client.get_or_create_collection(
                name="conversation_memory",
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"} # Use cosine distance for similarity
            )
            
            count = cls._collection.count()
            logging.info(colored(f"  - Memory: Initialized with {count} stored memories.", "blue"))
            cls._initialized = True
            
        except Exception as e:
            logging.error(f"Failed to initialize Memory DB: {e}", exc_info=True)
            cls._initialized = False

    @staticmethod
    def get_delim() -> str:
        return 'memory'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "memory",
            "description": ("Searches the agent's long-term memory using semantic search. "
                          "Perform semantic queries, filter by metadata (e.g., WHERE topics CONTAINS 'docker'), "
                          "and limit results. Results are sorted by latest first by default."),
            "example": "<memory>what did we discuss about fixing the docker-compose file? WHERE topics CONTAINS 'docker' LIMIT 3</memory>"
        }

    @staticmethod
    def run(content: str) -> str:
        """Executes the memory search query."""
        memory._initialize_db()
        if not memory._initialized:
            return json.dumps({"status": "error", "message": "Memory database is not initialized."})

        try:
            parsed_query = memory._parse_query(content.strip())
            results = memory._query_memory(
                semantic_query=parsed_query["semantic_query"],
                filters=parsed_query["filters"],
                limit=parsed_query["limit"]
            )

            if not results:
                return json.dumps({"status": "success", "message": "No relevant memories found."})

            response_str = f"Found {len(results)} relevant memories (sorted by latest):\n\n"
            for i, res in enumerate(results):
                response_str += f"--- Memory {i+1} ---\n"
                response_str += f"📅 {res['timestamp']}\n"
                response_str += f"📝 {res['summary']}\n"
                if res['metadata']:
                    # Safely format metadata, handling potential JSON strings
                    meta_parts = []
                    for k, v in res['metadata'].items():
                        try:
                            # If v is a string that looks like a list/dict, parse it for cleaner display
                            if isinstance(v, str) and (v.strip().startswith('[') or v.strip().startswith('{')):
                                parsed_v = json.loads(v)
                                meta_parts.append(f"{k}: {json.dumps(parsed_v)}")
                            else:
                                meta_parts.append(f"{k}: {v}")
                        except json.JSONDecodeError:
                             meta_parts.append(f"{k}: {v}")
                    metadata_str = ", ".join(meta_parts)
                    if metadata_str:
                        response_str += f"🏷️  {metadata_str}\n"
                response_str += f"⚡ Relevance: {res.get('relevance_score', 0):.3f}\n\n"

            return json.dumps({"status": "success", "result": response_str.strip()})
            
        except Exception as e:
            logging.error(f"Memory search failed: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": f"Memory search failed: {str(e)}"})
        
    @classmethod
    async def compact_and_store(cls, chat):
        """(Example Placeholder) Summarizes conversation turns and stores them in memory."""
        # This async method would be called by the main loop during compaction.
        # Its full implementation depends on LlmRouter, Chat, and Role from `core`.
        pass

    @classmethod
    def _add_memory(cls, summary_text: str, metadata: Dict[str, Any]):
        """Internal method to add a new memory to the ChromaDB collection."""
        if not cls._initialized: cls._initialize_db()
        try:
            memory_id = str(uuid.uuid4())
            sanitized_metadata = {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in metadata.items()}
            sanitized_metadata['timestamp'] = datetime.now().timestamp()
            
            cls._collection.add(ids=[memory_id], documents=[summary_text], metadatas=[sanitized_metadata])
        except Exception as e:
            logging.error(f"Failed to add memory: {e}")

    @classmethod
    def _query_memory(cls, semantic_query: str, filters: Optional[Dict] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Performs a hybrid search on the memory, ensuring results are sorted by date."""
        try:
            # Fetch more results than needed to ensure we can sort by date accurately after retrieval.
            query_params = {
                "n_results": limit * 3,
                "include": ["metadatas", "documents", "distances"]
            }
            if semantic_query: query_params["query_texts"] = [semantic_query]
            if filters: query_params["where"] = filters

            results = cls._collection.query(**query_params) if semantic_query else cls._collection.get(limit=query_params['n_results'], where=query_params.get('where'))

            if not results or not results.get('ids') or not results['ids'][0]: return []

            formatted = []
            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                ts_dt = datetime.fromtimestamp(meta.pop('timestamp', 0))
                
                # Calculate relevance score: 1 - distance (for cosine similarity)
                dist = (results.get('distances') or [[1.0]])[0][i] if 'distances' in results else 1.0
                
                formatted.append({
                    "timestamp_dt": ts_dt, # For sorting
                    "timestamp": ts_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": results['documents'][0][i],
                    "metadata": meta,
                    "relevance_score": 1.0 - dist
                })

            # Sort by timestamp (latest first) and then return the requested number of results.
            formatted.sort(key=lambda x: x['timestamp_dt'], reverse=True)
            return formatted[:limit]
        except Exception as e:
            logging.error(f"Memory query failed: {e}", exc_info=True)
            return []

    @classmethod
    def _parse_query(cls, text_input: str) -> dict:
        """Parses the natural language query into structured components."""
        query = {"semantic_query": text_input, "filters": {}, "limit": 5}
        
        limit_m = re.search(r'\sLIMIT\s+(\d+)\s*$', text_input, re.I)
        if limit_m:
            query["limit"] = int(limit_m.group(1))
            text_input = text_input[:limit_m.start()]

        where_m = re.search(r'\sWHERE\s+(.*)$', text_input, re.I)
        if where_m:
            clause = where_m.group(1).strip()
            # This simplified parser handles key CONTAINS 'value'.
            # NOTE: For ChromaDB, metadata filters are exact matches. Semantic search on the
            # query text is more effective for "contains"-style searches.
            for part in re.split(r'\s+AND\s+', clause, flags=re.I):
                filter_m = re.match(r"(\w+)\s+CONTAINS\s+'([^']*)'", part, re.I)
                if filter_m:
                    key, value = filter_m.groups()
                    # A true `CONTAINS` would require storing metadata as lists of strings
                    # and using the `$contains` operator. This is a simplification.
                    query["filters"][key.lower()] = value
            text_input = text_input[:where_m.start()]
        
        query["semantic_query"] = text_input.strip()
        return query