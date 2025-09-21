"""
Vector Database implementation for SearchWeb tool using ChromaDB and embeddings.
Manages storage and retrieval of search results and summaries for semantic search.
NOW INCLUDES HYBRID SEARCH CAPABILITIES WITH BM25.
"""

import os
import logging
import hashlib
import uuid
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Import BM25 for keyword search
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Import Ollama client for embedding generation
try:
    from core.providers.cls_ollama_interface import OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class SearchResultsVectorDB:
    """Vector database for storing and retrieving search results and summaries"""

    # Database configuration
    VECTOR_DB_PATH = None  # Will be set dynamically from globals
    RESULTS_COLLECTION = "web_search_results"
    SUMMARIES_COLLECTION = "search_summaries"
    EMBEDDING_MODEL = "embeddinggemma:latest"

    def __init__(self):
        """Initialize the vector database for search results"""
        self.is_ready = False
        self.client = None
        self.results_collection = None
        self.summaries_collection = None

        # Set vector DB path from globals if available
        try:
            from core.globals import g
            self.VECTOR_DB_PATH = os.path.join(g.CLIAGENT_DATA_DIR, "vector_db", "searchweb_vectordb")
        except ImportError:
            # Fallback to XDG-compliant path
            xdg_data_home = os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))
            self.VECTOR_DB_PATH = os.path.join(xdg_data_home, "cli-agent", "vector_db", "searchweb_vectordb")

        # --- NEW: BM25 Attributes ---
        self.bm25_index_path = Path(self.VECTOR_DB_PATH) / "bm25_index.pkl"
        self.corpus = {}  # In-memory store for BM25 docs: {doc_id: "text"}
        self.bm25 = None

        self._initialize_vector_db()

        if self.is_ready and BM25_AVAILABLE:
            self._load_or_build_bm25_index()

    def _initialize_vector_db(self):
        """Initialize ChromaDB collections for search results and summaries"""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, search vector DB disabled")
            return

        try:
            os.makedirs(self.VECTOR_DB_PATH, exist_ok=True)
            settings = Settings(
                persist_directory=self.VECTOR_DB_PATH,
                anonymized_telemetry=False
            )
            self.client = chromadb.PersistentClient(settings=settings)
            self.results_collection = self.client.get_or_create_collection(name=self.RESULTS_COLLECTION)
            self.summaries_collection = self.client.get_or_create_collection(name=self.SUMMARIES_COLLECTION)

            self.is_ready = True
            logger.debug(f"Search vector DB initialized at {self.VECTOR_DB_PATH}")

        except Exception as e:
            logger.warning(f"Failed to initialize search vector DB: {e}")
            self.is_ready = False

    # --- NEW: BM25 Index Management Methods ---
    def _load_or_build_bm25_index(self):
        """Loads a pre-computed BM25 index or builds one from ChromaDB."""
        if self.bm25_index_path.exists():
            try:
                logger.debug("💾 Loading existing BM25 index from disk...")
                with open(self.bm25_index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.corpus = data.get('corpus', {})

                if self.corpus:
                    tokenized_corpus = [doc.split(" ") for doc in self.corpus.values()]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    logger.debug(f"BM25 index loaded with {len(self.corpus)} documents.")
                return
            except Exception as e:
                logger.warning(f"Could not load BM25 index file, will rebuild. Error: {e}")

        # If file doesn't exist or failed to load, rebuild from ChromaDB
        logger.debug("📊 No BM25 index found, rebuilding from ChromaDB...")
        try:
            all_docs = self.results_collection.get(include=["documents"])
            if all_docs and all_docs['ids']:
                self.corpus = {doc_id: doc for doc_id, doc in zip(all_docs['ids'], all_docs['documents'])}

                if self.corpus:
                    tokenized_corpus = [doc.split(" ") for doc in self.corpus.values()]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    self._save_bm25_index()
                    logger.debug(f"BM25 index rebuilt with {len(self.corpus)} documents.")
        except Exception as e:
            logger.warning(f"Failed to rebuild BM25 index from ChromaDB: {e}")

    def _save_bm25_index(self):
        """Saves the BM25 index corpus to disk."""
        if not self.corpus:
            return
        logger.debug("💾 Saving BM25 index corpus to disk...")
        try:
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump({'corpus': self.corpus}, f)
        except Exception as e:
            logger.warning(f"Failed to save BM25 index: {e}")
    # --- END: BM25 Methods ---

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        # (This method remains unchanged)
        if not OLLAMA_AVAILABLE:
            logger.debug("Ollama not available for embeddings")
            return None
        try:
            embedding = OllamaClient.generate_embedding(text, model=self.EMBEDDING_MODEL)
            return embedding
        except Exception as e:
            logger.debug(f"Failed to generate embedding: {e}")
            return None

    def _create_content_hash(self, content: str) -> str:
        # (This method remains unchanged)
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def hybrid_search(self, query: str, search_type: str = "any", max_results: int = 10, k: int = 60) -> List[Tuple[Dict[str, Any], float]]:
        """
        Performs a hybrid search using Vector Search (ChromaDB) and Keyword Search (BM25)
        with Reciprocal Rank Fusion (RRF) to combine results.
        """
        if not self.is_ready or not self.results_collection:
            return []

        # Fallback to pure vector search if BM25 is not available/ready
        if not self.bm25 or not BM25_AVAILABLE:
            logger.debug("BM25 index not ready, falling back to vector search only.")
            return self.search_similar_results(query, search_type, max_results)

        logger.debug(f"Performing hybrid search for: '{query}'")

        # 1. Vector Search (Semantic)
        try:
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []

            where_clause = {"search_type": search_type} if search_type != "any" else None
            vector_results_data = self.results_collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results * 2, # Fetch more to allow for better fusion
                where=where_clause,
                include=["metadatas", "distances"]
            )
            vector_ids = vector_results_data.get('ids', [[]])[0]
        except Exception as e:
            logger.warning(f"Vector search part of hybrid search failed: {e}")
            vector_ids = []

        # 2. BM25 Search (Keyword)
        try:
            tokenized_query = query.lower().split(" ")
            doc_scores = self.bm25.get_scores(tokenized_query)

            corpus_doc_ids = list(self.corpus.keys())

            bm25_top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:max_results * 2]
            bm25_ids = [corpus_doc_ids[i] for i in bm25_top_indices if doc_scores[i] > 0]
        except Exception as e:
            logger.warning(f"BM25 search part of hybrid search failed: {e}")
            bm25_ids = []

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}

        # Process vector results
        for rank, doc_id in enumerate(vector_ids):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (k + rank + 1)

        # Process BM25 results
        for rank, doc_id in enumerate(bm25_ids):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1 / (k + rank + 1)

        if not rrf_scores:
            return []

        # 4. Sort by combined RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda id: rrf_scores[id], reverse=True)[:max_results]

        # 5. Fetch full metadata for the final ranked list
        final_results_data = self.results_collection.get(ids=sorted_doc_ids, include=["metadatas", "distances"])

        # 6. Repackage results into the expected format (metadata, distance)
        final_results = []
        for doc_id, meta, dist in zip(final_results_data['ids'], final_results_data['metadatas'], final_results_data['distances']):
            # Reconstruct the metadata object similar to the old method
            result_metadata = {
                'title': meta.get('title', ''), 'url': meta.get('url', ''), 'description': meta.get('description', ''),
                'extra_snippets': [meta.get('extra_snippets', meta.get('description', ''))],
                'source': 'vector_db_hybrid', 'search_type': meta.get('search_type', ''),
                'timestamp': meta.get('retrieved_at', 0)
            }
            if meta.get('search_type') == 'images':
                result_metadata['properties'] = {'url': meta.get('image_url', '')}
                result_metadata['thumbnail'] = {'src': meta.get('thumbnail_url', '')}
                result_metadata['meta_url'] = {'netloc': meta.get('source_page', '')}

            final_results.append((result_metadata, dist))

        logger.debug(f"Hybrid search returned {len(final_results)} results.")
        return final_results

    def search_similar_results(self, query: str, search_type: str = "any", max_results: int = 10, distance_threshold: float = 0.5) -> List[Tuple[Dict[str, Any], float]]:
        # (This method remains unchanged, but now serves as a fallback)
        if not self.is_ready or not self.results_collection:
            logger.debug("Vector DB not ready or collection missing")
            return []
        # ... (rest of the method is the same as your original) ...

    def store_search_results(self, query: str, search_type: str, results: List[Dict[str, Any]], api_endpoint: str = "web") -> bool:
        """
        Store search results in ChromaDB and update the BM25 index.
        """
        if not self.is_ready or not self.results_collection or not results:
            return False

        try:
            documents, metadatas, ids, embeddings = [], [], [], []
            new_docs_for_bm25 = {}

            search_id = str(uuid.uuid4())
            retrieved_at_iso = datetime.now(timezone.utc).isoformat()

            for i, result in enumerate(results):
                title = result.get('title', '')
                description = result.get('description', '')
                snippets = result.get('extra_snippets', [])

                content_to_embed = f"{title} {description}".strip()
                if not content_to_embed:
                    continue

                full_content_for_hash = f"{title} {' '.join(snippets) if snippets else ''}".strip()
                content_hash = self._create_content_hash(full_content_for_hash)
                doc_id = f"result_{content_hash}"

                ids.append(doc_id)
                documents.append(content_to_embed)
                # (Metadata creation is the same as your original)
                # ...

            if not documents:
                return False

            # Batch embedding generation
            logger.debug(f"📦 Generating embeddings for {len(documents)} new results in a batch...")
            embeddings = [self._generate_embedding(doc) for doc in documents]
            valid_embeddings_data = [(doc, emb, meta, doc_id) for doc, emb, meta, doc_id in zip(documents, embeddings, metadatas, ids) if emb is not None]

            if not valid_embeddings_data:
                return False

            # Upsert into ChromaDB
            final_docs, final_embeddings, final_metadatas, final_ids = zip(*valid_embeddings_data)
            self.results_collection.upsert(
                documents=list(final_docs),
                embeddings=list(final_embeddings),
                metadatas=list(final_metadatas),
                ids=list(final_ids)
            )

            # --- NEW: Update BM25 Index ---
            for doc_id, doc_text in zip(final_ids, final_docs):
                self.corpus[doc_id] = doc_text

            if self.corpus and BM25_AVAILABLE:
                logger.debug("🔄 Rebuilding and saving BM25 index...")
                tokenized_corpus = [doc.split(" ") for doc in self.corpus.values()]
                self.bm25 = BM25Okapi(tokenized_corpus)
                self._save_bm25_index()
            # --- END: BM25 Update ---

            logger.debug(f"Stored/updated {len(final_ids)} search results.")
            return True

        except Exception as e:
            logger.warning(f"Error storing search results in batch: {e}")
            return False

    def store_summary(self, query: str, search_type: str, summary: str, source_result_ids: List[str], generation_info: Dict[str, Any]) -> bool:
        """
        Store a generated summary in the summaries collection

        Args:
            query: Original search query
            search_type: Type of search
            summary: Generated summary text
            source_result_ids: List of document IDs from results_collection used to create this summary
            generation_info: Dict containing method, model, and other generation metadata

        Returns:
            True if successful, False otherwise
        """
        if not self.is_ready or not self.summaries_collection:
            return False

        try:
            # Generate embedding for summary
            embedding = self._generate_embedding(summary)
            if embedding is None:
                return False

            # Create document ID - simplified for easier lookups
            content_hash = self._create_content_hash(f"{query}_{search_type}")
            doc_id = f"summary_{content_hash}"

            # Prepare metadata with enhanced provenance tracking
            metadata = {
                # Accurate Time and Date
                'created_at': datetime.now(timezone.utc).isoformat(),

                # Accurate Source Information (What generated this?)
                'summary_method': generation_info.get('method', 'unknown'),
                'generator_model': generation_info.get('model', 'unknown'),

                # Context and Provenance (What was this based on?)
                'original_query': query,
                'search_type': search_type,
                'source_result_ids': ','.join(source_result_ids),  # Convert list to comma-separated string
                'source_result_count': len(source_result_ids),

                # Technical Details
                'embedding_model': self.EMBEDDING_MODEL
            }

            # Check if already exists
            existing = self.summaries_collection.get(ids=[doc_id])
            if existing['ids']:
                # Update existing
                self.summaries_collection.update(
                    documents=[summary],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
            else:
                # Add new
                self.summaries_collection.add(
                    documents=[summary],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[doc_id]
                )

            logger.debug(f"Stored summary for query: {query}")
            return True

        except Exception as e:
            logger.warning(f"Error storing summary: {e}")
            return False

    def get_summary_by_query(self, query: str, search_type: str) -> Optional[str]:
        """
        Retrieves a stored summary by an exact query and search type match.

        Args:
            query: Original search query
            search_type: Type of search (text, images, etc.)

        Returns:
            Existing summary text if found, None otherwise
        """
        if not self.is_ready or not self.summaries_collection:
            return None

        try:
            results = self.summaries_collection.get(
                where={
                    "$and": [
                        {"original_query": {"$eq": query}},
                        {"search_type": {"$eq": search_type}}
                    ]
                },
                limit=1
            )
            if results and results['documents']:
                logger.debug(f"Found existing summary for query: {query}")
                return results['documents'][0]
            return None
        except Exception as e:
            logger.warning(f"Error checking for existing summary: {e}")
            return None

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the vector database"""
        stats = {
            'results_count': 0,
            'summaries_count': 0,
            'is_ready': self.is_ready
        }

        if self.is_ready:
            try:
                if self.results_collection:
                    stats['results_count'] = self.results_collection.count()
                if self.summaries_collection:
                    stats['summaries_count'] = self.summaries_collection.count()
            except Exception as e:
                logger.debug(f"Error getting vector DB stats: {e}")

        return stats