# SearchWeb Vector Database Integration

## Overview

The SearchWeb tool provides semantic search caching with comprehensive provenance tracking and audit trails. Uses ChromaDB + embeddinggemma:latest for local vector storage, reducing API calls while building an auditable knowledge graph.

> **⚠️ Breaking Change**: The SearchWeb tool is now fully asynchronous. The `run()` method must be called with `await`.

## Architecture

### Core Components

1. **SearchResultsVectorDB** (`shared/utils/search/search_vector_db.py`)
   - Manages two ChromaDB collections for search results and summaries
   - Handles embedding generation using Ollama's `embeddinggemma:latest` model
   - Provides semantic search and similarity matching

2. **Enhanced SearchWeb Tool** (`tools/searchweb.py`)
   - Integrates vector database queries before API calls
   - Stores fresh results and generated summaries
   - Handles duplicate detection and result merging

### Database Collections

#### 1. Web Search Results (`web_search_results`)
Raw search data with enhanced provenance metadata:

```json
{
  "document": "title + description + snippets (for embedding)",
  "metadata": {
    "retrieved_at": "2025-01-15T14:30:45.123Z",
    "source_api": "BraveSearchAPI",
    "source_api_endpoint": "web|images",
    "source_query": "original search query",
    "search_id": "uuid-linking-results-from-same-api-call",
    "original_rank": 3,
    "url": "https://example.com",
    "title": "Result title",
    "description": "Result description",
    "search_type": "text|images|files|cited text|any",
    "embedding_model": "embeddinggemma:latest"
  },
  "embedding": [768-dimensional vector]
}
```

#### 2. Search Summaries (`search_summaries`)
AI-generated summaries with audit trail to source documents:

```json
{
  "document": "generated summary text",
  "metadata": {
    "created_at": "2025-01-15T14:31:12.456Z",
    "summary_method": "llm|cited|image_list",
    "generator_model": "claude-3-sonnet-20240229",
    "original_query": "search query",
    "search_type": "text|images|files|cited text|any",
    "source_result_ids": ["result_hash1", "result_hash2"],
    "source_result_count": 8,
    "embedding_model": "embeddinggemma:latest"
  },
  "embedding": [768-dimensional vector]
}
```

## Query Processing Flow

```
User Query → Vector Search (similarity ≥ 0.8) → API Call (if needed) → Ranking → Summary Generation → Audit Trail Storage
```

### Execution Steps

1. **Vector Search**: Query `web_search_results` with embeddinggemma embeddings, filter by similarity ≥ 0.8
2. **API Supplement**: If cached < MAX_RESULTS(10), call BraveSearchAPI async, batch-embed new results
3. **Intelligent Ranking**: Sort by (fresh=True, similarity_score DESC), limit to MAX_RESULTS
4. **Summary Phase**: Check existing summaries → Generate if needed → Store with source_result_ids
5. **Audit Trail**: All operations logged with ISO timestamps, model tracking, provenance links

## Configuration

```python
VECTOR_DB_PATH = "~/.cli-agent/searchweb_vectordb"
EMBEDDING_MODEL = "embeddinggemma:latest"
SIMILARITY_THRESHOLD = 0.8
MAX_RESULTS = 10
```

**Dependencies**: ChromaDB, Ollama, Brave API Key

## Usage

```xml
<searchweb><query>python web frameworks</query><type>text</type></searchweb>
```
→ LLM summary with source traceability

```xml
<searchweb><query>mountain landscapes</query><type>images</type></searchweb>
```
→ Formatted image list with direct URLs

```xml
<searchweb><query>machine learning</query><type>cited text</type></searchweb>
```
→ Numbered citations with URLs

## Performance Benefits

- **Cache Efficiency**: Similar queries skip API calls (similarity ≥ 0.8)
- **Batch Processing**: ~50% faster embedding generation for multiple results
- **Async Operations**: Non-blocking API + LLM calls
- **Duplicate Prevention**: Automatic summary deduplication saves LLM costs
- **Progressive Knowledge**: Database improves over time without intervention

## Storage & Debugging

**Location**: `~/.cli-agent/searchweb_vectordb/`

**Debug Output**:
```
🔎 Performing 'text' search for: 'python web frameworks'...
📊 Found 8 high-quality cached results (similarity ≥ 0.8)
🌐 Retrieved 20 fresh results from Brave API
💾 Stored new results in vector database
✅ Found and using existing summary from vector database
```

## Error Handling

**Graceful Degradation**: No ChromaDB → direct API calls | No Ollama → disable caching | No API key → cached only

## Maintenance

```bash
# View database statistics
python scripts/manage_search_cache.py stats

# Audit provenance trail for summaries
python scripts/manage_search_cache.py audit "search_term" --verbose

# List recent cache entries with enhanced metadata
python scripts/manage_search_cache.py list-recent --limit 10

# Remove entries older than 30 days
python scripts/manage_search_cache.py prune-by-date 30

# Clear all cache data
python scripts/manage_search_cache.py clear-all
```

**Programmatic Access**:
```python
from shared.utils.search.search_vector_db import SearchResultsVectorDB
stats = SearchResultsVectorDB().get_stats()
```

### ✅ Enhanced Provenance Tracking
- **ISO 8601 Timestamps**: Human-readable, unambiguous time tracking
- **API Source Tracking**: Explicit `source_api` and `source_api_endpoint` fields
- **Search Operation IDs**: `search_id` UUID linking results from same API call
- **Audit Trail**: `source_result_ids` connecting summaries to source documents
- **Model Tracking**: `generator_model` field capturing which LLM created summaries

### ✅ Audit Infrastructure
- **Provenance Queries**: `manage_search_cache.py audit` command traces summaries to sources
- **Enhanced Metadata Display**: Shows API endpoints, models, timestamps in listings
- **Complete Knowledge Graph**: Every summary traceable to exact source documents and models

## Troubleshooting

```bash
# Check dependencies
ollama list | grep embedding
python -c "import chromadb; print('ChromaDB available')"

# Test database
python -c "from shared.utils.search.search_vector_db import SearchResultsVectorDB; print(SearchResultsVectorDB().get_stats())"
```

**Common Issues**:
- No cached results → Similarity threshold too high (0.8) or model inconsistency
- Slow embeddings → Ollama overloaded or wrong model
- Database corruption → Delete `~/.cli-agent/searchweb_vectordb/`

## Integration

**Breaking Change**: SearchWeb is now async:
```python
# await searchweb.run(content)
# OLD: searchweb.run(content)
```

**Benefits**: Cache efficiency, batch processing, audit trails, progressive knowledge building without setup requirements.