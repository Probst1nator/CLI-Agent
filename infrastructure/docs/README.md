# Infrastructure Documentation

Supporting infrastructure for CLI-Agent including rate limiting, vector databases, and remote hosting capabilities.

## Architecture Overview

Infrastructure provides foundational services:
- **Rate Limiting**: API quota management across AI services
- **Vector Database**: Semantic search and embeddings storage
- **Remote Host**: Distributed computing and remote AI model hosting

## Directory Structure

```
infrastructure/
├── rate_limiting/
│   └── cls_rate_limit_tracker.py
├── vector_db/
│   └── vector_database.py
└── remote_host/
    └── (remote hosting components)
```

## Rate Limiting

### Rate Limit Tracker (`rate_limiting/cls_rate_limit_tracker.py`)

**Purpose**: Manage API call quotas and request throttling

**Features**:
- **Provider-specific limits**: Different limits per AI service
- **Request throttling**: Automatic rate control
- **Quota tracking**: Monitor usage against limits
- **Time window management**: Sliding window rate limiting
- **Retry logic**: Automatic retry with backoff

**Providers tracked**:
- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)
- Groq
- Brave Search
- Other external APIs

**Integration**:
- Used by `core/llm_router.py`
- Integrated with provider interfaces
- Automatic enforcement before API calls

**Rate limit patterns**:
```python
# Check if request allowed
if rate_limiter.is_allowed(provider="anthropic"):
    response = make_api_call()
else:
    wait_time = rate_limiter.time_until_allowed(provider="anthropic")
    # Wait or return error
```

**Configuration**:
- Per-provider limits configurable
- Time windows adjustable
- Burst allowances supported

## Vector Database

### Vector Database (`vector_db/vector_database.py`)

**Purpose**: Semantic search and embeddings storage using ChromaDB

**Features**:
- **Semantic search**: Find similar content by meaning, not just keywords
- **Multiple collections**: Organize embeddings by domain
- **Embedding generation**: Automatic via Ollama or other providers
- **Similarity matching**: Cosine similarity, L2 distance
- **Metadata filtering**: Search with constraints
- **Persistent storage**: ChromaDB-backed persistence

**Use cases**:

1. **Tool Discovery** (ToolVectorDB):
   - Embed tool descriptions
   - Find relevant tools based on query
   - Context-aware suggestions

2. **Playbook Management**:
   - Store multi-step workflow templates
   - Retrieve similar procedures
   - Strategic guidance

3. **Search Caching** (SearchResultsVectorDB):
   - Cache web search results
   - Semantic retrieval of past searches
   - Reduce API calls
   - See: `/shared/docs/SEARCHWEB_VECTOR_DB.md`

**Integration**:
```python
from infrastructure.vector_db.vector_database import VectorDatabase

# Create/connect to collection
db = VectorDatabase(collection_name="tools")

# Add documents
db.add_documents(
    ids=["tool1", "tool2"],
    documents=["Tool for X", "Tool for Y"],
    metadatas=[{"type": "file"}, {"type": "web"}]
)

# Query
results = db.query(
    query_text="find file tools",
    n_results=5
)
```

**Storage**:
- Default location: `./chroma/` (ChromaDB data)
- Configurable path
- Persistent across sessions

**Embedding providers**:
- Ollama (default: embeddinggemma:latest)
- OpenAI embeddings
- Custom embedding functions

## Remote Host

### Remote AI Model Hosting (`remote_host/`)

**Purpose**: Distributed computing and remote inference capabilities

**Features**:
- **Remote model execution**: Run models on remote servers
- **Distributed load**: Balance across multiple hosts
- **Network inference**: API-based remote calls
- **Fallback support**: Try remote, fall back to local

**Use cases**:
- Offload heavy models to powerful servers
- Access models not available locally
- Distribute inference load
- Development/production separation

**Configuration**:
- Host endpoints configurable
- Authentication support
- Timeout and retry settings

## Integration with Core

### LLM Router Integration

Infrastructure services integrate with LLM router:

```python
# Rate limiting before API call
if not rate_limiter.is_allowed(provider):
    raise RateLimitException()

# Vector DB for tool discovery
relevant_tools = tool_vector_db.query(user_query)

# Remote host as provider
remote_provider = RemoteHostProvider(endpoint="https://...")
```

### Provider System Integration

Provider interfaces use infrastructure:
- Rate limiting before requests
- Remote hosts as additional providers
- Vector DB for model discovery caching

## Monitoring and Statistics

### Rate Limiting Stats

```python
stats = rate_limiter.get_statistics()
# Returns:
# - Requests per provider
# - Rate limit hits
# - Average wait times
# - Quota utilization
```

### Vector DB Stats

```python
stats = vector_db.get_statistics()
# Returns:
# - Collection size
# - Query performance
# - Cache hit rate
# - Storage usage
```

## Configuration

### Rate Limiting Config

Typically in `core/globals.py` or environment:
```python
RATE_LIMITS = {
    "anthropic": {"calls": 50, "period": 60},  # 50/min
    "openai": {"calls": 60, "period": 60},     # 60/min
    "groq": {"calls": 100, "period": 60},      # 100/min
}
```

### Vector DB Config

```python
VECTOR_DB_PATH = "./chroma"
EMBEDDING_MODEL = "embeddinggemma:latest"
```

### Remote Host Config

```python
REMOTE_HOSTS = [
    {"url": "https://host1.example.com", "api_key": "..."},
    {"url": "https://host2.example.com", "api_key": "..."},
]
```

## Scaling Considerations

### Rate Limiting

- Adjust limits based on API tier
- Implement token bucket for bursts
- Monitor and alert on limit hits

### Vector Database

- Index optimization for large collections
- Partition collections by domain
- Regular maintenance and compaction

### Remote Hosting

- Load balancing across hosts
- Health checks for availability
- Graceful degradation on failures

## Error Handling

### Rate Limit Errors

```python
try:
    response = make_api_call()
except RateLimitException as e:
    wait_time = e.retry_after
    # Wait or queue for later
```

### Vector DB Errors

```python
try:
    results = vector_db.query(query)
except ChromaDBException as e:
    logger.error(f"Vector DB error: {e}")
    # Fall back to keyword search
```

### Remote Host Errors

```python
try:
    response = remote_host.generate(prompt)
except RemoteHostException as e:
    logger.warning(f"Remote host failed: {e}")
    # Fall back to local model
```

## Performance Optimization

### Rate Limiting

- Cache rate limit state in memory
- Batch requests when possible
- Use request queuing for smoothing

### Vector Database

- Precompute embeddings for static content
- Use appropriate similarity metrics
- Limit result set size
- Enable collection caching

### Remote Hosting

- Connection pooling
- Request batching
- Async/await for concurrency
- Caching of remote responses

## Related Documentation

- Vector DB search caching: `/shared/docs/SEARCHWEB_VECTOR_DB.md`
- LLM router: `/core/docs/LLM_ROUTER_API.md`
- Provider system: `/core/providers/docs/README.md`
- Infrastructure status: `/docs/INFRASTRUCTURE_STATUS.md` (may be moved here)

## Future Enhancements

### Rate Limiting

- Distributed rate limiting (Redis-based)
- Per-user quotas
- Dynamic limit adjustment based on usage patterns

### Vector Database

- Multi-modal embeddings (text + image)
- Hybrid search (semantic + keyword)
- Real-time indexing

### Remote Hosting

- Kubernetes-based auto-scaling
- Multi-region deployment
- Advanced load balancing
