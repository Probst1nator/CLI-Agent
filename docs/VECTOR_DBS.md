# Vector Database System in CLI-Agent

This document explains how embeddings and vector databases are used throughout the CLI-Agent framework to provide intelligent, context-aware functionality.

## Core Infrastructure

### Vector Database (`infrastructure/vector_db/vector_database.py`)
- **ToolVectorDB**: Central embedding storage using ChromaDB
- **Embedding Model**: `embeddinggemma:latest` (with fallbacks to `nomic-embed-text:latest`, `bge-m3:latest`)
- **Storage**: Persistent ChromaDB collections for tools, guidance, and documents
- **Learning**: Records user selections to improve future recommendations

### Embedding Generation (`core/providers/cls_ollama_interface.py`)
```python
OllamaClient.generate_embedding(text, model="embeddinggemma:latest")
```
- **Models Supported**: EmbeddingGemma, Nomic-Embed-Text, BGE-M3
- **Fallback System**: Automatic model switching on failures
- **Caching**: Embeddings cached for performance optimization

## Key Features Using Vector Search

### 1. Smart Tool Selection
**Location**: `agent/utils_manager/utils_manager.py`
```python
def get_relevant_utils(self, query: str, top_k: int = 3)
```

**Purpose**: Semantic search through available utility tools
- User query: "extract text from PDF" 
- System finds: PDFUtils, TextExtractor, DocumentProcessor
- **Intelligence**: Learns from selections to improve recommendations
- **Fallback**: Returns all tools if vector search fails

### 2. Strategic Playbook System  
**Location**: `agent/playbook/playbook_manager.py`
```python
def get_relevant_playbook(self, query: str, threshold: float = 0.75)
```

**Purpose**: Matches queries to predefined workflow templates
- User query: "debug networking issue"
- System finds: Network troubleshooting playbook (if >75% similarity)
- **Embeddings**: Each playbook embedded by strategic intent
- **Threshold-based**: Only suggests high-confidence matches

### 3. RAG Document Retrieval
**Location**: `shared/utils/rag/rag_utils.py`
```python
def retrieve_and_augment(collection, user_query, retrieve_n_results=5)
```

**Purpose**: Semantic search through document collections
- **Process**: Embed query → ChromaDB search → context augmentation
- **Applications**: PDFs, documentation, knowledge bases
- **Reranking**: Cross-encoder reranking for improved relevance
- **Integration**: Feeds relevant context to LLM responses

### 4. Command History Intelligence
**Location**: `shared/cmd_execution.py`

**Purpose**: Semantic search through bash command history
- **Storage**: Command embeddings with execution context
- **Search**: Find similar commands by intent, not just keywords
- **Example**: Query "large files" finds `find`, `du`, `ncdu` commands

### 5. Audio Processing Context
**Location**: `shared/audio/audio_utils.py`

**Purpose**: Embeds audio transcription commands for tool routing
- **Integration**: Links audio processing to utility discovery
- **Context**: Audio commands get semantic understanding

## Search Applications

### Web Search Integration (`tools/searchweb.py`)
- **Hybrid Search**: Keyword + semantic search combination
- **Context Awareness**: Understands search intent beyond exact terms

### Memory System (`tools/disabled/memory.py`)  
- **Long-term Memory**: Semantic search through conversation history
- **Context Retrieval**: Find relevant past interactions
- **Ollama Integration**: Uses local embedding models

## User Experience Benefits

### Smart Discovery
```
User: "convert video to audio"
System: → FFmpeg tools, audio utilities (automatic discovery)
```

### Contextual Guidance  
```
User: "database connection failing"
System: → Database debugging playbooks + relevant tools
```

### Semantic Command Search
```
User: "find large files"
System: → `find / -size +100M`, `du -sh *`, `ncdu` (from history)
```

### Document Intelligence
```
User: "API authentication" 
System: → Relevant sections from all documentation
```

## Technical Implementation

### Embedding Dimensions
- **EmbeddingGemma**: 768 dimensions
- **Nomic-Embed-Text**: 768 dimensions  
- **BGE-M3**: 1024 dimensions

### Similarity Calculation
- **Method**: Cosine similarity
- **Implementation**: NumPy-based vector operations
- **Performance**: Cached embeddings for repeated queries

### Error Handling
- **Embedding Failures**: Automatic fallback to alternative models
- **Vector Search Failures**: Graceful degradation to non-semantic methods
- **Dimension Mismatches**: Model compatibility checking

### Storage
- **ChromaDB Collections**: Persistent storage across sessions
- **Metadata**: Associated context, timestamps, usage statistics
- **Learning Data**: User selection patterns for recommendation improvement

## Configuration

### Environment Setup
```python
# core/globals.py
DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"
```

### Model Availability Check
```bash
curl -s http://localhost:11434/api/tags
curl -s -X POST http://localhost:11434/api/embed -d '{"model": "embeddinggemma:latest", "input": "test"}'
```

## Troubleshooting

### Common Issues
1. **"Model does not support embeddings"**: Update Ollama version or use fallback models
2. **Dimension mismatches**: Reset vector database singleton
3. **Performance issues**: Check embedding cache configuration
4. **Empty search results**: Verify model availability and ChromaDB initialization

### Diagnostics
- Check `vector_db.is_ready` status
- Monitor embedding generation times
- Verify ChromaDB collection counts
- Test with fallback models

The vector database system provides an intelligent semantic layer that understands user intent and delivers contextually relevant suggestions across all CLI-Agent features, transforming it from a simple tool collection into an intelligent assistant.