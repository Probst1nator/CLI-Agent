# CLI-Agent RAG System: Guidance, Tools & Strategic Hints

## Executive Summary

The CLI-Agent implements a sophisticated Retrieval-Augmented Generation (RAG) system that provides contextual guidance, tool recommendations, and strategic planning through three integrated components:

1. **ToolVectorDB**: Vector-based tool discovery and guidance hints
2. **PlaybookManager**: Strategic planning with multi-step workflows  
3. **UtilsManager Integration**: Real-time context-aware suggestions

This system enables the agent to automatically discover relevant tools, provide contextual guidance, and suggest strategic approaches based on user queries using semantic similarity matching.

## System Architecture

```
CLI-Agent RAG Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                        User Input                               │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  v
┌─────────────────────────────────────────────────────────────────┐
│                   Context Analysis                              │
│                 (main.py:1762-1820)                            │
└─────┬─────────┬─────────────────────────┬─────────────────────┘
      │         │                         │
      v         v                         v
┌─────────┐ ┌─────────┐             ┌─────────────┐
│  Tool   │ │Guidance │             │  Strategic  │
│Retrieval│ │ Hints   │             │  Playbooks  │
│         │ │         │             │             │
│Vector DB│ │Vector DB│             │Vector DB    │
└─────────┘ └─────────┘             └─────────────┘
      │         │                         │
      v         v                         v
┌─────────────────────────────────────────────────────────────────┐
│                    Prompt Enhancement                           │
│          • RELEVANT TOOLS section                              │
│          • GUIDANCE HINTS section                              │
│          • STRATEGIC HINTS section                             │
└─────────────────────────────────────────────────────────────────┘
```

## Component 1: ToolVectorDB (`py_classes/cls_vector_db.py`)

### Purpose
Vector-based semantic search for tool discovery and contextual guidance storage.

### Key Features

#### Embedding Models
- **Primary**: Ollama BGE-M3 (1024 dimensions)
- **Fallback**: SentenceTransformers all-MiniLM-L6-v2 (384 dimensions)
- **Caching**: Persistent embedding cache for performance

```python
def __init__(self, model_name: str = "bge-m3", use_ollama: bool = True):
    # Auto-detects Ollama availability, falls back to sentence-transformers
    if use_ollama:
        test_embedding = OllamaClient.generate_embedding("test", model=model_name)
        # Sets is_ready=True if successful
```

#### Tool Registration & Indexing
Tools are automatically indexed with rich metadata:

```python
def add_tool(self, tool_class: Type[UtilBase]):
    # Creates embeddings from:
    # - Tool name and description
    # - Keywords (adaptive + hardcoded)
    # - Use cases and code examples
    # - Usage examples extracted from metadata
    
    text_to_embed = f"Tool name: {name}. Description: {description}. Keywords: {keywords}. Use cases: {use_cases}"
    embedding = self._get_embedding(text_to_embed)
```

#### Vector Search Algorithm
Efficient cosine similarity search with dimension consistency:

```python
def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    query_embedding = self._get_embedding(query)
    similarities_2d = self.cos_sim(query_embedding.reshape(1, -1), self.vectors)
    similarities = similarities_2d[0] if similarities_2d.ndim > 1 else similarities_2d
    top_indices = np.argsort(-similarities)[:k]  # Descending sort
```

#### Adaptive Learning System
The system evolves based on user selections:

```python
def record_tool_selection(self, tool_name: str, query: str):
    # Records successful tool selections
    # Updates adaptive keywords
    # Refreshes tool embeddings immediately
    adaptive_keywords.record_tool_selection(actual_class_name, query)
    self._refresh_tool_embedding(actual_class_name)
```

#### Guidance Hints Database
Stores successful user prompts as guidance examples:

```python
def add_guidance_example(self, tool_name: str, user_prompt: str):
    # Creates semantic embeddings for user prompts
    # Links prompts to successful tool usage
    text_to_embed = f"An example of a user task that requires the '{tool_name}' tool: {user_prompt}"
    embedding = self._get_embedding(text_to_embed)
```

### Hardcoded Bootstrap Hints
Essential system knowledge for cold start:

```python
HARDCODED_HINTS = {
    "python": ["Remember that you can execute python code...```python\nprint('Hello world')\n```"],
    "ip address": ["To display the current IP address, use...```python\nimport socket\n...```"],
    "code": ["editfile", "writefile", "viewfiles", "Remember that you can run any bash code..."],
    # ... more bootstrap hints
}
```

## Component 2: PlaybookManager (`py_classes/cls_playbook_manager.py`)

### Purpose
Strategic multi-step workflow planning using predefined and learned playbooks.

### Default Playbooks
Comprehensive strategies for common complex tasks:

```python
DEFAULT_PLAYBOOKS = [
    {
        "name": "Website Scraping and Information Extraction",
        "trigger_intent": "scrape, download, or extract specific information from a website or URL",
        "thoughts": [
            "First, I must determine if the website offers a public API...",
            "If no API is available, I will use `curl` or Python's `requests`...",
            "After fetching the HTML, I will use `BeautifulSoup`...",
            "Be mindful of `robots.txt` and terms of service...",
            "Finally, save the extracted information..."
        ]
    },
    {
        "name": "Large Log File Analysis", 
        "trigger_intent": "process, search, filter, or analyze a large text or log file",
        "thoughts": [
            "To avoid memory issues, process it line-by-line...",
            "For simple patterns, use `grep`, `awk`, and `sed`...",
            "For complex logic, use Python with file handle iteration...",
            "Start by examining first few lines with `head`..."
        ]
    }
    # ... 5 total default playbooks
]
```

### Playbook Matching Algorithm
Semantic similarity matching with configurable threshold:

```python
def get_relevant_playbook(self, query: str, threshold: float = 0.75) -> Optional[Dict[str, Any]]:
    query_embedding = self.vector_db._get_embedding(query)
    all_playbook_embeddings = np.array([p["embedding"] for p in self.playbooks])
    similarity_matrix = self.vector_db.cos_sim(query_embedding.reshape(1, -1), all_playbook_embeddings)
    
    best_index = np.argmax(scores)
    highest_score = scores[best_index]
    
    if highest_score >= threshold:
        return self.playbooks[best_index]  # Returns full playbook with thoughts
```

### Strategic Hints Integration
Playbooks provide structured thinking frameworks:

```python
# In main.py:1800-1810
playbook = playbook_manager.get_relevant_playbook(user_context, threshold=0.7)
if playbook:
    playbook_thoughts = playbook.get("thoughts", [])
    prompt_subfix += f"\n\n# STRATEGIC HINTS (SUGGESTED PLAN)\n"
    for i, thought in enumerate(playbook_thoughts):
        prompt_subfix += f"{i+1}. {thought}\n"
```

## Component 3: Main Agent Integration (`main.py`)

### Context Analysis Pipeline
The agent processes user input through three RAG layers:

```python
# main.py:1757-1820 - Context-aware hint generation
user_context = user_input if user_input else ""

# Layer 1: Tool Suggestions
guidance_prompt = utils_manager.get_relevant_tools_prompt(user_context, top_k=5)
if guidance_prompt:
    logging.info(colored("🔧 Suggested utilities based on your request:", "cyan"))
    prompt_subfix += f"\n\n# RELEVANT TOOLS\n{guidance_prompt}"

# Layer 2: Guidance Hints  
guidance_hints = utils_manager.get_relevant_guidance(user_context, top_k=2)
if guidance_hints:
    logging.info(colored("💡 Relevant guidance hints:", "cyan"))
    prompt_subfix += "\n\n# GUIDANCE HINTS (EXAMPLES)\n" + "\n\n".join(guidance_texts)

# Layer 3: Strategic Planning
playbook = playbook_manager.get_relevant_playbook(user_context, threshold=0.7)
if playbook:
    logging.info(colored("📖 Strategic guidance found:", "magenta"))
    prompt_subfix += f"\n\n# STRATEGIC HINTS (SUGGESTED PLAN)\n"
```

### User Experience Flow

1. **User Input**: "help me process a large log file to find errors"

2. **Tool Discovery**: 
   ```
   🔧 Suggested utilities based on your request:
      - viewfiles
      - editfile
      - SearchWeb
   ```

3. **Guidance Hints**:
   ```
   💡 Relevant guidance hints:
      • file: writefile, editfile, viewfiles
      • read: To perform a read operation you can use `cat`...
   ```

4. **Strategic Planning**:
   ```
   📖 Strategic guidance found:
      • Strategy: Large Log File Analysis
   
   # STRATEGIC HINTS (SUGGESTED PLAN)
   1. To avoid memory issues, process it line-by-line
   2. For simple patterns, use `grep`, `awk`, and `sed`
   3. For complex logic, use Python with file handle iteration
   4. Start by examining first few lines with `head`
   ```

### Persistence & Caching

#### Embedding Cache
```python
# Persistent embedding cache for performance
self.embedding_cache_path = cache_dir / "embedding_cache.json"
cache_key = hashlib.sha256(f"{self.model_name}:{text}").hexdigest()
```

#### Database Persistence
```python
# Tool/hint database
self.db_path = cache_dir / "tool_hint_db.json"

# Playbook database  
self.db_path = cache_dir / "playbook_db.json"
```

## Advanced Features

### Adaptive Learning
- **Tool Selection Recording**: Tracks which tools work for specific queries
- **Keyword Evolution**: Updates tool keywords based on successful usage
- **Real-time Embedding Updates**: Immediately refreshes tool embeddings

### Error Handling & Fallbacks
- **Model Fallback**: Ollama → SentenceTransformers → Dummy vectors
- **Dimension Consistency**: Validates embedding dimensions before operations  
- **Graceful Degradation**: Continues operation even if one component fails

### Performance Optimizations
- **Vectorized Operations**: Uses numpy for efficient similarity calculations
- **Batch Processing**: Processes multiple embeddings simultaneously
- **Singleton Pattern**: Shares instances across the application
- **Lazy Loading**: Initializes components only when needed

## Usage Examples

### Tool Discovery Query
```python
# User: "I need to extract text from a PDF file"
results = vector_db.search("extract text from PDF", top_k=3)
# Returns: [ViewFiles, EditFile, WriteFile] with similarity scores
```

### Guidance Retrieval
```python  
# User: "How do I read a large file?"
hints = vector_db.get_relevant_guidance("read large file", top_k=2)
# Returns: Guidance about file reading techniques and code examples
```

### Strategic Planning
```python
# User: "I need to scrape data from a website"  
playbook = playbook_manager.get_relevant_playbook("scrape website data", threshold=0.7)
# Returns: "Website Scraping and Information Extraction" playbook with 5 strategic steps
```

## System Benefits

### For Users
- **Contextual Assistance**: Relevant tools and guidance appear automatically
- **Strategic Thinking**: Complex tasks get structured planning approaches
- **Learning System**: Improves recommendations based on usage patterns

### For Developers  
- **Extensible**: Easy to add new tools, hints, and playbooks
- **Observable**: Rich logging and debugging information
- **Maintainable**: Clean separation of concerns between components

### For the AI Agent
- **Enhanced Context**: Rich, structured information for decision making  
- **Reduced Hallucination**: Grounded recommendations based on actual tools
- **Improved Planning**: Strategic frameworks for complex multi-step tasks

This RAG system transforms the CLI-Agent from a reactive tool executor into a proactive assistant that understands context, provides relevant guidance, and suggests strategic approaches for complex tasks.