# Ollama Infrastructure Interaction Patterns

This document outlines the agentic interaction patterns learned for working with Ollama infrastructure in the CLI-Agent project.

## Architecture Overview

The CLI-Agent interacts with Ollama through a multi-layered architecture:

```
CLI-Agent Application
├── LlmSelector (UI Layer)
├── Globals (Caching & Background Discovery)
├── LlmRouter (Model Management)
└── OllamaInterface (Direct API Interaction)
```

## Core Data Structures

### Model Discovery Cache Structure
```json
{
  "model_base_name": {
    "downloaded": boolean,
    "variants": {
      "model_key:tag": {
        "downloaded": boolean,
        "size_str": "5.2GB",
        "context_window": 128000,
        "modified_at": "2024-01-01T00:00:00Z"
      }
    }
  }
}
```

### Model Status Information
```python
{
    'size': int,                    # Size in bytes from Ollama API
    'modified_at': datetime,        # Last modified timestamp
    'downloaded': boolean,          # Whether model is locally available
    'size_str': str,               # Human-readable size (e.g., "5.2GB")
    'context_window': int          # Token context window size
}
```

## Interaction Patterns

### 1. Background Model Discovery

**Pattern**: Asynchronous background caching with persistent storage

```python
# Start background discovery at application startup
def start_background_model_discovery(self, force_refresh: bool = False):
    # Load persistent cache immediately for instant UI
    persistent_cache = self.load_persistent_model_cache()
    if persistent_cache:
        self._model_discovery_cache = persistent_cache
        
    # Check cache freshness (6 hours fresh, 24 hours stale)
    cache_age_hours = self.get_persistent_cache_age_hours()
    if not force_refresh and cache_age_hours < 6:
        return  # Skip background refresh for fresh cache
    
    # Start async background refresh
    asyncio.create_task(self._background_model_discovery())
```

**Key Insights**:
- Immediate cache loading prevents UI blocking
- Smart cache aging reduces unnecessary API calls
- Persistent storage survives application restarts

### 2. Model Status Detection

**Pattern**: Hierarchical status checking with fallback logic

```python
def determine_download_status(model_key, ollama_status):
    model_base_name = model_key.split(':')[0]
    status_info = ollama_status.get(model_base_name, {})
    
    # Check both base-level and variant-level status
    is_downloaded_base = status_info.get('downloaded', False)
    variants = status_info.get('variants', {})
    variant_info = variants.get(model_key, {})
    is_downloaded_variant = variant_info.get('downloaded', False)
    
    # Model is downloaded if EITHER level indicates downloaded
    return is_downloaded_base or is_downloaded_variant
```

**Key Insights**:
- Models can be marked downloaded at base or variant level
- Always check both levels to avoid false negatives
- Fallback logic handles incomplete data gracefully

### 3. Live Update System

**Pattern**: Timestamp-based change detection with real-time UI updates

```python
def check_for_live_updates(self):
    from py_classes.globals import g
    
    # Compare timestamps to detect new data
    if (g._model_discovery_timestamp > self.last_update_timestamp and 
        g._model_discovery_cache is not None):
        
        # Update UI with new data
        self._rebuild_model_data_with_cache(g._model_discovery_cache)
        self._apply_search_filter()
        self.last_update_timestamp = g._model_discovery_timestamp
        return True
    return False
```

**Key Insights**:
- Timestamp comparison is more efficient than data comparison
- UI updates are atomic (rebuild + refilter)
- Always enabled for seamless user experience

### 4. Size Information Handling

**Pattern**: Multi-source size aggregation with display formatting

```python
def format_model_status_with_size(is_downloaded, variant_info):
    size_str = variant_info.get('size_str', '')
    
    if is_downloaded:
        status_text = "✓ Downloaded"
        if size_str and any(unit in size_str.upper() for unit in ['GB', 'MB', 'TB']):
            status_text += f" ({size_str})"
    else:
        status_text = "⬇ Downloadable"
        if size_str and any(unit in size_str.upper() for unit in ['GB', 'MB', 'TB']):
            status_text += f" ({size_str})"
    
    return status_text
```

**Key Insights**:
- Size information comes from web scraping, not Ollama API
- Display size for both downloaded and downloadable models
- Filter out parameter counts (only show GB/MB/TB)

### 5. API Error Handling

**Pattern**: Graceful degradation with fallback mechanisms

```python
async def safe_ollama_discovery():
    try:
        # Primary: Try Ollama API
        models = await ollama_interface.get_models()
        web_models = await ollama_interface.get_web_models()
        return merge_model_data(models, web_models)
    except OllamaConnectionError:
        # Fallback: Use cached data
        return load_persistent_cache() or {}
    except Exception as e:
        # Last resort: Empty data with logging
        logging.warning(f"Model discovery failed: {e}")
        return {}
```

**Key Insights**:
- Always provide fallback for network failures
- Log errors but don't crash the UI
- Empty data is better than broken UI

### 6. Performance Optimization

**Pattern**: Atomic cache operations with minimal blocking

```python
def save_persistent_model_cache(self, data: Dict[str, Any]) -> None:
    """Atomic cache save with error handling."""
    cache_file = Path.home() / ".cli-agent" / "model_cache.json"
    cache_file.parent.mkdir(exist_ok=True)
    
    # Write to temporary file first
    temp_file = cache_file.with_suffix('.tmp')
    try:
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        # Atomic move
        temp_file.replace(cache_file)
    except Exception as e:
        logging.error(f"Failed to save model cache: {e}")
        if temp_file.exists():
            temp_file.unlink()
```

**Key Insights**:
- Atomic writes prevent corrupted cache files
- Temporary file pattern ensures data integrity
- Error cleanup prevents file system pollution

## Best Practices

### 1. Cache Management
- **Fresh Cache**: < 6 hours (skip background refresh)
- **Stale Cache**: 6-24 hours (background refresh, use cache immediately)
- **Expired Cache**: > 24 hours (force refresh, show loading)

### 2. Error Recovery
- Network failures → Use cached data
- Corrupted cache → Regenerate from scratch
- API timeouts → Retry with exponential backoff

### 3. UI Responsiveness
- Load cache immediately on startup
- Show data instantly, refresh in background
- Update UI atomically when new data arrives

### 4. Data Consistency
- Always check both base and variant download status
- Merge web scraping data with API data
- Validate data before displaying to user

### 5. Resource Efficiency
- Use persistent storage to avoid redundant API calls
- Background tasks don't block UI thread
- Smart refresh scheduling based on cache age

## Integration Points

### With LlmRouter
```python
# Get available models including dynamic Ollama models
available_llms = Llm.get_available_llms(
    exclude_guards=True, 
    include_dynamic=True
)
```

### With Globals
```python
# Access cached model data
if g._model_discovery_cache:
    model_status = g._model_discovery_cache
    timestamp = g._model_discovery_timestamp
```

### With OllamaInterface
```python
# Direct API interaction
ollama_client = OllamaInterface()
local_models = await ollama_client.get_models()
web_models = await ollama_client.get_web_models()
```

## Common Pitfalls and Solutions

### Pitfall: Only checking base-level download status
**Problem**: Models show as downloadable when they're actually downloaded
**Solution**: Check both `status_info['downloaded']` and `variant_info['downloaded']`

### Pitfall: Blocking UI with synchronous API calls
**Problem**: UI freezes during model discovery
**Solution**: Use background tasks with immediate cache loading

### Pitfall: Corrupted cache files during write
**Problem**: Application crashes on startup due to malformed JSON
**Solution**: Use atomic writes with temporary files

### Pitfall: Outdated cache persisting indefinitely
**Problem**: New models don't appear in UI
**Solution**: Implement cache aging with automatic refresh

### Pitfall: Size information missing for downloaded models
**Problem**: Inconsistent display between downloaded and downloadable models
**Solution**: Show size for both states when available

## Future Enhancements

1. **Model Dependency Tracking**: Track which models depend on others
2. **Bandwidth-Aware Caching**: Adjust refresh frequency based on connection speed
3. **Model Usage Analytics**: Track which models are used most frequently
4. **Predictive Preloading**: Preload models likely to be used
5. **Health Monitoring**: Track Ollama service health and auto-restart if needed

This document serves as a reference for maintaining and extending Ollama integration in the CLI-Agent project.