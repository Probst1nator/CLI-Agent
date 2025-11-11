# Shared Utilities Documentation

The shared utilities provide cross-cutting functionality used throughout CLI-Agent, including audio processing, web utilities, path resolution, logging, and command execution.

## Architecture Overview

Shared utilities follow a domain-organized structure:
- **Audio**: Speech recognition, TTS, audio processing
- **Web**: HTTP requests, web scraping, search integration
- **Search**: Brave search, vector DB caching
- **Python**: Code execution and environment manipulation
- **Logging**: Centralized log management and retention
- **Path resolution**: Cross-platform path handling
- **File utilities**: PDF processing, file format support

## Directory Structure

```
shared/
├── utils_audio.py        # Main audio processing utilities
├── audio_router.py       # Audio routing (CLI vs GUI/Web)
├── audio/
│   └── audio_utils.py    # Legacy audio utilities
├── utils/
│   ├── web/
│   │   └── web_server.py  # HTTP endpoints and web interfaces
│   ├── search/
│   │   ├── brave_search.py      # Web search API
│   │   └── search_vector_db.py  # Search result caching
│   ├── python/
│   │   └── python_utils.py  # Code execution
│   ├── log_manager.py       # Log retention and cleanup
│   ├── pdf_utils.py         # PDF processing
│   ├── shutdown_utils.py    # Graceful shutdown
│   ├── display_utils.py     # UI formatting
│   ├── logging_utils.py     # Logging configuration
│   └── tool_logging.py      # Per-tool logging
├── path_resolver.py      # Path resolution and expansion
├── common_utils.py       # General utilities
├── cmd_execution.py      # Command execution
└── dia_helper.py         # Dialog and interaction helpers
```

## Audio System

### Main Audio Utilities (`utils_audio.py`)

Enhanced audio processing with multiple backends:

**Transcription**:
- **Faster-Whisper**: Optimized local transcription
  - Anti-hallucination parameters for German
  - GTX 1080 Ti compatibility (float16 precision)
  - Multiple model sizes (tiny, base, small, medium)
- **OpenAI Whisper**: API-based fallback
- **Google Speech Recognition**: Secondary fallback

**Text-to-Speech**:
- **Piper**: High-quality local TTS
- **Kokoro**: Alternative local engine
- **Google TTS**: Cloud-based option

**Intelligent fallback**:
```
Faster-Whisper → OpenAI Whisper → Google Speech Recognition
```

**Audio debugging**:
- Automatic audio file saving when errors occur
- Detailed logging of transcription attempts
- Device selection and configuration

### Audio Router (`audio_router.py`)

Routes TTS playback based on execution mode:

**Routing logic**:
- **CLI mode** (`g.GUI_MODE = False`): Python native playback
- **GUI/API mode** (`g.GUI_MODE = True`): WebSocket streaming to frontend

**Functions**:
```python
route_tts_audio(text, voice)  # Automatic routing
generate_tts_audio(text, voice)  # Generate without playback
```

**Integration**:
- Seamless with visualresponse WebSocket
- Browser-based playback in GUI mode
- Native system audio in CLI mode

See: `shared/docs/AUDIO_SYSTEM.md`

## Web Utilities (`utils/web/`)

### Web Server (`web_server.py`)

HTTP endpoints and web-based interfaces:
- Request handling
- Response formatting
- Static file serving
- Basic routing

## Search System (`utils/search/`)

### Brave Search (`brave_search.py`)

Web search API integration:
- Query formatting
- Result parsing
- Rate limiting
- Error handling

### Search Vector DB (`search_vector_db.py`)

**SearchResultsVectorDB class** for intelligent search caching:

**Features**:
- **Semantic caching**: ChromaDB-based similarity matching
- **Two collections**:
  - `web_search_results` - Individual search results
  - `search_summaries` - Query-level summaries
- **Embeddings**: Uses embeddinggemma:latest via Ollama
- **Progressive knowledge**: Builds understanding over time

**Benefits**:
- Reduces redundant API calls
- Faster responses for similar queries
- Knowledge accumulation across sessions

**Usage**:
```python
# Cache search results
db.cache_search_results(query, results, summary)

# Retrieve similar results
cached = db.get_similar_search_results(query, n_results=5)
```

See: `shared/docs/SEARCHWEB_VECTOR_DB.md`

## Python Utilities (`utils/python/`)

### Python Utils (`python_utils.py`)

Code execution and environment manipulation:
- Code execution in isolated contexts
- Syntax validation
- Package management
- Environment inspection

## Logging System

### Log Manager (`utils/log_manager.py`)

Centralized log retention and cleanup:

**Features**:
- **Automatic rotation**: File size limits
- **Retention policies**: Configurable by log type
  - Debug logs: 7 days
  - App logs: 30 days
  - Tool logs: 14 days
  - Audit logs: 90 days
- **Background cleanup**: Scheduled automatic cleanup
- **Statistics**: Comprehensive usage tracking

**Log locations**:
```
~/.cli-agent/logs/
├── session.log          # Main backend (DEBUG level, 30d)
├── tools_<name>.log     # Per-tool logs (14d, 20MB max)
└── debug_*.log          # Debug logs (7d)

~/.cli-agent/audit_logs/
└── sessions/            # Session audits (90d)
```

**Usage**:
```python
log_manager.cleanup_old_logs()  # Manual cleanup
log_manager.get_statistics()    # Usage stats
```

See: `shared/docs/LOGGING.md`

### Tool Logging (`utils/tool_logging.py`)

Per-tool logging infrastructure:

**Features**:
- Separate log file per tool
- Automatic logger creation
- Consistent formatting
- Independent retention

**Usage**:
```python
from shared.utils.tool_logging import get_logger_for_tool_module

logger = get_logger_for_tool_module(__name__)
logger.info("Tool operation")
```

### Logging Configuration (`utils/logging_utils.py`)

Logging formatters and setup:

**Formatters**:
- `ConsoleFormatter` - Colorized console output
- `SimpleFormatter` - Plain text
- `FileFormatter` - Detailed file logging

**Setup**:
```python
setup_logging(debug=True, log_file=True)
```

## File Utilities

### PDF Utils (`utils/pdf_utils.py`)

Consolidated PDF processing:

**Functions**:
- `extract_pdf_content(pdf_path)` - Extract text and metadata
- `clean_pdf_text(text)` - Remove artifacts and normalize
- `get_cache_file_path(pdf_path)` - Caching support

**Features**:
- Text extraction with structure preservation
- Metadata extraction (author, title, etc.)
- Caching for repeated access
- Error handling

Extracted from duplicated code in `common_utils.py` and `audio_utils.py`.

See: `shared/docs/FILE_FORMAT_SUPPORT.md`

## Path Resolution

### Path Resolver (`path_resolver.py`)

Cross-platform path handling:

**Features**:
- **Tilde expansion**: `~/foo` → `/home/user/foo`
- **Relative paths**: `./bar` → `/current/dir/bar`
- **Environment variables**: `$HOME/baz`
- **Cross-platform**: Works on Linux, macOS, Windows

**Functions**:
```python
resolve_path(path_str)  # Full resolution
expand_path(path_str)   # Tilde/env expansion
normalize_path(path)    # Clean and normalize
```

## Shutdown Utilities

### Shutdown Utils (`utils/shutdown_utils.py`)

Graceful shutdown handling:

**Features**:
- Extracted from main.py and main_refactored.py
- Supports legacy and modern architecture
- Proper log cleanup
- Context management

**Usage**:
```python
perform_graceful_shutdown()
```

## Display Utilities

### Display Utils (`utils/display_utils.py`)

UI and display formatting:

**Functions**:
- `colorize_model_name(name)` - Colorized model names
- `print_startup_summary()` - Startup information
- `show_unified_bindings_display()` - Keybinding help

Extracted from main.py for reusability.

## Common Utilities

### Common Utils (`common_utils.py`)

General shared functionality:
- Data processing helpers
- String manipulation
- File operations
- Miscellaneous utilities

### Dialog Helper (`dia_helper.py`)

Interactive prompts and dialogs:
- User confirmation prompts
- Input validation
- Interactive workflows

### Command Execution (`cmd_execution.py`)

Safe command execution:
- Shell command execution
- Process management
- Output capture
- Error handling
- Security considerations

## Persistent Storage

All shared utilities respect the persistent storage pattern:

```
~/.cli-agent/
├── logs/           # Managed by log_manager
├── settings.json   # Permissions and config
├── projects/       # Per-project data
└── audit_logs/     # Session audits
```

See: `shared/docs/PERSISTENT_FILES.md`

## Integration with Components

Shared utilities are used by:
- **Agent**: Audio routing, logging, display
- **API**: Audio APIs, logging
- **Frontend**: TTS playback (via audio router)
- **Core**: Logging, file management
- **Tools**: Tool logging, path resolution, PDF processing

## Key Design Principles

### 1. Domain Organization

Utilities grouped by domain (web/, search/, python/):
- Clear responsibility boundaries
- Easy to find relevant utility
- Logical import structure

### 2. Minimal Dependencies

Shared utilities avoid heavy dependencies:
- Can be imported without pulling in entire stack
- Fast import times
- Modular design

### 3. Cross-Platform

All utilities work on Linux, macOS, Windows:
- Path handling abstraction
- Platform-specific fallbacks
- Consistent behavior

### 4. Logging First

All utilities integrate with logging system:
- Per-tool logging where appropriate
- Consistent log formatting
- Automatic retention management

## Related Documentation

- Logging system: `shared/docs/LOGGING.md`
- Search vector DB: `shared/docs/SEARCHWEB_VECTOR_DB.md`
- File format support: `shared/docs/FILE_FORMAT_SUPPORT.md`
- Persistent files: `shared/docs/PERSISTENT_FILES.md`
- Audio system: `shared/docs/AUDIO_SYSTEM.md`
