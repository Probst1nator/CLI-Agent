# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Testing
```bash
# Run tests at different levels
python run_tests.py --quick      # Quick core tests (Level 2: core infrastructure)
python run_tests.py --full       # Full test suite (all levels)
python run_tests.py --verbose    # Detailed test output

# Run specific pytest tests
python -m pytest tests/shared/test_xml_parsing.py -v
python -m pytest tests/shared/test_xml_edge_cases.py -v --tb=line
```

### Code Quality
```bash
# Code formatting
black .

# Linting
ruff check .
```

### Running the Agent
```bash
# Basic CLI mode
python main.py

# With web GUI
python main.py --gui

# As API only
python main.py --api

# With voice/speech
python main.py -s               # speech mode (transcription + TTS)
python main.py -v               # voice mode (transcription only)

# Debug mode (verbose logging to console)
python main.py --debug

# Isolated mode (Docker container with full isolation)
python main.py --isolated       # First run builds image (~2-3 min)
python main.py --isolated --gui # Subsequent runs start instantly (~2-5 sec)
python main.py --isolated -m "Your message"
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install packages in development mode
pip install -e core/
pip install -e shared/

# Environment setup
cp .env.example .env
# Edit .env with your API keys
```

## Architecture Overview

### Recent Architectural Changes (2025-10)

**Infrastructure Cleanup**: The multi-user SaaS architecture has been marked as experimental and is no longer the primary deployment model. Focus has shifted to:
- **Local/isolated mode** as the primary deployment option
- **Single-user web interface** for personal use
- **Docker isolated mode** for secure, containerized execution
- Multi-user infrastructure remains available but is experimental (see `docs/INFRASTRUCTURE_STATUS.md`)

**Script Organization**: Scripts have been reorganized:
- `saas_scripts/`: Multi-user infrastructure deployment scripts (experimental)
- `user_scripts/`: User-facing setup and utility scripts (setup_cli_agent.sh, manage_search_cache.py, etc.)

### Core Design Pattern: AppContext as Single Source of Truth

The codebase has transitioned from argparse-dependent commands to an **AppContext-based architecture**:

- **`agent/core/app_context.py`**: Centralized application context with `AgentSettings` dataclass for all runtime configuration
- **Command Pattern**: All commands use `self.context.settings` instead of args parameter
- **Dependency Injection**: Components receive context, not global state
- **Testability**: Commands are isolated and testable without argparse mocking

### Three-Tier Architecture

1. **Backend Layer** (`api/`)
   - `server.py`: aiohttp web server with WebSocket, auto-start/stop, session audit logging
   - `chat_api.py`: Chat operations (load, new, archives, message editing) with WebSocket sync events
   - `audio_api.py`: Voice/audio (transcription, TTS) for browser-based voice interaction
   - `tools_api.py`, `profile_api.py`, `agent_state_api.py`: Additional API endpoints
   - Session management with GDPR-compliant audit logging (content hashing, IP tracking)

2. **Frontend Layer** (`frontend/`)
   - Separated from backend (moved from `tools/visual_assets/`)
   - `js/chat-state.js`: State management enabling chat loading without page reload
   - `js/audio-manager.js`: Browser audio recording/TTS playback via MediaRecorder API
   - `js/websocket.js`: Event-driven WebSocket with session ID propagation
   - `js/chat-history.js`: Sidebar with BM25 search and snippet highlighting

3. **Agent Layer** (`agent/`, `core/`, `shared/`, `tools/`)
   - Modern command system with proper separation of concerns
   - LLM routing with multi-provider support (OpenAI, Anthropic, Google, Groq, Ollama)
   - Tool system with vector-based discovery and context-aware suggestions
   - Permission system with path-aware rules

### Key Directories

- **`core/`**: AI infrastructure (LLM routing, chat, providers, permissions, caching)
- **`shared/`**: Shared utilities (audio, web, search, path resolution, logging)
- **`agent/`**: Agent architecture (commands, GUI, event management, app context)
- **`tools/`**: Individual tool implementations (bash, file operations, search, visual response)
- **`api/`**: Backend API layer (separate from tool system)
- **`frontend/`**: Web UI assets (HTML, CSS, JS)
- **`tests/`**: Comprehensive test suite organized by component
- **`saas_scripts/`**: Multi-user infrastructure deployment scripts (experimental)
- **`user_scripts/`**: User-facing setup and utility scripts
- **`docker_files/`**: Dockerfiles for various deployment modes

### LLM Routing Architecture

**`core/llm_router.py`** is the singleton that manages all LLM interactions:

- **Multi-provider support**: Automatically discovers and routes to available LLM providers
- **Dynamic model discovery**: Lazy-loads models to avoid startup delays
- **Caching**: SQLite-based cache manager (`core/cache/llm_cache_manager.py`)
- **Retry logic**: Automatic failover between providers on errors
- **Rate limiting**: Provider-specific rate limit handling
- **Streaming support**: Real-time token streaming with interrupt detection

**Provider interfaces** (`core/providers/`):
- `cls_anthropic_interface.py`: Claude models
- `cls_openai_interface.py`: GPT models
- `cls_google_interface.py`: Gemini models
- `cls_groq_interface.py`: Groq models
- `cls_nvidia_interface.py`: NVIDIA models

### Chat System (`core/chat.py`)

- **Provider-agnostic message format**: Converts between Ollama, Gemini, Anthropic, OpenAI formats
- **Message dataclass**: Role, content, images, metadata
- **Compression**: Smart context window management via LLM-based summarization
- **Persistence**: Save/load chat history to JSON with archiving
- **Metadata tracking**: Extensible metadata system for timestamps, IDs, etc.

### Command System Architecture

Modern architecture with complete argparse decoupling:

1. **Command Registry** (`agent/core/command_registry.py`): Single source of truth for command definitions
2. **Base Command** (`agent/commands/base_command.py`): Abstract base with context-based execution
3. **Command Handler** (`agent/commands/command_handler.py`): Central dispatcher replacing switch-case logic
4. **Command Categories**:
   - Simple commands: `/exit`, `/reset`, `/multiline`, toggles
   - LLM selection: `/planner_llm`, `/router_llm`, `/fallbacks`, `/config`
   - Profile management: `/profile_save`, `/profile_load`, `/profile_list`, `/profile_delete`
   - Permissions: `/perms` with direct actions and guided wizards
   - Screenshot: `/screenshot` for visual input
   - Help: `/help` with dynamic generation

### Input System (`agent/core/`)

Unified input management supporting multiple sources:

- **`input_manager.py`**: Abstraction layer for all input sources (CLI, GUI, Web, Voice, Queue)
- **`input_sources.py`**: Concrete implementations with standardized event handling
- **`input_compatibility.py`**: Bridge between legacy and new systems
- **`command_processor.py`**: Specialized processors for commands, text, interrupts, images (argparse-free)
- **`unified_completion.py`**: Command completion across all interfaces

### Tool System Architecture

**Auto-discovery and invocation** (`tools/` directory):
- Tools are Python modules auto-discovered at startup from `tools/` directory
- Each tool implements a standard 3-method contract for consistent integration
- LLM receives tool descriptions in system prompt and outputs XML-tagged calls
- Agent loop parses XML, invokes tool's `run()` method, returns result to LLM

**The Tool Contract** (required for all tools):

Each tool must implement exactly three static methods:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `get_delim()` | `() -> str` | Returns the XML tag name (e.g., "bash" → `<bash>...</bash>`) |
| `get_tool_info()` | `() -> dict` | Returns tool metadata: `{"name": str, "description": str, "example": str}` |
| `run(content)` | `(content: str) -> str` or `async (content: str) -> str` | Executes tool logic, receives inner XML content, returns result string |

**Synchronous vs Asynchronous Tools**:
- **Sync tools** (`def run(content: str) -> str`): Simple operations, file I/O, quick commands
- **Async tools** (`async def run(content: str) -> str`): Network calls, LLM operations, long-running tasks
- Main loop automatically detects and handles both patterns using `asyncio.iscoroutinefunction()`

**XML Parsing Patterns**:
```python
# Simple tag extraction
title = re.search(r'<title>(.*?)</title>', content, re.DOTALL).group(1).strip()

# Optional parameters with defaults
timeout_match = re.search(r'<timeout>(\d+)</timeout>', content)
timeout = int(timeout_match.group(1)) if timeout_match else 30

# Boolean flags
use_cache = True
cache_match = re.search(r'<cache>(.*?)</cache>', content, re.DOTALL)
if cache_match and cache_match.group(1).strip().lower() == 'false':
    use_cache = False
```

**Tool-Specific Logging**:
```python
from shared.utils.tool_logging import get_logger_for_tool_module
logger = get_logger_for_tool_module(__name__)
# Logs automatically go to ~/.cli-agent/logs/tools_<toolname>.log
```

**Vector-based tool discovery** (RAG system):
- Tools indexed in ChromaDB vector database for semantic search
- Context-aware tool suggestions based on conversation history
- Permission system integration with path-aware rules
- Tool manager UI for enable/disable toggling

**Key tools**:
- `bash.py`: Command execution with permission checks, persistent session, timeout handling
- `readfile.py`, `editfile.py`, `writefile.py`: File operations with path resolution
- `grepsearch.py`, `searchresource.py`: Code search and resource inspection utilities
- `searchweb.py`: Web search with LLM-powered summarization and result caching
- `googlekeep.py`: Google Keep API integration with OAuth 2.0 authentication
- `visualresponse.py`: Delegates to API server for visual UI rendering

**See `tools/CLAUDE.md` for detailed implementation guide with examples.**

### Logging Architecture (`docs/LOGGING.md`)

**Always-on file logging at DEBUG level** (even without `--debug` flag):

- **Session logs**: `~/.cli-agent/logs/session.log` (30 day retention, 50MB max)
- **Tool logs**: `~/.cli-agent/logs/tools_<toolname>.log` (14 day retention, 20MB max)
- **Audit logs**: `~/.cli-agent/audit_logs/sessions/` (90 day retention)
- **Automatic cleanup**: Runs on startup and every 24 hours
- **Log manager**: `shared/utils/log_manager.py` with retention policies

**Log formats**:
- Console: Clean, colored output without timestamps (INFO level by default)
- File: `0:12.345 [DEBUG   ] Message` with elapsed time from startup

**TTS debugging**: Search logs for `TTS:` markers to trace voice flow.

### WebSocket Communication Pattern

**Session initialization flow**:
1. Client connects → Server sends `session_init` with session_id
2. Client stores session_id and includes in all HTTP requests via `X-Session-ID` header
3. Server broadcasts sync events: `chat_replaced`, `chat_cleared`, `message_edited`
4. Frontend re-renders without page reload

**Streaming message events**: `message_start` → `message_chunk` → `message_end`

### Permission System (`core/permissions/`)

**Enhanced permission architecture** (`enhanced_permissions.py`):
- **Path-aware rules**: Granular control over file/directory access
- **Tool-level permissions**: Enable/disable tools individually
- **Persistent rules**: Saved across sessions
- **Interactive prompts**: User confirmation for sensitive operations
- **Permission command**: `/perms add dev grepsearch` for direct rule creation

### Audio System (`shared/utils_audio.py`, `shared/audio_router.py`)

**Voice pipeline**:
1. Browser recording → MediaRecorder API → `POST /api/audio/upload`
2. Backend transcription using Whisper (local or cloud)
3. Agent processes text and generates response
4. `POST /api/audio/tts` → TTS generation (Piper or cloud)
5. Audio stream → Browser playback

**Supported modes**:
- `-s`: Speech mode (transcription + TTS)
- `-v`: Voice mode (just transcription)
- Browser-based recording for web UI

## Important Implementation Details

### Backend/Frontend Separation

See `docs/VISUALRESPONSE_ARCHITECTURE.md` for full details. Key points:

- **API layer separate from tools**: `api/` contains web server and endpoints, `tools/` contains agent logic
- **Auto-start/stop server**: Starts when visualresponse is called, stops when last client disconnects (unless `--gui`)
- **Standard error responses**: All endpoints return `{status, data}` or `{status, error, code}` format
- **WebSocket sync events**: Enable real-time UI updates without page reload
- **Session audit logging**: GDPR-compliant with content hashing (not full text) in audit logs

### Main Entry Point (`main.py`)

**The stable, production entry point** (3,273 lines):
- Contains all recent bug fixes and active development
- Full user interaction loop and tool invocation system
- Warning suppression system for clean output (unless `--debug`)
- AppContext initialization from CLI args
- Comprehensive error handling and signal management

**Note**: `main_refactored.py` is experimental and not ready for production.

### Testing Infrastructure

**Hierarchical test framework** (`run_tests.py`):
- **Level 0 (INTEGRATION)**: End-to-end tests
- **Level 1 (TOOLS)**: Tool-specific tests
- **Level 2 (CORE)**: Core infrastructure tests (default for `--quick`)
- **Level 3 (SHARED)**: Shared utilities tests
- **Level 4 (UNIT)**: Detailed unit tests

**Test organization**:
- `tests/core/`: LLM router, model discovery, permissions
- `tests/shared/`: Path resolver, XML parsing
- `tests/tools/`: Tool-specific tests
- `tests/frontend/`: Frontend component tests

### Code Style and Linting

**Configured in `pyproject.toml`**:
- **Black**: Line length 88, Python 3.8+ target
- **Ruff**: Python 3.11 target, ignores E402, E501, E701, E722
- **Mypy**: Python 3.11, warn on return any, ignore missing imports

### Environment Variables

Required for LLM providers:
- `ANTHROPIC_API_KEY`: Claude models
- `OPENAI_API_KEY`: GPT models
- `GOOGLE_API_KEY`: Gemini models
- `GROQ_API_KEY`: Groq models
- `BRAVE_API_KEY`: Web search (optional)

## Common Workflows

### Adding a New Command

1. Create command class in `agent/commands/` inheriting from `BaseCommand`
2. Implement `execute(self, args: str) -> CommandResult` using `self.context.settings`
3. Register in `agent/core/command_registry.py` with metadata
4. Add to command completion in `unified_completion.py` if needed

### Adding a New Tool

**Quick Reference:**
1. Create `tools/your_tool.py` with the 3-method contract (`get_delim`, `get_tool_info`, `run`)
2. Add dependencies to `requirements.txt` if needed
3. Add environment variables to `.env.example` for configuration
4. Implement XML parsing for input parameters
5. Add tool-specific logging using `get_logger_for_tool_module(__name__)`
6. Create comprehensive tests in `tests/tools/test_your_tool.py`
7. Tool will be auto-discovered at startup - no registration needed!

**Detailed Steps:**

1. **Create the tool file** (`tools/your_tool.py`):
   ```python
   from shared.utils.tool_logging import get_logger_for_tool_module
   logger = get_logger_for_tool_module(__name__)

   class your_tool:
       @staticmethod
       def get_delim() -> str:
           return 'your_tool'

       @staticmethod
       def get_tool_info() -> dict:
           return {
               "name": "your_tool",
               "description": "What your tool does and when to use it",
               "example": "<your_tool><param>value</param></your_tool>"
           }

       @staticmethod
       async def run(content: str) -> str:  # or 'def' for sync
           # Parse XML parameters
           # Execute logic
           # Return result string
           pass
   ```

2. **Add dependencies** to `requirements.txt`:
   ```bash
   # Your Tool Dependencies
   your-library>=1.0.0  # Brief description
   ```

3. **Add configuration** to `.env.example`:
   ```bash
   # Your Tool Configuration
   YOUR_TOOL_API_KEY=your_api_key_here
   YOUR_TOOL_CONFIG_PATH=~/.cli-agent/your_tool_config.json
   ```

4. **Create tests** (`tests/tools/test_your_tool.py`):
   ```python
   from tools.your_tool import your_tool

   def test_your_tool_basic():
       result = your_tool.run("<param>test</param>")
       assert "expected" in result

   # Assign test level for hierarchical testing
   test_your_tool_basic.level = 2
   ```

5. **Run tests**:
   ```bash
   python -m pytest tests/tools/test_your_tool.py -v
   ```

**For detailed implementation patterns, examples, and best practices, see `tools/CLAUDE.md`**

### Adding a New API Endpoint

1. Add endpoint function to appropriate `api/*_api.py` file
2. Register route in `api/server.py`
3. Use standard error response format
4. Add WebSocket broadcast if state changes
5. Update frontend to consume endpoint

### Debugging TTS Issues

1. Run agent (logs are automatic): `python main.py -s`
2. Tail logs: `tail -f ~/.cli-agent/logs/session.log | grep "TTS:"`
3. Check for:
   - `voice_mode=False` → Enable with `-s`
   - `No visualresponse blocks found` → Check XML structure
   - `skip_tts=True` → Visual-only content
   - `response_text=0 chars` → Extraction failed

### Working with Chat History

- **Save**: Chat auto-saves on new chat or exit
- **Load**: Use `/chat load <id>` or web UI sidebar
- **Archive**: Old chats stored in `~/.cli-agent/chat_history/`
- **Search**: Web UI has BM25 search with snippet highlighting

### Using Isolated Mode

Isolated mode runs CLI-Agent in a Docker container with full filesystem isolation:

1. **First run** (builds cached image):
   ```bash
   python main.py --isolated
   # Takes ~2-3 minutes to build image
   ```

2. **Subsequent runs** (instant startup):
   ```bash
   python main.py --isolated -m "Your message"
   python main.py --isolated --gui
   # Takes ~2-5 seconds (reuses cached image)
   ```

3. **Source code auto-sync**: Latest code is automatically synced into container on every run
4. **Persistent data**: `~/.cli-agent/` directory is mounted for chat history and config
5. **Working directory access**: Current directory is accessible at `/workspace` in container

**When to use isolated mode**:
- Testing changes in a clean environment
- Running untrusted code safely
- Ensuring consistent dependencies
- Debugging environment-specific issues

**Rebuilding image** (when requirements.txt changes):
```bash
docker rmi cli-agent:isolated-cache
python main.py --isolated  # Rebuilds automatically
```

## Critical Notes

1. **Use `main.py`, not `main_refactored.py`** for development and deployment
2. **Always-on logging**: Logs are captured at DEBUG level even without `--debug` flag
3. **AppContext over argparse**: Commands must use `self.context.settings`, not args parameter
4. **API separation**: Backend API (`api/`) is separate from tool system (`tools/`)
5. **WebSocket state sync**: Frontend re-renders via events, no page reload needed
6. **Permission checks**: All file/tool operations must check permissions
7. **Provider failover**: LLM router automatically retries with different providers
8. **Session audit**: All web sessions logged with IP, timestamps, chat snapshots (GDPR-compliant)
9. **Isolated mode**: Use `--isolated` flag for Docker containerization with auto-sync of source code
10. **Multi-user infrastructure**: Experimental status - not production-ready (see `docs/INFRASTRUCTURE_STATUS.md`)

## Related Documentation

- `docs/LOGGING.md`: Comprehensive logging system documentation
- `docs/ISOLATED_MODE.md`: Docker isolated mode documentation
- `docs/VISUALRESPONSE_ARCHITECTURE.md`: Modern web backend/frontend architecture
- `docs/INFRASTRUCTURE_STATUS.md`: Multi-user deployment status and options
- `docs/experimental/`: Experimental features and deployment modes (not production-ready)
- `INDEX.md`: Complete file index with recent changes
- `README.md`: Installation and basic usage
