# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Quick Commands

### Testing & Quality
```bash
python run_tests.py --quick      # Core tests (Level 2)
python run_tests.py --full       # Full suite
python -m pytest tests/path/test_file.py -v

black .                          # Format
ruff check .                     # Lint (don't proactively fix unless requested)
```

### Running
```bash
python main.py                   # CLI mode
python main.py --gui             # Web UI
python main.py -s                # Speech (transcription + TTS)
python main.py --debug           # Verbose console (logs always at DEBUG in files)
python main.py --isolated        # Docker container (auto-sync source)
```

### Setup
```bash
pip install -r requirements.txt
pip install -e core/ -e shared/
cp .env.example .env             # Add API keys
```

## Logging (Critical)

**Always check logs FIRST before re-running!** All logs captured at DEBUG level in files regardless of `--debug` flag.

### Quick Access
```bash
# View recent logs
tail -f ~/.cli-agent/logs/session.log        # Backend (always DEBUG)
tail -f ~/.cli-agent/logs/tools_<name>.log   # Per-tool logs

# Find errors
grep -E "(ERROR|Failed|Exception)" ~/.cli-agent/logs/session.log | tail -50

# Trace specific request
grep "req_1730123456_abc123" ~/.cli-agent/logs/session.log

# Debug patterns
grep "TTS:" ~/.cli-agent/logs/session.log           # Voice/TTS
grep "generation_id" ~/.cli-agent/logs/session.log  # Live updates
grep "Model.*selected" ~/.cli-agent/logs/session.log # LLM routing
```

**Log locations:**
- `~/.cli-agent/logs/session.log` - Main backend (30d retention, 50MB max)
- `~/.cli-agent/logs/tools_<name>.log` - Per-tool (14d, 20MB)
- `~/.cli-agent/audit_logs/sessions/` - Session audits (90d)

See `docs/LOGGING.md` for full documentation.

## Architecture

### Key Concepts

**AppContext Pattern** (`agent/core/app_context.py`):
- Single source of truth for runtime config via `AgentSettings` dataclass
- Commands use `self.context.settings` instead of argparse
- Dependency injection, no global state

**Three-Tier Structure**:
1. **Backend** (`api/`): aiohttp server, WebSocket, session audit, API endpoints
2. **Frontend** (`frontend/`): HTML/CSS/JS, chat state management, audio recording
3. **Agent** (`agent/`, `core/`, `tools/`): Command system, LLM routing, tool execution

**Storage** (`~/.cli-agent/`):
```
~/.cli-agent/
  projects/{sanitized-path}/    # Per-project isolation
    metadata.json               # Project info
    chat_history/               # Archived chats
    recent_chat.json            # Active session
  logs/                         # Global logs
  settings.json                 # Permissions config
  tool_vectordb/                # Tool discovery
```

Implementation: `shared/utils/project_storage.py`

**LLM Routing** (`core/llm_router.py`):
- Multi-provider (Anthropic, OpenAI, Google, Groq, Ollama)
- Auto-discovery, lazy-load, failover, caching, streaming
- Provider interfaces in `core/providers/`

**Command System** (`agent/commands/`, `agent/core/command_registry.py`):
- Registry pattern with BaseCommand classes
- Context-based execution (no argparse dependency)
- Categories: simple, LLM selection, profiles, permissions, screenshot

**Tool System** (`tools/`):
- Auto-discovery from `tools/` directory
- 3-method contract: `get_delim()`, `get_tool_info()`, `run(content)`
- Sync or async execution (auto-detected)
- Vector-based discovery (ChromaDB)
- Tool-specific logging: `get_logger_for_tool_module(__name__)`

See `tools/CLAUDE.md` for implementation guide.

**Permissions** (`core/permissions/`):
- Config-based: `~/.cli-agent/settings.json`
- Pattern matching: `bash(npm:*)`, `readfile(./src/**)`
- Precedence: deny → ask → allow
- Commands: `/perms view|add|remove`

**Audio** (`shared/utils_audio.py`, `shared/audio_router.py`):
- Browser MediaRecorder → Whisper → Agent → TTS → Playback
- Modes: `-s` (speech), `-v` (voice only)

### Key Directories

- `core/`: LLM routing, chat, providers, permissions, caching
- `shared/`: Utilities (audio, web, search, path resolution, logging)
- `agent/`: Commands, GUI, event management, app context
- `tools/`: Tool implementations (auto-discovered)
- `api/`: Backend API layer
- `frontend/`: Web UI assets
- `tests/`: Hierarchical test suite (levels 0-4)
- `docs/`: Architecture, logging, workflows

## Common Workflows

### Debugging
1. **Check logs first** (don't re-run):
   ```bash
   grep -E "(ERROR|Failed)" ~/.cli-agent/logs/session.log | tail -50
   ```
2. Identify issue from error messages
3. Trace with request ID if available
4. Check tool-specific logs: `~/.cli-agent/logs/tools_<name>.log`
5. Only re-run if logs insufficient (use `--debug` for console output)

### Add Command
1. Create class in `agent/commands/` inheriting `BaseCommand`
2. Implement `execute(self, args: str)` using `self.context.settings`
3. Register in `agent/core/command_registry.py`
4. Add completion in `unified_completion.py` if needed

### Add Tool
1. Create `tools/your_tool.py`:
   ```python
   from shared.utils.tool_logging import get_logger_for_tool_module
   logger = get_logger_for_tool_module(__name__)

   class your_tool:
       @staticmethod
       def get_delim() -> str:
           return 'your_tool'

       @staticmethod
       def get_tool_info() -> dict:
           return {"name": "your_tool", "description": "...", "example": "..."}

       @staticmethod
       async def run(content: str) -> str:  # or 'def' for sync
           # Parse XML, execute, return result
           pass
   ```
2. Add deps to `requirements.txt`, config to `.env.example`
3. Create tests in `tests/tools/test_your_tool.py`
4. Auto-discovered at startup (no registration needed)

See `tools/CLAUDE.md` for patterns and examples.

### Add API Endpoint
1. Add function to `api/*_api.py`
2. Register route in `api/server.py`
3. Use standard `{status, data}` or `{status, error, code}` format
4. Broadcast WebSocket event if state changes
5. Update frontend to consume

### Isolated Mode (Docker)
```bash
python main.py --isolated        # First run builds image (~2-3 min)
python main.py --isolated --gui  # Subsequent runs instant (~2-5 sec)
```
- Source auto-syncs into container
- `~/.cli-agent/` mounted for persistence
- Current dir at `/workspace`
- Rebuild: `docker rmi cli-agent:isolated-cache`

## Critical Notes

1. **Use `main.py`** (not `main_refactored.py`)
2. **Logs always at DEBUG level** in files - check logs before re-running
3. **AppContext over argparse** - use `self.context.settings`
4. **API separation** - `api/` vs `tools/`
5. **Permission checks** required for file/tool operations
6. **Provider failover** automatic in LLM router
7. **Session audit** GDPR-compliant (content hashing, not full text)
8. **Don't proactively fix linter warnings** unless requested

## Public Repository Sync

Public version at https://github.com/Probst1nator/CLI-Agent (safe tools only).

**Workflow:** `main` → PR to `public-sync` → filter → `git_public/main`

```bash
gh pr create --base public-sync --head main              # Create review PR
git checkout public-sync && git pull                     # After merge
python user_scripts/sync_public_repo.py --prepare        # Filter
python user_scripts/sync_public_repo.py --push           # Release
```

See `docs/RELEASE_WORKFLOW.md` for full workflow.

## Documentation

- `docs/LOGGING.md` - Comprehensive logging system
- `docs/ISOLATED_MODE.md` - Docker isolation
- `docs/VISUALRESPONSE_ARCHITECTURE.md` - Web backend/frontend
- `docs/INFRASTRUCTURE_STATUS.md` - Deployment options
- `docs/RELEASE_WORKFLOW.md` - Public sync workflow
- `tools/CLAUDE.md` - Tool implementation guide
- `INDEX.md` - Complete file index
- `README.md` - Installation and usage
