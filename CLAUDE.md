# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLI-Agent is a multi-provider AI agent framework for autonomous task execution and coding assistance. It supports CLI, web interface, and isolated Docker execution.

## Build & Run Commands

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e core/ -e shared/

# Run modes
python main.py              # CLI mode
python main.py --gui        # Web interface (localhost:8765)
python main.py --api        # API server only
python main.py --isolated   # Docker container (first run builds image ~2-3min, then ~2-5s)
python main.py --debug      # Verbose logging

# Testing
python run_tests.py --quick           # CORE level tests (default)
python run_tests.py --full            # All test levels
python -m pytest tests/tools/test_<name>.py -v  # Single test file

# Code quality
ruff check .              # Linting
ruff format .             # Formatting
pyright                   # Type checking
```

## Architecture

### Deployment Options

1. **Local Development** - Direct execution with `python main.py`
2. **Isolated Mode** - Single Docker container with `--isolated` flag for secure execution

### Key Directories

- **core/** - LLM routing (`llm_router.py`), chat management (`chat.py`), provider integrations, permissions system
- **api/** - FastAPI backend: `server.py` (WebSocket/REST), `audio_api.py` (Whisper/TTS)
- **agent/** - Agent loop (`core/agent_loop.py`), command processing (`core/command_processor.py`), input handling
- **tools/** - Auto-discovered tool implementations (see `tools/CLAUDE.md` for tool development guide)
- **frontend/** - Vanilla JS web UI (see `frontend/CLAUDE.md` for frontend architecture)
- **shared/** - Audio processing (`utils_audio.py`), Docker runner, search providers, RAG utilities

### Tool System

Tools are auto-discovered from `tools/` directory. Each tool implements three static methods:
- `get_delim() -> str` - XML tag name
- `get_tool_info() -> dict` - Name, description, example for system prompt
- `run(content: str) -> str` - Execute tool and return result

See `tools/CLAUDE.md` for detailed implementation guide.

### Frontend Architecture

Vanilla JavaScript with WebSocket communication. Module load order in `index.html` is critical:
1. logger.js → utilities → tool-registry
2. Core modules: renderers → websocket → chat-state → audio
3. main.js (orchestrator) must be last

See `frontend/CLAUDE.md` for component details.

### Test Levels

`run_tests.py` uses hierarchical levels:
- **INTEGRATION (0)** - End-to-end tests
- **TOOLS (1)** - Tool-specific tests
- **CORE (2)** - Core infrastructure (default for `--quick`)
- **SHARED (3)** - Shared utilities
- **UNIT (4)** - Detailed unit tests

## Environment Configuration

Required API keys (at least one LLM provider):
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `NVIDIA_API_KEY`

Optional:
- `BRAVE_API_KEY` - Web search
- `HF_TOKEN` - HuggingFace for local models
- `OLLAMA_HOSTS` - Local Ollama instances

See `.env.example` for full configuration.

## Key Patterns

- **Role-based model selection**: PLANNER, ROUTER, OVERRIDE, AUTOCOMPLETE, FALLBACK roles in LLM routing
- **Permission-first**: File/system operations gated by `core/permissions/` checks
- **WebSocket-driven state**: Frontend syncs via events (chat_replaced, chat_cleared, message_edited)
- **Lazy initialization**: Heavy imports deferred until needed
- **Singleton sessions**: Stateful tools use class-level variables for persistent connections

## Code Style

- Python 3.11+ with type hints
- Ruff for linting/formatting (line-length 88)
- Pyright for type checking
- Prettier/ESLint for frontend JS
- `ruff check . --fix` is safe; never use `--unsafe-fixes` (removes variable assignments, breaks code)
