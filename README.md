# CLI-Agent

A powerful command-line AI agent framework for autonomous task execution and coding assistance.

# CLI Interface
![CLI Interface](assets/agent_demo.png)

# Web Interface
![CLI Interface](assets/agent_gui_demo.png)

## Overview

CLI-Agent is a comprehensive framework that provides:
- Multi-provider LLM integration (OpenAI, Anthropic, Google Gemini, Groq, Ollama)
- Advanced agent architecture with Monte Carlo Action Engine
- Vector database for smart tool retrieval and contextual guidance
- **Multi-user web service architecture with isolated Docker containers**
- Computational notebook capabilities
- Rate limiting and enhanced permission systems
- Audio processing and speech recognition
- Remote host capabilities for distributed computing

## Features

- **Multi-LLM Support**: Seamlessly switch between different AI providers
- **Smart Tool Management**: Vector-based tool discovery and context-aware suggestions
- **Interactive Permissions**: Path-aware permission system with persistent rules
- **Multi-User Architecture**: UUID-based container isolation for web deployment
- **Computational Notebook**: Jupyter-style code execution and cell management
- **Audio Integration**: Speech recognition, TTS, and audio processing
- **Remote Hosting**: Distributed AI model hosting capabilities
- **Comprehensive Testing**: Full test suite with multiple testing levels

## Deployment Options

### 1. Local Development (Single User)
```bash
python main.py              # CLI mode
python main.py --gui        # Web interface
python main.py --api        # API server only
```

### 2. Isolated Mode (Docker - Single Run) 🆕
Run in a fully isolated Docker container with a single flag:

```bash
# First run (builds cached image, ~2-3 min)
python main.py --isolated

# Subsequent runs (instant start, ~2-5 sec)
python main.py --isolated -m "Your message here"
python main.py --isolated --gui  # Web interface in container
```

**Benefits**: Full isolation, no dependency conflicts, auto-syncs latest code
**See:** [Isolated Mode Docs](docs/ISOLATED_MODE.md) for details.

### 3. Multi-User Production (Docker)
Deploy CLI-Agent as a web service where each user gets an isolated container:

```bash
# Build all images
./saas_scripts/build_all.sh

# Deploy infrastructure
./saas_scripts/deploy_infrastructure.sh
```

**See:** [Multi-User Architecture Docs](docs/MULTI_USER_ARCHITECTURE.md) for details.

**Note:** Multi-user architecture is experimental. For production use, see [Deployment Guide](docs/DEPLOYMENT.md).

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CLI-Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the packages in development mode:
```bash
pip install -e core/
pip install -e shared/
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Basic Usage
```bash
python main.py
```

### Running with TUI (Terminal User Interface)
```bash
python tui_main.py
```

### Running Tests
```bash
python run_tests.py --quick
python run_tests.py --full
```

## Architecture

### Single-User Architecture

![Architecture Diagram](docs/architecture-diagram.png)

- `core/`: Core AI infrastructure (LLM routing, chat, providers)
- `shared/`: Shared utilities (audio, web, search, RAG)
- `tools/`: Individual tool implementations (file operations, search, etc.)
- `api/`: Backend API layer (WebSocket, REST endpoints)
- `frontend/`: Web UI assets (HTML, CSS, JavaScript)
- `tests/`: Comprehensive test suite including tool testing framework
- `docs/`: Documentation and architecture guides

### Multi-User Architecture

```
Traefik → Frontend (Nginx) → Container Manager (FastAPI)
                            → Backend Containers (per-UUID)
```

Each user gets:
- Unique UUID (client-generated)
- Isolated Docker container
- 7-day data retention (configurable)
- Independent session state (multi-tab support)

**See:**
- [Multi-User Architecture](docs/MULTI_USER_ARCHITECTURE.md) - Architecture details
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment

## Configuration

The agent uses environment variables for configuration:
- `ANTHROPIC_API_KEY`: Anthropic/Claude API key
- `OPENAI_API_KEY`: OpenAI API key
- `GOOGLE_API_KEY`: Google Gemini API key
- `GROQ_API_KEY`: Groq API key
- `BRAVE_API_KEY`: Brave Search API key

## Auto-Generated Test Image

![Daily Celebration](docs/daily-celebration.png)

## Contributing

1. Run tests: `python run_tests.py`
2. Check code formatting: `black .`
3. Run linting: `ruff check .`

## License

This project is licensed under the Apache License, Version 2.0.