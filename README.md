# CLI-Agent

A powerful command-line AI agent framework for autonomous task execution and coding assistance.

## Overview

CLI-Agent is a comprehensive framework that provides:
- Multi-provider LLM integration (OpenAI, Anthropic, Google Gemini, Groq, Ollama)
- Advanced agent architecture with Monte Carlo Action Engine
- Vector database for smart tool retrieval and contextual guidance
- Computational notebook capabilities
- Rate limiting and enhanced permission systems
- Audio processing and speech recognition
- Remote host capabilities for distributed computing

## Features

- **Multi-LLM Support**: Seamlessly switch between different AI providers
- **Smart Tool Management**: Vector-based tool discovery and context-aware suggestions
- **Interactive Permissions**: Path-aware permission system with persistent rules
- **Computational Notebook**: Jupyter-style code execution and cell management
- **Audio Integration**: Speech recognition, TTS, and audio processing
- **Remote Hosting**: Distributed AI model hosting capabilities
- **Comprehensive Testing**: Full test suite with multiple testing levels

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

### Running with TUI
```bash
python tui_main.py
```

### Running Tests
```bash
python test.py --quick
python test.py --full
```

## Architecture

- `core/`: Core AI infrastructure (LLM routing, chat, providers)
- `shared/`: Shared utilities (audio, web, search, RAG)
- `main_cli_agent/`: Main agent implementation
- `py_classes/`: Legacy classes (remote host, interfaces)
- `tests/`: Comprehensive test suite
- `docs/`: Documentation and architecture guides

## Configuration

The agent uses environment variables for configuration:
- `ANTHROPIC_API_KEY`: Anthropic/Claude API key
- `OPENAI_API_KEY`: OpenAI API key
- `GOOGLE_API_KEY`: Google Gemini API key
- `GROQ_API_KEY`: Groq API key
- `BRAVE_API_KEY`: Brave Search API key

## Contributing

1. Run tests: `python test.py`
2. Check code formatting: `black .`
3. Run linting: `ruff check .`

## License

This project is licensed under the GNU Affero General Public License v3 (AGPL-3.0).