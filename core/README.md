# CLI-Agent Core Infrastructure

This package contains the core AI and infrastructure components for CLI-Agent.

## Installation

```bash
# Install in editable mode for development
pip install -e .

# Install with all AI providers
pip install -e ".[providers]"
```

## Components

- **LLM Router**: Multi-provider LLM routing and management
- **Chat System**: Conversation management and role handling
- **AI Strengths**: Capability-based model selection
- **Globals**: Configuration and settings management
- **Providers**: AI service provider implementations

## Usage

```python
from core import LlmRouter, Chat, Role, AIStrengths, g
```