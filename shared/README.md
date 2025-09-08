# CLI-Agent Shared Infrastructure

This package contains the shared utilities and infrastructure components used across all CLI-Agent tools.

## Installation

```bash
# Install in editable mode for development
pip install -e .

# Install with optional dependencies
pip install -e ".[pdf,audio,ai]"
```

## Components

- **Path Resolution**: Consistent import path handling
- **Utilities**: Common helper functions
- **Audio Processing**: Shared audio utilities  
- **Command Execution**: Process management helpers

## Usage

```python
from shared import PathResolver, setup_cli_agent_imports
from shared.common_utils import extract_blocks
from shared.dia_helper import get_dia_model
```