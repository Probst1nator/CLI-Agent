# Agent Subsystem Documentation

The agent subsystem is the core of CLI-Agent's interactive functionality, providing the command system, GUI components, and application context management.

## Architecture Overview

The agent follows a modern, context-based architecture built around the **AppContext pattern** - a dependency injection system that eliminates global state and enables proper testing.

### Key Components

1. **Core (`agent/core/`)**: Application initialization, context management, input handling
2. **Commands (`agent/commands/`)**: Modular command system with base classes and handlers
3. **GUI (`agent/gui/`)**: Prompt_toolkit-based terminal UI with layouts and widgets

## Core Architecture (`agent/core/`)

### AppContext Pattern (app_context.py)

Single source of truth for runtime configuration:
- **AgentSettings dataclass**: Centralized configuration replacing argparse dependencies
- **Dependency injection**: Clean component initialization without globals
- **Lazy loading**: Heavy dependencies loaded on-demand for fast startup

See: `agent/core/docs/APP_CONTEXT.md` for detailed implementation guide.

### Input Management System

Unified input handling across all interfaces (CLI, GUI, Web, Voice):
- `input_manager.py` - Central input coordination and routing
- `input_sources.py` - Concrete implementations (CLI, GUI, Web, Voice, Queue)
- `input_compatibility.py` - Bridge for legacy code migration

### Initialization & Main Loop

- `initialization.py` - Environment setup, logging config, context creation
- `agent_loop.py` - Main conversation loop with error handling and command processing
- `cli_args.py` - CLI argument parsing extracted from monolithic main.py

### Command System

- `command_registry.py` - Single source of truth for command definitions
- `command_processor.py` - Command processing with specialized processors
- `unified_completion.py` - Tab completion across all interfaces

## Command System (`agent/commands/`)

Modern command architecture with complete decoupling from argparse:

### Base Infrastructure

- `base_command.py` - Abstract base class with protocols and type safety
- `command_handler.py` - Central dispatcher replacing switch-case logic

### Command Categories

**Simple Commands** (`simple_commands.py`):
- LLM selection (PlannerLlmCommand, RouterLlmCommand, FallbacksCommand)
- Configuration display (ConfigCommand)
- Permission management (PermissionCommand)
- UI toggles (multiline, exit, reset)

**Profile Commands** (`profile_commands.py`):
- Save/load LLM configurations
- Profile management (list, delete)

**Complex Commands**:
- `screenshot_command.py` - Screen capture integration
- `help_command.py` - Dynamic help generation
- `regenerate_command.py` - Chat history management

See: `agent/commands/docs/COMMAND_SYSTEM.md` for implementation guide.

## GUI Components (`agent/gui/`)

Prompt_toolkit-based terminal interface:

### Widget System

- `widgets.py` - ChatDisplay, StatusPanel, InputArea, ChatMessage
- Standardized event handling
- Integration with unified input system

### Layout Management

- `layouts.py` - Main, debug, minimal, compact layouts
- Dynamic layout switching
- Status panel integration

### Event Management

- `event_manager.py` - Standardized event types
- Session lifecycle management
- Input coordination

See: `agent/gui/docs/LAYOUTS.md` for layout system details.

## Legacy Components

These components are migrated from py_classes but not yet refactored:

- `notebook/computational_notebook.py` - Jupyter-style execution
- `playbook/playbook_manager.py` - Multi-step workflows
- `utils_manager/utils_manager.py` - Tool orchestration
- `text_painter/stream_painter.py` - Colorized output
- `llm_selection/llm_selector.py` - Interactive model selection

## Key Achievements

### AppContext Refactor (2025-01)

**Major architectural improvement**:
- ✅ Commands fully decoupled from argparse
- ✅ All commands use `self.context.settings` instead of args
- ✅ Eliminated "dummy args" hack
- ✅ Commands now completely testable in isolation
- ✅ Foundation for config from multiple sources (CLI, files, GUI, env)

**Benefits**:
- Clear separation: CLI parsing vs runtime configuration
- No more duplicate command execution bugs
- Clean testing without argparse mocking

## Related Documentation

- Application context: `agent/core/docs/APP_CONTEXT.md`
- Command implementation: `agent/commands/docs/COMMAND_SYSTEM.md`
- GUI layouts: `agent/gui/docs/LAYOUTS.md`
- Main architecture: `/ARCHITECTURE.md`
- Backend integration: `/api/docs/README.md`
