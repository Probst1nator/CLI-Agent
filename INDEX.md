# Project File Index

## 🏗️ Modern Architecture (Post-Migration & Refactoring)

### 🎯 **RECENT MAJOR ARCHITECTURAL ACHIEVEMENT**: AppContext as Single Source of Truth

**Completed**: Full transition from argparse-dependent command system to AppContext-based architecture:
- ✅ **AgentSettings dataclass**: Centralized configuration with `load_from_args()` method
- ✅ **Command decoupling**: All commands now use `self.context.settings` instead of args parameter
- ✅ **BaseCommand refactor**: Execute signature simplified, removing argparse dependency
- ✅ **CommandHandler cleanup**: Eliminated args parameter from handle_command method
- ✅ **Processor cleanup**: Removed `_create_dummy_args` function from CommandInputProcessor
- ✅ **Main.py integration**: AppContext populated from args at startup, single initialization point

**Benefits Achieved**:
- Commands are now completely testable in isolation (no argparse mocking needed)
- Clear separation of concerns: CLI parsing vs runtime configuration
- Eliminated dual command processing pipelines that caused duplicate execution bugs
- Foundation for configuration from multiple sources (CLI, config files, GUI, environment)
- Removed "dummy args" hack and transitional code patterns

- `agent/` - Modern agent architecture with organized components:
  
  **Refactored Core Architecture**
  - `core/app_context.py` - Application context replacing global state singleton with dependency injection pattern, providing centralized configuration, lazy-loaded components, and proper typing. **MAJOR REFACTOR**: Now includes `AgentSettings` dataclass as single source of truth for all runtime configuration, eliminating command dependency on argparse and enabling clean command isolation
  - `core/initialization.py` - Clean application initialization logic extracted from monolithic main.py with environment setup, logging configuration, and context creation
  - `core/agent_loop.py` - Main conversation loop extracted from main.py with proper error handling, command processing, and AI interaction management
  - `core/lazy_imports.py` - Startup time optimization through lazy loading of heavy dependencies, demonstrating ~1.5s potential startup time savings
  - `core/cli_args.py` - CLI argument parsing extracted from main.py with exact same logic, enabling reuse across different entry points (main.py, main_refactored.py)
  - `core/config_utils.py` - LLM configuration functions extracted from main.py, consolidating apply_default_llm_config() with preserved logic
  - `core/command_system.py` - : Unified command definition system and command completion extracted from main.py. Provides centralized command management with aliases, descriptions, and prompt toolkit integration
  - `core/input_manager.py` - : Unified input management system providing clean abstraction layer for all input sources (CLI, GUI, Web, Voice) with standardized event handling, source registration, and processor routing
  - `core/input_sources.py` - : Concrete input source implementations for CLI (prompt_toolkit), GUI (event-driven), Web (browser-based), Voice (speech recognition), and Queue (programmatic) input methods
  - `core/command_processor.py` - : Command processing system integrating with unified input manager, providing specialized processors for commands, text, interrupts, and images with proper error handling. **MAJOR REFACTOR**: Removed `_create_dummy_args` function, eliminated argparse dependency from command processing pipeline
  - `core/input_compatibility.py` - : Compatibility layer bridging legacy and new input systems, enabling gradual migration while maintaining backwards compatibility with existing get_user_input_with_bindings interface
  - `core/input_handlers.py` - : Input handling utilities now serving as compatibility wrappers that delegate to the unified input system while preserving original function interfaces for seamless transition
  - `core/unified_completion.py` - : Unified command completion system working across all interfaces (CLI, GUI, Web) with consistent behavior, real-time command discovery, and JSON serialization for web clients
  - `core/command_registry.py` - : Single source of truth for command definitions, centralizing command metadata and bridging static definitions with dynamic implementations through categorized command registry. : Added unified LLM selection commands (`/planner_llm`, `/router_llm`, `/fallbacks`) and configuration display command (`/config`) replacing WebSocket-based selection with command-based approach
  - `core/migration_helper.py` - : Migration utilities for transitioning main.py from legacy input handling to unified system, providing drop-in replacements, monkey-patching capabilities, and gradual migration support
  
  **Command System Architecture**
  - `commands/base_command.py` - Abstract base class and protocols for all CLI commands with proper typing, consistent interfaces, and command registration system. **MAJOR REFACTOR**: Execute method signature simplified to remove argparse dependency, commands now access configuration via `self.context.settings`
  - `commands/command_handler.py` - Central command dispatcher replacing massive switch-case logic from get_user_input_with_bindings with clean, modular command routing. **MAJOR REFACTOR**: Eliminated args parameter from handle_command, now uses context-based configuration exclusively
  - `commands/screenshot_command.py` - Screenshot command implementation demonstrating the new command pattern with proper error handling and response types
  - `commands/help_command.py` - Help command showing integration with command handler for dynamic help generation
  - `commands/regenerate_command.py` - Regenerate command with chat history management and proper state handling
  - `commands/simple_commands.py` - Collection of simple command implementations (multiline, exit, reset, toggle commands) with consistent interfaces. : Added unified LLM selection commands (`PlannerLlmCommand`, `RouterLlmCommand`, `FallbacksCommand`), configuration display command (`ConfigCommand`), and replaced `ClearCommand` with command-based approach integrating with web UI selectboxes. **NEW**: Enhanced `PermissionCommand` with simple command-based permission management supporting direct actions like `/perms add dev grepsearch` and guided wizards
  
  **GUI Components**
  - `gui/layouts.py` - Layout definitions and management for prompt_toolkit-based GUI including main layout, debug layout, minimal layout, and compact layout options
  - `gui/widgets.py` - : Custom GUI widgets with standardized event handling including ChatDisplay for conversation history, StatusPanel for system information, InputArea for user input, and ChatMessage for message formatting
  - `gui/event_manager.py` - : Standardized GUI event handling and session management system providing unified event types, session lifecycle management, input coordination, and event processing for all GUI interfaces
  - `gui/__init__.py` - GUI package initialization providing clean imports for layout and widget components
  
  
  **Legacy Components (Pre-Refactor)**
  - `notebook/computational_notebook.py` - Jupyter-style code execution and cell management (migrated from py_classes)
  - `playbook/playbook_manager.py` - Strategic multi-step workflow templates and procedural guidance (migrated from py_classes)  
  - `utils_manager/utils_manager.py` - Orchestration and management of utility tools (migrated from py_classes)
  - `text_painter/stream_painter.py` - Colorized streaming text output with syntax highlighting (migrated from py_classes)
  - `llm_selection/llm_selector.py` - Interactive LLM selection and capability filtering (migrated from py_classes)
  - `utils_manager/util_base.py` - Base utility class providing common functionality (migrated from py_classes)

- `infrastructure/` - Supporting infrastructure organized by purpose:
  - `rate_limiting/` - Rate limiting tracker and enforcement across AI services (migrated from py_classes)
  - `vector_db/vector_database.py` - Semantic search and embeddings storage using ChromaDB (migrated from py_classes)
  - `remote_host/` - Remote AI model hosting and distributed computing capabilities (migrated from py_classes)

- `shared/utils/` - Structured shared utilities organized by domain:
  - `web/web_server.py` - HTTP endpoints and web-based interfaces (migrated from py_classes/utils)
  - `search/brave_search.py` - Web search API integration (migrated from py_classes/utils)
  - `search/search_vector_db.py` - : SearchResultsVectorDB class for semantic search caching using ChromaDB and embeddinggemma:latest embeddings. Manages two collections (web_search_results + search_summaries) with similarity-based retrieval, automatic result storage, and progressive knowledge building for the SearchWeb tool
  
  - `python/python_utils.py` - Code execution and Python environment manipulation (migrated from py_classes/utils)
  - `log_manager.py` - Centralized log retention and cleanup management system with automatic rotation, retention policies, background cleanup scheduling, and comprehensive statistics for all CLI-Agent logging operations
  - `pdf_utils.py` - : Consolidated PDF processing utilities extracted from duplicated functions in shared/common_utils.py and shared/audio/audio_utils.py. Provides clean_pdf_text(), extract_pdf_content(), get_cache_file_path(), and related functions with caching support
  - `shutdown_utils.py` - : Graceful shutdown utilities extracted from main.py and main_refactored.py, supporting both legacy and modern architecture patterns with proper log cleanup and context management
  - `display_utils.py` - : UI and display utilities extracted from main.py including colorize_model_name(), print_startup_summary(), and show_unified_bindings_display() for consistent formatting across the application
  - `logging_utils.py` - : Logging formatters and configuration extracted from main.py. Provides ConsoleFormatter, SimpleFormatter, FileFormatter classes and setup_logging() function with retention management

- `shared/audio/audio_utils.py` - Audio processing, TTS, and speech recognition (migrated from py_methods/utils)

## 📋 Legacy Structure (Deprecated)

- ⚠️ `py_classes/` - **DEPRECATED**: Legacy flat organization, see DEPRECATED.md for migration guide
- ⚠️ `py_methods/` - **DEPRECATED**: Legacy utilities, see DEPRECATED.md for migration guide

## 🧪 Testing & Development

- `test.py` - Unified, user-facing entrypoint for the entire test suite. This script provides a simple command-line interface with modes like `--quick`, `--full`, and `--verbose`, orchestrating the underlying test infrastructure to provide clear, actionable results for developers. This is the primary command to run for validating changes.

- `test_gui_components.py` - : Comprehensive test suite for GUI components validating ChatDisplay, StatusPanel, InputArea widgets, layout systems, command integration, and core CLI-Agent integration without requiring terminal interface. Provides component-level testing for the prompt_toolkit GUI implementation.

- `run_tests.py` - The core test runner infrastructure that powers `test.py`. This module is responsible for the low-level mechanics of test discovery (scanning for `test_*` functions), hierarchical execution based on defined levels (e.g., unit, integration), and aggregating results. Developers typically don't run this script directly.

- `profiling.py` - A diagnostic utility script for profiling Python import times. By running this script, developers can identify which modules or dependencies are contributing most to application startup latency, helping to optimize the initial load time.

- `.env.example` - A template file that defines the necessary environment variables, such as API keys and configuration settings. Developers should copy this to a `.env` file and populate it with their own credentials, keeping sensitive information out of version control.

- `requirements.txt` - A standard file listing all necessary Python packages for the project. It is used with `pip install -r requirements.txt` to ensure a consistent and reproducible development environment.

- `CLAUDE.md` - AI-assisted development guide containing high-level project rules, architecture patterns, and key commands.

- `README.md` - The main project documentation, serving as the official entry point for new contributors (currently a placeholder).

- `ARCHITECTURE.md` - Comprehensive analysis of the monorepo structure and detailed strategy for decomposing it into 3 separate projects corresponding to tools/* directories, including dependency mapping, migration complexity assessment, and recommended decomposition approaches.

- `main.py` - **CURRENT/ACTIVE**: The main entrypoint and orchestration script for the CLI-Agent. Contains the main execution loop, user input handling, and all recent bug fixes including the ComputationalNotebook architecture fix. : Input system consolidated - removed unused `get_gui_input()` function and added consistent `get_cli_input()` function. All input now flows through the unified input management system. **FIXED**: Added router LLM initialization during startup (lines 1561-1574) to ensure `g.router_llm` is properly loaded from configuration, matching planner LLM initialization pattern. This is the **stable, production entry point** that should be used for all current development and deployment.

- `main_refactored.py` - **⚠️ EXPERIMENTAL**: Clean, modular main entrypoint demonstrating the future refactored architecture. At only ~188 lines, this showcases the new agent/core components with dependency injection and proper separation of concerns. **Status: Work in progress** - still depends on main.py imports and lacks recent bug fixes. For stable operation, use main.py instead.

- `main_prompt_toolkit_gui.py` - : Enhanced GUI entry point using prompt_toolkit for conversational AI interface. Provides a superior alternative to Textual TUI attempts by building on existing prompt_toolkit infrastructure. Features full-screen chat interface with streaming responses, command processing, browser input monitoring, and integration with the complete CLI-Agent conversation system.

- `enhanced_permissions.py` - Enhanced permission system providing interactive prompts with persistent "always allow" functionality. Supports path-specific permissions, directory-based rules, and intelligent permission scoping with options like exact file, directory pattern, project-wide, or tool-wide permissions. Permission prompts now display tool arguments in addition to file paths and actions.

- `permissions_manager.py` - Persistent permission storage manager handling tool-specific and path-specific permission rules. Manages JSON-based configuration storage in user's home directory with support for directory patterns, exact path matching, and permission inheritance.

- `path_detector.py` - XML block analyzer for detecting file paths in tool commands by scanning for tags containing 'path' substring. Provides intelligent action descriptions and determines permission requirements based on tool types and detected file operations.

- `tool_context.py` - Global tool context manager for cross-tool communication, particularly handling base64 image data transfer from tools to chat context for vision model integration.

- `requirements.txt` - A component-specific list of Python dependencies required for the Main CLI Agent. This file extends the project's root `requirements.txt`, adding extra packages necessary for advanced features like interactive prompts (`prompt-toolkit`), clipboard operations (`pyperclip`), and audio/PDF processing.

- `__init__.py` - Makes the `main_cli_agent` directory a recognizable Python package. It contains package-level metadata, including the version number and a high-level summary of the tool's purpose.

- `assets/podcast.svg` - An SVG icon graphic depicting a podcast-related symbol with chat bubbles and user icons, used for UI elements related to podcast functionality.

- `assets/screenshot.png` - A screenshot image file used for documentation or testing purposes.

- `assets/test_image.png` - A test image file showing a YouTube video screenshot of an autumn scene, used for testing image processing and analysis functionality.

- `cliagent_instructions.md` - Step-by-step guide for creating a VS Code extension that exposes currently open files via HTTP server, along with corresponding Python script to retrieve this information for external tool integration.

- `core/ai_strengths.py` - Enum definition for AI model capabilities, categorizing different strengths like UNCENSORED, VISION, and LOCAL processing with numeric values.

- `core/chat.py` - Core Chat class for managing conversational AI interactions, supporting multiple message roles, colorized terminal output, debug windows, serialization, and conversion to various AI API formats (Ollama, OpenAI, Groq, Gemini, Gemma).

- `core/globals.py` - Singleton global state manager handling application configuration, file paths, LLM settings, debug flags, model discovery caching, persistent storage with cross-platform compatibility, and enhanced temporary file cleanup using the integrated file manager.

- `core/file_manager.py` - Centralized temporary file lifecycle management system providing coordinated file creation, tracking, and cleanup across all CLI-Agent components with configurable TTL policies, automatic background cleanup, context managers, and comprehensive statistics to prevent file accumulation.

- `core/__init__.py` - Package initialization file providing compatibility imports between new core location and legacy py_classes modules, exposing key classes like Chat, LlmRouter, AIStrengths, and global state.

- `core/llm.py` - LLM (Large Language Model) class managing AI model properties, capabilities, and discovery across providers (Google, Groq, Ollama), with background caching, parallel discovery, and dynamic context window fetching.

- `core/llm_router.py` - Comprehensive LLM router singleton managing model selection, request routing, caching, stream processing, error handling, retry logic, and provider integration across multiple AI services with activity monitoring and fine-tuning data collection.

- `core/providers/cls_anthropic_interface.py` - Anthropic API provider implementation for Claude models, handling chat format conversion, image processing (base64/URL), system prompts, and streaming responses with unified interface integration.

- `core/providers/cls_google_interface.py` - Google Gemini API provider implementation with chat format conversion, temperature control, thinking budget support for Gemini 2.x models, rate limiting integration, and streaming response generation.

- `core/providers/cls_groq_interface.py` - Groq API provider implementation with custom exception handling (TimeoutException, RateLimitException), rate limiting support, and streaming response generation for fast inference models.

- `core/providers/cls_human_as_interface.py` - Human-in-the-loop interface allowing human operators to respond instead of AI models, with speech recognition support for interactive testing and evaluation scenarios.

- `core/providers/cls_nvidia_interface.py` - NVIDIA NeMo API provider implementation using OpenAI-compatible interface for NVIDIA's LLM services with base64 image support and temperature control.

- `core/providers/cls_ollama_interface.py` - Comprehensive Ollama local LLM provider with model discovery, host management, context window detection, date parsing utilities, and streaming response generation for self-hosted models.

- `core/providers/cls_openai_interface.py` - OpenAI API provider implementation with streaming response generation, temperature control, and speech recognition support for GPT models and OpenAI services.

- `core/providers/cls_whisper_interface.py` - Whisper audio provider implementation for speech transcription and text-to-speech functionality using local Whisper models and Kokoro TTS integration.

- `core/providers/cls_piper_interface.py` - Piper TTS provider implementation for high-quality local text-to-speech synthesis with voice selection, multilingual support, and audio generation capabilities.

- `Dockerfile.test` - Docker container configuration for testing CLI-Agent functionality with Ubuntu 22.04 base, complete dependency installation, and non-root user setup for isolated testing environment.

- `docker_files/Dockerfile.isolated` - Production Docker container for secure, isolated CLI-Agent deployment with Ubuntu 22.04, security hardening, non-root user (UID 1000), comprehensive dependencies, and isolated entrypoint script for standalone operation.

- `docker_files/docker-compose.isolated.yml` - Container orchestration configuration for isolated CLI-Agent deployment with network isolation (internet-only access), persistent Docker volumes, resource limits, health checks, and systemd integration for production use.

- `docs/AGENT_RAG.md` - Documentation of the CLI-Agent's RAG (Retrieval-Augmented Generation) system architecture, covering ToolVectorDB, PlaybookManager, and UtilsManager integration for contextual guidance and strategic planning.

- `docs/APIS.md` - Comprehensive API documentation for LLM providers (Google Gemini, Groq, Ollama), detailing data structures, parameters, and model discovery endpoints for dynamic model integration.

- `docs/DOCKER_ISOLATED.md` - Comprehensive documentation for CLI-Agent isolated Docker container setup with network isolation, security hardening, UFW firewall configuration, troubleshooting guide, and production-ready deployment instructions including port mapping (7860→7861) and dependency management.

- `docs/CURRENT_AND_IDEAL_ARCHITECTURE.md` - Architecture analysis documenting duplicate files between py_classes and core directories, with findings on AI provider interfaces and proposed monorepo structure improvements.

- `docs/SEARCHWEB_VECTOR_DB.md` - : Comprehensive documentation for SearchWeb vector database integration using ChromaDB and embeddinggemma:latest. Details semantic search architecture, two-collection system (search results + summaries), query flow, performance benefits, configuration, and troubleshooting for the intelligent caching system that reduces API calls and builds progressive knowledge base.

- `docs/OLLAMA_INFO.md` - Documentation for Ollama integration, covering local model hosting, configuration, and offline LLM capabilities.

- `license` - GNU Affero General Public License v3 (AGPL-3.0) governing the distribution and modification of the CLI-Agent project as open source software.

- `pyproject.toml` - Python project configuration for code formatting (Black) and linting (Ruff) tools, setting line length, target Python version, and rule selections for code quality.

- `py_classes/agent_template.py` - Template and example code for creating agent instances with web search capabilities, demonstrating the agent instantiation and usage patterns.

- `py_classes/cls_computational_notebook.py` - Computational notebook class for managing Jupyter-style code execution, cell management, and interactive computing workflows.

- `py_classes/cls_llm_selection.py` - LLM selection and discovery interface providing model enumeration, capability filtering, and dynamic model recommendation based on user requirements.

- `py_classes/cls_playbook_manager.py` - Playbook management system for storing and retrieving strategic multi-step workflows, task templates, and procedural guidance using vector similarity matching.

- `py_classes/cls_pyaihost_interface.py` - PyAI host interface for remote AI model execution, supporting distributed computing and remote inference capabilities.

- `py_classes/cls_rate_limit_tracker.py` - Rate limiting tracker for managing API call quotas, request throttling, and provider-specific rate limit enforcement across multiple AI services.

- `py_classes/cls_text_stream_painter.py` - Text streaming utility for colorizing and formatting streaming text output with syntax highlighting and visual enhancements for terminal display.

- `py_classes/cls_util_base.py` - Base utility class providing common functionality and abstract interfaces for various utility implementations and tool integrations.

- `py_classes/cls_util_manager.py` - Utility manager for orchestrating and managing various utility tools, providing centralized access and coordination for system utilities.

- `py_classes/cls_vector_db.py` - Vector database implementation for semantic search, embeddings storage, and similarity matching using ChromaDB for RAG and context retrieval.

- `py_classes/lazy_import.py` - Lazy import utility for deferring module imports until needed, improving startup performance and handling optional dependencies gracefully.

- `py_classes/unified_interfaces.py` - Unified abstract interfaces defining common contracts for AI providers, audio providers, and other service integrations to ensure consistent API patterns.

- `py_classes/utils/cls_utils_python.py` - Python utility functions for code execution, syntax validation, package management, and Python environment manipulation within the agent system.

- `py_classes/utils/cls_utils_rag.py` - RAG (Retrieval-Augmented Generation) utilities for document processing, embedding generation, context extraction, and semantic search integration.

- `py_classes/utils/cls_utils_web.py` - Web utility functions for HTTP requests, web scraping, URL processing, and web content extraction with proper error handling and rate limiting.

- `py_classes/utils/cls_utils_web_server.py` - Web server utilities for creating HTTP endpoints, serving content, handling API requests, and managing web-based interfaces for the agent system.

- `py_classes/utils/cls_utils_youtube.py` - YouTube-specific utilities for video processing, metadata extraction, transcript retrieval, and YouTube API integration for content analysis.

- `py_classes/wip/cls_youtube_scraper.py` - Work-in-progress YouTube scraper implementation for extracting video data, comments, and metadata from YouTube content (experimental).

- `py_methods/utils.py` - General utility functions and helper methods used across the project for common operations, data processing, and system interactions.

- `shared/cmd_execution.py` - Command execution utilities for running system commands, shell operations, and process management with proper error handling and security considerations.

- `shared/common_utils.py` - Shared common utility functions used across multiple components, providing standardized helper methods for cross-module functionality.

- `shared/dia_helper.py` - Dialog helper utilities for interactive prompts, user input handling, and conversational interface management within the CLI agent.

- `shared/path_resolver.py` - Path resolution utilities for handling file paths, directory navigation, relative path conversion, and cross-platform path management.

- `shared/utils_audio.py` - Enhanced audio processing utilities with integrated Faster-Whisper transcription, anti-hallucination parameters optimized for German, GTX 1080 Ti compatibility, intelligent fallback system (OpenAI Whisper + Google Speech Recognition), and automatic audio debugging for speech recognition and TTS functionality.

- `shell_scripts/run_cli_agent.sh` - Shell script for launching the main CLI agent with proper environment setup and configuration.

- `shell_scripts/run_clipboard_to_podcast.sh` - Shell script for converting clipboard content into podcast format using text-to-speech and audio processing.

- `shell_scripts/run_minimal_agent.sh` - Shell script for running a minimal version of the CLI agent with reduced functionality for lightweight usage.

- `shell_scripts/run_screen_agent.sh` - Shell script for launching the agent with screen capture and visual processing capabilities.

- `shell_scripts/setup_bash_alias.sh` - Setup script for creating bash aliases to easily access CLI agent functionality from the command line.

- `shell_scripts/setup_desktop_entry.sh` - Setup script for creating desktop entries and system integration for GUI access to the CLI agent.

- `shell_scripts/setup_dia_environment.sh` - Environment setup script for configuring the development and runtime environment for the CLI agent system.

- `scripts/manage_search_cache.py` - : Cache management utility for the SearchWeb Vector Database providing commands for viewing statistics, pruning old entries, listing recent cache entries, and clearing cache data with comprehensive maintenance capabilities.

- `scripts/setup_isolated_container.sh` - Automated setup script for CLI-Agent isolated Docker container with prerequisite checking, image building, environment file creation, systemd service installation, and comprehensive status reporting with colored output.

- `scripts/cli-agent-isolated.service` - Systemd service definition for CLI-Agent isolated container providing auto-restart on boot, proper dependency management, timeout handling, and Docker Compose integration for production deployment.

- `scripts/verify_isolated_setup.sh` - Comprehensive verification script for CLI-Agent isolated container setup, testing systemd service, container status, web UI accessibility, dependencies, port mapping, firewall configuration, and internet connectivity with colored output and detailed reporting.

- `tests/README.md` - Testing documentation explaining the test structure, running procedures, and testing methodology for the CLI agent project.

- `tests/test_cross_tool.py` - Integration tests for cross-tool communication and coordination between different agent components and utilities.

- `tests/test_file_copier.py` - Unit tests for file copying functionality, path resolution, and smart file management features.

- `tests/test_path_resolver.py` - Unit tests for path resolution utilities, testing cross-platform path handling and directory navigation.

- `tests/test_smart_paster.py` - Unit tests for smart paste functionality, content analysis, and intelligent clipboard processing.

- `tests/test_type_hints.py` - Type checking tests ensuring proper type annotations and static type validation across the codebase.

- `tests/test_audio_transcription.py` - Audio transcription tests for speech-to-text functionality including device selection, German language support, and scientific vocabulary handling. Documents known issues with Faster-Whisper tiny model failing on German scientific terms.

- `tests/test_voice_pipeline.py` - Simple voice pipeline test generating synthetic audio using espeak TTS and testing transcription accuracy across German/English with different voice types (12 test files total).

- `tests/test_german_transcription_pipeline.py` - Focused German scientific vocabulary transcription test comparing Faster-Whisper model sizes (tiny/base/small/medium) on real German audio to identify optimal models for scientific terms.

- `tests/audio/` - Test audio files directory containing real-world transcription test cases:
  - `german_transcription_failure.wav` - Audio file demonstrating Faster-Whisper failure on German scientific vocabulary ("kannst du mir erklären was Wasserstoffbrückenbindungen sind" incorrectly transcribed as "Das ist das, das ist das, das ist das.")
  - `pipeline/` - Generated test audio files for systematic testing (12 files: 3 German + 3 English sentences × 2 voice types)

- `tests/core/test_model_discovery.py` - Core tests for LLM model discovery, provider integration, and dynamic model enumeration functionality.

- `tests/core/test_piper_interface.py` - Core tests for Piper TTS interface functionality, voice selection, audio generation, and quality verification.

- `tools/file_copier/main.py` - Main executable for the file copying tool with intelligent path resolution, duplicate detection, and smart paste functionality for efficient file management.

- `tools/file_copier/smart_paster.py` - Smart paste implementation for analyzing clipboard content, detecting file types, and providing intelligent suggestions for content placement and processing.

- `tools/file_copier/ai_path_finder.py` - AI-powered path finding utility using language models to intelligently resolve file paths and suggest optimal file locations based on context.

- `tools/file_copier/gui.py` - Graphical user interface for the file copier tool, providing user-friendly interaction for file management operations with visual feedback.

- `tools/taskmanager.py` - Background process management tool for managing processes started with <bash><background>true</background>, providing listing, killing, and output monitoring capabilities for long-running background tasks.

- `tools/todos.py` - Todo management functionality for the CLI agent, providing task tracking, reminder systems, and workflow organization capabilities.

- `tools/readfile.py` - Enhanced file reading tool with line-indexed access capabilities, supporting range reading (e.g., lines 10-20) and single line access for precise file inspection and analysis.

- `tools/editfile.py` - Enhanced file editing tool supporting both traditional find-and-replace operations and precise line-specific editing for surgical code modifications.

- `tools/insertline.py` - Line insertion tool for adding new content at specific positions in files, supporting insertion before/after specific line numbers or at file beginning/end.

- `tools/deletelines.py` - Line deletion tool for removing specific lines or line ranges from files with precise control over deletion scope and detailed feedback.

- `tools/writefile.py` - File writing tool for creating new files or completely overwriting existing files with specified content, with automatic directory creation.

- `tools/permissions.py` - Advanced permission management tool for viewing, editing, and removing specific saved permissions with interactive management capabilities and detailed listing.

- `utils/architectnewutil.py` - Architecture utility for creating new system components, scaffolding code structures, and managing system architecture evolution.

- `utils/generateimage.py` - Image generation utility integrating with AI image generation services for creating visual content within agent workflows.


- `utils/homeassistant.py` - Home Assistant integration utility for smart home automation, device control, and IoT interaction through the CLI agent.


- `utils/removefile.py` - File removal utility with safety checks, backup options, and intelligent deletion policies for secure file management.

- `utils/searchweb.py` - Web search utility providing internet search capabilities, result processing, and web information retrieval for agent queries.

- `utils/showuser.py` - User interface utility for displaying information to users, formatting output, and managing interactive display elements.

- `utils/takescreenshot.py` - Screenshot capture utility for screen recording, visual documentation, and image analysis integration within agent workflows.

- `utils/tts.py` - Text-to-speech utility for converting text content to audio, supporting multiple voices and languages for accessibility and audio interfaces.

- `utils/updatefile.py` - File update utility for modifying existing files, applying patches, and managing content updates with version control integration.

- `utils/viewfiles.py` - File viewing utility for displaying file contents, supporting multiple formats, and providing formatted output for various file types.

- `utils/viewimage.py` - Image viewing utility for displaying images, extracting metadata, and providing image analysis capabilities within the agent system.

- `utils/web_fetch.py` - Web content fetching utility for retrieving web pages, handling HTTP requests, and processing web-based information sources.

- `tools/configure_tts.py` - TTS configuration utility for interactive setup and management of text-to-speech engine preferences, voice selection, and audio output settings.

- `tools/tts_tester.py` - Comprehensive TTS testing and benchmarking utility for evaluating all available TTS engines, comparing quality, and configuring optimal TTS setup.

- `tools/podcast_generator/main.py` - Podcast generation tool for creating audio content from text, managing podcast workflows, and automating audio content production.

- `core/cli_agent_core.egg-info/dependency_links.txt` - Empty dependency links file for the core package build configuration.

- `core/cli_agent_core.egg-info/PKG-INFO` - Package metadata for cli-agent-core including version 1.0.0, dependencies (OpenAI, Anthropic, etc.), and installation instructions.

- `core/cli_agent_core.egg-info/requires.txt` - Package requirements specification for the core package dependencies and optional provider extras.

- `core/cli_agent_core.egg-info/SOURCES.txt` - Source file manifest listing all files included in the core package distribution.

- `core/cli_agent_core.egg-info/top_level.txt` - Top-level module specification for the core package indicating which modules are exposed at package level.

- `core/pyproject.toml` - Core package build configuration with setuptools metadata, dependencies, and optional extras for AI providers.

- `core/README.md` - Core package documentation explaining the AI infrastructure components, installation, and usage examples.

- `shared/cli_agent_shared.egg-info/dependency_links.txt` - Empty dependency links file for the shared package build configuration.

- `shared/cli_agent_shared.egg-info/PKG-INFO` - Package metadata for cli-agent-shared v1.0.0 with optional extras for PDF, audio, and AI functionality, including installation and usage instructions.

- `shared/cli_agent_shared.egg-info/requires.txt` - Requirements specification for the shared package with optional extras for PDF processing, audio handling, and AI integration.

- `shared/cli_agent_shared.egg-info/SOURCES.txt` - Source file manifest listing all files included in the shared package distribution for cross-tool utilities.

- `shared/cli_agent_shared.egg-info/top_level.txt` - Top-level module specification for the shared package indicating exposed modules for import path resolution and utilities.

- `shared/pyproject.toml` - Shared package build configuration with setuptools metadata, dependencies, and optional extras for specialized functionality.

- `shared/README.md` - Shared package documentation covering common utilities, path resolution, audio processing, and command execution helpers.

- `tools/file_copier/.file_copier_config.json` - Configuration file storing file copier presets, selected files, filter settings, and exclusion patterns for different project directories.

- `tools/file_copier/.gitignore` - Git ignore rules for file copier tool excluding cache, temp, outputs directories and backup files.

- `tools/file_copier/__init__.py` - Package initialization for file copier tool v1.0.0 with intelligent file discovery and AI-powered file finding capabilities.

- `tools/file_copier/LICENSE` - MIT License for the file copier tool component, separate license for this specific tool within the larger AGPL project.

- `tools/file_copier/requirements.txt` - Dependencies for file copier tool including pyperclip for clipboard access, PyMuPDF for PDF reading, and odfpy for ODT files.

- `tools/__init__.py` - Tools directory package initialization defining available tools registry (file_copier, podcast_generator, main_cli_agent) for future expansion.

- `__init__.py` - Main CLI agent tool package initialization v1.0.0 providing primary command-line interface with full AI capabilities and tool integration.

- `utils/__init__.py` - Utilities package initialization for main CLI agent importing various tools like UpdateFile, RemoveFile, ViewFiles, SearchWeb, and other utility functions.

- `tools/podcast_generator/__init__.py` - Podcast generator tool package initialization v1.0.0 for AI-powered podcast generation from text, PDFs, or clipboard data with multiple TTS engine support.

- `tools/podcast_generator/requirements.txt` - Dependencies for podcast generator including pydub for audio manipulation, google-genai for TTS, prompt-toolkit for CLI interface, and PyMuPDF for PDF extraction.

- `py_classes/__init__.py` - Empty package initialization file for legacy py_classes module structure.

- `shared/__init__.py` - Shared utilities package initialization providing compatibility imports for dia_helper, common_utils, AIFixPath, and PathResolver with fallback mechanisms.

- `tests/__init__.py` - Comprehensive test suite initialization v1.0.0 defining hierarchical testing framework with levels 0-4 from integration to unit tests.

- `tests/shared/__init__.py` - Shared utilities test package initialization for testing common helper functions used across CLI-Agent tools.

- `tests/tools/__init__.py` - Tool-specific test package initialization for testing individual CLI-Agent tools like file copier, podcast generator, and main CLI agent.

- `tests/core/__init__.py` - Core infrastructure test package initialization for testing LLM routing, chat systems, and global configuration components.

- `py_classes/remote_host/__init__.py` - Remote host package initialization v0.1.0 providing remote hosting capabilities for various CLI-Agent services.

- `py_classes/remote_host/services/__init__.py` - Remote host services package initialization containing individual service modules for remote host functionality.

- `profiling.py` - Python import profiling utility for measuring and analyzing module import times to optimize application startup performance and identify slow-loading dependencies.