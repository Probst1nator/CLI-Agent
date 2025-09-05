# Project File Index

## 🏗️ Modern Architecture (Post-Migration)

- `agent/` - Modern agent architecture with organized components:
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
  - `rag/rag_utils.py` - Document processing and semantic search utilities (migrated from py_classes/utils)  
  - `python/python_utils.py` - Code execution and Python environment manipulation (migrated from py_classes/utils)
  - `youtube/youtube_utils.py` - Video processing and metadata extraction (migrated from py_classes/utils)

- `shared/audio/audio_utils.py` - Audio processing, TTS, and speech recognition (migrated from py_methods/utils)

## 📋 Legacy Structure (Deprecated)

- ⚠️ `py_classes/` - **DEPRECATED**: Legacy flat organization, see DEPRECATED.md for migration guide
- ⚠️ `py_methods/` - **DEPRECATED**: Legacy utilities, see DEPRECATED.md for migration guide

## 🧪 Testing & Development

- `test.py` - Unified, user-facing entrypoint for the entire test suite. This script provides a simple command-line interface with modes like `--quick`, `--full`, and `--verbose`, orchestrating the underlying test infrastructure to provide clear, actionable results for developers. This is the primary command to run for validating changes.

- `run_tests.py` - The core test runner infrastructure that powers `test.py`. This module is responsible for the low-level mechanics of test discovery (scanning for `test_*` functions), hierarchical execution based on defined levels (e.g., unit, integration), and aggregating results. Developers typically don't run this script directly.

- `profiling.py` - A diagnostic utility script for profiling Python import times. By running this script, developers can identify which modules or dependencies are contributing most to application startup latency, helping to optimize the initial load time.

- `.env.example` - A template file that defines the necessary environment variables, such as API keys and configuration settings. Developers should copy this to a `.env` file and populate it with their own credentials, keeping sensitive information out of version control.

- `requirements.txt` - A standard file listing all necessary Python packages for the project. It is used with `pip install -r requirements.txt` to ensure a consistent and reproducible development environment.

- `CLAUDE.md` - AI-assisted development guide containing high-level project rules, architecture patterns, and key commands.

- `README.md` - The main project documentation, serving as the official entry point for new contributors (currently a placeholder).

- `ARCHITECTURE.md` - Comprehensive analysis of the monorepo structure and detailed strategy for decomposing it into 3 separate projects corresponding to tools/* directories, including dependency mapping, migration complexity assessment, and recommended decomposition approaches.

- `main.py` - The primary entrypoint and orchestration script for the agentic coding assistant. This file initializes the agent's core "Monte Carlo Action Engine," which uses a council of LLMs for specialized tasks: proposing multiple action paths (Branch Generators), voting on the best path (Branch Evaluators), and vetoing unsafe code (Execution Guards). It manages the main execution loop, handles user input, and enriches the agent's context by retrieving tool usage hints and strategic "playbooks" from a vector database. Now includes enhanced path-aware permission system for tool execution.

- `enhanced_permissions.py` - Enhanced permission system providing interactive prompts with persistent "always allow" functionality. Supports path-specific permissions, directory-based rules, and intelligent permission scoping with options like exact file, directory pattern, project-wide, or tool-wide permissions.

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

- `core/globals.py` - Singleton global state manager handling application configuration, file paths, LLM settings, debug flags, model discovery caching, and persistent storage with cross-platform compatibility.

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

- `Dockerfile.test` - Docker container configuration for testing CLI-Agent functionality with Ubuntu 22.04 base, complete dependency installation, and non-root user setup for isolated testing environment.

- `docs/AGENT_RAG.md` - Documentation of the CLI-Agent's RAG (Retrieval-Augmented Generation) system architecture, covering ToolVectorDB, PlaybookManager, and UtilsManager integration for contextual guidance and strategic planning.

- `docs/APIS.md` - Comprehensive API documentation for LLM providers (Google Gemini, Groq, Ollama), detailing data structures, parameters, and model discovery endpoints for dynamic model integration.

- `docs/CURRENT_AND_IDEAL_ARCHITECTURE.md` - Architecture analysis documenting duplicate files between py_classes and core directories, with findings on AI provider interfaces and proposed monorepo structure improvements.

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

- `shared/utils_audio.py` - Audio processing utilities for speech recognition, audio file handling, sound playback, and text-to-speech functionality integration.

- `shell_scripts/run_cli_agent.sh` - Shell script for launching the main CLI agent with proper environment setup and configuration.

- `shell_scripts/run_clipboard_to_podcast.sh` - Shell script for converting clipboard content into podcast format using text-to-speech and audio processing.

- `shell_scripts/run_minimal_agent.sh` - Shell script for running a minimal version of the CLI agent with reduced functionality for lightweight usage.

- `shell_scripts/run_screen_agent.sh` - Shell script for launching the agent with screen capture and visual processing capabilities.

- `shell_scripts/setup_bash_alias.sh` - Setup script for creating bash aliases to easily access CLI agent functionality from the command line.

- `shell_scripts/setup_desktop_entry.sh` - Setup script for creating desktop entries and system integration for GUI access to the CLI agent.

- `shell_scripts/setup_dia_environment.sh` - Environment setup script for configuring the development and runtime environment for the CLI agent system.

- `tests/README.md` - Testing documentation explaining the test structure, running procedures, and testing methodology for the CLI agent project.

- `tests/test_cross_tool.py` - Integration tests for cross-tool communication and coordination between different agent components and utilities.

- `tests/test_file_copier.py` - Unit tests for file copying functionality, path resolution, and smart file management features.

- `tests/test_path_resolver.py` - Unit tests for path resolution utilities, testing cross-platform path handling and directory navigation.

- `tests/test_smart_paster.py` - Unit tests for smart paste functionality, content analysis, and intelligent clipboard processing.

- `tests/test_type_hints.py` - Type checking tests ensuring proper type annotations and static type validation across the codebase.

- `tests/core/test_model_discovery.py` - Core tests for LLM model discovery, provider integration, and dynamic model enumeration functionality.

- `tools/file_copier/main.py` - Main executable for the file copying tool with intelligent path resolution, duplicate detection, and smart paste functionality for efficient file management.

- `tools/file_copier/smart_paster.py` - Smart paste implementation for analyzing clipboard content, detecting file types, and providing intelligent suggestions for content placement and processing.

- `tools/file_copier/ai_path_finder.py` - AI-powered path finding utility using language models to intelligently resolve file paths and suggest optimal file locations based on context.

- `tools/file_copier/gui.py` - Graphical user interface for the file copier tool, providing user-friendly interaction for file management operations with visual feedback.

- `tools/todos.py` - Todo management functionality for the CLI agent, providing task tracking, reminder systems, and workflow organization capabilities.

- `utils/architectnewutil.py` - Architecture utility for creating new system components, scaffolding code structures, and managing system architecture evolution.

- `utils/generateimage.py` - Image generation utility integrating with AI image generation services for creating visual content within agent workflows.

- `utils/_goldenstandardutil.py` - Golden standard utility for benchmarking, validation, and quality assurance in agent operations and output quality.

- `utils/homeassistant.py` - Home Assistant integration utility for smart home automation, device control, and IoT interaction through the CLI agent.

- `utils/process_manager.py` - Process management utility for handling system processes, monitoring resource usage, and managing concurrent operations.

- `utils/removefile.py` - File removal utility with safety checks, backup options, and intelligent deletion policies for secure file management.

- `utils/searchweb.py` - Web search utility providing internet search capabilities, result processing, and web information retrieval for agent queries.

- `utils/showuser.py` - User interface utility for displaying information to users, formatting output, and managing interactive display elements.

- `utils/takescreenshot.py` - Screenshot capture utility for screen recording, visual documentation, and image analysis integration within agent workflows.

- `utils/tts.py` - Text-to-speech utility for converting text content to audio, supporting multiple voices and languages for accessibility and audio interfaces.

- `utils/updatefile.py` - File update utility for modifying existing files, applying patches, and managing content updates with version control integration.

- `utils/viewfiles.py` - File viewing utility for displaying file contents, supporting multiple formats, and providing formatted output for various file types.

- `utils/viewimage.py` - Image viewing utility for displaying images, extracting metadata, and providing image analysis capabilities within the agent system.

- `utils/web_fetch.py` - Web content fetching utility for retrieving web pages, handling HTTP requests, and processing web-based information sources.

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