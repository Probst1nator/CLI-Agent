# LLM Router Microservice Extraction Plan

**Document Version:** 1.0
**Date:** 2025-11-11
**Status:** Ready for Implementation
**Estimated Effort:** 5-6 days (1 senior engineer)

---

## Executive Summary

This document outlines a comprehensive plan to extract the AI provider components from the CLI-Agent project into a standalone, reusable Python package called **`llm-router`**. The goal is to create a professional, independently versioned library that can be:

- Installed via `pip install llm-router`
- Reused across multiple projects without copying code
- Maintained and updated centrally with automatic propagation to dependents
- Deployed with proper semantic versioning and CI/CD automation

### Current State
- **Location**: `core/llm_router.py`, `core/providers/`, `core/cache/`, `core/llm.py` (~5,000 LOC)
- **Providers**: 8 implementations (Anthropic, OpenAI, Google, Groq, Ollama, NVIDIA, HuggingFace, Human)
- **Features**: Multi-provider routing, intelligent failover, SQLite caching, model discovery, streaming
- **Dependencies**: Tightly coupled to CLI-Agent's global state, permissions, and utilities

### Target State
- **Package Name**: `llm-router` (PyPI)
- **Repository**: New standalone GitHub repository
- **Architecture**: Python library with clean API and configuration-based initialization
- **Consumption**: `from llm_router import LLMRouter, RouterConfig`
- **Versioning**: Semantic versioning with automated PyPI releases

---

## Table of Contents

1. [Architecture Analysis](#1-architecture-analysis)
2. [Package Design](#2-package-design)
3. [Repository Setup](#3-repository-setup)
4. [Refactoring Strategy](#4-refactoring-strategy)
5. [Migration Plan](#5-migration-plan)
6. [Testing Strategy](#6-testing-strategy)
7. [Release Workflow](#7-release-workflow)
8. [Open Questions & Uncertainties](#8-open-questions--uncertainties)
9. [Success Criteria](#9-success-criteria)
10. [Timeline & Milestones](#10-timeline--milestones)

---

## 1. Architecture Analysis

### 1.1 Current Component Structure

```
CLI-Agent/
├── core/
│   ├── llm_router.py (2,647 lines)           # Main orchestration & routing
│   ├── llm.py (1,001 lines)                  # Model representation & discovery
│   ├── chat.py                               # Chat message handling
│   ├── lm_capabilities.py                    # Capability enums
│   ├── cache/
│   │   └── llm_cache_manager.py (382 lines)  # SQLite caching
│   ├── providers/                            # 8 provider implementations
│   │   ├── cls_anthropic_interface.py
│   │   ├── cls_google_interface.py
│   │   ├── cls_groq_interface.py
│   │   ├── cls_openai_interface.py
│   │   ├── cls_ollama_interface.py (600+ lines)
│   │   ├── cls_nvidia_interface.py
│   │   ├── cls_huggingface_local_interface.py
│   │   └── cls_human_as_interface.py
│   └── permissions/
│       └── settings_manager.py               # Permission system (optional)
└── py_classes/
    └── unified_interfaces.py                 # AIProviderInterface base class
```

### 1.2 Key Dependencies to Decouple

| Dependency | Current Usage | Refactoring Approach | Complexity |
|------------|---------------|----------------------|------------|
| **`core.globals.g`** | Singleton global state for config | Replace with `RouterConfig` instance | **HIGH** |
| **`agent.text_painter.stream_painter`** | Token colorization | Remove or make optional callback | **LOW** |
| **`main.check_interrupt()`** | Ctrl+C handling | Replace with standard signal handlers | **MEDIUM** |
| **`py_classes.unified_interfaces`** | Provider base class | Move to `llm_router.providers.base` | **LOW** |
| **Hardcoded paths** | `~/.cli-agent/*` | Use `config.cache_dir` | **LOW** |
| **Permissions system** | Optional usage | Make completely optional | **LOW** |

### 1.3 Critical Features to Preserve

✅ **Multi-Provider Support**: All 8 providers with automatic failover
✅ **Streaming**: Unified handling of 6+ different stream formats
✅ **Caching**: SQLite-based response caching with migration support
✅ **Model Discovery**: Parallel discovery from multiple providers
✅ **Role-Based Selection**: Different models for different task types
✅ **Error Handling**: Provider-specific error recovery
✅ **Rate Limiting**: Integration points (currently optional)
✅ **Generation Tracking**: Real-time metrics and telemetry

---

## 2. Package Design

### 2.1 Proposed Directory Structure

```
llm-router/                               # New repository
├── README.md                             # Comprehensive usage guide
├── CHANGELOG.md                          # Version history
├── LICENSE                               # MIT or Apache 2.0
├── pyproject.toml                        # Modern Python packaging
├── setup.py                              # Backward compatibility
├── requirements.txt                      # Production dependencies
├── requirements-dev.txt                  # Development dependencies
├── .env.example                          # Configuration template
├── .gitignore
├── .github/
│   └── workflows/
│       ├── test.yml                      # CI tests on PRs
│       ├── release.yml                   # Automated PyPI publishing
│       └── lint.yml                      # Code quality checks
│
├── src/
│   └── llm_router/
│       ├── __init__.py                   # Public API exports
│       ├── version.py                    # __version__ = "0.1.0"
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── router.py                 # LLMRouter (refactored from llm_router.py)
│       │   ├── model.py                  # LLM class (from llm.py)
│       │   ├── chat.py                   # Chat & Message classes
│       │   ├── capabilities.py           # LMCapabilities enum
│       │   └── config.py                 # RouterConfig dataclass
│       │
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py                   # BaseProvider interface
│       │   ├── anthropic.py              # Claude API
│       │   ├── google.py                 # Gemini API
│       │   ├── groq.py                   # Groq API
│       │   ├── openai.py                 # GPT API
│       │   ├── nvidia.py                 # NVIDIA Cloud
│       │   ├── ollama.py                 # Ollama local
│       │   ├── huggingface.py            # HF local inference
│       │   └── human.py                  # Testing interface
│       │
│       ├── cache/
│       │   ├── __init__.py
│       │   ├── manager.py                # CacheManager (from llm_cache_manager.py)
│       │   └── backends.py               # Future: Redis, Memcached
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logger.py                 # Standalone logging
│           ├── stream_processor.py       # Stream format normalization
│           └── errors.py                 # Custom exceptions
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                       # Pytest fixtures
│   ├── unit/
│   │   ├── test_router.py
│   │   ├── test_config.py
│   │   ├── test_cache.py
│   │   └── providers/
│   │       ├── test_anthropic.py
│   │       ├── test_openai.py
│   │       └── ...
│   ├── integration/
│   │   ├── test_multi_provider.py
│   │   ├── test_failover.py
│   │   └── test_streaming.py
│   └── fixtures/
│       ├── mock_responses.json
│       └── test_models.json
│
├── examples/
│   ├── 01_simple_completion.py
│   ├── 02_chat_history.py
│   ├── 03_streaming.py
│   ├── 04_model_discovery.py
│   ├── 05_vision_models.py
│   ├── 06_local_inference.py
│   └── 07_advanced_config.py
│
└── docs/
    ├── index.md
    ├── quickstart.md
    ├── api_reference.md
    ├── configuration.md
    ├── providers.md
    ├── caching.md
    └── migration_from_cli_agent.md
```

### 2.2 Public API Design

#### Core Imports

```python
# Top-level imports (src/llm_router/__init__.py)
from llm_router.core.router import LLMRouter, LLMRole
from llm_router.core.model import LLM, LLMCapabilities
from llm_router.core.chat import Chat, Message, Role
from llm_router.core.config import RouterConfig
from llm_router.version import __version__

__all__ = [
    "LLMRouter",
    "LLMRole",
    "LLM",
    "LLMCapabilities",
    "Chat",
    "Message",
    "Role",
    "RouterConfig",
    "__version__",
]
```

#### Configuration Class

```python
# src/llm_router/core/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

@dataclass
class RouterConfig:
    """Central configuration for LLM Router"""

    # ============================================================
    # API Keys
    # ============================================================
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None

    # ============================================================
    # Local Models
    # ============================================================
    ollama_hosts: list[str] = field(default_factory=lambda: ["localhost:11434"])
    huggingface_models_dir: Optional[Path] = None
    huggingface_device: str = "auto"  # auto, cuda, cpu, mps
    huggingface_max_memory_gb: int = 16

    # ============================================================
    # Caching
    # ============================================================
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "llm-router")
    enable_cache: bool = True
    cache_max_age_days: int = 30

    # ============================================================
    # Behavior
    # ============================================================
    force_local: bool = False
    force_model: Optional[str] = None  # Force specific model
    default_temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 3
    verbose: bool = False

    # ============================================================
    # Model Selection
    # ============================================================
    role_preferences: dict[str, list[str]] = field(default_factory=dict)
    fallback_chain: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, dotenv_path: Optional[Path] = None) -> "RouterConfig":
        """Load configuration from environment variables

        Args:
            dotenv_path: Path to .env file (defaults to .env in current directory)

        Returns:
            RouterConfig instance with values from environment

        Example:
            >>> config = RouterConfig.from_env()
            >>> config = RouterConfig.from_env(Path("/path/to/.env"))
        """
        if dotenv_path:
            load_dotenv(dotenv_path)
        else:
            load_dotenv()  # Load from .env in current directory

        return cls(
            # API Keys
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY"),

            # Local Models
            ollama_hosts=os.getenv("OLLAMA_HOSTS", "localhost:11434").split(","),
            huggingface_models_dir=Path(
                os.getenv("HF_MODELS_DIR", "~/.cache/huggingface")
            ).expanduser(),
            huggingface_device=os.getenv("HF_DEVICE", "auto"),
            huggingface_max_memory_gb=int(os.getenv("HF_MAX_MEMORY_GB", "16")),

            # Caching
            cache_dir=Path(
                os.getenv("LLM_CACHE_DIR", "~/.cache/llm-router")
            ).expanduser(),
            enable_cache=os.getenv("LLM_ENABLE_CACHE", "true").lower() == "true",
            cache_max_age_days=int(os.getenv("LLM_CACHE_MAX_AGE_DAYS", "30")),

            # Behavior
            force_local=os.getenv("FORCE_LOCAL", "false").lower() == "true",
            force_model=os.getenv("LLM_FORCE_MODEL"),
            default_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.0")),
            timeout=float(os.getenv("LLM_TIMEOUT", "60.0")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            verbose=os.getenv("LLM_VERBOSE", "false").lower() == "true",
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings

        Returns:
            List of warning messages (empty if all valid)
        """
        warnings = []

        # Check if at least one provider is configured
        has_cloud = any([
            self.anthropic_api_key,
            self.openai_api_key,
            self.google_api_key,
            self.groq_api_key,
            self.nvidia_api_key,
        ])
        has_local = len(self.ollama_hosts) > 0 or self.huggingface_models_dir

        if not has_cloud and not has_local:
            warnings.append(
                "No providers configured. Set at least one API key or local model source."
            )

        if self.force_local and not has_local:
            warnings.append(
                "FORCE_LOCAL=true but no local providers configured (Ollama or HuggingFace)."
            )

        # Validate paths
        if not self.cache_dir.exists():
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                warnings.append(f"Cannot create cache directory {self.cache_dir}: {e}")

        return warnings
```

#### Main Router Class

```python
# src/llm_router/core/router.py
from typing import Optional, Union, AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum

from llm_router.core.config import RouterConfig
from llm_router.core.chat import Chat, Message, Role
from llm_router.core.model import LLM, LLMCapabilities
from llm_router.cache.manager import CacheManager
from llm_router.utils.logger import get_logger

class LLMRole(Enum):
    """Role-based model selection for different task types"""
    PLANNER = "planner"        # Complex planning tasks
    ROUTER = "router"          # Intent classification, routing decisions
    OVERRIDE = "override"      # User-forced model selection
    AUTOCOMPLETE = "autocomplete"  # Code/text completion
    FALLBACK = "fallbacks"     # Default fallback chain

@dataclass
class GenerationResponse:
    """Standardized response from any provider"""
    text: str
    model_used: str
    provider: str
    tokens: int
    cached: bool
    generation_time: float
    metadata: dict = field(default_factory=dict)

class LLMRouter:
    """Intelligent LLM routing with automatic failover and caching

    Examples:
        >>> # Simple initialization
        >>> config = RouterConfig.from_env()
        >>> router = LLMRouter(config)

        >>> # Generate completion
        >>> response = await router.generate("What is 2+2?")
        >>> print(response.text)

        >>> # With chat history
        >>> chat = Chat()
        >>> chat.add_message(Role.USER, "Hello!")
        >>> response = await router.generate(chat, temperature=0.7)

        >>> # Streaming
        >>> async for token in router.generate_stream("Write a poem"):
        >>>     print(token, end="", flush=True)
    """

    def __init__(self, config: RouterConfig):
        """Initialize router with configuration

        Args:
            config: RouterConfig instance with API keys and settings

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.logger = get_logger("router", config.verbose)

        # Validate configuration
        warnings = config.validate()
        for warning in warnings:
            self.logger.warning(warning)

        # Initialize cache
        self._cache = CacheManager(config) if config.enable_cache else None

        # Initialize providers (lazy loading)
        self._providers = {}
        self._models: list[LLM] = []

        self.logger.info(f"LLM Router initialized (v{__version__})")

    async def generate(
        self,
        messages: Union[str, Chat],
        preferred_models: Optional[list[str]] = None,
        role: LLMRole = LLMRole.FALLBACK,
        capabilities: Optional[list[LLMCapabilities]] = None,
        temperature: Optional[float] = None,
        images: Optional[list[str]] = None,
        thinking_budget: Optional[int] = 4096,
        exclude_reasoning_tokens: bool = True,
        force_local: Optional[bool] = None,
        max_retries: Optional[int] = None,
    ) -> GenerationResponse:
        """Generate a completion with automatic provider failover

        Args:
            messages: Single message string or Chat history
            preferred_models: List of model keys to try in order (e.g., ["claude-sonnet-4", "gpt-4"])
            role: Task role for model selection (PLANNER, ROUTER, etc.)
            capabilities: Required model capabilities (VISION, UNCENSORED, etc.)
            temperature: Sampling temperature (0-1, default: config.default_temperature)
            images: List of base64-encoded images (for vision models)
            thinking_budget: Max tokens for reasoning (for models with extended thinking)
            exclude_reasoning_tokens: Whether to exclude reasoning from response
            force_local: Override config.force_local for this request
            max_retries: Override config.max_retries for this request

        Returns:
            GenerationResponse with text, metadata, and telemetry

        Raises:
            ValueError: If no suitable models found
            TimeoutError: If all models timeout
            Exception: If all models fail after retries
        """
        # Implementation here (extracted from llm_router.py)
        pass

    async def generate_stream(
        self,
        messages: Union[str, Chat],
        **kwargs  # Same params as generate()
    ) -> AsyncIterator[str]:
        """Stream tokens as they're generated

        Args:
            messages: Single message or Chat history
            **kwargs: Same parameters as generate()

        Yields:
            Individual tokens as strings

        Example:
            >>> async for token in router.generate_stream("Count to 10"):
            >>>     print(token, end="", flush=True)
        """
        # Implementation here
        pass

    async def discover_models(
        self,
        capabilities: Optional[list[LLMCapabilities]] = None,
        force_local: bool = False,
        refresh: bool = False,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> list[LLM]:
        """Discover available models across all configured providers

        Args:
            capabilities: Filter by required capabilities
            force_local: Only discover local models
            refresh: Force re-discovery (ignore cache)
            progress_callback: Called with (status_message, progress_0_to_1)

        Returns:
            List of available LLM models

        Example:
            >>> def progress(msg, pct):
            >>>     print(f"{msg}: {pct*100:.0f}%")
            >>> models = await router.discover_models(progress_callback=progress)
        """
        # Implementation here (extracted from llm.py)
        pass

    def get_model_info(self, model_key: str) -> Optional[LLM]:
        """Get metadata for a specific model

        Args:
            model_key: Model identifier (e.g., "claude-sonnet-4")

        Returns:
            LLM instance with metadata, or None if not found
        """
        pass

    def clear_cache(self) -> int:
        """Clear response cache

        Returns:
            Number of entries removed
        """
        if self._cache:
            return self._cache.clear()
        return 0

    def get_cache_stats(self) -> dict:
        """Get cache statistics

        Returns:
            Dict with keys: total_entries, unique_models, db_size_mb, hit_rate, etc.
        """
        if self._cache:
            return self._cache.get_statistics()
        return {}
```

### 2.3 Usage Examples

#### Example 1: Simple Completion

```python
# examples/01_simple_completion.py
import asyncio
from llm_router import LLMRouter, RouterConfig

async def main():
    # Initialize from environment (.env file)
    config = RouterConfig.from_env()
    router = LLMRouter(config)

    # Generate completion
    response = await router.generate(
        messages="What is the capital of France?",
        temperature=0
    )

    print(f"Answer: {response.text}")
    print(f"Model used: {response.model_used}")
    print(f"Tokens: {response.tokens}")
    print(f"Cached: {response.cached}")

if __name__ == "__main__":
    asyncio.run(main())
```

#### Example 2: Chat History

```python
# examples/02_chat_history.py
import asyncio
from llm_router import LLMRouter, RouterConfig, Chat, Role

async def main():
    config = RouterConfig.from_env()
    router = LLMRouter(config)

    # Create chat with history
    chat = Chat()
    chat.add_message(Role.SYSTEM, "You are a helpful coding assistant.")
    chat.add_message(Role.USER, "How do I reverse a list in Python?")

    response = await router.generate(
        messages=chat,
        preferred_models=["claude-sonnet-4", "gpt-4"],
        temperature=0.7
    )

    # Add response to history
    chat.add_message(Role.ASSISTANT, response.text)

    # Continue conversation
    chat.add_message(Role.USER, "Now show me how to do it in JavaScript")
    response = await router.generate(chat)

    print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

#### Example 3: Streaming

```python
# examples/03_streaming.py
import asyncio
from llm_router import LLMRouter, RouterConfig

async def main():
    config = RouterConfig.from_env()
    router = LLMRouter(config)

    print("Generating story...\n")

    async for token in router.generate_stream(
        messages="Write a short story about a robot learning to feel emotions.",
        temperature=0.9,
        preferred_models=["claude-sonnet-4"]
    ):
        print(token, end="", flush=True)

    print("\n\nDone!")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 3. Repository Setup

### 3.1 Initial Repository Creation

```bash
# Create new repository
gh repo create llm-router --public \
    --description "Intelligent multi-provider LLM routing library with failover and caching" \
    --clone

cd llm-router

# Initialize git flow structure
git checkout -b develop

# Create directory structure
mkdir -p src/llm_router/{core,providers,cache,utils}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p examples
mkdir -p docs
mkdir -p .github/workflows

# Create __init__.py files
touch src/llm_router/__init__.py
touch src/llm_router/{core,providers,cache,utils}/__init__.py
touch tests/__init__.py
touch tests/{unit,integration}/__init__.py
```

### 3.2 Essential Configuration Files

#### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-router"
dynamic = ["version"]
description = "Intelligent multi-provider LLM routing with failover and caching"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "you@example.com"}
]
keywords = ["llm", "ai", "claude", "gpt", "gemini", "ollama", "router", "anthropic", "openai"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "anthropic>=0.25.0",
    "openai>=1.0.0",
    "google-generativeai>=0.3.0",
    "groq>=0.4.0",
    "ollama>=0.1.0",
    "python-dotenv>=1.0.0",
    "aiohttp>=3.8.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
local = [
    "transformers>=4.48.0",
    "torch>=2.0.0",
    "accelerate>=0.20.0",
]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "pre-commit>=3.0.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/llm-router"
Documentation = "https://llm-router.readthedocs.io"
Repository = "https://github.com/yourusername/llm-router"
Issues = "https://github.com/yourusername/llm-router/issues"
Changelog = "https://github.com/yourusername/llm-router/blob/main/CHANGELOG.md"

[tool.setuptools.dynamic]
version = {attr = "llm_router.version.__version__"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
llm_router = ["py.typed"]

[tool.black]
line-length = 100
target-version = ["py39", "py310", "py311", "py312"]
include = '\.pyi?$'

[tool.ruff]
line-length = 100
target-version = "py39"
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by black)
]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-v --tb=short --strict-markers"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

#### .env.example

```bash
# ============================================================
# LLM Router Configuration
# Copy this file to .env and fill in your API keys
# ============================================================

# ============================================================
# CLOUD PROVIDERS
# Get your API keys from:
# - Anthropic: https://console.anthropic.com/
# - OpenAI: https://platform.openai.com/api-keys
# - Google: https://ai.google.dev/
# - Groq: https://console.groq.com/
# - NVIDIA: https://build.nvidia.com/
# ============================================================

ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
NVIDIA_API_KEY=

# ============================================================
# LOCAL MODELS
# ============================================================

# Comma-separated list of Ollama hosts
OLLAMA_HOSTS=localhost:11434

# HuggingFace local inference (requires transformers, torch)
HF_MODELS_DIR=~/.cache/huggingface/models
HF_DEVICE=auto  # Options: auto, cuda, cpu, mps
HF_MAX_MEMORY_GB=16

# ============================================================
# CACHING
# ============================================================

LLM_CACHE_DIR=~/.cache/llm-router
LLM_ENABLE_CACHE=true
LLM_CACHE_MAX_AGE_DAYS=30

# ============================================================
# BEHAVIOR
# ============================================================

# Only use local models (Ollama, HuggingFace)
FORCE_LOCAL=false

# Force a specific model (overrides all selection logic)
# LLM_FORCE_MODEL=claude-sonnet-4

# Default temperature for all requests (0.0-1.0)
LLM_DEFAULT_TEMPERATURE=0.0

# Request timeout in seconds
LLM_TIMEOUT=60

# Maximum retry attempts per model
LLM_MAX_RETRIES=3

# Enable verbose debug logging
LLM_VERBOSE=false
```

#### .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Testing
.pytest_cache/
.coverage
.coverage.*
htmlcov/
.tox/
.nox/

# Type checking
.mypy_cache/
.dmypy.json
dmypy.json
.pytype/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Distribution
*.tar.gz
*.whl

# Ruff
.ruff_cache/
```

### 3.3 GitHub Actions Workflows

#### Test Workflow (`.github/workflows/test.yml`)

```yaml
name: Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run linters
        run: |
          black --check .
          ruff check .
          mypy src/

      - name: Run tests
        run: |
          pytest tests/ -v --cov=llm_router --cov-report=xml --cov-report=html

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
```

#### Release Workflow (`.github/workflows/release.yml`)

```yaml
name: Release to PyPI

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write

jobs:
  release:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
          pip install -e ".[dev]"

      - name: Run tests
        run: pytest tests/ -v

      - name: Build package
        run: python -m build

      - name: Check package
        run: twine check dist/*

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*
          generate_release_notes: true
          draft: false
          prerelease: false
```

### 3.4 Branch Protection Rules

Configure on GitHub (`Settings > Branches > Add rule`):

**For `main` branch:**
- ✅ Require pull request reviews before merging (1 approval)
- ✅ Require status checks to pass before merging
  - Required checks: `test (3.9)`, `test (3.10)`, `test (3.11)`, `test (3.12)`
- ✅ Require branches to be up to date before merging
- ✅ Do not allow bypassing the above settings
- ❌ Allow force pushes
- ❌ Allow deletions

---

## 4. Refactoring Strategy

### 4.1 Critical Code Changes

#### Change 1: Replace Global State with Config

**Before (CLI-Agent):**
```python
# core/llm_router.py
from core.globals import g

class LlmRouter:
    @classmethod
    async def generate_completion(cls, chat, ...):
        force_local = g.FORCE_LOCAL
        if g.LLM:
            preferred_models = [g.LLM]

        cache_dir = Path.home() / ".cli-agent"
```

**After (llm-router):**
```python
# src/llm_router/core/router.py
from llm_router.core.config import RouterConfig

class LLMRouter:
    def __init__(self, config: RouterConfig):
        self.config = config

    async def generate(self, messages, ...):
        force_local = self.config.force_local
        if self.config.force_model:
            preferred_models = [self.config.force_model]

        cache_dir = self.config.cache_dir
```

#### Change 2: Remove TextStreamPainter Dependency

**Before:**
```python
from agent.text_painter.stream_painter import TextStreamPainter

painter = TextStreamPainter()
for token in stream:
    painter.paint_streamed_token(token, model_key)
```

**After (Option 1 - Remove completely):**
```python
# Simply yield tokens, let user handle formatting
for token in stream:
    yield token
```

**After (Option 2 - Optional callback):**
```python
class LLMRouter:
    async def generate_stream(
        self,
        messages,
        token_callback: Optional[Callable[[str, str], str]] = None,
        **kwargs
    ):
        for token in stream:
            if token_callback:
                token = token_callback(token, model_key)
            yield token

# User can provide custom formatter
def my_formatter(token: str, model: str) -> str:
    return f"\033[92m{token}\033[0m"  # Green color

async for token in router.generate_stream(
    "Hello",
    token_callback=my_formatter
):
    print(token, end="")
```

#### Change 3: Standardize Provider Interface

**Before:**
```python
# py_classes/unified_interfaces.py
class AIProviderInterface(BaseProviderInterface):
    @abstractmethod
    def generate_response(self, chat, model_key, temperature, silent_reason, thinking_budget):
        pass
```

**After:**
```python
# src/llm_router/providers/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator, Union, Optional

class BaseProvider(ABC):
    """Abstract base class for all LLM providers"""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0):
        self.api_key = api_key
        self.timeout = timeout

    @abstractmethod
    async def generate(
        self,
        messages: Union[Chat, str],
        model_key: str,
        temperature: float,
        thinking_budget: Optional[int] = None,
        images: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        """Generate tokens asynchronously

        Args:
            messages: Chat history or single message
            model_key: Model identifier
            temperature: Sampling temperature (0-1)
            thinking_budget: Max reasoning tokens (for models with extended thinking)
            images: Base64-encoded images (for vision models)

        Yields:
            Individual tokens as strings
        """
        pass

    @abstractmethod
    async def get_available_models(self) -> list[dict]:
        """Return list of models this provider supports

        Returns:
            List of dicts with keys: model_key, context_window, capabilities
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(configured={self.api_key is not None})"
```

#### Change 4: Replace Interrupt Handling

**Before:**
```python
from main import check_interrupt

for token in stream:
    check_interrupt()  # Raises UserInterruptedException if Ctrl+C
    yield token
```

**After:**
```python
import signal
import asyncio

class InterruptHandler:
    """Handle graceful interruption of generation"""

    def __init__(self):
        self._interrupted = False
        self._original_handler = None

    def __enter__(self):
        self._original_handler = signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, *args):
        signal.signal(signal.SIGINT, self._original_handler)

    def _handler(self, signum, frame):
        self._interrupted = True

    @property
    def interrupted(self) -> bool:
        return self._interrupted

# Usage in router
async def generate_stream(self, ...):
    with InterruptHandler() as handler:
        for token in stream:
            if handler.interrupted:
                raise KeyboardInterrupt("Generation interrupted by user")
            yield token
```

### 4.2 File-by-File Migration Map

| CLI-Agent Source | llm-router Destination | Changes Required |
|------------------|------------------------|------------------|
| `core/llm_router.py` | `src/llm_router/core/router.py` | Replace globals, remove TextStreamPainter, add config |
| `core/llm.py` | `src/llm_router/core/model.py` | Remove global imports, update paths |
| `core/chat.py` | `src/llm_router/core/chat.py` | Extract standalone, add serialization |
| `core/lm_capabilities.py` | `src/llm_router/core/capabilities.py` | No changes (already standalone) |
| `core/cache/llm_cache_manager.py` | `src/llm_router/cache/manager.py` | Update paths from config |
| `core/providers/cls_anthropic_interface.py` | `src/llm_router/providers/anthropic.py` | Implement BaseProvider, accept API key in init |
| `core/providers/cls_google_interface.py` | `src/llm_router/providers/google.py` | Same as above |
| `core/providers/cls_groq_interface.py` | `src/llm_router/providers/groq.py` | Same as above |
| `core/providers/cls_openai_interface.py` | `src/llm_router/providers/openai.py` | Same as above |
| `core/providers/cls_ollama_interface.py` | `src/llm_router/providers/ollama.py` | Same as above, accept hosts list |
| `core/providers/cls_nvidia_interface.py` | `src/llm_router/providers/nvidia.py` | Same as above |
| `core/providers/cls_huggingface_local_interface.py` | `src/llm_router/providers/huggingface.py` | Same as above, accept models_dir |
| `core/providers/cls_human_as_interface.py` | `src/llm_router/providers/human.py` | Same as above |
| `py_classes/unified_interfaces.py` | `src/llm_router/providers/base.py` | Extract only AIProviderInterface |

### 4.3 Dependency Resolution

#### Internal Dependencies to Extract

```python
# These need to be copied/adapted from CLI-Agent
- Chat class (core/chat.py)
- Role enum (core/chat.py)
- LMCapabilities enum (core/lm_capabilities.py)
- Basic utility functions (path normalization, date parsing, etc.)
```

#### Internal Dependencies to Remove/Replace

```python
# Remove these CLI-Agent specific dependencies
- core.globals.g → RouterConfig instance
- agent.text_painter.stream_painter → Optional callback or remove
- main.check_interrupt → signal.SIGINT handler
- core.permissions.settings_manager → Optional feature
```

#### External Dependencies to Add

All current external dependencies can be retained:
```
anthropic>=0.25.0
openai>=1.0.0
google-generativeai>=0.3.0
groq>=0.4.0
ollama>=0.1.0
python-dotenv>=1.0.0
aiohttp>=3.8.0
pydantic>=2.0.0
```

---

## 5. Migration Plan

### 5.1 Phase 1: Repository Setup (Day 1)

**Tasks:**
1. Create GitHub repository `llm-router`
2. Set up directory structure
3. Create configuration files (pyproject.toml, .gitignore, .env.example)
4. Set up GitHub Actions workflows
5. Configure branch protection rules
6. Create initial documentation structure

**Deliverables:**
- Empty repository with proper structure
- CI/CD pipeline configured
- README.md with placeholder content

**Commands:**
```bash
# Create repo
gh repo create llm-router --public --clone
cd llm-router

# Create structure
mkdir -p src/llm_router/{core,providers,cache,utils}
mkdir -p tests/{unit,integration,fixtures}
mkdir -p examples docs .github/workflows

# Copy config files (from this plan)
# ... create pyproject.toml, .gitignore, etc. ...

# Initial commit
git add .
git commit -m "chore: initial repository setup"
git push origin main

# Create develop branch
git checkout -b develop
git push origin develop

# Set develop as default branch (on GitHub)
```

### 5.2 Phase 2: Core Extraction (Days 2-3)

**Tasks:**
1. Extract and refactor `core/chat.py` → `src/llm_router/core/chat.py`
2. Extract `core/lm_capabilities.py` → `src/llm_router/core/capabilities.py`
3. Create `src/llm_router/core/config.py` from scratch
4. Extract `core/cache/llm_cache_manager.py` → `src/llm_router/cache/manager.py`
5. Create `src/llm_router/utils/logger.py` for standalone logging
6. Write unit tests for each component

**Deliverables:**
- Functional Chat, Config, Cache, Capabilities modules
- Unit tests with >80% coverage
- Updated imports to use llm_router namespace

**Testing:**
```bash
# Run tests after each component
pytest tests/unit/test_chat.py -v
pytest tests/unit/test_config.py -v
pytest tests/unit/test_cache.py -v
```

### 5.3 Phase 3: Provider Migration (Days 3-4)

**Tasks:**
1. Create `src/llm_router/providers/base.py` with BaseProvider interface
2. Migrate each provider (8 total):
   - `cls_anthropic_interface.py` → `anthropic.py`
   - `cls_openai_interface.py` → `openai.py`
   - `cls_google_interface.py` → `google.py`
   - `cls_groq_interface.py` → `groq.py`
   - `cls_ollama_interface.py` → `ollama.py`
   - `cls_nvidia_interface.py` → `nvidia.py`
   - `cls_huggingface_local_interface.py` → `huggingface.py`
   - `cls_human_as_interface.py` → `human.py`
3. Update each to accept config in `__init__` (no global state)
4. Write unit tests with mocked API responses

**Deliverables:**
- All 8 providers implementing BaseProvider
- Unit tests for each provider
- Mock fixtures for testing

**Testing:**
```bash
# Test each provider
pytest tests/unit/providers/ -v
```

### 5.4 Phase 4: Router Migration (Day 4-5)

**Tasks:**
1. Extract `core/llm.py` → `src/llm_router/core/model.py`
2. Extract `core/llm_router.py` → `src/llm_router/core/router.py`
3. Refactor to use RouterConfig instead of globals
4. Remove TextStreamPainter dependency
5. Implement interrupt handling
6. Update model discovery logic
7. Write integration tests

**Deliverables:**
- Fully functional LLMRouter class
- Model discovery working
- Streaming support
- Integration tests

**Testing:**
```bash
# Integration tests (requires API keys)
ANTHROPIC_API_KEY=sk-... pytest tests/integration/ -v
```

### 5.5 Phase 5: Documentation & Examples (Day 5-6)

**Tasks:**
1. Write comprehensive README.md
2. Create 7 usage examples (examples/*.py)
3. Write API reference documentation
4. Create migration guide for CLI-Agent
5. Add docstrings to all public methods
6. Set up Sphinx/MkDocs documentation

**Deliverables:**
- Complete README with quickstart
- Working examples
- API documentation
- Migration guide

### 5.6 Phase 6: Release & CLI-Agent Integration (Day 6)

**Tasks:**
1. Final testing across all Python versions
2. Create v0.1.0 tag
3. Publish to PyPI
4. Update CLI-Agent to use llm-router package
5. Remove old code from CLI-Agent
6. Update CLI-Agent requirements.txt

**Deliverables:**
- Published package on PyPI
- CLI-Agent using llm-router
- No duplicate code

**Commands:**
```bash
# In llm-router repo
git checkout main
git tag v0.1.0
git push --tags  # Triggers GitHub Action → PyPI

# In CLI-Agent repo
pip uninstall -y llm-router
pip install llm-router==0.1.0

# Update imports
# ... (see section 5.7) ...

# Remove old code
rm -rf core/llm_router.py core/llm.py core/providers/ core/cache/

# Test CLI-Agent
python main.py -m "test message"
python run_tests.py --full
```

### 5.7 CLI-Agent Code Updates

**Before:**
```python
# CLI-Agent imports (OLD)
from core.llm_router import LlmRouter
from core.chat import Chat, Role
from core.llm import Llm
from core.lm_capabilities import LMCapabilities
```

**After:**
```python
# CLI-Agent imports (NEW)
from llm_router import LLMRouter, RouterConfig, Chat, Role, LLM, LLMCapabilities

# Initialize once (in main.py or app initialization)
llm_config = RouterConfig.from_env()
llm_router = LLMRouter(llm_config)

# Pass router instance to commands/tools that need it
# (via AppContext or dependency injection)
```

**Example Usage Update:**

```python
# OLD CLI-Agent code
response = await LlmRouter.generate_completion(
    chat=chat,
    preferred_models=["claude-sonnet-4"],
    temperature=0.7,
    role=LLMRole.PLANNER
)

# NEW with llm-router package
response = await llm_router.generate(
    messages=chat,
    preferred_models=["claude-sonnet-4"],
    temperature=0.7,
    role=LLMRole.PLANNER
)

# Response object is different
# OLD: response is a string
# NEW: response is GenerationResponse object
text = response.text
model_used = response.model_used
```

---

## 6. Testing Strategy

### 6.1 Test Structure

```
tests/
├── conftest.py                    # Pytest fixtures
├── unit/                          # Fast, isolated tests
│   ├── test_config.py            # Config loading, validation
│   ├── test_chat.py              # Chat/Message classes
│   ├── test_cache.py             # Cache manager
│   ├── test_capabilities.py      # Capability enum
│   └── providers/
│       ├── test_anthropic.py     # Mocked API tests
│       ├── test_openai.py
│       ├── test_google.py
│       └── ...
├── integration/                   # Requires API keys
│   ├── test_router_real.py       # Real API calls
│   ├── test_failover.py          # Multi-provider failover
│   ├── test_streaming.py         # Streaming responses
│   └── test_caching.py           # End-to-end caching
└── fixtures/
    ├── mock_responses.json        # Sample API responses
    └── test_models.json           # Test model metadata
```

### 6.2 Unit Test Examples

```python
# tests/unit/test_config.py
import os
import pytest
from pathlib import Path
from llm_router import RouterConfig

def test_config_from_env(monkeypatch):
    """Test loading config from environment variables"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-456")
    monkeypatch.setenv("FORCE_LOCAL", "true")

    config = RouterConfig.from_env()

    assert config.anthropic_api_key == "sk-test-123"
    assert config.openai_api_key == "sk-test-456"
    assert config.force_local is True

def test_config_validation_warns_no_providers():
    """Test that validation warns when no providers configured"""
    config = RouterConfig()  # No API keys
    warnings = config.validate()

    assert len(warnings) > 0
    assert "No providers configured" in warnings[0]

def test_config_creates_cache_dir(tmp_path):
    """Test that cache directory is created if missing"""
    cache_dir = tmp_path / "test_cache"
    config = RouterConfig(cache_dir=cache_dir)

    warnings = config.validate()

    assert cache_dir.exists()
    assert len(warnings) == 0  # Should not warn about created dir

# tests/unit/test_chat.py
from llm_router import Chat, Message, Role

def test_chat_add_message():
    """Test adding messages to chat"""
    chat = Chat()
    chat.add_message(Role.USER, "Hello")
    chat.add_message(Role.ASSISTANT, "Hi there!")

    assert len(chat.messages) == 2
    assert chat.messages[0].role == Role.USER
    assert chat.messages[1].content == "Hi there!"

def test_chat_serialization():
    """Test chat to/from dict"""
    chat = Chat()
    chat.add_message(Role.SYSTEM, "You are helpful")
    chat.add_message(Role.USER, "Help me", images=["base64img"])

    data = chat.to_dict()
    restored = Chat.from_dict(data)

    assert len(restored.messages) == 2
    assert restored.messages[1].images == ["base64img"]

# tests/unit/providers/test_anthropic.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from llm_router.providers.anthropic import AnthropicProvider
from llm_router import Chat, Role

@pytest.fixture
def mock_anthropic_client(monkeypatch):
    """Mock Anthropic API client"""
    mock_client = MagicMock()
    mock_stream = MagicMock()
    mock_stream.text_stream = iter(["Hello", " ", "world"])
    mock_client.messages.stream.return_value.__enter__.return_value = mock_stream

    monkeypatch.setattr("anthropic.Anthropic", lambda *args, **kwargs: mock_client)
    return mock_client

@pytest.mark.asyncio
async def test_anthropic_generate(mock_anthropic_client):
    """Test Anthropic provider generation"""
    provider = AnthropicProvider(api_key="sk-test")

    tokens = []
    async for token in provider.generate(
        messages="Hello",
        model_key="claude-sonnet-4",
        temperature=0.7
    ):
        tokens.append(token)

    assert "".join(tokens) == "Hello world"
    mock_anthropic_client.messages.stream.assert_called_once()
```

### 6.3 Integration Test Examples

```python
# tests/integration/test_router_real.py
import pytest
import os
from llm_router import LLMRouter, RouterConfig

@pytest.fixture
def router():
    """Create router with real config (requires API keys)"""
    config = RouterConfig.from_env()
    return LLMRouter(config)

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires ANTHROPIC_API_KEY"
)
@pytest.mark.asyncio
async def test_simple_completion(router):
    """Test real completion with Anthropic"""
    response = await router.generate(
        messages="What is 2+2? Answer with just the number.",
        preferred_models=["claude-sonnet-4"],
        temperature=0
    )

    assert "4" in response.text
    assert response.model_used == "claude-sonnet-4"
    assert response.tokens > 0

@pytest.mark.integration
@pytest.mark.asyncio
async def test_failover(router):
    """Test failover to next provider on error"""
    # Force first model to fail, second to succeed
    response = await router.generate(
        messages="Hello",
        preferred_models=["invalid-model", "claude-sonnet-4"],
        temperature=0
    )

    assert response.model_used == "claude-sonnet-4"
    assert len(response.text) > 0
```

### 6.4 Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Core Modules**: 90% (router, config, cache)
- **Providers**: 70% (harder to test without real APIs)
- **Integration**: Focus on critical paths, not coverage %

**Check coverage:**
```bash
pytest tests/ --cov=llm_router --cov-report=html --cov-report=term
open htmlcov/index.html
```

---

## 7. Release Workflow

### 7.1 Semantic Versioning

Follow [SemVer 2.0.0](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes (API incompatibility)
- **MINOR** (0.X.0): New features (backward compatible)
- **PATCH** (0.0.X): Bug fixes (backward compatible)

**Examples:**
- `0.1.0` → `0.2.0`: Add Azure OpenAI provider (new feature)
- `0.2.0` → `1.0.0`: Change `generate()` signature (breaking)
- `1.0.0` → `1.0.1`: Fix caching bug (patch)

### 7.2 Version Bumping

Use [`bump2version`](https://github.com/c4urself/bump2version):

```bash
# Install
pip install bump2version

# Create .bumpversion.cfg
cat > .bumpversion.cfg << EOF
[bumpversion]
current_version = 0.1.0
commit = True
tag = True
tag_name = v{new_version}

[bumpversion:file:src/llm_router/version.py]
search = __version__ = "{current_version}"
replace = __version__ = "{new_version}"

[bumpversion:file:pyproject.toml]
search = version = "{current_version}"
replace = version = "{new_version}"
EOF

# Bump version
bump2version patch  # 0.1.0 → 0.1.1
bump2version minor  # 0.1.1 → 0.2.0
bump2version major  # 0.2.0 → 1.0.0
```

### 7.3 Release Checklist

**Before Release:**
- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Linters passing (`black .`, `ruff check .`, `mypy src/`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped (`bump2version minor`)
- [ ] Examples tested manually

**Release Process:**
```bash
# 1. Ensure you're on develop with latest changes
git checkout develop
git pull

# 2. Create release branch
git checkout -b release/v0.2.0

# 3. Update CHANGELOG.md
cat >> CHANGELOG.md << EOF
## [0.2.0] - 2025-11-15

### Added
- Azure OpenAI provider support
- Streaming progress callbacks

### Fixed
- Cache key collision on similar prompts
EOF

# 4. Bump version
bump2version minor  # Creates commit and tag

# 5. Push and create PR to main
git push origin release/v0.2.0
git push --tags
gh pr create --base main --title "Release v0.2.0"

# 6. Merge PR → Triggers GitHub Actions
# - Runs tests
# - Builds wheel
# - Publishes to PyPI
# - Creates GitHub release

# 7. Merge main back to develop
git checkout main
git pull
git checkout develop
git merge main
git push
```

### 7.4 PyPI Setup (One-Time)

```bash
# 1. Create PyPI account: https://pypi.org/account/register/

# 2. Generate API token: https://pypi.org/manage/account/token/
#    Scope: Entire account or specific project

# 3. Add to GitHub Secrets
#    Settings > Secrets and variables > Actions > New repository secret
#    Name: PYPI_API_TOKEN
#    Value: pypi-AgENdGVzdC5weXBpLm9yZy...

# 4. Test with TestPyPI first
#    Upload to test.pypi.org before real PyPI
```

### 7.5 Post-Release

```bash
# Update dependent projects
cd ../CLI-Agent
pip install --upgrade llm-router==0.2.0

# Update requirements.txt
sed -i 's/llm-router==.*/llm-router==0.2.0/' requirements.txt

# Test
python main.py -m "test new version"

# Commit
git add requirements.txt
git commit -m "chore: upgrade llm-router to v0.2.0"
git push
```

---

## 8. Open Questions & Uncertainties

### 8.1 Critical Decisions Needed

#### ❓ Question 1: Rate Limiting Integration

**Context:** CLI-Agent has optional integration with `infrastructure.rate_limiting.cls_enhanced_rate_limit_tracker`

**Options:**
1. **Extract rate limiting into llm-router** - Makes package more complex
2. **Make it pluggable** - Accept optional rate limiter in RouterConfig
3. **Remove entirely** - Let users handle rate limiting externally
4. **Use provider-native limits** - Rely on Groq, OpenAI built-in handling

**Recommendation:** **Option 2 (Pluggable)** - Add optional `rate_limiter` parameter to RouterConfig that accepts any object with `.check_limit(provider, model)` method.

**Impact:** Low - Optional feature, won't block v0.1.0

---

#### ❓ Question 2: Permissions System

**Context:** CLI-Agent has pattern-based permissions in `core/permissions/settings_manager.py`

**Options:**
1. **Include permissions** - Full pattern matching for model/provider restrictions
2. **Make optional** - Include but disabled by default
3. **Remove entirely** - Not relevant for library usage

**Recommendation:** **Option 3 (Remove)** - Permissions are more relevant to CLI/agent tools than library usage. Users can implement their own restrictions.

**Impact:** Low - Simplifies codebase, rarely used by router

---

#### ❓ Question 3: Caching Strategy for Multi-Instance

**Context:** Current SQLite cache works for single process, but multiple router instances might conflict

**Options:**
1. **Keep SQLite with file locking** - Works but slow for concurrent access
2. **Add Redis backend** - Requires external service
3. **Use SQLite WAL mode** - Better concurrent performance (already implemented)
4. **In-memory only** - Fast but no persistence

**Recommendation:** **Option 3 (Current implementation)** is fine for v0.1.0. Add Redis as optional backend in v0.2.0.

**Impact:** Low - Current implementation adequate, future enhancement

---

#### ❓ Question 4: Async vs Sync API

**Context:** Current implementation is fully async. Some users may prefer sync API.

**Options:**
1. **Async only** - Modern, efficient, but requires asyncio knowledge
2. **Sync only** - Simpler but blocks on I/O
3. **Both** - Wrapper around async version
4. **Async with sync helper** - `asyncio.run()` wrapper for convenience

**Recommendation:** **Option 4** - Primary API is async, provide `router.generate_sync()` helper:

```python
class LLMRouter:
    async def generate(self, ...):
        """Async version (primary)"""

    def generate_sync(self, ...):
        """Sync wrapper for convenience"""
        return asyncio.run(self.generate(...))
```

**Impact:** Medium - Requires additional API surface, but valuable for usability

---

#### ❓ Question 5: Model Discovery Caching

**Context:** Model discovery is slow (3-5 seconds), but models change (Ollama downloads)

**Options:**
1. **Cache to disk** - Persist discovered models, refresh on demand
2. **Cache in memory** - Only for router instance lifetime
3. **No caching** - Always discover (slow)
4. **Hybrid** - Memory cache with TTL, optional disk persistence

**Recommendation:** **Option 4** - Memory cache with 5-minute TTL, optional disk cache:

```python
config = RouterConfig(
    model_cache_ttl=300,  # 5 minutes
    persist_model_cache=True,  # Save to disk
    model_cache_file=cache_dir / "models.json"
)
```

**Impact:** Medium - Improves performance, adds complexity

---

### 8.2 Technical Uncertainties

#### ⚠️ Uncertainty 1: Provider API Stability

**Issue:** Provider SDKs (Anthropic, OpenAI, etc.) change frequently

**Mitigation:**
- Pin major versions in requirements.txt (e.g., `anthropic>=0.25.0,<1.0.0`)
- Add compatibility layer for breaking changes
- Test against multiple SDK versions in CI

**Risk:** Medium - May require frequent updates

---

#### ⚠️ Uncertainty 2: Ollama Multi-Host Complexity

**Issue:** Ollama provider has complex host discovery, reachability checking (600+ lines)

**Questions:**
- Is all this complexity necessary?
- Can we simplify to basic round-robin?
- Should host health checking be optional?

**Recommendation:** Review and simplify for v0.1.0, keep advanced features optional

**Risk:** Low - Ollama users are likely advanced, can configure manually

---

#### ⚠️ Uncertainty 3: HuggingFace Local Inference

**Issue:** HuggingFace provider requires large dependencies (transformers, torch, ~2GB)

**Questions:**
- Should this be a separate package (`llm-router-local`)?
- Is local inference even needed (Ollama is better for this)?
- How many users actually use it?

**Recommendation:** Keep as optional dependency (`pip install llm-router[local]`), consider deprecating in favor of Ollama

**Risk:** Low - Optional feature

---

#### ⚠️ Uncertainty 4: Streaming Format Diversity

**Issue:** Each provider has different streaming format (6+ variations)

**Questions:**
- Is current normalization robust enough?
- What happens when providers add new stream formats?
- Should we have per-provider stream processors?

**Recommendation:** Abstract stream processing into provider-specific methods, with fallback to generic processor

**Risk:** Medium - Fragile if providers change formats

---

### 8.3 Documentation Gaps

The following need comprehensive documentation:

- [ ] **Migration guide** - Step-by-step for existing CLI-Agent users
- [ ] **Provider configuration** - Detailed setup for each provider
- [ ] **Error handling** - How to catch and handle provider errors
- [ ] **Cost tracking** - How to monitor API spending (if supported)
- [ ] **Rate limiting** - Best practices for production use
- [ ] **Custom providers** - How to implement BaseProvider for new APIs
- [ ] **Deployment** - Docker, Kubernetes, serverless examples

---

## 9. Success Criteria

### 9.1 Functional Requirements

- [ ] **All 8 providers working** - Anthropic, OpenAI, Google, Groq, Ollama, NVIDIA, HuggingFace, Human
- [ ] **Streaming support** - Async iteration over tokens
- [ ] **Failover logic** - Automatic retry with next provider
- [ ] **Caching** - SQLite-based response caching
- [ ] **Model discovery** - Parallel discovery from all providers
- [ ] **Configuration** - Environment-based and programmatic
- [ ] **Error handling** - Graceful degradation on failures

### 9.2 Quality Metrics

- [ ] **Test coverage ≥80%** - Unit + integration tests
- [ ] **All linters passing** - black, ruff, mypy
- [ ] **Python 3.9-3.12 support** - Tested in CI
- [ ] **Zero critical bugs** - No blocking issues in v0.1.0
- [ ] **Documentation complete** - README, examples, API reference

### 9.3 Performance Targets

- [ ] **Router initialization <100ms** - Fast startup
- [ ] **Model discovery <5s** - Parallel discovery
- [ ] **Cache hit <10ms** - Fast SQLite lookups
- [ ] **Streaming latency <500ms** - Time to first token
- [ ] **Memory usage <200MB** - Base overhead

### 9.4 CLI-Agent Integration

- [ ] **Clean installation** - `pip install llm-router`
- [ ] **No breaking changes** - CLI-Agent works after migration
- [ ] **All tests passing** - CLI-Agent test suite passes
- [ ] **No duplicate code** - Old provider code removed
- [ ] **Version pinned** - requirements.txt specifies exact version

### 9.5 Release Checklist

- [ ] **Published to PyPI** - Installable via pip
- [ ] **GitHub release created** - With changelog and artifacts
- [ ] **Documentation deployed** - ReadTheDocs or GitHub Pages
- [ ] **Examples tested** - All 7 examples work
- [ ] **CLI-Agent updated** - Using llm-router package
- [ ] **Announcement** - GitHub discussions, Reddit, etc.

---

## 10. Timeline & Milestones

### 10.1 Estimated Schedule (1 Senior Engineer)

| Phase | Duration | Start | End | Deliverable |
|-------|----------|-------|-----|-------------|
| **Phase 1: Setup** | 0.5 days | Day 1 AM | Day 1 PM | Repository, CI/CD, structure |
| **Phase 2: Core** | 1.5 days | Day 1 PM | Day 3 AM | Chat, Config, Cache, Capabilities |
| **Phase 3: Providers** | 1.5 days | Day 3 AM | Day 4 PM | All 8 providers migrated |
| **Phase 4: Router** | 1.5 days | Day 4 PM | Day 6 AM | LLMRouter, model discovery |
| **Phase 5: Docs** | 0.5 days | Day 6 AM | Day 6 PM | README, examples, API docs |
| **Phase 6: Release** | 0.5 days | Day 6 PM | Day 7 AM | PyPI publish, CLI-Agent update |
| **Total** | **6 days** | | | Production-ready v0.1.0 |

### 10.2 Milestones

#### Milestone 1: Foundation (End of Day 2)
- ✅ Repository created and configured
- ✅ Core modules extracted (Chat, Config, Cache)
- ✅ Unit tests for core modules
- ✅ CI/CD pipeline working

#### Milestone 2: Providers Complete (End of Day 4)
- ✅ All 8 providers migrated
- ✅ BaseProvider interface defined
- ✅ Unit tests for each provider
- ✅ Integration tests passing (with real APIs)

#### Milestone 3: Router Functional (End of Day 5)
- ✅ LLMRouter class working
- ✅ Model discovery functional
- ✅ Streaming support
- ✅ Failover logic tested

#### Milestone 4: Release Ready (End of Day 6)
- ✅ Documentation complete
- ✅ Examples working
- ✅ Published to PyPI
- ✅ CLI-Agent integrated

### 10.3 Risk Mitigation

**High-Risk Items:**
1. **Provider API changes during migration** - Test early, pin versions
2. **Complex refactoring of LlmRouter** - Incremental migration, parallel testing
3. **CLI-Agent integration issues** - Maintain backward compatibility layer

**Contingency Plans:**
- Add 1-2 buffer days for unexpected issues
- Prioritize core functionality (cloud providers) over optional features (HuggingFace)
- Ship v0.1.0 with minimal features, iterate quickly

---

## 11. Post-Launch Roadmap

### Version 0.2.0 (Planned Features)

- [ ] **Redis cache backend** - Distributed caching for multi-instance
- [ ] **Azure OpenAI provider** - Additional cloud provider
- [ ] **Cost tracking** - Track API spending per model
- [ ] **Async batch processing** - Process multiple requests concurrently
- [ ] **Request/response logging** - Audit trail for debugging
- [ ] **Custom retry policies** - User-defined retry strategies

### Version 0.3.0 (Advanced Features)

- [ ] **Model fine-tuning support** - Upload/manage fine-tuned models
- [ ] **A/B testing** - Compare responses from different models
- [ ] **Response validation** - Schema-based output validation
- [ ] **Prompt templates** - Reusable prompt library
- [ ] **Usage analytics** - Dashboard for cost/performance metrics

### Version 1.0.0 (Production Ready)

- [ ] **Stability guarantees** - Semantic versioning, deprecation policy
- [ ] **Enterprise features** - SSO, RBAC, audit logs
- [ ] **Performance optimizations** - Connection pooling, batching
- [ ] **Comprehensive documentation** - Video tutorials, cookbooks
- [ ] **Professional support** - Dedicated support channel

---

## 12. Handoff Instructions

### For the Senior Engineer

**Getting Started:**

1. **Review this document thoroughly** - Familiarize yourself with the plan
2. **Clone CLI-Agent repository** - Understand current implementation
   ```bash
   git clone https://github.com/Probst1nator/CLI-Agent.git
   cd CLI-Agent
   ```
3. **Read source files** - Focus on these key files:
   - `core/llm_router.py` (main router logic)
   - `core/llm.py` (model discovery)
   - `core/providers/` (all provider implementations)
   - `core/cache/llm_cache_manager.py` (caching)
4. **Set up development environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
5. **Test current implementation** - Verify everything works before extraction
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   python main.py -m "Hello, test the LLM router"
   ```

**Execution Order:**

1. Start with Phase 1 (Repository Setup) - Get infrastructure right first
2. Proceed sequentially through phases - Each builds on previous
3. Run tests after each phase - Don't accumulate technical debt
4. Update this document - Note any deviations or discoveries
5. Ask questions early - Don't wait until blocked

**Communication:**

- **Daily updates** - Brief status report (what's done, what's next, blockers)
- **Document decisions** - If you deviate from plan, note why
- **Track time** - Compare actual vs estimated (helps future planning)
- **Raise uncertainties** - Sections 8.1-8.4 need decisions

**Resources:**

- **CLI-Agent docs**: `~/repos/CLI-Agent/docs/`
- **Provider docs**:
  - [Anthropic](https://docs.anthropic.com/)
  - [OpenAI](https://platform.openai.com/docs/)
  - [Google Gemini](https://ai.google.dev/docs)
  - [Groq](https://console.groq.com/docs)
  - [Ollama](https://github.com/ollama/ollama/blob/main/docs/api.md)
- **Packaging**: [Python Packaging Guide](https://packaging.python.org/)
- **GitHub Actions**: [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)

**Contact:**

- For clarifications on architecture: [Reference this document, Section 1]
- For design decisions: [Reference this document, Section 8]
- For timeline concerns: [Reference this document, Section 10]

---

## 13. Appendix

### A. Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Conditional | None | Claude API key |
| `OPENAI_API_KEY` | Conditional | None | GPT API key |
| `GOOGLE_API_KEY` | Conditional | None | Gemini API key |
| `GROQ_API_KEY` | Conditional | None | Groq API key |
| `NVIDIA_API_KEY` | Conditional | None | NVIDIA Cloud API key |
| `OLLAMA_HOSTS` | Optional | `localhost:11434` | Comma-separated Ollama hosts |
| `HF_MODELS_DIR` | Optional | `~/.cache/huggingface` | HuggingFace models directory |
| `HF_DEVICE` | Optional | `auto` | Device for local inference (auto/cuda/cpu/mps) |
| `HF_MAX_MEMORY_GB` | Optional | `16` | Max memory for HuggingFace models |
| `LLM_CACHE_DIR` | Optional | `~/.cache/llm-router` | Cache directory |
| `LLM_ENABLE_CACHE` | Optional | `true` | Enable response caching |
| `LLM_CACHE_MAX_AGE_DAYS` | Optional | `30` | Cache entry expiration |
| `FORCE_LOCAL` | Optional | `false` | Only use local models |
| `LLM_FORCE_MODEL` | Optional | None | Force specific model |
| `LLM_DEFAULT_TEMPERATURE` | Optional | `0.0` | Default sampling temperature |
| `LLM_TIMEOUT` | Optional | `60` | Request timeout (seconds) |
| `LLM_MAX_RETRIES` | Optional | `3` | Max retry attempts |
| `LLM_VERBOSE` | Optional | `false` | Enable debug logging |

**Note:** At least one cloud API key OR local provider must be configured.

### B. CLI-Agent Files to Extract

**Core files (must extract):**
```
core/llm_router.py            → src/llm_router/core/router.py
core/llm.py                   → src/llm_router/core/model.py
core/chat.py                  → src/llm_router/core/chat.py
core/lm_capabilities.py       → src/llm_router/core/capabilities.py
core/cache/llm_cache_manager.py → src/llm_router/cache/manager.py
```

**Provider files (all must extract):**
```
core/providers/cls_anthropic_interface.py → src/llm_router/providers/anthropic.py
core/providers/cls_google_interface.py    → src/llm_router/providers/google.py
core/providers/cls_groq_interface.py      → src/llm_router/providers/groq.py
core/providers/cls_openai_interface.py    → src/llm_router/providers/openai.py
core/providers/cls_ollama_interface.py    → src/llm_router/providers/ollama.py
core/providers/cls_nvidia_interface.py    → src/llm_router/providers/nvidia.py
core/providers/cls_huggingface_local_interface.py → src/llm_router/providers/huggingface.py
core/providers/cls_human_as_interface.py  → src/llm_router/providers/human.py
```

**Supporting files (partial extraction):**
```
py_classes/unified_interfaces.py → src/llm_router/providers/base.py (extract AIProviderInterface only)
```

### C. External Dependencies (requirements.txt)

```txt
# LLM Provider SDKs
anthropic>=0.25.0,<1.0.0
openai>=1.0.0,<2.0.0
google-generativeai>=0.3.0,<1.0.0
groq>=0.4.0,<1.0.0
ollama>=0.1.0,<1.0.0

# Configuration
python-dotenv>=1.0.0

# Async HTTP
aiohttp>=3.8.0

# Data validation
pydantic>=2.0.0

# Optional: Local inference (install with pip install llm-router[local])
# transformers>=4.48.0
# torch>=2.0.0
# accelerate>=0.20.0
```

### D. Useful Commands

```bash
# Development
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"  # Editable install with dev dependencies

# Testing
pytest tests/ -v                          # All tests
pytest tests/unit/ -v                     # Unit tests only
pytest tests/integration/ -v -m "not slow" # Fast integration tests
pytest tests/ --cov=llm_router --cov-report=html  # With coverage

# Code quality
black .                                   # Format code
ruff check .                              # Lint
ruff check . --fix                        # Auto-fix linting issues
mypy src/                                 # Type checking

# Documentation
mkdocs serve                              # Local docs server
mkdocs build                              # Build static docs

# Release
bump2version patch                        # Bump patch version
bump2version minor                        # Bump minor version
bump2version major                        # Bump major version
git push && git push --tags               # Push with tags

# Package
python -m build                           # Build wheel + sdist
twine check dist/*                        # Validate package
twine upload --repository testpypi dist/* # Upload to test PyPI
twine upload dist/*                       # Upload to real PyPI
```

---

## Document Control

**Version History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-11 | Claude Code | Initial comprehensive plan |

**Review Status:**

- [ ] Technical review completed
- [ ] Architecture approved
- [ ] Timeline validated
- [ ] Resources allocated
- [ ] Ready for implementation

**Approvals:**

- **Technical Lead**: _____________________ Date: _______
- **Project Manager**: ____________________ Date: _______

---

**END OF DOCUMENT**

*This plan is a living document. Update as implementation progresses and new information emerges.*
