# Core Infrastructure Documentation

The core infrastructure provides the foundation for CLI-Agent's AI capabilities, including LLM routing, provider integration, chat management, and permission systems.

## Architecture Overview

Core is the heart of CLI-Agent's AI infrastructure:
- **LLM Router**: Multi-provider model selection and request routing
- **Provider Interfaces**: Unified API for Anthropic, OpenAI, Google, Groq, Ollama
- **Chat Management**: Conversation state and serialization
- **Permission System**: Security and user consent management
- **Caching**: Response and model discovery caching

## Directory Structure

```
core/
├── llm_router.py         # Central LLM routing and model selection
├── llm.py                # LLM model class and capabilities
├── chat.py               # Chat conversation management
├── globals.py            # Global state singleton (being phased out)
├── file_manager.py       # Temporary file lifecycle management
├── llm_profiles.py       # Save/load LLM configurations
├── lm_capabilities.py    # Model capability enums
├── providers/            # AI provider implementations
│   ├── cls_anthropic_interface.py
│   ├── cls_openai_interface.py
│   ├── cls_google_interface.py
│   ├── cls_groq_interface.py
│   ├── cls_ollama_interface.py
│   ├── cls_nvidia_interface.py
│   ├── cls_whisper_interface.py
│   └── cls_piper_interface.py
├── permissions/          # Permission system
│   ├── enhanced_permissions.py
│   ├── permissions_manager.py
│   └── path_detector.py
└── cache/                # Caching infrastructure
```

## LLM Router (`llm_router.py`)

Central orchestration for AI model requests:

### Key Features

- **Multi-provider support**: Anthropic, OpenAI, Google, Groq, Ollama, NVIDIA
- **Auto-discovery**: Dynamic model enumeration from all providers
- **Lazy loading**: Providers loaded on-demand for fast startup
- **Failover**: Automatic fallback on errors
- **Caching**: Response and model list caching
- **Streaming**: Real-time token streaming
- **Activity monitoring**: Track model usage

### Model Selection

Three selection modes:
1. **Override mode**: Specific model for current request
2. **Router LLM**: Model for tool selection and routing
3. **Planner LLM**: Model for main conversation

**Fallback chain**:
- Try configured model
- Fall back to fallback list
- Try discovered models
- Return error if all fail

### Caching Strategy

- **Model discovery**: 1-hour cache (reduces startup time)
- **Provider lists**: Cached per provider
- **Response cache**: Optional (disabled by default)

See: `core/docs/LLM_ROUTER_API.md`

## LLM Model Class (`llm.py`)

Represents individual AI models:

### Properties

- **ID and name**: Unique identifier and display name
- **Provider**: Which service hosts the model
- **Capabilities**: Vision, uncensored, local, etc.
- **Context window**: Maximum tokens
- **Discovery metadata**: Last updated, availability

### Model Discovery

- **Background fetching**: Parallel provider queries
- **Caching**: 1-hour TTL for discovered models
- **Dynamic context**: Ollama context window detection
- **Capability detection**: Automatic based on model name/metadata

## Chat Management (`chat.py`)

Core Chat class for conversation state:

### Features

- **Multiple message roles**: system, user, assistant, tool
- **Colorized output**: Terminal-friendly display
- **Message editing**: `update_message(index, content)`
- **Serialization**: JSON with base64_images persistence
- **Multi-format conversion**: Ollama, OpenAI, Groq, Gemini formats
- **Compression**: LLM-based conversation summarization

### Message Format

```python
{
    "role": "user|assistant|system|tool",
    "content": "Message text",
    "base64_images": ["..."],  # Optional images
    "tool_calls": [...]         # Optional tool invocations
}
```

### Conversation Compression

Reduce token usage while preserving context:
- **Density levels**: 'dense', 'detailed'
- **Custom prompts**: Configurable compression instructions
- **LLM-powered**: Uses configured model for summarization

## Provider System (`core/providers/`)

Unified interface for multiple AI services:

### Provider Interfaces

Each provider implements the unified interface:
- `generate(messages, **kwargs)` - Text generation
- `stream(messages, **kwargs)` - Streaming responses
- `list_models()` - Model discovery

### Anthropic (`cls_anthropic_interface.py`)

Claude models (Claude 3, Claude Opus, etc.):
- Chat format conversion
- Image processing (base64/URL)
- System prompt handling
- Streaming with proper event parsing

### OpenAI (`cls_openai_interface.py`)

GPT models:
- Streaming response generation
- Temperature control
- Speech recognition (Whisper via API)

### Google (`cls_google_interface.py`)

Gemini models:
- Chat format conversion
- Temperature control
- **Thinking budget** support for Gemini 2.x
- Rate limiting integration

### Groq (`cls_groq_interface.py`)

Fast inference:
- Custom exception handling (TimeoutException, RateLimitException)
- Rate limiting support
- Streaming responses

### Ollama (`cls_ollama_interface.py`)

Local models:
- Model discovery
- Host management (localhost + remote)
- Context window detection
- Date parsing for model metadata

### NVIDIA (`cls_nvidia_interface.py`)

NVIDIA NeMo:
- OpenAI-compatible interface
- Base64 image support

### Audio Providers

**Whisper** (`cls_whisper_interface.py`):
- Speech transcription
- Local Whisper models
- Kokoro TTS integration

**Piper** (`cls_piper_interface.py`):
- High-quality local TTS
- Voice selection
- Multilingual support

See: `core/providers/docs/README.md`

## Permission System (`core/permissions/`)

Security and user consent management:

### Components

**Enhanced Permissions** (`enhanced_permissions.py`):
- Interactive permission prompts
- Persistent "always allow" functionality
- Path-specific permissions
- Directory-based rules
- Permission scoping (file, directory, project, tool-wide)

**Permissions Manager** (`permissions_manager.py`):
- JSON-based config storage (`~/.cli-agent/settings.json`)
- Tool-specific rules
- Path-specific rules
- Pattern matching

**Path Detector** (`path_detector.py`):
- XML block analysis for file paths
- Tag detection ('path' substring)
- Action descriptions
- Permission requirement determination

### Permission Patterns

```json
{
  "bash": {
    "npm:*": "allow",
    "docker:*": "allow"
  },
  "readfile": {
    "./src/**": "allow",
    "/etc/passwd": "deny"
  }
}
```

### Precedence

1. **Deny** rules (highest priority)
2. **Ask** rules (prompt user)
3. **Allow** rules (automatic approval)

See: `core/permissions/docs/PERMISSION_SYSTEM.md`

## File Manager (`file_manager.py`)

Centralized temporary file lifecycle:

### Features

- **Coordinated file creation** across all components
- **Automatic tracking** of temporary files
- **TTL policies**: Configurable time-to-live
- **Background cleanup**: Automatic old file removal
- **Context managers**: Clean resource management
- **Statistics**: File usage tracking

### Usage

```python
with file_manager.create_temp_file(suffix='.txt', ttl=3600) as f:
    f.write('temp content')
# File automatically tracked for cleanup
```

## LLM Profiles (`llm_profiles.py`)

Save and load LLM configurations:

### Features

- **Simple numeric IDs** (1-6 for web UI slots)
- **Complete config**: Override, router, planner, fallback models
- **Persistent storage**: JSON files
- **Automatic loading**: On startup and on-demand

### Profile Format

```python
{
  "override_llm": "claude-3-opus",
  "router_llm": "gpt-4",
  "planner_llm": "gemini-pro",
  "fallback_llm_names": ["claude-3-sonnet", "gpt-3.5"]
}
```

### Commands

- `/profile_save <id>` - Save current config
- `/profile_load <id>` - Load config from slot
- `/profile_list` - Show all profiles
- `/profile_delete <id>` - Delete profile

## Global State (`globals.py`)

**Note**: Being phased out in favor of AppContext pattern

Current responsibilities:
- Application configuration
- File paths
- LLM settings
- Debug flags
- Model discovery caching
- Temporary file cleanup

**Migration target**: Move to agent/core/app_context.py

## Caching (`core/cache/`)

Response and metadata caching for performance.

## Key Design Principles

### 1. Provider Abstraction

All providers implement the same interface:
```python
interface LLMProvider:
    def generate(messages, **kwargs) -> str
    def stream(messages, **kwargs) -> Iterator[str]
    def list_models() -> List[str]
```

### 2. Lazy Loading

Heavy dependencies loaded on-demand:
- Provider modules imported when first used
- Model lists fetched and cached
- ~1.5s startup time savings

### 3. Failover Resilience

Multiple fallback strategies:
- Try configured model
- Fall back to fallback list
- Try discovered models
- Graceful error handling

### 4. Separation of Concerns

- **llm_router.py**: Orchestration and routing
- **llm.py**: Model representation
- **providers/**: Provider-specific logic
- **chat.py**: Conversation state

## Related Documentation

- LLM Router API: `core/docs/LLM_ROUTER_API.md`
- Provider APIs: `core/docs/APIS.md`
- Provider system: `core/providers/docs/README.md`
- Permission system: `core/permissions/docs/PERMISSION_SYSTEM.md`
- Permission guide: `core/permissions/docs/PERMISSION_SYSTEM_GUIDE.md`
- Profile validation: `core/docs/PROFILE_VALIDATION.md`
- HuggingFace local: `core/docs/HUGGINGFACE_LOCAL_PROVIDER.md`
- Chat serialization: `core/docs/CHAT_SERIALIZATION.md`
