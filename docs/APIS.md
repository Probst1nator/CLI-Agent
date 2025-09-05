# LLM Provider APIs Documentation

This document provides comprehensive documentation for the three LLM provider APIs supported by CLI-Agent: Google (Gemini), Groq, and Ollama. Each section details the data structures, parameters, and information provided by their respective model discovery endpoints.

## Overview

CLI-Agent supports dynamic model discovery from three different LLM providers:

1. **Google Generative AI (Gemini)** - Cloud-based AI models with vision and text capabilities
2. **Groq** - High-performance inference platform for open-source models  
3. **Ollama** - Local model hosting platform for running LLMs offline

Each provider exposes different model information through their APIs, which is documented below.

---

## Google Generative AI API

### Endpoint
The Google API uses the `genai.list_models()` method from the `google-generativeai` library.

### Authentication
Requires `GEMINI_API_KEY` environment variable.

### Implementation Location
`core/providers/cls_google_interface.py:get_available_models()`

### Data Structure

Each model returned by the Google API contains the following fields:

```python
{
    'name': str,                           # Model identifier (e.g., "models/gemini-1.5-pro")
    'display_name': str,                   # Human-readable name
    'description': str,                    # Model description  
    'supported_generation_methods': List[str],  # List of supported methods (e.g., ["generateContent"])
    'version': str,                        # Model version
    'input_token_limit': int | None,       # Maximum input tokens (if available)
    'output_token_limit': int | None,      # Maximum output tokens (if available)
    'temperature': float | None,           # Default temperature setting
    'top_p': float | None,                 # Default top_p setting
    'top_k': int | None                    # Default top_k setting
}
```

### Field Descriptions

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `name` | `str` | Unique model identifier used in API calls | `"models/gemini-1.5-pro"` |
| `display_name` | `str` | Human-friendly model name | `"Gemini 1.5 Pro"` |
| `description` | `str` | Detailed description of model capabilities | `"Mid-size multimodal model..."` |
| `supported_generation_methods` | `List[str]` | Available generation methods | `["generateContent", "streamGenerateContent"]` |
| `version` | `str` | Model version identifier | `"001"` |
| `input_token_limit` | `int \| None` | Maximum input context window | `1048576` |
| `output_token_limit` | `int \| None` | Maximum output tokens per response | `8192` |
| `temperature` | `float \| None` | Default randomness control (0.0-2.0) | `1.0` |
| `top_p` | `float \| None` | Default nucleus sampling parameter | `0.95` |
| `top_k` | `int \| None` | Default top-k sampling parameter | `40` |

### Example Response

```python
[
    {
        'name': 'models/gemini-1.5-pro-latest',
        'display_name': 'Gemini 1.5 Pro Latest',
        'description': 'Latest version of Gemini 1.5 Pro with enhanced capabilities',
        'supported_generation_methods': ['generateContent', 'streamGenerateContent'],
        'version': '002',
        'input_token_limit': 2097152,
        'output_token_limit': 8192,
        'temperature': 1.0,
        'top_p': 0.95,
        'top_k': 40
    }
]
```

---

## Groq API

### Endpoint
`https://api.groq.com/openai/v1/models` (REST API)

### Authentication
Requires `GROQ_API_KEY` environment variable passed as Bearer token.

### Implementation Location
`core/providers/cls_groq_interface.py:get_available_models()`

### Data Structure

Each model returned by the Groq API contains the following fields:

```python
{
    'id': str,                             # Model identifier (e.g., "llama-3.3-70b-versatile")
    'object': str,                         # Object type (typically "model")
    'created': int,                        # Creation timestamp (Unix epoch)
    'owned_by': str,                       # Model developer/owner
    'root': str,                           # Root model identifier
    'parent': str | None,                  # Parent model (if applicable)
    'context_window': int | None,          # Maximum context window size
    'max_model_len': int | None,           # Alternative context window field
    'description': str,                    # Model description
    'pricing': Dict[str, Any],             # Pricing information
    'supported_modalities': List[str]      # Supported input/output modalities
}
```

### Field Descriptions

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `id` | `str` | Unique model identifier used in API calls | `"llama-3.3-70b-versatile"` |
| `object` | `str` | API object type | `"model"` |
| `created` | `int` | Model creation timestamp (Unix) | `1702953960` |
| `owned_by` | `str` | Organization or entity that owns the model | `"Meta"` |
| `root` | `str` | Base model identifier | `"llama-3.3-70b-versatile"` |
| `parent` | `str \| None` | Parent model if this is a fine-tuned variant | `null` |
| `context_window` | `int \| None` | Maximum input context length | `131072` |
| `max_model_len` | `int \| None` | Alternative field for context window | `131072` |
| `description` | `str` | Model description and capabilities | `"Llama 3.3 70B Versatile model"` |
| `pricing` | `Dict[str, Any]` | Pricing structure for API usage | `{"input": 0.59, "output": 0.79}` |
| `supported_modalities` | `List[str]` | Types of input/output supported | `["text"]` |

### Example Response

```python
[
    {
        'id': 'llama-3.3-70b-versatile',
        'object': 'model',
        'created': 1702953960,
        'owned_by': 'Meta',
        'root': 'llama-3.3-70b-versatile',
        'parent': null,
        'context_window': 131072,
        'max_model_len': 131072,
        'description': 'Llama 3.3 70B Versatile is a text generation model',
        'pricing': {
            'input': 0.59,
            'output': 0.79,
            'currency': 'USD',
            'unit': '1M tokens'
        },
        'supported_modalities': ['text']
    }
]
```

---

## Ollama API

### Endpoint
Uses the `ollama.Client().list()` method from the `ollama` Python library to query local Ollama instances.

### Authentication
No authentication required (local API).

### Implementation Location
`core/providers/cls_ollama_interface.py:get_downloaded_models()`
`core/llm_router.py:discover_models_with_progress()` (Ollama section)

### Data Structure

#### Raw Ollama Response
The Ollama API returns model information in this structure:

```python
{
    'name': str,                           # Full model name with tag (e.g., "llama3:8b")
    'size': int,                           # Model size in bytes
    'modified_at': str | None              # Last modification timestamp (ISO format)
}
```

#### Processed Model Information
After processing by CLI-Agent, each Ollama model contains:

```python
{
    'name': str,                           # Model name (e.g., "llama3:8b")
    'context_window': int,                 # Estimated or retrieved context window
    'strengths': List[str],                # Inferred capabilities (e.g., ["VISION", "UNCENSORED"])
    'provider': str,                       # Always "OllamaClient"
    'host': str,                           # Ollama host where model is available
    'size': int,                           # Model size in bytes
    'modified_at': str | None,             # Last modification timestamp
    'raw_info': Dict[str, Any]             # Original API response data
}
```

### Field Descriptions

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `name` | `str` | Full model name including tag | `"llama3:8b"` |
| `context_window` | `int` | Maximum context length (estimated/fetched) | `128000` |
| `strengths` | `List[str]` | Inferred capabilities from model name patterns | `["VISION"]` |
| `provider` | `str` | Provider identifier | `"OllamaClient"` |
| `host` | `str` | Ollama host where model is downloaded | `"localhost"` |
| `size` | `int` | Model file size in bytes | `4661224192` |
| `modified_at` | `str \| None` | Last modification date (ISO 8601 format) | `"2024-03-15T10:30:00Z"` |
| `raw_info` | `Dict[str, Any]` | Original unprocessed API response | `{"name": "llama3:8b", ...}` |

### Capability Inference Rules

CLI-Agent automatically infers model capabilities based on name patterns:

| Pattern | Strength | Example Models |
|---------|----------|----------------|
| `vision`, `vl`, `visual` | `VISION` | `qwen2.5vl:3b`, `llava:7b` |
| `uncensored`, `dolphin`, `wizard` | `UNCENSORED` | `dolphin-mixtral:8x7b` |

### Host Discovery

Ollama models are discovered from multiple hosts defined in `core/globals.py:DEFAULT_OLLAMA_HOSTS`:
- Local instances (localhost, 127.0.0.1)
- Network instances (custom IP addresses)

### Example Response

```python
[
    {
        'name': 'qwen2.5vl:3b',
        'context_window': 128000,
        'strengths': ['VISION'],
        'provider': 'OllamaClient',
        'host': 'localhost',
        'size': 4661224192,
        'modified_at': '2024-03-15T10:30:00Z',
        'raw_info': {
            'name': 'qwen2.5vl:3b',
            'size': 4661224192,
            'modified_at': '2024-03-15T10:30:00Z'
        }
    }
]
```

---

## Discovery Process Flow

### Manual Discovery Trigger
Users can trigger manual model discovery by pressing `u` in the LLM selector interface.

### Discovery Sequence
1. **Google**: Query `genai.list_models()` with API key
2. **Groq**: HTTP GET to `/models` endpoint with Bearer token  
3. **Ollama**: Query each configured host's `/api/tags` endpoint

### Progress Tracking
Each provider reports discovery status through callbacks:
- `starting` - Discovery process begins
- `success` - Discovery completed successfully  
- `error` - Discovery failed with error
- `skipped` - Discovery skipped (e.g., missing API key)

### Caching and Persistence
- Discovered models are cached locally in JSON format
- Cache includes timestamps and model metadata
- Models are persisted across CLI-Agent sessions
- Cache is updated on each manual discovery

### Error Handling
- Missing API keys result in `skipped` status
- Network errors result in `error` status  
- Malformed responses are logged and handled gracefully
- Discovery continues for other providers even if one fails

---

## Usage in CLI-Agent

### Integration Points
- **LLM Router** (`core/llm_router.py`): Central model discovery and management
- **LLM Selector** (`py_classes/cls_llm_selection.py`): User interface for model selection
- **Provider Interfaces**: Individual API implementations

### Model Selection
Discovered models are automatically integrated into the LLM selector interface, allowing users to:
- View all available models from all providers
- See model capabilities (vision, context window, etc.)
- Select models for different use cases (beams, evaluation, guard)

### Dynamic Updates
The system supports adding new models without code changes:
- New Google models appear automatically when released
- New Groq models are discovered from their API
- New Ollama models are found when downloaded locally

This architecture ensures CLI-Agent stays current with the latest available models across all supported providers.