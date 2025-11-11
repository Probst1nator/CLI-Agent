# LLM Router API - Design Document

## Overview

This document outlines the design for extracting `core/llm_router.py` functionality into a self-contained API service.

## Technology Choice: FastAPI vs aiohttp

### FastAPI (Recommended)
- ✅ Auto-generated OpenAPI documentation
- ✅ Built-in Pydantic validation and type safety
- ✅ Better for standard REST APIs
- ✅ Easier OpenAI API compatibility
- ❌ Slightly higher overhead than aiohttp

### aiohttp (Current Stack)
- ✅ Already used in `api/server.py`
- ✅ Lower overhead, more control
- ✅ Existing WebSocket infrastructure
- ❌ More manual validation, no auto-docs

**Decision:** Use FastAPI for the LLM router API as a separate service. Benefits from type safety, auto-generated docs for integration, and better OpenAI API compatibility.

## Core Endpoints

### Model Management
- `GET /models` - List available models across all providers
- `GET /models/discover` - Trigger model discovery/refresh
- `GET /providers` - List available providers and their status

### Chat/Completion
- `POST /chat/completions` - Main completion endpoint (OpenAI-compatible)
- `POST /chat/stream` - Streaming completions with SSE
- `POST /chat/interrupt` - Interrupt active generation

### Provider Management
- `GET /providers/{provider}/models` - List models for specific provider
- `POST /providers/{provider}/test` - Test provider connectivity
- `GET /providers/health` - Health check for all providers

### Cache Management
- `GET /cache/stats` - Cache hit/miss statistics
- `DELETE /cache` - Clear cache
- `GET /cache/{key}` - Get cached response for key

### Configuration
- `GET /config` - Get current router configuration
- `PUT /config` - Update router settings (temperature, max_tokens, etc.)
- `GET /config/providers` - Get provider-specific settings

### Retry/Failover
- `GET /retry/stats` - Get retry statistics per provider
- `POST /retry/reset` - Reset retry counters

## Additional Considerations

### 1. Authentication & Rate Limiting
- API keys for router access (separate from provider keys)
- Per-client rate limits
- Provider quota tracking
- Multi-tenant support if needed

### 2. Provider Key Management
- Secure storage for provider API keys (environment variables, vault)
- Key rotation support
- Multi-tenant key isolation if needed
- Separate client auth from provider auth

### 3. Observability
- Token usage tracking per provider/model
- Latency metrics per provider
- Error rate monitoring
- Cost tracking (tokens × pricing per provider)
- Prometheus/OpenTelemetry integration

### 4. Streaming Architecture
- SSE (Server-Sent Events) vs WebSocket for streaming
- Client disconnect handling (stop generation immediately)
- Partial response recovery on connection drops
- Backpressure handling

### 5. Failover Strategy
- Configurable retry policies per provider
- Provider priority ordering (primary/fallback)
- Circuit breaker pattern for failing providers
- Exponential backoff with jitter
- Provider health scoring

### 6. Message Format Conversion
- Leverage existing `core/chat.py` conversion logic
- Support for Ollama, Gemini, Anthropic, OpenAI formats
- Unified API format (OpenAI-compatible recommended)
- Provider-specific passthrough mode for advanced features

### 7. Context Window Management
- Auto-truncation for long contexts
- Smart compression (leverage existing LLM-based summarization)
- Context budget estimation before API calls
- Warning on context overflow

### 8. Deployment Considerations
- Separate process from main agent (microservice architecture)
- Docker containerization with multi-stage builds
- Health checks for Kubernetes/orchestration
- Horizontal scaling considerations
- Shared cache backend (Redis) for multi-instance deployments

## Implementation Notes

### Leveraging Existing Code
- `core/llm_router.py`: Core routing logic (singleton pattern)
- `core/chat.py`: Message format conversion
- `core/cache/llm_cache_manager.py`: SQLite-based caching
- `core/providers/`: Provider interface implementations

### Migration Path
1. Create FastAPI wrapper around existing `LLMRouter` singleton
2. Expose endpoints incrementally
3. Add authentication/rate limiting layer
4. Deploy as separate service
5. Update main agent to consume API instead of direct router

### API Response Format
Use standard format (matching existing `api/` layer):
```json
{
  "status": "success",
  "data": { ... }
}
```

Error format:
```json
{
  "status": "error",
  "error": "Error message",
  "code": "ERROR_CODE"
}
```

## Related Documentation
- `BACKEND_FRONTEND_ARCHITECTURE.md`: Current API architecture
- `CLAUDE.md`: LLM routing architecture section
- `core/llm_router.py`: Implementation reference
