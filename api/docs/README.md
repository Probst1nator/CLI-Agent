# Backend API Documentation

The API layer provides a clean separation between the backend server and frontend, enabling web-based interaction with CLI-Agent.

## Architecture Overview

Modern backend/frontend separation achieved in January 2025:
- **aiohttp** web server with WebSocket support
- **REST API** for chat, audio, tools, and profile management
- **Session management** with GDPR-compliant audit logging
- **Real-time sync** via WebSocket events

## Key Components

### Server Infrastructure (`api/server.py`)

~520 lines of core server functionality:

- **WebSocket management**: Real-time bidirectional communication
- **Session tracking**: Per-client session IDs and lifecycle management
- **Audit logging**: SessionAuditLogger with IP, user-agent, chat snapshots
- **State broadcasting**: Real-time events to all connected clients
- **Auto-start/stop**: Starts with `--gui`, stops when last client disconnects

**Key features**:
- Session initialization on WebSocket connect
- Image upload via WebSocket (base64 transmission)
- Graceful shutdown with proper cleanup
- Static file serving for frontend assets

### Chat API (`api/chat_api.py`)

~330 lines managing chat operations:

**Endpoints**:
- `POST /api/chat/load/{id}` - Load archived chat
- `POST /api/chat/new` - Start new chat
- `GET /api/chat/current` - Fetch current state
- `GET /api/chat/archives` - List all saved chats
- `DELETE /api/chat/{id}` - Delete chat
- `POST /api/chat/message/edit` - Edit message in current chat

**WebSocket events**:
- `chat_replaced` - Chat loaded from archive
- `chat_cleared` - New chat started
- `message_edited` - Message updated

Enables **chat loading without page reload** via ChatStateManager integration.

### Audio API (`api/audio_api.py`)

~180 lines for voice interaction:

**Endpoints**:
- `POST /api/audio/upload` - Browser microphone recording upload
- `POST /api/audio/tts` - Text-to-speech with streaming
- `GET /api/audio/voice-status` - Voice mode status

**Features**:
- Transcription using shared audio utilities (Faster-Whisper, OpenAI, Google)
- TTS generation with multiple engines (Piper, Kokoro, Google)
- Streaming audio playback
- Integration with browser MediaRecorder API

### Agent State API (`api/agent_state_api.py`)

Real-time agent configuration:

**Endpoints**:
- `GET /api/agent/state` - Current agent settings
- `POST /api/agent/voice-mode` - Toggle voice mode
- `POST /api/agent/auto-execute` - Toggle auto-execution

**Features**:
- Real-time sync with WebSocket
- Persistent state management

### Tools API (`api/tools_api.py`)

Tool configuration and management:

**Endpoints**:
- `GET /api/tools` - List available tools
- `POST /api/tools/toggle` - Enable/disable tools
- `POST /api/tools/upload` - File uploads from tool manager

### Profile API (`api/profile_api.py`)

LLM profile management:

**Endpoints**:
- `GET /api/profiles` - List saved profiles (slots 1-6)
- `POST /api/profiles/save/{id}` - Save current config to slot
- `POST /api/profiles/load/{id}` - Load profile from slot

**Features**:
- Simple numeric IDs (1-6 for web UI slots)
- Stores override, router, planner, fallback models
- Persistent JSON storage

## Session Management

### Session Lifecycle

1. **Client connects** → WebSocket handshake
2. **Session created** → Unique session_id assigned
3. **Session ID propagation** → Sent to frontend, used in HTTP headers
4. **Session tracking** → All requests associated with session
5. **Client disconnects** → Session cleanup, server auto-stop (if no --gui)

### Audit Logging (GDPR-Compliant)

SessionAuditLogger in `api/server.py`:
- **Session metadata**: IP, user-agent, timestamps
- **Chat snapshots**: Hashed content, not full text
- **Event tracking**: Session start/end, chat loads
- **Privacy-first**: Minimal PII, hashed identifiers

Storage: `~/.cli-agent/audit_logs/sessions/`

## WebSocket Communication

### Event Types

**Streaming** (message generation):
- `message_start` - LLM begins response
- `message_chunk` - Incremental content
- `message_end` - Response complete

**Chat sync** (state management):
- `chat_replaced` - Archived chat loaded
- `chat_cleared` - New chat started
- `message_edited` - Message updated

**Image handling**:
- `image_upload` - Base64 screenshot transmission

### Event Emitter Pattern

Frontend WebSocket client (`frontend/js/websocket.js`):
```javascript
websocket.on('chat_replaced', (data) => {
    chatStateManager.loadChatState(data.chat);
});
```

## Integration with Frontend

The API layer cleanly separates from frontend (`/frontend/`):
- API serves REST endpoints and WebSocket
- Frontend is static HTML/CSS/JS served by API
- No tight coupling - could use different frontend

See: `/frontend/docs/README.md` for frontend architecture.

## Integration with Agent

The API layer delegates to agent commands:
- `/api/chat/new` → triggers chat clear in agent
- `/api/audio/upload` → uses `shared/utils_audio.py` transcription
- LLM selection → uses `core/llm_router.py`

## Migration from Legacy

**Before** (visualresponse.py):
- Single file doing server + tool + WebSocket + UI
- Tight coupling, hard to test
- No frontend state management
- CLI-only voice interaction

**After** (api/ + frontend/):
- Clean separation of concerns
- Modular, testable endpoints
- Frontend state management (ChatStateManager)
- Browser-based voice via MediaRecorder

## Related Documentation

- Backend/Frontend architecture: `api/docs/BACKEND_FRONTEND_ARCHITECTURE.md`
- Chat state sync: `api/docs/CHAT_STATE_SYNC.md`
- Session sharing: `api/docs/SESSION_SHARING.md`
- Frontend integration: `/frontend/docs/README.md`
- Legacy architecture: `/docs/VISUALRESPONSE_ARCHITECTURE.md` (historical)
