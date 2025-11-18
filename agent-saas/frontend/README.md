# Agent Management SaaS - Frontend

Web-based UI for managing multiple agent sessions with real-time chat interface.

## Overview

Split-panel interface for the Agent SaaS infrastructure:
- **Left Panel**: Session list with create/terminate controls
- **Right Panel**: Real-time chat interface with selected agent

## Features

- ✅ **Session Management**: Create, list, select, and terminate agent sessions
- ✅ **Multi-Agent Support**: Choose from available agents (minimal, developer, etc.)
- ✅ **Real-time Chat**: WebSocket streaming for live agent responses
- ✅ **Tool Visualization**: See tool executions and results in real-time
- ✅ **Dark/Light Theme**: Toggle between themes (matches OLD system)
- ✅ **Fully Independent**: No dependencies on OLD CLI-Agent frontend

## Directory Structure

```
frontend/
├── index.html              # Main HTML structure
├── css/
│   └── main.css           # Styling (matches OLD system theme)
├── js/
│   ├── logger.js          # Logging utility (copied from OLD)
│   ├── identity-manager.js # UUID management (copied from OLD)
│   ├── session-manager.js  # Session list sidebar
│   ├── chat-interface.js   # Chat panel
│   ├── agent-websocket.js  # WebSocket client
│   └── main.js            # App initialization
├── serve.py               # Development HTTP server
└── README.md              # This file
```

## Quick Start (Standalone Testing)

### Option 1: Python HTTP Server

```bash
cd ~/repos/CLI-Agent/agent-saas/frontend
python3 serve.py

# Access at: http://localhost:8000
```

### Option 2: Any HTTP Server

```bash
cd ~/repos/CLI-Agent/agent-saas/frontend

# Node.js http-server
npx http-server -p 8000

# Python SimpleHTTPServer
python3 -m http.server 8000

# PHP built-in server
php -S localhost:8000
```

**Note**: Backend API calls to `/api/agents/*` will fail in standalone mode. Backend must be running for full functionality.

## API Integration

The frontend expects the Agent Container Manager API to be available at:

```
GET    /api/agents/list                        # List available agents
POST   /api/agents/sessions/create             # Create new session
GET    /api/agents/sessions/list?user_uuid=... # List user sessions
DELETE /api/agents/sessions/{session_id}       # Terminate session
WS     /api/agents/sessions/{session_id}/ws    # WebSocket chat
```

See `../docs/API.md` for complete API documentation.

## Components

### 1. Session Manager (`session-manager.js`)

Handles left sidebar session list:
- Fetches and displays user sessions
- Filters sessions by status (all/running/stopped)
- Creates new sessions (agent type selector)
- Terminates sessions
- Auto-refreshes every 30 seconds

### 2. Chat Interface (`chat-interface.js`)

Handles right panel chat:
- Displays selected session info
- Renders user/agent/system messages
- Formats markdown-like content
- Shows tool executions and results
- Clears chat history
- Manages input and send button states

### 3. Agent WebSocket (`agent-websocket.js`)

WebSocket client for agent communication:
- Connects to agent container via session ID
- Streams LLM responses in real-time
- Receives tool execution updates
- Handles reconnection (max 5 attempts)
- Updates connection status indicator

### 4. Main App (`main.js`)

Orchestrates all components:
- Initializes identity manager (UUID)
- Creates component instances
- Sets up theme toggle
- Auto-refreshes sessions
- Handles global errors

## Message Types

WebSocket protocol (Agent → Frontend):

```json
{
  "type": "llm_chunk",
  "content": "partial response...",
  "is_complete": false
}

{
  "type": "tool_execution",
  "tool_name": "readfile",
  "content": "/path/to/file"
}

{
  "type": "tool_result",
  "tool_name": "readfile",
  "content": "file contents..."
}

{
  "type": "final_response",
  "content": "complete agent response",
  "is_complete": true
}

{
  "type": "error",
  "content": "error message"
}

{
  "type": "history",
  "messages": [...]
}
```

WebSocket protocol (Frontend → Agent):

```json
{
  "type": "chat_message",
  "content": "user message"
}

{
  "type": "get_history"
}

{
  "type": "clear_history"
}
```

## Styling

CSS variables for theming (matching OLD system):

```css
:root {
    --bg-color: #121212;
    --surface-color: #1e1e1e;
    --text-color: #e0e0e0;
    --border-color: #333;
    --primary-color: #007acc;
    --status-running: #28a745;
    --status-stopped: #6c757d;
    --status-error: #dc3545;
}

html[data-theme='light'] {
    --bg-color: #f5f5f5;
    --surface-color: #ffffff;
    --text-color: #212121;
    /* ... */
}
```

Theme persisted in `localStorage`.

## Development

### Testing Locally

1. **Start backend** (Agent Container Manager):
   ```bash
   cd ~/repos/CLI-Agent/agent-saas
   uvicorn agent_container_manager:app --port 8081 --reload
   ```

2. **Start frontend**:
   ```bash
   cd frontend
   python3 serve.py
   ```

3. **Open browser**: `http://localhost:8000`

### Modifying Components

**Add new agent type display:**
Edit `session-manager.js`, update `createAgentTypeCard()` method.

**Change chat formatting:**
Edit `chat-interface.js`, update `formatContent()` method.

**Add new WebSocket message type:**
Edit `agent-websocket.js`, update `handleMessage()` switch statement.

### Browser Console

Enable debug logging:
```javascript
// In browser console
localStorage.setItem('logLevel', 'DEBUG');
location.reload();
```

## Deployment

### Production (Docker)

1. **Build frontend image**:
   ```bash
   cd ~/repos/CLI-Agent/docker_files
   docker build -f Dockerfile.agent-frontend -t cli-agent-agent-frontend:latest .
   ```

2. **Deploy via docker-compose**:
   ```yaml
   frontend-agents:
     image: cli-agent-agent-frontend:latest
     networks:
       - cli-agent-network
     labels:
       - "traefik.enable=true"
       - "traefik.http.routers.frontend-agents.rule=PathPrefix(`/agents/`)"
   ```

3. **Access**: `http://your-domain/agents/`

### Traefik Integration

Traefik routes:
- `/agents/` → Frontend (static files)
- `/agents/api/agents/*` → Agent Container Manager

See `../docker_files/traefik/dynamic.yml` for routing configuration.

## Known Limitations (Current MVP)

1. **No chat history persistence**: Chat cleared when switching sessions
2. **No markdown rendering**: Basic text formatting only
3. **No file uploads**: Agent can't receive files from user
4. **No audio support**: Text-only interface
5. **No session rename**: Sessions identified by ID only
6. **No session search**: Linear list only
7. **No resource monitoring**: No CPU/memory display

These are intentional for MVP and can be added in future phases.

## Troubleshooting

### "Disconnected" status
- Check Agent Container Manager is running on port 8081
- Verify WebSocket URL is correct
- Check browser console for errors

### Sessions not loading
- Verify `/api/agents/sessions/list` returns data
- Check User UUID is valid
- Confirm backend database is accessible

### Agent creation fails
- Check available agents: `/api/agents/list`
- Verify agent Docker images are built
- Review backend logs for container creation errors

### WebSocket connection fails
- Check session exists and is running
- Verify agent container is reachable
- Review browser console WebSocket errors

## Related Documentation

- `../README.md` - Agent SaaS overview
- `../docs/API.md` - Complete API reference
- `../docs/AGENT_SAAS_ARCHITECTURE.md` - System architecture
- `../../frontend/CLAUDE.md` - OLD frontend (for comparison)

## License

Same as main CLI-Agent project.
