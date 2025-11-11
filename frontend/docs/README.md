# Frontend Documentation

The frontend provides a modern, browser-based interface for CLI-Agent with real-time chat, voice interaction, and tool management.

## Architecture Overview

Modern frontend architecture separated from backend (January 2025):
- **Static HTML/CSS/JS** served by backend
- **WebSocket** for real-time communication
- **State management** with ChatStateManager
- **Modular JavaScript** with clean separation of concerns

## Directory Structure

```
frontend/
├── index.html           # Main entry point
├── css/
│   ├── main.css         # Core styles (~3,344 lines)
│   └── components/
│       └── loading-tools.css  # Tool execution indicators
├── js/
│   ├── main.js          # Main controller
│   ├── websocket.js     # WebSocket client
│   ├── chat-state.js    # State management (ChatStateManager)
│   ├── audio-manager.js # Voice interaction (AudioManager)
│   ├── chat-history.js  # Chat sidebar with BM25 search
│   ├── screen-capture.js # Screen/window/tab capture
│   ├── llm-selector.js  # Model selection UI
│   ├── tool-manager.js  # Tool configuration UI
│   ├── renderers.js     # Content rendering (Markdown, code, viz)
│   ├── logger.js        # Frontend logging
│   └── js-utils.js      # Shared utilities
└── libs/                # Third-party libraries
    ├── jquery/
    ├── marked/          # Markdown parsing
    ├── mermaid/         # Diagram rendering
    ├── plotly/          # Chart visualization
    └── 3dmol/           # Molecular visualization
```

## Core Modules

### Main Controller (`js/main.js`)

Central coordination for all frontend functionality:
- **WebSocket initialization** and connection management
- **Event handlers** for user interactions
- **Screen capture** base64 transmission via `handleScreenCaptureBase64()`
- **Module coordination** between chat, audio, tools, LLM selector

### State Management (`js/chat-state.js`)

**ChatStateManager class** - Frontend state without page reloads:

**Features**:
- Message state management (add, update, clear)
- **Chat loading** from archives without reload
- **WebSocket sync** handlers (`chat_replaced`, `chat_cleared`)
- Error handling with user feedback

**Key methods**:
```javascript
loadChatState(chatData)  // Load archived chat
clearChat()              // Start new chat
addMessage(role, content) // Add message to state
updateMessage(index, content) // Edit existing message
```

**Integration**:
- Syncs with backend via `/api/chat/load/{id}`
- Responds to WebSocket events from backend
- Updates UI in real-time

See: `frontend/docs/STATE_MANAGEMENT.md`

### Audio Manager (`js/audio-manager.js`)

**AudioManager class** (~300 lines) - Browser-based voice interaction:

**Features**:
- **Recording**: Browser microphone via MediaRecorder API
- **Upload**: Audio chunks to `/api/audio/upload`
- **Transcription**: Backend Whisper integration
- **TTS playback**: Streaming audio from `/api/audio/tts`
- **Conflict handling**: Stop TTS when user speaks
- **Resource cleanup**: Proper stream/blob management

**Workflow**:
1. User clicks voice button
2. Browser requests microphone permission
3. MediaRecorder captures audio
4. Upload to backend for transcription
5. Display transcribed text
6. Send to agent for processing
7. Receive TTS response
8. Play audio in browser

### WebSocket Client (`js/websocket.js`)

Event emitter pattern for real-time communication:

**Features**:
- **Connection management** with auto-reconnect
- **Session initialization** (stores session_id)
- **Event routing** for message streaming and chat sync
- **Session ID propagation** to HTTP requests

**Event handlers**:
```javascript
websocket.on('message_chunk', handleChunk);
websocket.on('chat_replaced', handleChatLoad);
websocket.on('message_edited', handleEdit);
```

### Chat History Sidebar (`js/chat-history.js`)

**Features** (~320 lines):
- **BM25 search** with snippet highlighting
- **Archive browsing** with date grouping
- **Chat loading** via ChatStateManager
- **New chat** creation
- **Delete** functionality
- **Real-time sync** with WebSocket events

**Integration**:
- Uses ChatStateManager for state updates
- Calls `/api/chat/load/{id}` and `/api/chat/new`
- Highlights search matches in snippets

### Screen Capture (`js/screen-capture.js`)

**ScreenCaptureManager class** - Browser screen/window/tab capture:

**Features**:
- **Screen Capture API** with cross-browser support (Chrome, Firefox)
- **Permission handling** with user prompts
- **Base64 generation** for WebSocket transmission
- **Multiple capture modes**: screen, window, tab

**Workflow**:
1. User clicks screenshot button
2. Browser shows capture picker
3. User selects screen/window/tab
4. Capture frame as image
5. Convert to base64
6. Send via WebSocket `image_upload` event

## Content Rendering

### Renderers (`js/renderers.js`)

Handles rich content display:

**Markdown rendering**:
- Uses Marked.js library
- Syntax highlighting for code blocks
- Proper escaping and sanitization

**Visualizations**:
- **Plotly charts** - Interactive graphs
- **Mermaid diagrams** - Flowcharts, sequence diagrams
- **3Dmol** - Molecular structures

**Code blocks**:
- Syntax highlighting
- Copy button
- Language detection

## UI Components

### LLM Selector (`js/llm-selector.js`)

Model selection interface:
- **Planner LLM** dropdown
- **Router LLM** dropdown
- **Fallback models** multi-select
- **Profile slots** (1-6) with save/load
- Real-time sync with backend

### Tool Manager (`js/tool-manager.js`)

Tool configuration interface:
- **Enable/disable** tools
- **Permission management**
- **Tool discovery** from backend
- **File uploads** for tool data

## Styling

### Main Stylesheet (`css/main.css`)

~3,344 lines of comprehensive styling:
- **Dark/light themes** with CSS variables
- **Responsive design** for mobile/desktop
- **Chat interface** with message bubbles
- **Tool toggles** and loading indicators
- **Animation** for smooth interactions

### Component Styles

**Loading Tools** (`css/components/loading-tools.css`):
- Tool execution indicators
- Expand/collapse animations
- Progress feedback

## Third-Party Libraries

Located in `frontend/libs/`:

- **jQuery** - DOM manipulation
- **Marked** - Markdown parsing
- **Mermaid** - Diagram rendering
- **Plotly** - Chart visualization
- **3Dmol** - Molecular visualization
- **svg-pan-zoom** - SVG interaction

## WebSocket Events

### Events Handled by Frontend

**Message streaming**:
- `message_start` - LLM begins response
- `message_chunk` - Incremental content
- `message_end` - Response complete

**Chat state**:
- `chat_replaced` - Archived chat loaded
- `chat_cleared` - New chat started
- `message_edited` - Message updated

**Images**:
- `image_upload` - Screenshot transmitted

## Integration with Backend

Clean separation via API:
- **REST endpoints**: `/api/chat/*`, `/api/audio/*`, `/api/tools/*`
- **WebSocket**: Real-time events at `/ws`
- **Static serving**: Frontend assets served by backend

See: `/api/docs/README.md`

## Key Features

### Chat Without Reload

ChatStateManager enables:
- Load archived chats instantly
- Edit messages in place
- Start new chats without page refresh
- Seamless state transitions

### Voice Interaction

AudioManager enables:
- Browser-based recording (no plugins)
- Transcription via backend Whisper
- TTS playback in browser
- Full voice conversation mode

### Screen Capture

ScreenCaptureManager enables:
- Capture screen/window/tab from browser
- No desktop app required
- Direct integration with agent vision models
- Base64 transmission via WebSocket

## Migration from Legacy

**Before** (tools/visual_assets/):
- Frontend mixed with tool implementation
- No state management (page reloads required)
- CLI-only voice
- Limited chat history

**After** (frontend/):
- Clean separation from backend
- Full state management
- Browser-based voice
- Rich chat history with search

## Development

### Adding New Features

1. **Create module** in `frontend/js/`
2. **Import in main.js**
3. **Add styles** to appropriate CSS file
4. **Integrate** with WebSocket for real-time updates

### Testing

- Browser console for JavaScript errors
- Network tab for API calls
- WebSocket frames for real-time events
- Frontend logging via logger.js

## Related Documentation

- Backend API: `/api/docs/README.md`
- State management: `frontend/docs/STATE_MANAGEMENT.md`
- Backend/Frontend separation: `/api/docs/BACKEND_FRONTEND_ARCHITECTURE.md`
- WebSocket events: `/api/docs/CHAT_STATE_SYNC.md`
