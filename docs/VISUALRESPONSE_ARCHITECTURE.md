# Visual Response System Architecture

## Overview

The CLI-Agent Visual Response system provides a sophisticated web-based interface for real-time interaction with AI agents. It implements a multi-process, event-driven architecture with comprehensive security, real-time communication, and container orchestration capabilities.

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Main Process  │◄──►│  Visual Server   │◄──►│  Browser Client │
│   (CLI Agent)   │    │   (Subprocess)   │    │   (Frontend)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   Tool Execution          WebSocket Server         React/Vue Support
   Queue Management        HTTP API Endpoints       Real-time Updates
   Permission System       Static Asset Serving     Interactive UI
```

### Technology Stack

**Server-Side:**
- **Language**: Python 3.8+
- **Web Framework**: aiohttp (async)
- **Process Management**: multiprocessing
- **Communication**: WebSocket + REST APIs
- **Containerization**: Docker

**Client-Side:**
- **Base**: HTML5, CSS3, ES6+ JavaScript
- **Libraries**: jQuery, Marked, Mermaid, Plotly, 3DMol.js
- **Frameworks**: React 18, Vue 3 (for dynamic content)
- **Build Tools**: Babel (runtime compilation)

## Component Architecture

### 1. Server Components

#### A. Visual Server Session (`tools/visualresponse.py`)

**Primary Responsibilities:**
- Subprocess lifecycle management
- Queue-based inter-process communication
- Health monitoring and auto-restart
- Browser session coordination

**Key Classes:**
```python
class VisualServerSession:
    """Main server session manager"""
    def __init__(self):
        self.output_queue = multiprocessing.Queue()  # Main → Server
        self.input_queue = multiprocessing.Queue()   # Server → Main
        self.process = multiprocessing.Process(...)

class visualresponse:
    """Tool interface for content creation"""
    @staticmethod
    def run(content: str) -> str:
        # Parses <type> and <content> tags
        # Creates unique visualization IDs
        # Sends to browser via session
```

#### B. Web Server Process (`_server_process_main_loop`)

**aiohttp Application Structure:**
```python
app = web.Application()
app.router.add_get('/', index_handler)              # Main UI
app.router.add_get('/ws', websocket_handler)        # WebSocket endpoint
app.router.add_get('/api/llms', llm_data_handler)   # LLM model data
app.router.add_get('/api/tools', tools_handler)     # Tool enumeration
app.router.add_post('/upload', upload_handler)      # File uploads
app.router.add_static('/css/', ...)                 # Static assets
app.router.add_static('/js/', ...)                  # JavaScript modules
```

**WebSocket Message Types:**
- `config` - Connection initialization and continue mode
- `permission_request` - Tool permission prompts
- `user_message` - User input from web interface
- `assistant_message` - AI responses and visualizations
- `console_log` - Real-time console output streaming
- `loading_tool_start/complete` - Live tool execution status
- `interrupt_signal` - User cancellation requests

### 2. Client Components

#### A. Application Controller (`js/main.js`)

**Core Class:**
```javascript
class VisualResponseApp {
    constructor() {
        this.contentRenderer = new ContentRenderer();
        this.websocketManager = new WebSocketManager();
        this.llmSelector = new LLMSelector();
        this.toolManager = new ToolManager();
    }

    // Coordinates all components
    // Manages application lifecycle
    // Handles theme switching and preferences
}
```

#### B. WebSocket Manager (`js/websocket.js`)

**Key Features:**
- Automatic reconnection with exponential backoff
- Message history persistence and replay for new connections
- Live tool execution tracking with real-time updates
- Console log streaming with ANSI sequence cleaning

**Message Flow:**
```javascript
// Outbound: User → Server
{
    type: 'user_message',
    message: 'User input text',
    toolConfig: { disabledTools: [...] }
}

// Inbound: Server → Client
{
    type: 'assistant_message',
    action: 'create',
    id: 'viz-1234567890',
    content: '<html>...</html>',
    model: 'gemini-2.5-flash',
    tool_info: [...]
}
```

#### C. Content Renderer (`js/renderers.js`)

**Security Features:**
- Sandboxed iframe execution for HTML content
- JavaScript enable/disable controls per message
- Content Security Policy enforcement
- Script tag removal and event handler stripping

**Rendering Pipeline:**
```javascript
function renderHTML(targetElement, htmlContent, messageId) {
    // 1. Decode HTML entities if needed
    // 2. Check JavaScript execution permissions
    // 3. Create sandboxed iframe or safe container
    // 4. Apply security filtering
    // 5. Render with appropriate restrictions
}
```

#### D. Component Modules

**LLM Selector (`js/llm-selector.js`):**
- Dynamic model discovery and selection
- Real-time sync with server configuration
- Support for router, planner, and fallback models

**Tool Manager (`js/tool-manager.js`):**
- Dynamic tool discovery and enumeration
- Per-tool enable/disable controls
- Integration with user preferences

**Logger (`js/logger.js`):**
- Client-side logging with multiple levels
- Performance timing and metrics
- Error tracking and reporting

### 3. Multi-Instance Architecture

#### A. Session Manager (`scripts/session_manager.py`)

**FastAPI-based Orchestration:**
```python
class InstanceManager:
    @staticmethod
    def get_available_instance() -> Optional[str]:
        # 1. Check existing idle instances
        # 2. Create new instance if under limit
        # 3. Use least loaded as fallback

    @staticmethod
    def create_new_instance() -> str:
        # 1. Generate unique instance ID
        # 2. Create Docker container
        # 3. Setup volumes and networking
        # 4. Register in health tracking
```

#### B. Container Orchestration (`docker_files/`)

**Docker Compose Structure:**
```yaml
services:
  session-manager:          # Port 7860 - Main entry point
    image: cli-agent-session-manager

  cli-agent-instance-1:     # Port 7861 - Instance 1
    image: cli-agent:isolated
    volumes:
      - cli-agent-data-1:/workspace/data
      - cli-agent-logs-1:/workspace/logs
      - cli-agent-config-1:/workspace/.cli-agent
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

## Communication Protocol

### 1. WebSocket Protocol

**Connection Lifecycle:**
```javascript
// 1. Connection establishment
ws = new WebSocket('ws://localhost:7860/ws');

// 2. Configuration exchange
{
    type: 'config',
    action: 'connection_init',
    continue_mode: true/false
}

// 3. Message history replay (if continue_mode=true)
// 4. Real-time bidirectional communication
// 5. Graceful disconnection handling
```

### 2. REST API Endpoints

**LLM Data API (`/api/llms`):**
```json
{
    "status": "success",
    "models": [
        {
            "id": "gemini-2.5-flash",
            "name": "gemini-2.5-flash",
            "provider": "Google",
            "context_window": 1000000,
            "capabilities": ["vision", "reasoning"],
            "status": "available"
        }
    ],
    "current_configuration": {
        "selected_llms": ["gemini-2.5-flash", "claude-3-sonnet"],
        "planner_llm": "gemini-2.5-flash",
        "router_llm": null
    }
}
```

**Tools API (`/api/tools`):**
```json
{
    "status": "success",
    "tools": [
        {
            "name": "readfile",
            "delim": "readfile",
            "displayName": "Read File",
            "description": "Read contents of a file",
            "example": "<readfile>path/to/file.txt</readfile>"
        }
    ]
}
```

### 3. File Upload Protocol

**Multipart Upload (`/upload`):**
- Filename sanitization and collision handling
- Temporary storage in `g.CLIAGENT_TEMP_STORAGE_PATH`
- Security validation and size limits
- Return absolute file paths for tool usage

## Security Architecture

### 1. Content Security Policy

```http
Content-Security-Policy:
    default-src 'self';
    script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com https://cdn.plot.ly;
    style-src 'self' 'unsafe-inline';
    img-src 'self' data: blob: https://* http://localhost:*;
    frame-src 'self' data: blob:;
    connect-src 'self' ws: wss:;
```

### 2. JavaScript Sandboxing

**Multi-Level Control:**
- Global JavaScript toggle for all messages
- Per-message JavaScript enable/disable
- Iframe sandboxing without `allow-scripts` for disabled content
- Script tag removal and event handler stripping

**Safe Rendering Pipeline:**
```javascript
if (!jsEnabled && hasJS) {
    // Remove <script> tags
    safeContent = content.replace(/<script[\s\S]*?<\/script>/gi, '<!-- Script removed -->');
    // Remove event handlers
    safeContent = safeContent.replace(/\son\w+\s*=\s*["'][^"']*["']/gi, '');
    // Remove javascript: URLs
    safeContent = safeContent.replace(/javascript:[^"'\s>]*/gi, '#');
}
```

### 3. Input Validation

**File Upload Security:**
- Filename sanitization: `[a-zA-Z0-9 ._-]` only
- Path traversal prevention
- Unique filename generation for collisions
- Temporary directory isolation

**Message Validation:**
- JSON schema validation for WebSocket messages
- Type checking for message fields
- Size limits on content (implicit)

## Performance Characteristics

### 1. Resource Usage

**Per Instance:**
- Memory: 2GB limit (Docker)
- CPU: 1 core limit
- Storage: Persistent volumes for data/logs/config
- Network: WebSocket + HTTP traffic

**Scaling Limits:**
- Maximum instances: Configurable (default 5)
- Session timeout: 30 minutes default
- Connection monitoring: 30-second intervals

### 2. Performance Optimizations

**Client-Side:**
- Lazy loading of heavy libraries
- Efficient DOM manipulation with minimal reflows
- WebSocket connection pooling and reuse
- Local storage for message history persistence

**Server-Side:**
- Async I/O with aiohttp
- Process isolation for fault tolerance
- Queue-based IPC for reliability
- Static asset caching with appropriate headers

## Deployment Patterns

### 1. Single Instance Deployment

```bash
# Direct execution
python main.py --gui

# Docker container
docker run -p 7860:7860 cli-agent:isolated python main.py --gui
```

### 2. Multi-Instance Deployment

```bash
# Docker Compose orchestration
docker-compose -f docker-compose.multi-simple.yml up -d

# Includes:
# - Session manager (port 7860)
# - 3 CLI-Agent instances (ports 7861-7863)
# - Automatic session routing
# - Health monitoring and restart
```

### 3. Production Deployment

**Requirements:**
- Docker and Docker Compose
- Persistent storage volumes
- Network connectivity for AI APIs
- Resource monitoring and alerting

**Security Considerations:**
- Firewall configuration for exposed ports
- TLS termination at reverse proxy
- Container security scanning
- Log aggregation and monitoring

## Monitoring and Observability

### 1. Health Checks

**Container Level:**
```bash
# Health check command
curl -f http://localhost:7860/ || exit 1

# Intervals: 30s check, 10s timeout, 3 retries
```

**Application Level:**
- WebSocket connection monitoring
- Process lifecycle tracking
- Queue depth monitoring
- Memory and CPU usage tracking

### 2. Logging

**Log Streams:**
- Main process: Agent execution logs
- Visual server: HTTP/WebSocket access logs
- Browser console: Client-side error logs
- Container logs: System-level diagnostics

**Log Aggregation:**
- File-based logging with rotation
- Real-time streaming to browser console
- Structured logging with timestamps and IDs

## Future Architectural Considerations

### 1. Scalability Enhancements

**Horizontal Scaling:**
- Load balancer integration
- Session affinity management
- Distributed state storage
- Auto-scaling based on demand

**Performance Improvements:**
- CDN integration for static assets
- Redis caching for session data
- Database persistence for message history
- WebSocket connection pooling

### 2. Observability Improvements

**Metrics and Monitoring:**
- Prometheus metrics collection
- Grafana dashboards for visualization
- Distributed tracing with OpenTelemetry
- Error tracking with Sentry

**Operational Improvements:**
- Blue-green deployment support
- Rolling updates with zero downtime
- Backup and disaster recovery
- Compliance and audit logging

### 3. Security Enhancements

**Advanced Security:**
- OAuth2/OIDC authentication
- Role-based access control (RBAC)
- API rate limiting and throttling
- Advanced threat detection

**Compliance:**
- GDPR data protection compliance
- SOC 2 Type II audit readiness
- PCI DSS for payment processing
- HIPAA for healthcare data

This architecture provides a solid foundation for a production-ready AI agent interface with strong security, scalability, and maintainability characteristics.