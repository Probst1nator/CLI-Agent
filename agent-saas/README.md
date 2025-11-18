# Agent SaaS Infrastructure

Multi-session agent container management system for CLI-Agent. Allows users to spawn, manage, and interact with individual agents (minimal_agent, developer_agent, etc.) in fully isolated Docker containers.

## Overview

The Agent SaaS infrastructure provides:

- **Multi-Agent Support**: Spawn different agent types (minimal, developer, custom)
- **Multi-Session**: Multiple concurrent agent sessions per user
- **Full Isolation**: Each session runs in its own Docker container
- **Real-time Communication**: WebSocket streaming for live agent responses
- **Session Management**: Create, list, and terminate agent sessions
- **Resource Management**: Automatic cleanup, storage monitoring, retention policies

## Architecture

```
User → Frontend (agent-saas) → Agent Container Manager → Individual Agent Containers
                                         ↓
                              Docker (isolated per-session)
```

**Components**:
- `agent_discovery.py`: Discovers available agents from `./agents/` directory
- `agent_container_manager.py`: FastAPI server managing container lifecycle
- `frontend_agent/`: Split-panel UI (sessions list + chat interface)
- `docker/`: Dockerfiles for agent images

**Separation from Main CLI-Agent**:
- Separate database: `/data/agent-saas/agent_sessions.db`
- Separate networks: `agent-saas-user-{hash}`
- Separate API routes: `/api/agents/*`
- Separate frontend: `/agent-saas`

## Quick Start

### 1. Install Dependencies

```bash
cd agent-saas
pip install -r requirements.txt
```

### 2. Build Agent Docker Images

First, ensure the agents exist:
```bash
ls ../agents/
# Should show: minimal_agent, developer_agent
```

Build images (Phase 2 - not yet implemented):
```bash
cd docker
./build-all.sh
```

### 3. Create Docker Network

```bash
docker network create agent-saas-network
```

### 4. Start Agent Container Manager

```bash
# Development mode (standalone)
python agent_container_manager.py

# Production mode (with Uvicorn)
uvicorn agent_container_manager:app --host 0.0.0.0 --port 8081 --reload
```

### 5. Test Agent Discovery

```bash
python agent_discovery.py
```

Output:
```
=== Discovered Agents ===

👨‍💻 Developer Agent
  Name: developer_agent
  Category: development
  Tools (7): bash, editfile, git, github, grepsearch, readfile, writefile
  ...

⚡ Minimal Agent
  Name: minimal_agent
  Category: analysis
  Tools (4): grepsearch, readfile, think, todos
  ...
```

## API Usage

### List Available Agents

```bash
curl http://localhost:8081/api/agents/list
```

### Create Session

```bash
curl -X POST http://localhost:8081/api/agents/sessions/create \
  -H "Content-Type: application/json" \
  -d '{
    "user_uuid": "550e8400-e29b-41d4-a716-446655440000",
    "agent_type": "minimal_agent"
  }'
```

Response:
```json
{
  "status": "success",
  "session": {
    "session_id": "sess_20231117120000_a1b2c3",
    "agent_type": "minimal_agent",
    "container_name": "agent-minimal_agent-abc123",
    "created_at": "2023-11-17T12:00:00Z",
    "status": "running"
  }
}
```

### List User Sessions

```bash
curl "http://localhost:8081/api/agents/sessions/list?user_uuid=550e8400-e29b-41d4-a716-446655440000"
```

### Connect via WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8081/api/agents/sessions/sess_20231117120000_a1b2c3/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Agent:', data.content);
};

ws.send(JSON.stringify({
  type: 'chat_message',
  content: 'Hello, agent!'
}));
```

### Terminate Session

```bash
curl -X DELETE http://localhost:8081/api/agents/sessions/sess_20231117120000_a1b2c3
```

## Configuration

### Environment Variables

```bash
# Session limits
MAX_SESSIONS_PER_USER=5           # Max concurrent sessions per user

# Storage monitoring
STORAGE_THRESHOLD=90              # Trigger cleanup at 90% usage

# Retention
DEFAULT_RETENTION_DAYS=7          # Auto-cleanup after 7 days inactivity

# CORS (for frontend)
CORS_ALLOWED_ORIGINS="http://localhost:7860,http://127.0.0.1:7860"

# LLM API keys (passed to agent containers)
GOOGLE_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### Database

Location: `/data/agent-saas/agent_sessions.db`

Schema:
```sql
-- Sessions table
CREATE TABLE agent_sessions (
    session_id TEXT PRIMARY KEY,
    user_uuid TEXT NOT NULL,
    agent_type TEXT NOT NULL,
    container_id TEXT NOT NULL,
    container_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_activity TEXT NOT NULL,
    status TEXT DEFAULT 'running',
    retention_days INTEGER DEFAULT 7
);

-- Activity log
CREATE TABLE activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT,
    FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
);
```

## Directory Structure

```
agent-saas/
├── agent_discovery.py              # Agent enumeration
├── agent_container_manager.py      # Container lifecycle manager (FastAPI)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── docs/
│   ├── API.md                      # API documentation
│   └── AGENT_SAAS_ARCHITECTURE.md  # Architecture deep-dive
├── frontend_agent/                 # Frontend UI (Phase 4)
│   ├── index.html
│   ├── css/
│   └── js/
└── docker/                         # Agent Docker images (Phase 2)
    ├── Dockerfile.agent-base
    ├── Dockerfile.minimal-agent
    ├── Dockerfile.developer-agent
    └── build-all.sh
```

## Container Isolation

Each agent session runs in a fully isolated Docker container:

**Network**:
- Isolated per-user network: `agent-saas-user-{uuid_hash}`
- No cross-user communication possible
- External API access allowed (for LLM providers)

**Resources**:
- CPU: 100% of 1 core
- Memory: 1GB limit
- Processes: 50 max
- Storage: Shared host disk (monitored)

**Security**:
- No new privileges
- Minimal capabilities (CHOWN, SETUID, SETGID only)
- AppArmor default profile
- Separate volumes per session

**Volumes**:
- `agent-{session_hash}-data`: Persistent data
- `agent-{session_hash}-logs`: Agent logs
- `agent-{session_hash}-config`: Configuration

## Lifecycle Management

### Session Creation
1. Validate user UUID and agent type
2. Check session limit (default: 5 per user)
3. Check storage availability
4. Create Docker volumes with labels
5. Spawn container with resource limits
6. Register in database
7. Wait for container readiness (max 30s)

### Activity Tracking
- Every WebSocket message updates `last_activity`
- Every HTTP request updates `last_activity`
- Used for retention policy enforcement

### Cleanup
- **Hourly**: Cleanup expired sessions (default: 7 days inactivity)
- **Continuous**: Storage monitoring (trigger aggressive cleanup at 90%)
- **On Termination**: Stop container, delete volumes, update database

### Retention Policy
- Default: 7 days of inactivity
- Configurable per-session
- Automatic cleanup via background task
- Manual termination available

## Monitoring

### Health Check

```bash
curl http://localhost:8081/health
```

Response:
```json
{
  "status": "healthy",
  "active_sessions": 12,
  "storage_usage": 45.2,
  "storage_available_gb": 120.5,
  "agents_available": 2
}
```

### Logs

Location: `/logs/agent-saas/agent_container_manager.log`

Format:
```
2023-11-17 12:00:00 [INFO] agent_container_manager: Session created: sess_20231117120000_a1b2c3
2023-11-17 12:05:30 [DEBUG] agent_container_manager: WebSocket connection for session sess_20231117120000_a1b2c3
2023-11-17 13:00:00 [INFO] agent_container_manager: Running cleanup loop...
```

### Metrics

Available via `/health` endpoint:
- `active_sessions`: Number of running sessions
- `storage_usage`: Disk usage percentage
- `storage_available_gb`: Free disk space
- `agents_available`: Number of discoverable agents

## Development

### Adding a New Agent

1. Create agent directory:
   ```bash
   mkdir ../agents/my_agent
   cd ../agents/my_agent
   ```

2. Create required files:
   ```
   my_agent/
   ├── main.py          # Agent loop
   ├── chat.py          # Message management
   ├── llm_router.py    # LLM provider
   ├── tools/           # Tool implementations
   ├── requirements.txt
   └── README.md
   ```

3. Agent automatically discovered on next scan:
   ```bash
   python agent_discovery.py
   ```

4. Build Docker image (Phase 2):
   ```bash
   cd docker
   ./build-agent-image.sh my_agent
   ```

### Testing

```bash
# Test agent discovery
python agent_discovery.py

# Test container manager (unit tests - TODO)
pytest tests/test_agent_container_manager.py

# Test API endpoints
curl http://localhost:8081/health
curl http://localhost:8081/api/agents/list
```

### Debugging

```bash
# View logs
tail -f /logs/agent-saas/agent_container_manager.log

# List running containers
docker ps | grep agent-

# Inspect container
docker inspect agent-minimal_agent-abc123

# View container logs
docker logs agent-minimal_agent-abc123

# Connect to container
docker exec -it agent-minimal_agent-abc123 /bin/bash
```

## Deployment

### Standalone (Development)

```bash
uvicorn agent_container_manager:app --host 0.0.0.0 --port 8081 --reload
```

### With Docker Compose (Production)

Add to `docker-compose.infrastructure.yml`:

```yaml
services:
  agent-container-manager:
    image: cli-agent-saas-manager:latest
    container_name: agent-container-manager
    restart: unless-stopped
    ports:
      - "8081:8081"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - agent-saas-data:/data
      - agent-saas-logs:/logs
    networks:
      - agent-saas-network
    environment:
      - MAX_SESSIONS_PER_USER=5
      - STORAGE_THRESHOLD=90
      - DEFAULT_RETENTION_DAYS=7
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
```

### Traefik Integration

```yaml
labels:
  - "traefik.enable=true"
  - "traefik.http.routers.agent-manager.rule=Host(`example.com`) && PathPrefix(`/api/agents`)"
  - "traefik.http.services.agent-manager.loadbalancer.server.port=8081"
```

## Security

### Authentication
- JWT tokens required for all endpoints (except `/health`)
- Tokens contain user UUID
- Issued by main container manager's `/auth/login`

### Container Isolation
- No shared networks between users
- No root privileges
- Minimal Linux capabilities
- AppArmor enforcement

### Input Validation
- UUID format validation (RFC 4122 UUID v4)
- Agent type validation (must exist in discovery)
- SQL injection prevention (parameterized queries)

### Resource Limits
- CPU: 100% of 1 core per container
- Memory: 1GB per container
- Processes: 50 per container
- Storage: Monitored globally (cleanup at 90%)

## Troubleshooting

### "Unknown agent type" Error
```bash
# Verify agent exists
ls ../agents/
python agent_discovery.py
```

### "Maximum sessions per user reached"
```bash
# List and terminate old sessions
curl "http://localhost:8081/api/agents/sessions/list?user_uuid=YOUR_UUID"
curl -X DELETE "http://localhost:8081/api/agents/sessions/SESS_ID"
```

### Container Won't Start
```bash
# Check Docker daemon
docker info

# Check image exists
docker images | grep cli-agent-

# Check logs
docker logs agent-minimal_agent-abc123
tail -f /logs/agent-saas/agent_container_manager.log
```

### WebSocket Connection Fails
```bash
# Check container is running
docker ps | grep agent-

# Check network connectivity
docker network inspect agent-saas-network

# Test direct connection
wscat -c ws://container-name:8000/ws
```

## Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Agent discovery system
- [x] Container lifecycle management
- [x] Database schema
- [x] API endpoints
- [x] Documentation

### Phase 2: Agent Containerization (Next)
- [ ] Dockerfile.agent-base
- [ ] Per-agent Dockerfiles
- [ ] Build scripts
- [ ] Agent server wrapper (CLI → WebSocket/HTTP)

### Phase 3: Real-time Communication
- [ ] WebSocket streaming implementation
- [ ] Chat persistence
- [ ] Session recovery

### Phase 4: Frontend UI
- [ ] Split-panel layout
- [ ] Session manager (left panel)
- [ ] Chat interface (right panel)
- [ ] Agent selector dialog

### Phase 5: Integration & Testing
- [ ] Traefik routing
- [ ] End-to-end tests
- [ ] Load testing
- [ ] Security audit

### Phase 6: Production-Ready
- [ ] Monitoring & alerting
- [ ] Usage metrics
- [ ] Billing integration
- [ ] Documentation updates

## Contributing

When adding features to the Agent SaaS infrastructure:

1. **Agent Discovery**: Update `agent_discovery.py` for new agent metadata
2. **Container Manager**: Update `agent_container_manager.py` for new lifecycle logic
3. **API**: Update `docs/API.md` for new endpoints
4. **Frontend**: Update `frontend_agent/` for UI changes

## Related Documentation

- `docs/API.md`: Complete API reference
- `docs/AGENT_SAAS_ARCHITECTURE.md`: Architecture deep-dive
- `../agents/CLAUDE.md`: Guide for creating agents
- `../docker_files/docs/DEPLOYMENT.md`: Deployment guide

## License

Same as main CLI-Agent project.

## Support

For issues and questions:
- Check logs: `/logs/agent-saas/agent_container_manager.log`
- Health check: `curl http://localhost:8081/health`
- API docs: `docs/API.md`
- Architecture: `docs/AGENT_SAAS_ARCHITECTURE.md`
