# Agent SaaS API Documentation

This document describes the REST and WebSocket APIs for the Agent SaaS infrastructure.

## Base URL

```
http://<host>:8081
```

For production deployments with Traefik:
```
http://<domain>/api/agents
```

## Authentication

All endpoints (except `/health`) require JWT authentication via the `Authorization` header:

```
Authorization: Bearer <jwt-token>
```

The JWT token contains the user's UUID and is obtained from the main container manager's `/auth/login` endpoint.

## API Endpoints

### 1. List Available Agents

Get a list of all discoverable agents.

**Endpoint**: `GET /api/agents/list`

**Response**:
```json
{
  "status": "success",
  "count": 2,
  "agents": [
    {
      "name": "minimal_agent",
      "display_name": "Minimal Agent",
      "description": "A minimal agent with 4 essential tools",
      "icon": "⚡",
      "category": "analysis",
      "tool_count": 4,
      "tools": ["think", "todos", "readfile", "grepsearch"],
      "has_write_access": false,
      "docker_image": "cli-agent-minimal_agent:latest"
    },
    {
      "name": "developer_agent",
      "display_name": "Developer Agent",
      "description": "Full-featured development agent",
      "icon": "👨‍💻",
      "category": "development",
      "tool_count": 7,
      "tools": ["readfile", "writefile", "editfile", "grepsearch", "bash", "git", "github"],
      "has_write_access": true,
      "docker_image": "cli-agent-developer_agent:latest"
    }
  ]
}
```

**Fields**:
- `name`: Internal agent identifier (used in API calls)
- `display_name`: Human-readable name for UI
- `description`: Short description of agent capabilities
- `icon`: Emoji/icon for UI display
- `category`: Agent category (`development`, `analysis`, `cognitive`, `general`)
- `tool_count`: Number of available tools
- `tools`: List of tool names
- `has_write_access`: Whether agent can modify files/execute commands
- `docker_image`: Docker image name for container creation

---

### 2. Create Agent Session

Spawn a new agent container session for the authenticated user.

**Endpoint**: `POST /api/agents/sessions/create`

**Request Body**:
```json
{
  "user_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "agent_type": "minimal_agent"
}
```

**Response**:
```json
{
  "status": "success",
  "session": {
    "session_id": "sess_20231117120000_a1b2c3",
    "agent_type": "minimal_agent",
    "container_name": "agent-minimal_agent-abc123def456",
    "created_at": "2023-11-17T12:00:00Z",
    "status": "running"
  }
}
```

**Error Responses**:

`400 Bad Request` - Invalid agent type:
```json
{
  "detail": "Unknown agent type: invalid_agent. Available: ['minimal_agent', 'developer_agent']"
}
```

`429 Too Many Requests` - Session limit reached:
```json
{
  "detail": "Maximum sessions per user reached (5). Please terminate existing sessions."
}
```

`500 Internal Server Error` - Container creation failed:
```json
{
  "detail": "Session creation failed: <error message>"
}
```

**Notes**:
- Default session limit: 5 per user (configurable via `MAX_SESSIONS_PER_USER` env var)
- Container creation typically takes 2-5 seconds
- Session ID format: `sess_<timestamp>_<random>`

---

### 3. List User Sessions

Get all sessions for the authenticated user.

**Endpoint**: `GET /api/agents/sessions/list`

**Query Parameters**:
- `user_uuid` (required): User's UUID
- `status` (optional): Filter by status (`running`, `stopped`, `terminated`)

**Example**:
```
GET /api/agents/sessions/list?user_uuid=550e8400-e29b-41d4-a716-446655440000
GET /api/agents/sessions/list?user_uuid=550e8400-e29b-41d4-a716-446655440000&status=running
```

**Response**:
```json
{
  "status": "success",
  "count": 2,
  "sessions": [
    {
      "session_id": "sess_20231117120000_a1b2c3",
      "agent_type": "minimal_agent",
      "created_at": "2023-11-17T12:00:00Z",
      "last_activity": "2023-11-17T12:05:30Z",
      "status": "running"
    },
    {
      "session_id": "sess_20231117110000_d4e5f6",
      "agent_type": "developer_agent",
      "created_at": "2023-11-17T11:00:00Z",
      "last_activity": "2023-11-17T11:30:00Z",
      "status": "running"
    }
  ]
}
```

**Session Statuses**:
- `running`: Active container, accepting requests
- `stopped`: Container stopped but not deleted
- `terminated`: Container and resources deleted

---

### 4. Terminate Session

Stop and delete an agent session.

**Endpoint**: `DELETE /api/agents/sessions/{session_id}`

**Example**:
```
DELETE /api/agents/sessions/sess_20231117120000_a1b2c3
```

**Response**:
```json
{
  "status": "success",
  "session_id": "sess_20231117120000_a1b2c3"
}
```

**Notes**:
- Stops container (10 second timeout)
- Removes container
- Deletes all associated volumes (data, logs, config)
- Updates database status to `terminated`
- Idempotent: safe to call multiple times

---

### 5. WebSocket Proxy

Bi-directional WebSocket connection to agent container for real-time chat.

**Endpoint**: `WebSocket /api/agents/sessions/{session_id}/ws`

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:8081/api/agents/sessions/sess_20231117120000_a1b2c3/ws');
```

**Protocol**:

Client → Agent (send message):
```json
{
  "type": "chat_message",
  "content": "What files are in this directory?"
}
```

Agent → Client (response stream):
```json
{
  "type": "chat_response",
  "content": "Let me check...",
  "is_complete": false
}
```

```json
{
  "type": "chat_response",
  "content": "There are 5 files: main.py, config.json, README.md, ...",
  "is_complete": true
}
```

**WebSocket Events**:
- Client messages are forwarded to agent container
- Agent responses are streamed back to client
- Connection automatically updates `last_activity` timestamp
- Ping/pong heartbeat every 30 seconds

**Error Codes**:
- `1008`: Invalid session ID format
- `1011`: Session not found or proxy error

---

### 6. HTTP Proxy

HTTP proxy for agent container REST endpoints (chat history, metadata, etc.).

**Endpoint**: `GET|POST|PUT|DELETE|PATCH /api/agents/sessions/{session_id}/http/{path}`

**Example**:
```
GET /api/agents/sessions/sess_20231117120000_a1b2c3/http/chat/history
POST /api/agents/sessions/sess_20231117120000_a1b2c3/http/chat/send
```

**Notes**:
- All HTTP methods supported
- Query parameters preserved
- Request headers forwarded
- Response streamed back to client
- 30 second timeout
- Updates `last_activity` timestamp

**Error Responses**:
- `404 Not Found`: Session doesn't exist
- `502 Bad Gateway`: Agent container unreachable or error

---

### 7. Health Check

Check service health and stats.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "active_sessions": 12,
  "storage_usage": 45.2,
  "storage_available_gb": 120.5,
  "agents_available": 2
}
```

**Fields**:
- `status`: Service status (`healthy`, `degraded`, `unhealthy`)
- `active_sessions`: Number of running sessions
- `storage_usage`: Disk usage percentage
- `storage_available_gb`: Available disk space in GB
- `agents_available`: Number of discoverable agents

**Notes**:
- No authentication required
- Used by load balancers and monitoring systems
- Returns `200 OK` if healthy

---

## Error Handling

All errors follow a consistent format:

```json
{
  "detail": "Error message here"
}
```

**HTTP Status Codes**:
- `200 OK`: Success
- `400 Bad Request`: Invalid input (malformed UUID, unknown agent type, etc.)
- `401 Unauthorized`: Missing or invalid JWT token
- `404 Not Found`: Resource not found (session, agent, etc.)
- `429 Too Many Requests`: Rate limit or quota exceeded
- `500 Internal Server Error`: Server-side error
- `502 Bad Gateway`: Upstream service (agent container) error

---

## Rate Limits

**Session Creation**:
- Max 5 concurrent sessions per user (configurable)
- No rate limit on creation attempts

**API Requests**:
- No rate limits currently enforced
- WebSocket connections: max 1 per session

**Storage Limits**:
- When storage usage exceeds 90%, aggressive cleanup starts
- Oldest sessions terminated first

---

## Data Retention

**Session Retention**:
- Default: 7 days of inactivity
- Configurable per-session via `retention_days` field
- Automatic cleanup runs hourly

**Activity Tracking**:
- Every WebSocket message updates `last_activity`
- Every HTTP proxy request updates `last_activity`
- Manual activity updates not required

**Volume Cleanup**:
- Volumes deleted when session is terminated
- Orphaned volumes cleaned up during hourly audit
- Label-based discovery for robustness

---

## Container Isolation

**Network**:
- Each user gets isolated Docker network: `agent-saas-user-{hash}`
- Prevents cross-user container communication
- External API access allowed (for LLM providers)

**Resources**:
- CPU: 100% of 1 core (adjustable)
- Memory: 1GB limit
- Processes: 50 max
- Disk I/O: Best-effort (no hard limits)

**Security**:
- `no-new-privileges` enabled
- All capabilities dropped, only essential ones added
- AppArmor default profile applied
- Read-only root filesystem (coming soon)

---

## Examples

### Complete Session Workflow

```javascript
// 1. List available agents
const agentsResp = await fetch('/api/agents/list', {
  headers: { 'Authorization': 'Bearer ' + jwt }
});
const { agents } = await agentsResp.json();
console.log('Available agents:', agents);

// 2. Create session
const createResp = await fetch('/api/agents/sessions/create', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer ' + jwt,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    user_uuid: userUuid,
    agent_type: 'minimal_agent'
  })
});
const { session } = await createResp.json();
console.log('Session created:', session.session_id);

// 3. Connect WebSocket
const ws = new WebSocket(`ws://localhost:8081/api/agents/sessions/${session.session_id}/ws`);

ws.onopen = () => {
  console.log('WebSocket connected');
  ws.send(JSON.stringify({
    type: 'chat_message',
    content: 'Hello, agent!'
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log('Agent response:', msg.content);
};

// 4. Later: Terminate session
await fetch(`/api/agents/sessions/${session.session_id}`, {
  method: 'DELETE',
  headers: { 'Authorization': 'Bearer ' + jwt }
});
console.log('Session terminated');
```

---

## Integration with Main CLI-Agent

The Agent SaaS infrastructure is designed to run alongside the main CLI-Agent SaaS:

**Shared**:
- JWT authentication (same secret, same token format)
- User UUIDs (same localStorage-based identity)
- Traefik reverse proxy (different routes)

**Separate**:
- Database: `agent_sessions.db` vs `container_manager.db`
- Docker networks: `agent-saas-user-*` vs `cli-agent-user-*`
- API prefix: `/api/agents/*` vs `/api/backend/*`
- Frontend: `/agent-saas` vs `/`

**Traefik Routes**:
```yaml
# Main CLI-Agent
- "/api/backend/*" → container-manager:8080

# Agent SaaS
- "/api/agents/*" → agent-container-manager:8081
```

---

## Monitoring and Logging

**Logs Location**:
- `/logs/agent-saas/agent_container_manager.log`
- Rotating logs: 50MB max, 5 backups
- Format: `timestamp [level] name: message`

**Metrics** (Future):
- Active sessions per agent type
- Average session duration
- Storage usage trends
- API request rates

**Health Monitoring**:
- Endpoint: `GET /health`
- Check interval: 30 seconds recommended
- Alert on storage usage > 90%

---

## Troubleshooting

**Session Creation Fails**:
1. Check Docker images exist: `docker images | grep cli-agent-`
2. Check network exists: `docker network ls | grep agent-saas`
3. Check logs: `tail -f /logs/agent-saas/agent_container_manager.log`
4. Verify agent discovery: `python agent_discovery.py`

**WebSocket Connection Issues**:
1. Verify session exists: `GET /api/agents/sessions/list`
2. Check container status: `docker ps | grep agent-`
3. Test direct connection: `wscat -c ws://container-name:8000/ws`
4. Check firewall rules

**Storage Full**:
1. Check usage: `GET /health`
2. Manual cleanup: `DELETE /api/agents/sessions/{id}`
3. Prune orphaned volumes: `docker volume prune`
4. Increase retention days or reduce session limit

---

## Future Enhancements

- [ ] Session pause/resume
- [ ] Session snapshots and rollback
- [ ] Usage-based billing metrics
- [ ] WebRTC for low-latency communication
- [ ] Multi-container agents (agent + database + tools)
- [ ] Agent versioning and A/B testing
- [ ] Real-time session analytics
- [ ] Rate limiting per user tier
