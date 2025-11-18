# Agent SaaS Architecture

Deep-dive into the design and implementation of the Agent SaaS infrastructure.

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [System Architecture](#system-architecture)
4. [Component Details](#component-details)
5. [Data Flow](#data-flow)
6. [Security Model](#security-model)
7. [Resource Management](#resource-management)
8. [Comparison with Main CLI-Agent](#comparison-with-main-cli-agent)
9. [Deployment Architecture](#deployment-architecture)
10. [Future Enhancements](#future-enhancements)

---

## Overview

The Agent SaaS infrastructure enables users to spawn and manage multiple isolated agent sessions, each running a different specialized agent (minimal, developer, custom) in its own Docker container.

**Key Goals**:
- **Multi-Session**: Support multiple concurrent agent sessions per user
- **Multi-Agent**: Support different agent types with different tool sets
- **Full Isolation**: Complete isolation between users and between sessions
- **Real-time**: WebSocket streaming for live agent responses
- **Scalable**: Efficient resource usage and automatic cleanup
- **Secure**: Robust isolation and input validation

---

## Design Principles

### 1. Clean Separation from Main CLI-Agent

The Agent SaaS infrastructure is designed to coexist with the existing CLI-Agent SaaS without conflicts:

| Aspect | Main CLI-Agent | Agent SaaS |
|--------|----------------|------------|
| **Purpose** | Full-featured production agent | Specialized agent experiments |
| **Session Model** | 1 container per user | N containers per user |
| **Base Image** | `cli-agent-backend:latest` | Per-agent images |
| **Database** | `container_manager.db` | `agent_sessions.db` |
| **Network** | `cli-agent-user-{hash}` | `agent-saas-user-{hash}` |
| **API Prefix** | `/api/backend/*` | `/api/agents/*` |
| **Frontend** | `/` | `/agent-saas` |

### 2. No Legacy Mindset

Following the project's core principle: clean breaks over backward compatibility.

- No symlinks to agents
- No compatibility layers between versions
- Complete migrations when agents change

### 3. Hierarchical Documentation

Documentation lives close to code:
- `agent-saas/README.md`: User-facing setup guide
- `agent-saas/docs/API.md`: API reference
- `agent-saas/docs/AGENT_SAAS_ARCHITECTURE.md`: This file (deep-dive)

### 4. Fail-Safe Cleanup

Robust cleanup mechanisms prevent resource leaks:
- Label-based Docker resource tracking
- Idempotent cleanup operations
- Background cleanup tasks
- Storage-aware aggressive cleanup

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                         User Browser                        │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTPS (WebSocket + HTTP)
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      Traefik (Reverse Proxy)                │
│  Routes:                                                    │
│    /api/agents/*    → Agent Container Manager (port 8081)  │
│    /agent-saas      → Frontend (Nginx)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│          Agent Container Manager (FastAPI)                  │
│  - Agent Discovery                                          │
│  - Session Management (CRUD)                                │
│  - WebSocket/HTTP Proxy                                     │
│  - Lifecycle Management                                     │
│  - Database: agent_sessions.db                              │
└────────────────────────┬────────────────────────────────────┘
                         │ Docker API
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      Docker Daemon                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Session 1    │  │ Session 2    │  │ Session 3    │      │
│  │ minimal_agent│  │ dev_agent    │  │ minimal_agent│      │
│  │              │  │              │  │              │      │
│  │ User A       │  │ User A       │  │ User B       │      │
│  │ Network: A   │  │ Network: A   │  │ Network: B   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Network Topology

```
Internet
    ↓
Traefik (80, 443)
    ↓
┌───────────────────────────────────────────────┐
│         agent-saas-network (bridge)           │
│                                               │
│  ┌──────────────────────┐                     │
│  │ Agent Container      │                     │
│  │ Manager (port 8081)  │────┐                │
│  └──────────────────────┘    │                │
│                              │                │
└──────────────────────────────┼────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
┌───────────────────────┐     ┌───────────────────────┐
│ agent-saas-user-abc   │     │ agent-saas-user-def   │
│ (User A's network)    │     │ (User B's network)    │
│                       │     │                       │
│ ├─ Session 1 (minimal)│     │ ├─ Session 3 (minimal)│
│ └─ Session 2 (dev)    │     │                       │
└───────────────────────┘     └───────────────────────┘
```

**Key Points**:
- Each user gets an isolated bridge network
- Containers from different users cannot communicate
- Container Manager can reach all containers via shared network
- Containers can access external APIs (for LLM providers)

---

## Component Details

### 1. Agent Discovery (`agent_discovery.py`)

**Purpose**: Automatically discover available agents from `./agents/` directory.

**Discovery Process**:
1. Scan `./agents/` for subdirectories
2. Validate agent structure (must have `main.py`, `tools/`)
3. Extract metadata (description, tools, category)
4. Return `AgentDefinition` objects

**AgentDefinition Schema**:
```python
@dataclass
class AgentDefinition:
    name: str                    # e.g., "minimal_agent"
    display_name: str            # e.g., "Minimal Agent"
    description: str             # From README/CLAUDE.md
    tools: list[str]             # ["think", "todos", "readfile", ...]
    path: str                    # Absolute path
    docker_image: str            # "cli-agent-minimal_agent:latest"
    icon: str                    # "⚡"
    category: str                # "analysis", "development", etc.
    tool_count: int              # 4
    has_write_access: bool       # False (no bash/writefile)
```

**Metadata Extraction**:
- **Description**: First paragraph from README.md or CLAUDE.md
- **Tools**: Scan `tools/*.py` for classes with `get_delim()` method
- **Category**: Determined by tool types (dev tools → "development")
- **Icon**: Based on category or agent name

**Singleton Pattern**:
```python
# Global instance for easy access
discovery = get_discovery()  # Auto-created on first call

# Refresh if agents added/removed
refresh_discovery()
```

### 2. Agent Container Manager (`agent_container_manager.py`)

**Purpose**: FastAPI server managing agent container lifecycle.

**Core Classes**:

```python
class AgentContainerManager:
    """Main manager class"""

    def __init__(self):
        self.docker_client = docker.DockerClient()
        self.discovery = get_discovery()
        self.init_database()

    async def create_session(user_uuid, agent_type) -> AgentSession
    async def terminate_session(session_id)
    def list_user_sessions(user_uuid) -> list[AgentSession]
    def get_session(session_id) -> AgentSession

    # Cleanup
    async def cleanup_expired_sessions()
    async def aggressive_cleanup()  # When storage critical
```

**Session Schema**:
```python
@dataclass
class AgentSession:
    session_id: str              # "sess_20231117120000_abc123"
    user_uuid: str               # RFC 4122 UUID v4
    agent_type: str              # "minimal_agent"
    container_id: str            # Docker container ID
    container_name: str          # "agent-minimal_agent-abc123"
    created_at: datetime
    last_activity: datetime
    status: str                  # "running", "stopped", "terminated"
    retention_days: int          # Default: 7
```

**Database Schema**:
```sql
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

CREATE INDEX idx_user_uuid ON agent_sessions(user_uuid);
CREATE INDEX idx_agent_type ON agent_sessions(agent_type);
CREATE INDEX idx_last_activity ON agent_sessions(last_activity);
CREATE INDEX idx_status ON agent_sessions(status);
```

**Background Tasks**:
- **Cleanup Loop** (hourly): Remove expired sessions based on retention policy
- **Storage Monitoring** (10 min): Trigger aggressive cleanup if storage > 90%

### 3. Storage Monitor

**Purpose**: Track disk usage and prevent storage exhaustion.

```python
class StorageMonitor:
    def get_usage_percent() -> float
    def get_available_gb() -> float
    def is_critical() -> bool  # Returns True if > threshold
```

**Cleanup Strategy**:
- **Normal**: Hourly cleanup of sessions idle > retention_days
- **Aggressive**: When storage > 90%, delete oldest sessions until < 85%
- **Priority**: Oldest sessions deleted first (FIFO)

### 4. Session ID Generation

**Format**: `sess_<timestamp>_<random>`

Example: `sess_20231117120000_a1b2c3d4e5f6`

**Components**:
- `sess_`: Prefix for easy identification
- `20231117120000`: YYYYMMDDHHmmss timestamp
- `a1b2c3d4e5f6`: 12 random hex characters (from `secrets.token_hex(6)`)

**Properties**:
- **Unique**: Timestamp + random ensures no collisions
- **Sortable**: Timestamp prefix allows chronological sorting
- **Secure**: Uses `secrets` module (cryptographically strong)

---

## Data Flow

### Session Creation Flow

```
1. User clicks "New Session" → Selects agent type

2. Frontend → POST /api/agents/sessions/create
   {
     "user_uuid": "550e8400-...",
     "agent_type": "minimal_agent"
   }

3. Container Manager validates:
   - UUID format (RFC 4122)
   - Agent type exists
   - Session limit not exceeded
   - Storage available

4. Generate session ID:
   sess_20231117120000_a1b2c3

5. Create Docker resources:
   - Volumes: agent-{hash}-data, -logs, -config
   - Network: agent-saas-user-{user_hash} (if not exists)
   - Container: agent-minimal_agent-{hash}

6. Container started with:
   - Resource limits (1GB RAM, 1 CPU core)
   - Security constraints (no-new-privileges, minimal caps)
   - Environment variables (API keys)
   - Volume mounts

7. Register in database:
   INSERT INTO agent_sessions (...)

8. Wait for container ready (max 30s)

9. Return session details to frontend

10. Frontend displays session in list
```

### WebSocket Communication Flow

```
1. User sends message in chat

2. Frontend → WebSocket /api/agents/sessions/{id}/ws
   {
     "type": "chat_message",
     "content": "What files are here?"
   }

3. Container Manager:
   - Validates session exists
   - Updates last_activity timestamp
   - Establishes WebSocket to agent container

4. Container Manager → Agent Container
   ws://{container_name}:8000/ws

5. Agent processes message:
   - Calls LLM with tools
   - Executes tools (readfile, grepsearch, etc.)
   - Streams response chunks

6. Agent → Container Manager → Frontend
   {
     "type": "chat_response",
     "content": "There are 5 files: ...",
     "is_complete": false
   }

7. Frontend renders streamed response in real-time

8. Final message:
   {
     "type": "chat_response",
     "content": "...",
     "is_complete": true
   }

9. Chat history saved in agent container volume
```

### Session Termination Flow

```
1. User clicks "Terminate" button

2. Frontend → DELETE /api/agents/sessions/{id}

3. Container Manager:
   - Get session from database
   - Stop Docker container (10s timeout)
   - Remove container
   - Remove volumes (using label filters)
   - Update database status to 'terminated'
   - Log activity event

4. Cleanup user network if no sessions remain

5. Return success to frontend

6. Frontend removes session from list
```

---

## Security Model

### Input Validation

**UUID Validation**:
```python
UUID_PATTERN = re.compile(
    r'^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$',
    re.IGNORECASE
)

def validate_uuid(uuid: str) -> str:
    if not uuid or len(uuid) > 36:
        raise HTTPException(400, "Invalid UUID")
    if not UUID_PATTERN.match(uuid):
        raise HTTPException(400, "Invalid UUID format")
    return uuid
```

**Agent Type Validation**:
- Must exist in discovered agents
- No path traversal characters allowed
- Whitelisted against discovery results

**SQL Injection Prevention**:
- All queries use parameterized statements
- No string concatenation for SQL

### Container Isolation

**Network Isolation**:
- Each user gets dedicated network: `agent-saas-user-{hash}`
- Containers from different users cannot communicate
- Network created with `internal=False` for LLM API access

**Resource Isolation**:
```python
container = docker_client.containers.run(
    # CPU: 100% of 1 core
    cpu_quota=100000,
    cpu_period=100000,

    # Memory: 1GB
    mem_limit="1g",

    # Processes: max 50
    pids_limit=50,

    # Security
    security_opt=[
        "no-new-privileges:true",
        "apparmor=docker-default"
    ],

    # Capabilities: minimal set
    cap_drop=["ALL"],
    cap_add=["CHOWN", "SETUID", "SETGID"]
)
```

**Volume Isolation**:
- Separate volumes per session
- Labels for tracking: `agent-saas.session-id={id}`
- Automatic cleanup on termination

### Authentication

**JWT Token Flow**:
1. User logs in via main container manager: `/auth/login`
2. Receives JWT token with user UUID
3. Includes token in `Authorization: Bearer {token}` header
4. Agent Container Manager validates token signature
5. Extracts user UUID from token payload

**Token Validation** (handled by main container manager):
```python
def verify_jwt_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload  # Contains 'uuid', 'email', 'exp', 'iat'
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
```

### Rate Limiting

**Session Creation**:
- Max 5 concurrent sessions per user (configurable)
- Checked before container creation
- Returns `429 Too Many Requests` if exceeded

**Storage-Based Throttling**:
- When storage > 90%, aggressive cleanup triggered
- Oldest sessions terminated first
- Prevents disk exhaustion

---

## Resource Management

### Container Resource Limits

| Resource | Limit | Rationale |
|----------|-------|-----------|
| **CPU** | 100% (1 core) | Fair share, prevents monopolization |
| **Memory** | 1GB | Sufficient for LLM calls, prevents OOM |
| **Processes** | 50 | Prevents fork bombs |
| **Disk I/O** | Best-effort | No hard limits (future enhancement) |

### Storage Management

**Volume Structure**:
```
/var/lib/docker/volumes/
├── agent-{hash}-data/
│   └── chat_history.json
├── agent-{hash}-logs/
│   └── agent.log
└── agent-{hash}-config/
    └── settings.json
```

**Cleanup Triggers**:
1. **Hourly**: Retention policy (delete if idle > 7 days)
2. **On Demand**: Manual termination via API
3. **Aggressive**: Storage > 90% threshold

**Label-Based Discovery**:
```python
# Find all volumes for a session
volumes = docker_client.volumes.list(
    filters={"label": f"agent-saas.session-id={session_id}"}
)

# Cleanup is idempotent and robust
for volume in volumes:
    volume.remove()
```

### Database Management

**Write-Ahead Logging (WAL)**:
```python
conn.execute("PRAGMA journal_mode=WAL")
```

Benefits:
- Concurrent reads during writes
- Better performance
- Crash recovery

**Vacuum Strategy**:
- Automatic (SQLite auto-vacuum enabled)
- Periodic manual vacuum (future enhancement)

---

## Comparison with Main CLI-Agent

| Feature | Main CLI-Agent | Agent SaaS |
|---------|----------------|------------|
| **Use Case** | Production AI agent with full feature set | Experimental specialized agents |
| **Session Model** | 1 container per user UUID | N containers per user UUID |
| **Base Image** | `cli-agent-backend:latest` (700MB+) | Per-agent images (200-400MB) |
| **Tools** | 15+ tools (bash, readfile, git, github, etc.) | 3-7 tools per agent |
| **Permissions** | Full permission system (`/perms` commands) | Trust-based (future: permissions) |
| **Frontend** | Full-featured (chat, audio, tools, settings) | Minimal (session list + chat) |
| **API Prefix** | `/api/backend/*` | `/api/agents/*` |
| **Database** | `container_manager.db` | `agent_sessions.db` |
| **Network** | `cli-agent-user-{hash}` | `agent-saas-user-{hash}` |
| **Port** | 7860 (dynamic) | 8000 (dynamic) |
| **Memory Limit** | 2GB | 1GB |
| **Session Limit** | 1 per user | 5 per user (configurable) |
| **Retention** | Indefinite (manual cleanup) | 7 days auto-cleanup |

**Why Separate?**

1. **Different Lifecycles**: Main CLI-Agent is long-running, agents are ephemeral
2. **Different Scale**: Main CLI-Agent expects 1:1 user:container, agents expect N:1
3. **Different Images**: Main CLI-Agent needs all tools, agents need minimal subsets
4. **Experimentation**: Agent SaaS allows rapid prototyping without affecting production

---

## Deployment Architecture

### Development (Standalone)

```bash
# Terminal 1: Start Agent Container Manager
cd agent-saas
uvicorn agent_container_manager:app --reload --port 8081

# Terminal 2: Test API
curl http://localhost:8081/api/agents/list
```

### Production (Docker Compose + Traefik)

**docker-compose.agent-saas.yml**:
```yaml
version: '3.8'

services:
  agent-container-manager:
    image: cli-agent-saas-manager:latest
    container_name: agent-container-manager
    restart: unless-stopped
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
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.agent-manager.rule=Host(`example.com`) && PathPrefix(`/api/agents`)"
      - "traefik.http.services.agent-manager.loadbalancer.server.port=8081"

networks:
  agent-saas-network:
    driver: bridge

volumes:
  agent-saas-data:
  agent-saas-logs:
```

**Traefik Routes**:
```yaml
# traefik/dynamic.yml
http:
  routers:
    agent-manager:
      rule: "PathPrefix(`/api/agents`)"
      service: agent-manager
      middlewares:
        - strip-agents-prefix
        - security-headers

  middlewares:
    strip-agents-prefix:
      stripPrefix:
        prefixes:
          - "/api/agents"

  services:
    agent-manager:
      loadBalancer:
        servers:
          - url: "http://agent-container-manager:8081"
```

---

## Future Enhancements

### Phase 2: Agent Containerization
- [ ] Create `Dockerfile.agent-base` (shared Python environment)
- [ ] Per-agent Dockerfiles (minimal, developer, custom)
- [ ] Agent server wrapper (convert CLI to WebSocket/HTTP)
- [ ] Build scripts and CI/CD integration

### Phase 3: Advanced Features
- [ ] Session pause/resume (stop container, preserve volumes)
- [ ] Session snapshots (volume backups)
- [ ] Session rollback (restore from snapshot)
- [ ] Session cloning (duplicate for experimentation)

### Phase 4: Monitoring & Analytics
- [ ] Prometheus metrics export
- [ ] Grafana dashboards (sessions, storage, errors)
- [ ] Usage-based billing metrics (LLM tokens, uptime)
- [ ] Real-time session analytics

### Phase 5: Advanced Isolation
- [ ] Read-only root filesystem
- [ ] Custom seccomp profiles
- [ ] Network egress filtering (whitelist LLM APIs only)
- [ ] Resource quotas per user (total CPU/memory across all sessions)

### Phase 6: Multi-Container Agents
- [ ] Agent + database (e.g., agent with PostgreSQL sidecar)
- [ ] Agent + tool containers (e.g., agent + code execution sandbox)
- [ ] Docker Compose per session

### Phase 7: Performance Optimizations
- [ ] Container pooling (pre-warm containers)
- [ ] Image layer caching
- [ ] Connection pooling for database
- [ ] Redis for session state caching

---

## Design Decisions

### Why FastAPI?
- **Async**: Native WebSocket support, async database queries
- **Performance**: Faster than Flask for high concurrency
- **Validation**: Pydantic models for request/response validation
- **Documentation**: Auto-generated OpenAPI docs

### Why SQLite?
- **Simplicity**: No separate database server required
- **Performance**: WAL mode supports concurrent reads
- **Portability**: Single file database
- **Sufficient**: Not expecting millions of sessions

**Future**: Migrate to PostgreSQL if:
- Need replication
- Need advanced analytics
- Scale beyond 10K sessions

### Why Docker SDK (not subprocess)?
- **Type Safety**: Python objects instead of string parsing
- **Error Handling**: Exceptions instead of return codes
- **Robustness**: Handles Docker API changes
- **Maintainability**: Easier to read and test

### Why Label-Based Cleanup?
**Problem**: Name-based cleanup is fragile:
```python
# Fragile: What if name format changes?
volumes = [v for v in all_volumes if v.name.startswith(f"agent-{hash}")]
```

**Solution**: Label-based is robust:
```python
# Robust: Labels are immutable metadata
volumes = docker_client.volumes.list(
    filters={"label": f"agent-saas.session-id={session_id}"}
)
```

Benefits:
- Survives container renames
- Handles orphaned resources
- Self-documenting

---

## Conclusion

The Agent SaaS infrastructure provides a clean, isolated, and scalable system for managing multiple specialized agent sessions. It coexists peacefully with the main CLI-Agent SaaS while enabling rapid experimentation with new agent types.

Key achievements:
- ✅ Clean separation from main infrastructure
- ✅ Multi-session support (5 per user)
- ✅ Multi-agent support (dynamic discovery)
- ✅ Full isolation (network, resources, storage)
- ✅ Automatic cleanup (retention policy + storage monitoring)
- ✅ Comprehensive API (REST + WebSocket)
- ✅ Production-ready architecture

Next steps: Phase 2 (Agent Containerization) to enable actual agent execution.
