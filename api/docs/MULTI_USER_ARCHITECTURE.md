# CLI-Agent Multi-User Architecture

**Version:** 1.0.0
**Status:** Production Ready
**Last Updated:** 2025-10-12

## Overview

CLI-Agent's multi-user architecture enables hosting as a web service where each user gets an isolated Docker container. The system uses UUID-based identity management for privacy and scalability, with automatic container lifecycle management and resource optimization.

---

## Architecture Diagram

```
Internet → Traefik (Port 80/443)
├── /* → Frontend Container (Nginx)
├── /api/manager/* → Container Manager
└── /api/backend/{uuid}/* → Backend Container (proxy via Container Manager)

Frontend Container (Single Instance)
├── Static files (HTML/CSS/JS)
├── js/identity-manager.js (UUID-based user identification)
├── js/backend-connector.js (discovers backend via UUID)
└── js/container-controls.js (wipe/reload functionality)

Container Manager (FastAPI + Gunicorn)
├── UUID → Container mapping (SQLite with WAL mode)
├── Docker API integration (create/destroy/monitor)
├── WebSocket proxy (/api/backend/{uuid}/ws)
├── HTTP proxy (REST endpoints)
├── Activity tracking & cleanup loops
├── Storage monitoring & aggressive cleanup
├── Per-container CPU quota adjustment
└── Orphaned volume audit

Backend Containers (cli-agent-{uuid_short})
├── main.py --api (API-only, no static files)
├── Session-scoped state (multi-tab support)
├── Dynamic port allocation (Docker assigns)
└── Volumes: cli-agent-{uuid_short}-{data|logs|config}
```

---

## Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Identity System** | UUID-based (client-generated) | Avoids IP NAT conflicts, spoofing, dynamic IP issues |
| **Short-term Identity** | UUID stored in localStorage | Allows MVP without user accounts; migrates to JWT later |
| **Container Strategy** | One per UUID | User isolation, resource fairness |
| **Multi-Tab Support** | Session-scoped contexts | Independent chat histories per browser tab |
| **Frontend Architecture** | Separate shared container | Enables container lifecycle control from UI |
| **Backend Mode** | API-only (no frontend) | Backends serve WebSocket + REST only |
| **Discovery** | Container Manager proxy | Frontend → Manager → Backend by UUID |
| **Reverse Proxy** | Traefik + Container Manager | Single wildcard rule in Traefik, Manager handles routing |
| **Port Allocation** | Docker dynamic (no fixed range) | Eliminates 100-container port limit |
| **Volume Cleanup** | Docker labels | Idempotent cleanup, no brittle name parsing |
| **CPU Limits** | Per-container monitoring | Fair quotas based on individual usage, not global load |
| **Storage Cleanup** | Proactive at <10% free | Delete oldest containers first |
| **State Persistence** | Local volumes (7d retention) | No centralized archive (future: LLM router) |

---

## Components

### 1. Frontend Container (Nginx)

**Purpose:** Serves static web UI to all users

**Key Features:**
- Single shared instance (not per-user)
- Nginx serving static files from `frontend/`
- Client-side UUID generation via `crypto.randomUUID()`
- Automatic backend discovery
- Container lifecycle controls (wipe/reload)

**JavaScript Modules:**
- `identity-manager.js` - UUID generation and persistence
- `backend-connector.js` - Backend container discovery and connection
- `container-controls.js` - Wipe/reload functionality
- `websocket.js` - Modified to use BackendConnector for WebSocket URL

**Mode Detection:**
- **Local Mode:** `localhost` / `127.0.0.1` → Direct WebSocket connection
- **Containerized Mode:** Production hostname → Proxied via Container Manager

### 2. Container Manager (FastAPI)

**Purpose:** Manages container lifecycle and routes requests

**Key Features:**
- **WebSocket Proxy:** Bi-directional proxy at `/api/backend/{uuid}/ws`
- **HTTP Proxy:** REST endpoint proxy to backend containers
- **Container Discovery:** Creates containers on-demand via `/discover` endpoint
- **Activity Tracking:** Updates `last_activity` on every request
- **Resource Management:**
  - Storage monitoring (triggers cleanup at 90% usage)
  - Per-container CPU throttling (fair, not global)
  - Orphaned volume audit (hourly)
- **Docker Labels:** Tags all resources with `cli-agent.managed=true` and `cli-agent.user-uuid={uuid}`
- **Production Grade:** Gunicorn with 4 uvicorn workers

**Endpoints:**
- `GET /discover` - Get or create container for UUID
- `WS /api/backend/{uuid}/ws` - WebSocket proxy to backend
- `ANY /api/backend/{uuid}/*` - HTTP proxy to backend
- `POST /activity` - Update activity timestamp
- `GET /containers` - List all containers (admin)
- `DELETE /container/{uuid}` - Destroy container
- `POST /container/{uuid}/wipe` - Wipe and recreate
- `GET /health` - Health check

**Background Tasks:**
- **Cleanup Loop (hourly):** Remove expired containers + orphaned volumes
- **CPU Adjustment Loop (15min):** Per-container CPU monitoring in thread pool
- **Storage Monitoring (10min):** Aggressive cleanup if <10% free

### 3. Backend Containers (Per-UUID)

**Purpose:** Isolated CLI-Agent instance per user

**Key Features:**
- Runs `main.py --api` (API-only mode, no frontend serving)
- Session-scoped state for multi-tab support
- Health endpoint at `/api/health`
- Dynamic port allocation (no fixed range)
- 3 labeled volumes per container:
  - `cli-agent-{uuid_short}-data`
  - `cli-agent-{uuid_short}-logs`
  - `cli-agent-{uuid_short}-config`

**Resource Limits:**
- CPU: 25-100% of one core (dynamic, per-container monitored)
- Memory: 2GB max
- Storage: ~500MB base + user data

**Lifecycle:**
- Created on first request (via Container Manager)
- Auto-cleanup after 7 days of inactivity
- Can be manually wiped via UI button

### 4. Traefik Reverse Proxy

**Purpose:** TLS termination, rate limiting, routing

**Key Features:**
- **Single wildcard rule:** `PathPrefix('/api/backend')` → Container Manager
- No per-container routing rules (scalable)
- Rate limiting: 100 req/min, burst 200
- Security headers (XSS, frame options, etc.)
- Gzip compression
- Health checks for all services

**Routes:**
- `/*` → Frontend Container (priority 1, lowest)
- `/api/manager/*` → Container Manager
- `/api/backend/*` → Container Manager (which proxies to backend)

---

## Critical Implementation Details

### 1. UUID-Based Identity

**Why not IP-based?**
- ❌ NAT collision: Corporate users share one IP
- ❌ Dynamic IPs: Mobile/residential IPs change
- ❌ IPv6 complexity: Multiple temporary addresses per device
- ❌ Spoofing risk: X-Forwarded-For can be manipulated

**UUID Solution:**
```javascript
// frontend/js/identity-manager.js
function getOrCreateUserUUID() {
    let uuid = localStorage.getItem('cli-agent-user-uuid');
    if (!uuid) {
        uuid = crypto.randomUUID();
        localStorage.setItem('cli-agent-user-uuid', uuid);
    }
    return uuid;
}
```

**Security Note:** UUID in localStorage can be accessed by another user on same machine/browser. Risk is low; full mitigation requires JWT authentication. Collision probability is negligible (2^122).

**Migration Path:** When user accounts are implemented, UUID migrates to JWT claim. No architectural changes needed.

### 2. WebSocket Proxying (Critical)

**Problem:** Standard HTTP clients like `httpx` don't handle WebSocket connections.

**Solution:** Dedicated WebSocket proxy using `websockets` library:

```python
# api/container_manager.py
@app.websocket("/api/backend/{uuid}/ws")
async def websocket_proxy(websocket: WebSocket, uuid: str):
    """Bi-directional WebSocket proxy to backend container"""
    await websocket.accept()

    container_info = manager.get_container(uuid)
    backend_ws_url = f"ws://{container_info.container_name}:7860/ws"

    async with websockets.connect(backend_ws_url) as backend_ws:
        async def forward_to_backend():
            while True:
                data = await websocket.receive_text()
                await backend_ws.send(data)

        async def forward_to_client():
            while True:
                data = await backend_ws.recv()
                await websocket.send_text(data)

        await asyncio.gather(forward_to_client(), forward_to_backend())
```

### 3. Docker Labels (Not Name Parsing)

**Why labels?**
- ❌ Name parsing (`volume.name.split('-')[2]`) is brittle
- ✅ Labels are robust key-value metadata on Docker objects

**Label Everything on Creation:**
```python
labels = {
    "cli-agent.managed": "true",
    "cli-agent.user-uuid": uuid,
    "cli-agent.created": datetime.utcnow().isoformat()
}

# Create volumes with labels
self.docker_client.volumes.create(name=vol_name, labels=labels)

# Create container with labels
container = self.docker_client.containers.run(..., labels=labels)
```

**Label-Based Cleanup:**
```python
# Efficient label filtering
volumes = self.docker_client.volumes.list(
    filters={"label": f"cli-agent.user-uuid={uuid}"}
)

# Orphaned volume audit
managed_volumes = self.docker_client.volumes.list(
    filters={"label": "cli-agent.managed=true"}
)
```

### 4. Non-Blocking CPU Stats

**Problem:** `container.stats(stream=False)` is synchronous and takes 1-2 seconds per container. Running in loop blocks FastAPI event loop.

**Solution:** Run in thread pool:

```python
from fastapi.concurrency import run_in_threadpool

async def adjust_container_cpu_quotas(self):
    """Non-blocking CPU adjustment"""
    await run_in_threadpool(self._adjust_cpu_quotas_sync)

def _adjust_cpu_quotas_sync(self):
    """Synchronous CPU adjustment (runs in thread pool)"""
    for container_info in self.list_active_containers():
        stats = container.stats(stream=False)  # Blocking call
        # Calculate CPU usage and adjust quota
```

**Alternative:** Increase check interval from 5min to 15-30min (sustained patterns).

### 5. Per-Container CPU Monitoring (Not Global Punishing)

**Old approach:** Global load → reduce ALL container quotas → punishes all users for one heavy user

**New approach:** Monitor per-container CPU usage → throttle only heavy users

```python
if cpu_percent > 90:
    quota = 50000  # 50% of one core (throttle heavy user)
elif cpu_percent > 70:
    quota = 75000  # 75%
else:
    quota = 100000  # 100% (full core)
```

---

## Deployment

### Prerequisites

- Docker + Docker Compose installed
- `.env` file with API keys (see `.env.example`)
- At least 8GB RAM (for infrastructure + multiple backend containers)
- 50GB+ disk space recommended

### Quick Start

```bash
# 1. Build all images
./scripts/build_all.sh

# 2. Deploy infrastructure
./scripts/deploy.sh

# 3. Test multi-user functionality
./scripts/test_multi_user.sh

# 4. Access the system
# - Frontend: http://localhost
# - Traefik Dashboard: http://localhost:8080
# - Container Manager: http://localhost/api/manager/containers
```

### Manual Deployment

```bash
# Build images
docker build -f docker_files/Dockerfile.container-manager -t cli-agent-manager:latest .
docker build -f docker_files/Dockerfile.backend -t cli-agent-backend:latest .
docker build -f docker_files/Dockerfile.frontend -t cli-agent-frontend:latest .

# Create network
docker network create cli-agent-network

# Start services
docker-compose -f docker_files/docker-compose.infrastructure.yml up -d

# Check health
curl http://localhost/api/manager/health
```

### Monitoring

```bash
# View all logs
docker-compose -f docker_files/docker-compose.infrastructure.yml logs -f

# Individual service logs
docker logs -f cli-agent-traefik
docker logs -f cli-agent-container-manager
docker logs -f cli-agent-frontend

# List all backend containers
curl http://localhost/api/manager/containers

# Docker resource usage
docker stats
```

### Stopping Services

```bash
# Stop all infrastructure
docker-compose -f docker_files/docker-compose.infrastructure.yml down

# Stop and remove volumes (wipes all data)
docker-compose -f docker_files/docker-compose.infrastructure.yml down -v

# Remove all backend containers
docker ps -a --filter "name=cli-agent-" --format "{{.ID}}" | xargs docker rm -f

# Remove all labeled volumes
docker volume ls --filter "label=cli-agent.managed=true" --format "{{.Name}}" | xargs docker volume rm
```

---

## Resource Management

### Storage

- **Monitoring:** Checks disk usage every 10 minutes
- **Threshold:** Triggers aggressive cleanup at 90% usage
- **Cleanup Strategy:** Deletes oldest containers first (by `created_at`)
- **Orphaned Volumes:** Audited hourly via label filters

### CPU

- **Per-Container Monitoring:** Every 15 minutes
- **Fair Throttling:** Only heavy users (>90% CPU) are throttled to 50%
- **Non-Blocking:** Runs in thread pool to keep event loop responsive

### Memory

- **Backend Limit:** 2GB per container
- **Container Manager:** 1GB limit, 512MB reserved
- **Frontend:** 256MB limit

### Network

- **Bridge Network:** `cli-agent-network` with subnet `172.20.0.0/16`
- **Dynamic Ports:** Docker assigns ports automatically (no fixed range)
- **Internal Communication:** Containers communicate via container names

---

## Security Considerations

### Current Implementation

- UUID-based identity (localStorage)
- Rate limiting (100 req/min, burst 200)
- Security headers (XSS protection, frame deny, etc.)
- Docker labels for resource tracking
- Non-root user in backend containers
- Health check authentication (none - public)

### Future Enhancements

- **JWT Authentication:** Replace UUID with JWT tokens
- **User Accounts:** Login system with password hashing
- **API Key Authentication:** For Container Manager admin endpoints
- **TLS/HTTPS:** Let's Encrypt integration (Traefik ready)
- **Network Policies:** Restrict container-to-container communication

### Known Limitations

- UUID in localStorage can be shared on same machine
- No authentication on Container Manager admin endpoints
- Traefik dashboard publicly accessible (disable in production)
- No request payload validation

---

## Troubleshooting

### Container Discovery Fails

**Symptom:** Frontend shows "Backend discovery failed"

**Causes:**
1. Container Manager not running
2. Network connectivity issues
3. UUID header missing

**Solution:**
```bash
# Check Container Manager health
curl http://localhost/api/manager/health

# Check Container Manager logs
docker logs cli-agent-container-manager

# Verify network
docker network inspect cli-agent-network
```

### WebSocket Connection Drops

**Symptom:** Chat messages not sending/receiving

**Causes:**
1. WebSocket proxy failing
2. Backend container stopped
3. Traefik routing issue

**Solution:**
```bash
# Check WebSocket proxy logs
docker logs cli-agent-container-manager | grep "WebSocket"

# Check backend container status
docker ps --filter "name=cli-agent-" --format "{{.Names}} {{.Status}}"

# Test WebSocket directly (requires wscat)
wscat -c ws://localhost/api/backend/{uuid}/ws
```

### Storage Fills Up

**Symptom:** Aggressive cleanup running but disk still full

**Causes:**
1. Orphaned volumes not cleaned
2. Docker image layers accumulating
3. Log files growing

**Solution:**
```bash
# Check disk usage
df -h

# List all volumes
docker volume ls

# Remove orphaned volumes
docker volume prune -f --filter "label=cli-agent.managed=true"

# Remove unused Docker images
docker image prune -a -f

# Check Container Manager logs size
docker logs cli-agent-container-manager 2>&1 | wc -l
```

### Backend Container Won't Start

**Symptom:** Container created but health check fails

**Causes:**
1. Missing dependencies in Dockerfile
2. Port conflict
3. Volume mount permissions

**Solution:**
```bash
# Check container logs
docker logs cli-agent-{uuid_short}

# Inspect container
docker inspect cli-agent-{uuid_short}

# Check health check status
docker ps --filter "health=unhealthy" --format "{{.Names}} {{.Status}}"

# Test health endpoint manually
docker exec cli-agent-{uuid_short} curl -f http://localhost:7860/api/health
```

---

## Performance Tuning

### Container Manager Workers

Default: 4 workers (gunicorn)

```dockerfile
# Adjust in docker_files/Dockerfile.container-manager
CMD ["gunicorn", "-w", "8", ...]  # Increase to 8 for high load
```

### CPU Adjustment Interval

Default: 15 minutes

```python
# Adjust in api/container_manager.py
async def cpu_adjustment_loop():
    await asyncio.sleep(600)  # Change to 10 minutes
```

### Cleanup Interval

Default: 1 hour

```python
# Adjust in api/container_manager.py
async def cleanup_loop():
    await asyncio.sleep(1800)  # Change to 30 minutes
```

### Backend Resource Limits

```yaml
# Adjust in docker_files/docker-compose.infrastructure.yml (for per-IP containers, modify container_manager.py)
mem_limit: "4g"  # Increase from 2g
cpu_quota: 200000  # Increase to 2 full cores
```

---

## Migration Path

### Current State (MVP)
- UUID-based identity in localStorage
- No user accounts
- No persistent chat archiving
- No centralized LLM routing

### Phase 2 (User Accounts) - Q2 2026
- JWT authentication
- UUID migrates to JWT claim (no architectural change)
- Login/signup system
- Custom retention per user
- Multi-device session sync

### Phase 3 (Centralized LLM Router) - Q3 2026
- Separate service for LLM API calls
- Chat history archiving to S3/database
- Cost tracking per user
- Usage quotas and billing

### Phase 4 (Advanced Features) - Q4 2026
- Container pause/resume (instead of destroy)
- Hot migration between hosts
- Prometheus metrics export
- Grafana dashboards
- Multi-region deployment

---

## Related Documentation

- [Deployment Guide](DEPLOYMENT.md) - Detailed deployment instructions
- [Backend/Frontend Architecture](BACKEND_FRONTEND_ARCHITECTURE.md) - API layer separation
- [Logging](LOGGING.md) - Logging system and retention policies
- [Docker Setup](DOCKER_ISOLATED.md) - Original single-container setup (legacy)

---

**For questions or issues, see:** `docs/TROUBLESHOOTING.md` (to be created)
