# 🔴 CRITICAL SECURITY WARNING - NOT PRODUCTION-READY

**Status:** ⚠️ **EXPERIMENTAL - LOCAL USE ONLY**
**Last Updated:** 2025-10-21
**Severity:** CRITICAL vulnerabilities make this unsuitable for production deployment

---

## ⚠️ IMPORTANT NOTICE

**This application is NOT production-ready and should ONLY be used:**
- ✅ On trusted local networks (e.g., 192.168.x.x)
- ✅ For personal development and testing
- ✅ Behind additional authentication layers (VPN, reverse proxy with auth)

**DO NOT use this application for:**
- ❌ Public-facing internet services
- ❌ Multi-tenant SaaS deployments
- ❌ Handling sensitive user data
- ❌ Production environments without significant security hardening

---

## Executive Summary

The CLI-Agent multi-user infrastructure has **7 CRITICAL** and **12 HIGH-severity** security vulnerabilities that fundamentally compromise its security model. The application is designed as a container orchestration system that gives users isolated Docker containers, but lacks essential security controls including authentication, authorization, and proper multi-tenancy isolation.

**Estimated Effort to Production-Ready:** 8-13 weeks of dedicated security engineering

---

## 🔴 CRITICAL VULNERABILITIES (Deploy Blockers)

### 1. No Authentication System

**Issue:**
The entire application has ZERO authentication. Access control relies solely on a client-generated UUID stored in browser localStorage.

**Files Affected:**
- `frontend/js/identity-manager.js` - Client-side UUID generation
- `api/container_manager.py` - Trusts X-User-UUID header without validation

**Current "Authentication":**
```javascript
// frontend/js/identity-manager.js
getOrCreateUUID() {
    let uuid = localStorage.getItem('cli-agent-user-uuid');
    if (!uuid) {
        uuid = crypto.randomUUID();  // Client generates their own ID
        localStorage.setItem('cli-agent-user-uuid', uuid);
    }
    return uuid;
}
```

```python
# api/container_manager.py (line 556)
@app.get("/discover")
async def discover_container(request: Request):
    uuid = request.headers.get("X-User-UUID")  # No validation
    # Creates container for ANY UUID provided
```

**Why This Is Critical:**
- **Complete Identity Forgery:** Anyone can impersonate any user by changing the UUID in their browser
- **No Password Protection:** No username, password, or any form of credential verification
- **Session Hijacking:** UUIDs can be stolen from localStorage (shared browser/machine)
- **No Expiry:** UUIDs are permanent - stolen UUID grants permanent access

**Attack Scenarios:**
1. **Enumerate UUIDs:** Attacker guesses/discovers valid UUIDs and accesses victim containers
2. **Shared Computer:** User A can read User B's UUID from localStorage on same machine
3. **Network Sniffing:** Without HTTPS (see vulnerability #7), UUIDs transmitted in plaintext

**Impact:**
- Access to other users' containers and data
- Ability to delete other users' containers
- Impersonation of any user

**How to Fix (Production):**

Implement proper authentication system:
```python
# api/container_manager.py - Add JWT authentication
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            os.getenv("JWT_SECRET"),
            algorithms=["HS256"]
        )
        return payload["user_id"]
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

@app.get("/discover")
async def discover_container(
    request: Request,
    user_id: str = Depends(verify_token)
):
    # Now user_id is verified server-side
    container_info = await manager.get_or_create_container(user_id)
```

Add user registration/login endpoints with password hashing (bcrypt/argon2).

---

### 2. No Authorization on Administrative Endpoints

**Issue:**
Anyone can list all containers, delete any user's container, or wipe any user's data.

**Files Affected:** `api/container_manager.py`

**Vulnerable Endpoints:**
```python
# Line 686 - Lists ALL active containers (no auth check)
@app.get("/containers")
async def list_containers():
    containers = manager.list_active_containers()
    return JSONResponse({"containers": containers})

# Line 712 - Deletes ANY user's container (no ownership check)
@app.delete("/container/{uuid}")
async def delete_container(uuid: str):
    success = await manager.cleanup_container(uuid)
    return JSONResponse({"success": success})

# Line 719 - Wipes ANY user's data (no authorization)
@app.post("/container/{uuid}/wipe")
async def wipe_container_data(uuid: str):
    # Deletes volumes without verifying ownership
```

**Why This Is Critical:**
- **Information Disclosure:** Anyone can see all active containers, UUIDs, and usage patterns
- **Data Destruction:** Any user can delete other users' containers
- **Privacy Violation:** User activity and container metadata exposed

**Attack Example:**
```bash
# Attacker discovers all active users:
curl http://<YOUR_SERVER_IP>/api/manager/containers

# Response reveals victim UUIDs:
{
  "containers": [
    {"uuid": "victim-uuid-1234", "container_name": "cli-agent-abc123", ...},
    {"uuid": "victim-uuid-5678", "container_name": "cli-agent-def456", ...}
  ]
}

# Attacker deletes victim's container:
curl -X DELETE http://<YOUR_SERVER_IP>/api/manager/container/victim-uuid-1234
```

**How to Fix:**

1. **Add admin authentication:**
   ```python
   from functools import wraps

   def require_admin(func):
       @wraps(func)
       async def wrapper(*args, admin_key: str = Header(None), **kwargs):
           expected_key = os.getenv("ADMIN_API_KEY")
           if not expected_key or admin_key != expected_key:
               raise HTTPException(status_code=403, detail="Admin access required")
           return await func(*args, **kwargs)
       return wrapper

   @app.get("/containers")
   @require_admin
   async def list_containers(admin_key: str = Header(None)):
       # Now requires admin authentication
   ```

2. **Add to .env:**
   ```bash
   ADMIN_API_KEY=$(openssl rand -hex 32)
   ```

3. **Add ownership verification:**
   ```python
   @app.delete("/container/{uuid}")
   async def delete_container(uuid: str, user_id: str = Depends(verify_token)):
       container = manager.get_container(uuid)
       if container.uuid != user_id:
           raise HTTPException(status_code=403, detail="Not authorized")
       # Proceed with deletion
   ```

---

### 3. Exposed API Keys in .env File

**Issue:**
Live API keys for 10+ services stored in plaintext .env file, accessible to all user containers.

**File Affected:** `.env` (repository root)

**Exposed Credentials:**
```bash
GROQ_API_KEY=gsk_<REDACTED>
ANTHROPIC_API_KEY=sk-ant-<REDACTED>
OPENAI_API_KEY=sk-proj-<REDACTED>
NVIDIA_API_KEY=nvapi-<REDACTED>
GEMINI_API_KEY=AIzaSy<REDACTED>
YOUTUBE_API_KEY=AIzaSy<REDACTED>
BRAVE_API_KEY=<REDACTED>
HASS_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
HUGGINGFACE_TOKEN=hf_<REDACTED>
# + more...
```

**Additional Exposure in Git History:**
- Commit `e0069a3` contains an older Groq API key that was later removed
- This key exists **permanently** in git history and should be considered compromised

**Why This Is Critical:**
- **Cost Abuse:** All users share the same API keys - any user can exhaust quotas and incur unlimited costs on your account
- **No Cost Attribution:** Cannot track which user made which API calls
- **Container Inspection:** Keys visible via `docker inspect` to anyone with Docker access
- **No Rotation:** Keys stored in plaintext files, hard to rotate
- **Git History:** Historical keys permanently exposed

**Current Implementation:**
```python
# api/container_manager.py (lines 210-221)
api_key_vars = [
    "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    # ... all keys
]
for key in api_key_vars:
    value = os.environ.get(key)
    if value:
        env_vars[key] = value  # Passed to ALL containers
        logger.debug(f"Passing {key} to container")
```

**How to Fix:**

1. **Rotate ALL exposed keys immediately** (especially git history key)

2. **Use Docker secrets:**
   ```yaml
   # docker_files/docker-compose.infrastructure.yml
   secrets:
     anthropic_key:
       external: true
     openai_key:
       external: true

   services:
     container-manager:
       secrets:
         - anthropic_key
         - openai_key
   ```

   ```bash
   # Create secrets
   echo "new_api_key" | docker secret create anthropic_key -
   ```

3. **Implement per-user API keys:**
   - Store user-specific API keys in database (encrypted)
   - Inject only that user's keys into their container
   - Track usage per user for billing/limits

4. **Clean git history:**
   ```bash
   # Remove .env from all git history
   java -jar bfg.jar --delete-files .env
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   ```

---

### 4. Docker Socket Exposed with Write Access

**Issue:**
Container Manager has full write access to Docker socket - a critical container escape vector.

**File Affected:** `docker_files/docker-compose.infrastructure.yml` (line 41)

**Current Configuration:**
```yaml
container-manager:
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock  # WRITE ACCESS (not read-only)
```

**Why This Is Critical:**
- **Complete Container Escape:** Container Manager can create privileged containers and mount host filesystem
- **Root on Host:** Can execute arbitrary code as root on the host machine
- **Compromise All Containers:** Can access/modify any container on the system

**Attack Scenario (if Container Manager is compromised):**
```python
# Attacker code running in Container Manager:
import docker
client = docker.from_env()

# Create privileged container with host filesystem mounted
container = client.containers.run(
    "alpine",
    privileged=True,
    volumes={"/": {"bind": "/host", "mode": "rw"}},
    command="chroot /host /bin/bash -c 'cat /etc/shadow'"
)
# Now attacker has root access to host
```

**Why This Is Necessary:**
The Container Manager **requires** Docker socket access to create and manage user containers. This is an inherent risk of the Docker-in-Docker pattern used for container orchestration.

**Mitigations (Cannot Fully Eliminate Risk):**

1. **Run Container Manager as non-root:**
   ```dockerfile
   # docker_files/Dockerfile.container-manager
   RUN groupadd -g 999 docker && \
       useradd -m -u 1000 -g docker containermgr && \
       chown -R containermgr:docker /app

   USER containermgr
   ```

   **Note:** Host must add user to docker group:
   ```bash
   sudo usermod -aG docker containermgr
   ```

2. **Implement AppArmor/SELinux profiles:**
   ```python
   # When creating containers:
   container = client.containers.run(
       security_opt=["apparmor=docker-default", "no-new-privileges"]
   )
   ```

3. **Consider alternatives:**
   - **Sysbox:** Docker-in-Docker without privileged access
   - **Kata Containers:** VM-level isolation instead of containers
   - **Separate VM:** Run Container Manager in isolated VM

---

### 5. Container Manager Runs as Root

**Issue:**
The Container Manager container runs as root user with Docker socket access.

**File Affected:** `docker_files/Dockerfile.container-manager`

**Current Configuration:**
```dockerfile
FROM python:3.11-slim
# ... no USER directive ...
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", ...]
# Process runs as root (UID 0)
```

**Why This Is Critical:**
Combined with Docker socket access, this creates the highest-risk scenario:
- Root process with full Docker control
- If Container Manager is exploited → immediate root on host
- No privilege separation or defense in depth

**Contrast with Backend Containers (Good Practice):**
```dockerfile
# docker_files/Dockerfile.backend (lines 56-72)
RUN useradd -m -u 1000 -s /bin/bash cliagent && \
    chown -R cliagent:cliagent /workspace

USER cliagent  # ✅ Runs as non-root
```

**How to Fix:**

See mitigation in Vulnerability #4 (Run as non-root with docker group).

---

### 6. No Network Isolation Between User Containers

**Issue:**
All user containers are on the same Docker network and can communicate with each other.

**File Affected:** `api/container_manager.py` (line 234)

**Current Configuration:**
```python
container = self.docker_client.containers.run(
    network="cli-agent-network",  # All containers on same network
    # ... other settings
)
```

**Why This Is Critical:**
- **Cross-User Attacks:** User A can scan and connect to User B's container
- **Data Exfiltration:** Compromised container can reach other users' containers
- **No Tenant Isolation:** Defeats the purpose of multi-user isolation

**Attack Scenario:**
```bash
# From inside User A's container:
# Scan the network for other containers
for i in {0..255}; do
  curl -s http://172.18.0.$i:7860/api/health && echo "Found container at 172.18.0.$i"
done

# Connect to victim container
curl http://172.18.0.5:7860/api/chat -d '{"message":"access victim data"}'
```

**Current Network Configuration:**
```yaml
# docker_files/docker-compose.infrastructure.yml
networks:
  cli-agent-network:
    external: true  # Single shared network
```

**How to Fix:**

1. **Create per-user networks:**
   ```python
   # In container_manager.py
   def create_user_network(self, uuid: str) -> str:
       network_name = f"cli-agent-{uuid[:12]}"
       try:
           self.docker_client.networks.create(
               network_name,
               driver="bridge",
               internal=True  # No internet access
           )
       except docker.errors.APIError:
           pass  # Network already exists
       return network_name

   # When creating container:
   network = self.create_user_network(uuid)
   container = self.docker_client.containers.run(
       network=network,  # Isolated network per user
       # ...
   )
   ```

2. **Implement network policies:**
   - Use Docker network plugins (Calico, Weave)
   - Block inter-container traffic except to Container Manager

---

### 7. No TLS/HTTPS - All Traffic in Plaintext

**Issue:**
Traefik is configured for HTTPS but TLS is disabled. All traffic transmitted unencrypted.

**File Affected:** `docker_files/traefik/traefik.yml` (lines 18-25)

**Current Configuration:**
```yaml
websecure:
  address: ":443"
  # TLS configuration (uncomment for production)
  # http:
  #   tls:
  #     certResolver: letsencrypt  # COMMENTED OUT
```

**Why This Is Critical:**
- **Man-in-the-Middle Attacks:** Network eavesdropper can see all data
- **UUID Theft:** User UUIDs transmitted in plaintext headers
- **API Key Exposure:** If debug mode enabled, keys could leak in error messages
- **Session Hijacking:** WebSocket connections unencrypted
- **No Integrity:** Attacker can modify traffic in transit

**How to Fix:**

1. **Enable TLS in Traefik:**
   ```yaml
   # docker_files/traefik/traefik.yml
   websecure:
     address: ":443"
     http:
       tls:
         certResolver: letsencrypt

   certificatesResolvers:
     letsencrypt:
       acme:
         email: your-email@example.com
         storage: /letsencrypt/acme.json
         httpChallenge:
           entryPoint: web
   ```

2. **Force HTTPS redirect:**
   ```yaml
   # docker_files/traefik/dynamic.yml
   http:
     middlewares:
       security-headers:
         headers:
           sslRedirect: true  # Change from false to true
   ```

3. **Update frontend to use wss:// for WebSockets:**
   ```javascript
   // frontend/js/websocket.js
   const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
   this.ws = new WebSocket(`${protocol}//...`);
   ```

---

### 8. Traefik Dashboard Publicly Accessible Without Authentication

**Issue:**
Traefik dashboard exposed without any authentication (though bound to localhost).

**File Affected:** `docker_files/traefik/traefik.yml` (lines 4-6)

**Current Configuration:**
```yaml
api:
  dashboard: true
  insecure: true  # ⚠️ No authentication required
```

**Binding:**
```yaml
# docker-compose.infrastructure.yml (line 15)
ports:
  - "127.0.0.1:8080:8080"  # Localhost only (good)
```

**Why This Is Critical:**
While bound to localhost, the dashboard is still accessible:
- To anyone with SSH access to the server
- Via port forwarding: `ssh -L 8080:localhost:8080 user@server`
- If server is compromised

**Dashboard Exposes:**
- All routing rules and backend endpoints
- Real-time traffic metrics
- Container discovery patterns
- Infrastructure topology

**How to Fix:**

1. **Add BasicAuth:**
   ```yaml
   # docker_files/traefik/dynamic.yml
   http:
     middlewares:
       dashboard-auth:
         basicAuth:
           users:
             - "admin:$apr1$..." # Generate with htpasswd
   ```

   Generate password:
   ```bash
   docker run --rm httpd:2.4-alpine htpasswd -nbB admin "YourPasswordHere"
   ```

2. **Apply to dashboard route:**
   ```yaml
   # docker-compose.infrastructure.yml
   labels:
     - "traefik.http.routers.dashboard.middlewares=dashboard-auth@file"
   ```

3. **Or disable dashboard entirely:**
   ```yaml
   api:
     dashboard: false
   ```

---

## 🟡 HIGH PRIORITY ISSUES

### 9. No Input Validation on UUID Parameter

**Issue:**
UUID header is not validated before use in container names, volumes, and database queries.

**File:** `api/container_manager.py` (line 556)

**Current Code:**
```python
uuid = request.headers.get("X-User-UUID")
if not uuid:
    raise HTTPException(status_code=400, detail="Missing X-User-UUID header")
# No validation - directly used
```

**Risks:**
- **Command Injection:** Special characters in UUID could be used in shell commands
- **SQL Injection:** Though parameterized queries protect against this, validation is defense-in-depth
- **Container Name Injection:** Malicious UUID could create containers with special characters
- **Volume Name Issues:** Docker volume names could be malformed

**How to Fix:**

```python
import re

UUID_PATTERN = re.compile(
    r'^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$',
    re.IGNORECASE
)

def validate_uuid(uuid: str) -> str:
    if not uuid or len(uuid) > 36:
        raise HTTPException(status_code=400, detail="Invalid UUID")
    if not UUID_PATTERN.match(uuid):
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    return uuid

@app.get("/discover")
async def discover_container(request: Request):
    uuid = request.headers.get("X-User-UUID")
    uuid = validate_uuid(uuid)  # Validate before use
    # ... rest of function
```

---

### 10. No Rate Limiting on Container Creation

**Issue:**
Users can create unlimited containers by clearing localStorage and generating new UUIDs.

**File:** `api/container_manager.py` (lines 146-171)

**Current Code:**
```python
async def get_or_create_container(self, uuid: str) -> ContainerInfo:
    existing = self.get_container(uuid)
    if existing:
        return existing
    return await self.create_container(uuid)  # No limits
```

**Why This Is Critical:**
- **Resource Exhaustion:** Single user can create hundreds of containers
- **Disk Space DoS:** Each container has 3 volumes
- **Cost Escalation:** If using cloud hosting, unbounded resource usage
- **Image Pull DoS:** Repeated container creation triggers image pulls

**Attack Scenario:**
```javascript
// Malicious frontend script:
for (let i = 0; i < 1000; i++) {
    localStorage.setItem('cli-agent-user-uuid', crypto.randomUUID());
    fetch('/api/manager/discover');  // Creates new container each time
}
```

**How to Fix:**

1. **Add per-UUID container limit:**
   ```python
   MAX_CONTAINERS_PER_USER = 3

   async def get_or_create_container(self, uuid: str) -> ContainerInfo:
       existing = self.get_container(uuid)
       if existing:
           return existing

       # Check user's container count
       user_containers = self.get_containers_by_uuid(uuid)
       if len(user_containers) >= MAX_CONTAINERS_PER_USER:
           raise HTTPException(
               status_code=429,
               detail=f"Maximum {MAX_CONTAINERS_PER_USER} containers per user"
           )

       return await self.create_container(uuid)
   ```

2. **Add IP-based rate limiting in Traefik:**
   ```yaml
   # docker_files/traefik/dynamic.yml
   http:
     middlewares:
       container-creation-limit:
         rateLimit:
           average: 5
           burst: 10
           period: 5m
   ```

---

### 11. Insufficient Rate Limiting in Traefik

**Issue:**
Current rate limit of 100 requests/minute is too permissive for container operations.

**File:** `docker_files/traefik/dynamic.yml` (lines 6-11)

**Current Configuration:**
```yaml
rate-limit:
  rateLimit:
    average: 100  # Too high for container orchestration
    burst: 200
    period: 1m
```

**Why This Matters:**
- Container creation is expensive (CPU, memory, disk I/O)
- DoS attacks can overwhelm the system
- No differentiation between read vs. write operations

**How to Fix:**

```yaml
# Separate rate limits by operation type:
http:
  middlewares:
    general-rate-limit:
      rateLimit:
        average: 20
        burst: 40
        period: 1m

    container-ops-limit:
      rateLimit:
        average: 5
        burst: 10
        period: 5m

    read-only-limit:
      rateLimit:
        average: 100
        burst: 200
        period: 1m
```

Apply different limits to different routes in routing configuration.

---

### 12. No Container Resource Quotas

**Issue:**
Beyond memory/CPU limits, there are no disk I/O, network, or process limits.

**File:** `api/container_manager.py` (lines 237-240)

**Current Limits:**
```python
mem_limit="2g",
cpu_quota=cpu_quota,  # Dynamic 50-100% of one core
# Missing: disk I/O, network bandwidth, PID limits
```

**Why This Matters:**
- **Disk I/O Abuse:** User can saturate disk with heavy I/O
- **Network Flooding:** No bandwidth limits on egress
- **Fork Bomb:** No process limits allow unlimited process creation
- **Noisy Neighbor:** One user degrades performance for all

**How to Fix:**

```python
container = self.docker_client.containers.run(
    # Existing limits:
    mem_limit="2g",
    cpu_quota=cpu_quota,
    cpu_period=100000,

    # Add new limits:
    pids_limit=100,  # Max 100 processes

    # Disk I/O limits:
    device_write_bps=[
        docker.types.DeviceRequest(
            Path='/dev/sda',
            Rate='10mb'  # 10 MB/s write limit
        )
    ],

    # Network bandwidth (requires tc/iptables on host):
    # Implement via Docker network plugin or host-level controls
)
```

---

### 13. No Security Profiles (AppArmor/SELinux)

**Issue:**
Containers created without security profiles, increasing container escape risk.

**File:** `api/container_manager.py` (lines 223-241)

**Current Configuration:**
```python
container = self.docker_client.containers.run(
    # No security_opt specified
    image=BASE_IMAGE,
    # ... other settings
)
```

**Why This Matters:**
- No syscall filtering (seccomp)
- No mandatory access controls (AppArmor/SELinux)
- Easier kernel exploits
- No defense against privilege escalation

**How to Fix:**

```python
container = self.docker_client.containers.run(
    security_opt=[
        "no-new-privileges",  # Prevent privilege escalation
        "apparmor=docker-default",  # Apply AppArmor profile
        "seccomp=/path/to/seccomp-profile.json"  # Syscall filtering
    ],
    cap_drop=["ALL"],  # Drop all Linux capabilities
    cap_add=["CHOWN", "SETUID", "SETGID"],  # Add back only needed ones
    # ... other settings
)
```

Create custom seccomp profile to block dangerous syscalls (e.g., `mount`, `ptrace`).

---

### 14. No Egress Network Filtering

**Issue:**
User containers can make arbitrary outbound connections to any IP/port.

**Current Behavior:**
- Containers can connect to any internet host
- No whitelist of allowed destinations
- No port restrictions

**Why This Matters:**
- **Data Exfiltration:** Compromised container can send data to attacker's server
- **Command & Control:** Malware can phone home
- **Cryptocurrency Mining:** Containers can connect to mining pools
- **Internal Network Scanning:** Can scan your local network (192.168.x.x)

**Documentation vs. Reality:**
`docs/NETWORK_ACCESS.md` lists blocked ports (22, 21, 3306), but **no actual firewall rules implement these blocks**.

**How to Fix:**

1. **Host-level iptables rules:**
   ```bash
   # Block outbound SSH, FTP, MySQL from containers
   iptables -A FORWARD -o eth0 -p tcp --dport 22 -j DROP
   iptables -A FORWARD -o eth0 -p tcp --dport 21 -j DROP
   iptables -A FORWARD -o eth0 -p tcp --dport 3306 -j DROP

   # Whitelist only necessary services (AI APIs)
   iptables -A FORWARD -o eth0 -p tcp --dport 443 -j ACCEPT
   iptables -A FORWARD -o eth0 -p tcp --dport 80 -j ACCEPT
   iptables -A FORWARD -o eth0 -j DROP  # Drop all other outbound
   ```

2. **Docker network policies** (requires Calico or Weave plugin):
   ```yaml
   # Example Calico policy
   apiVersion: projectcalico.org/v3
   kind: NetworkPolicy
   metadata:
     name: allow-ai-apis-only
   spec:
     selector: app == 'cli-agent-backend'
     egress:
       - action: Allow
         destination:
           domains:
             - api.anthropic.com
             - api.openai.com
             # ... whitelist AI services
   ```

---

### 15. No CORS Configuration

**Issue:**
FastAPI application has no CORS middleware, potentially breaking cross-origin requests or allowing unwanted origins.

**File:** `api/container_manager.py`

**Current State:** No CORS configuration found.

**Why This Matters:**
- Frontend may not work if served from different origin
- No control over which websites can access the API
- CSRF attacks possible if origins not restricted

**How to Fix:**

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://<YOUR_SERVER_IP>"],  # Specific origin only
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-User-UUID", "Content-Type"],
)
```

---

## 🟢 MEDIUM PRIORITY RECOMMENDATIONS

### 16. Database Security

**Issue:**
SQLite database has no encryption and uses default file permissions.

**File:** `api/container_manager.py` (line 39)

**Current Configuration:**
```python
DB_PATH = Path("/data/container_manager.db")
# No encryption, no password protection
```

**Recommendations:**
- Use SQLCipher for encryption at rest
- Set restrictive file permissions (chmod 600)
- Implement database backup strategy
- Add audit logging for sensitive queries

---

### 17. Error Message Information Disclosure

**Issue:**
Exception details exposed to clients in error responses.

**Examples:**
```python
# Line 271
raise HTTPException(status_code=500, detail=f"Container creation failed: {e}")

# Line 574
raise HTTPException(status_code=500, detail=str(e))
```

**Why This Matters:**
- Internal paths disclosed
- Stack traces may reveal code structure
- Helps attackers understand system

**How to Fix:**
```python
# Production error handling:
try:
    # ... operation
except Exception as e:
    logger.error(f"Container creation failed: {e}", exc_info=True)
    raise HTTPException(
        status_code=500,
        detail="Internal server error" if production else str(e)
    )
```

---

### 18. No Audit Logging

**Issue:**
Current logging doesn't capture security-relevant events.

**What's Missing:**
- User authentication attempts (none exist)
- Container creation/deletion with user attribution
- Administrative actions
- Rate limit violations
- Failed authorization checks

**How to Fix:**

Implement comprehensive audit logging:
```python
class AuditLogger:
    def log_container_creation(self, uuid: str, ip: str, container_id: str):
        self.logger.info(
            "AUDIT: Container created",
            extra={
                "user_uuid": uuid,
                "ip_address": ip,
                "container_id": container_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

    def log_access_denied(self, uuid: str, resource: str, reason: str):
        self.logger.warning(
            "AUDIT: Access denied",
            extra={
                "user_uuid": uuid,
                "resource": resource,
                "reason": reason
            }
        )
```

Export logs to external system (Splunk, ELK stack) for security monitoring.

---

### 19. Frontend Security Issues

**File:** `frontend/js/renderers.js`

**Issues:**
1. **Unsafe innerHTML usage** (line 12) - Potential XSS
2. **Weak sandbox configuration** (line 165) - `allow-popups-to-escape-sandbox` breaks isolation
3. **Insufficient XSS sanitization** (lines 64-70) - Regex-based filtering is incomplete

**How to Fix:**

Replace innerHTML with DOM manipulation or use DOMPurify library:
```javascript
import DOMPurify from 'dompurify';

// Safe sanitization:
iframe.srcdoc = DOMPurify.sanitize(htmlContent, {
    ALLOWED_TAGS: ['p', 'div', 'span', 'code', 'pre'],
    ALLOWED_ATTR: ['class', 'id']
});
```

Strengthen sandbox:
```javascript
iframe.sandbox = jsEnabled ?
    'allow-scripts allow-forms allow-popups allow-downloads' :  // Remove escape
    'allow-forms allow-popups allow-downloads';
```

---

### 20. WebSocket Security

**File:** `frontend/js/websocket.js`

**Issues:**
- No origin validation on messages
- No message size limits
- Unvalidated postMessage with wildcard origin (`*`)

**How to Fix:**

```javascript
// Validate WebSocket messages:
this.ws.onmessage = (event) => {
    // Verify origin if needed
    const data = JSON.parse(event.data);

    // Validate message size
    if (event.data.length > 1024 * 1024) {  // 1MB limit
        console.error('Message too large');
        return;
    }

    this.handleMessage(data);
};

// Fix postMessage origin:
parent.postMessage(
    {type: 'resize', height: bestHeight},
    window.location.origin  // Not '*'
);
```

---

## Production Readiness Checklist

### Critical (Must Fix Before Any Production Use)

- [ ] **Implement authentication system** (JWT, OAuth2, or similar)
- [ ] **Add authorization checks** to all endpoints
- [ ] **Rotate ALL exposed API keys** (especially git history key)
- [ ] **Implement per-user API keys** (no sharing)
- [ ] **Run Container Manager as non-root**
- [ ] **Enable TLS/HTTPS** with Let's Encrypt
- [ ] **Implement network isolation** per user (separate networks)
- [ ] **Secure Traefik dashboard** with BasicAuth or disable

### High Priority (Required for Production)

- [ ] **Add UUID validation** with regex
- [ ] **Implement container limits** per user (max 3)
- [ ] **Add rate limiting** on container creation (5 per 5 min)
- [ ] **Strengthen Traefik rate limits** (reduce from 100/min)
- [ ] **Add resource quotas** (PID limits, disk I/O)
- [ ] **Apply security profiles** (AppArmor, seccomp)
- [ ] **Implement egress filtering** (whitelist AI APIs only)
- [ ] **Add CORS configuration** with specific origins

### Medium Priority (Hardening)

- [ ] **Encrypt database** with SQLCipher
- [ ] **Implement audit logging** for all security events
- [ ] **Add health monitoring** and alerting
- [ ] **Fix frontend XSS** vulnerabilities
- [ ] **Strengthen iframe sandboxing**
- [ ] **Add WebSocket origin validation**
- [ ] **Implement error sanitization** (no internal details in prod)
- [ ] **Create backup strategy** for database and volumes

### Testing & Validation

- [ ] **Run penetration testing**
- [ ] **Conduct security code review**
- [ ] **Implement container vulnerability scanning** (Trivy, Clair)
- [ ] **Set up intrusion detection** (Falco)
- [ ] **Test multi-tenancy isolation**
- [ ] **Verify egress filtering rules**
- [ ] **Load testing with resource limits**
- [ ] **Test authentication bypass scenarios**

---

## Alternative Architectures for Production

If the security requirements are too complex for Docker-in-Docker:

### Option 1: True VM Isolation
- Use Firecracker or Kata Containers instead of Docker
- Each user gets lightweight VM, not container
- Hardware-enforced isolation
- No shared kernel vulnerabilities

### Option 2: Managed Kubernetes
- Use GKE/EKS with proper RBAC
- Network policies for tenant isolation
- Secrets management built-in
- Production-grade multi-tenancy

### Option 3: Serverless Functions
- Convert to FaaS (AWS Lambda, Google Cloud Functions)
- No container orchestration needed
- Built-in isolation and scaling
- Pay-per-use pricing

---

## Estimated Implementation Timeline

To make this production-ready with proper security:

| Phase | Duration | Focus |
|-------|----------|-------|
| **Phase 1:** Authentication & Authorization | 2-3 weeks | JWT auth, user management, endpoint protection |
| **Phase 2:** Network Isolation | 1-2 weeks | Per-user networks, egress filtering |
| **Phase 3:** Secret Management | 1 week | Docker secrets, per-user API keys |
| **Phase 4:** Container Hardening | 2-3 weeks | Security profiles, resource limits, TLS |
| **Phase 5:** Testing & Audit | 2-4 weeks | Pentesting, code review, vulnerability scanning |

**Total:** 8-13 weeks (2-3 months) of dedicated security engineering

---

## Questions or Support

For security concerns or implementation questions:

1. **Read the Documentation:**
   - `docs/MULTI_USER_ARCHITECTURE.md` - Current architecture
   - `docs/INFRASTRUCTURE_STATUS.md` - Deployment status
   - `docs/NETWORK_ACCESS.md` - Network configuration

2. **Security Resources:**
   - [Docker Security Best Practices](https://docs.docker.com/engine/security/)
   - [OWASP Top 10](https://owasp.org/www-project-top-ten/)
   - [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
   - [Container Security Guide](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

3. **Professional Help:**
   - Consider hiring security consultant for production deployment
   - Implement bug bounty program once hardened
   - Regular security audits (quarterly recommended)

---

**REMEMBER:** This application is suitable for **local development and personal use** on trusted networks. Do NOT deploy to production or expose to the internet without implementing the security measures outlined in this document.

**Last Updated:** 2025-10-21
**Next Security Review:** Recommended before any production deployment
