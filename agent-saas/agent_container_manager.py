"""
Agent Container Manager for Agent SaaS Infrastructure

Manages lifecycle of agent-specific containers:
- Multi-session support (multiple containers per user)
- Agent-type specific container creation
- Session management and cleanup
- WebSocket and HTTP proxying to agent containers
"""

import asyncio
import hashlib
import logging
import os
import re
import shutil
import sqlite3
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import docker
import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from agent_discovery import get_discovery, AgentDiscovery

# Configure logging
LOG_DIR = Path("/logs/agent-saas")
LOG_DIR.mkdir(parents=True, exist_ok=True)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console_handler.setFormatter(console_formatter)

file_handler = RotatingFileHandler(
    LOG_DIR / "agent_container_manager.log",
    maxBytes=50 * 1024 * 1024,  # 50MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
file_handler.setFormatter(file_formatter)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path("/data/agent-saas/agent_sessions.db")
STORAGE_THRESHOLD = 90  # Percent
DEFAULT_RETENTION_DAYS = 7
MAX_SESSIONS_PER_USER = int(os.getenv("MAX_SESSIONS_PER_USER", "5"))

# CORS Configuration
DEFAULT_CORS = "http://localhost:7860,http://127.0.0.1:7860" if os.getenv("ENVIRONMENT") != "development" else "*"
ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", DEFAULT_CORS).split(",")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS]
logger.info(f"CORS allowed origins: {ALLOWED_ORIGINS}")

app = FastAPI(
    title="CLI-Agent SaaS - Agent Container Manager",
    description="Manages multi-session agent container lifecycle with WebSocket proxying",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH"],
    allow_headers=["X-User-UUID", "Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
)

# UUID validation pattern (RFC 4122 UUID v4)
UUID_PATTERN = re.compile(
    r'^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$',
    re.IGNORECASE
)


def validate_uuid(uuid: str) -> str:
    """Validate UUID format to prevent injection attacks"""
    if not uuid:
        raise HTTPException(status_code=400, detail="UUID cannot be empty")
    if len(uuid) > 36:
        raise HTTPException(status_code=400, detail="UUID too long")
    if not UUID_PATTERN.match(uuid):
        raise HTTPException(
            status_code=400,
            detail="Invalid UUID format. Expected UUID v4 format"
        )
    return uuid


@dataclass
class AgentSession:
    """Agent session information"""
    session_id: str
    user_uuid: str
    agent_type: str
    container_id: str
    container_name: str
    created_at: datetime
    last_activity: datetime
    status: str = "running"  # running, stopped, terminated
    retention_days: int = 7


class StorageMonitor:
    """Monitor disk storage usage"""

    def __init__(self, threshold_percent: int = 90):
        self.threshold = threshold_percent

    def get_usage_percent(self) -> float:
        """Get storage usage percentage"""
        stat = shutil.disk_usage('/')
        return (stat.used / stat.total) * 100

    def get_available_gb(self) -> float:
        """Get available storage in GB"""
        stat = shutil.disk_usage('/')
        return stat.free / (1024**3)

    def is_critical(self) -> bool:
        """Check if storage is critically low"""
        return self.get_usage_percent() > self.threshold


class AgentContainerManager:
    """Manages agent container lifecycle with multi-session support"""

    def __init__(self):
        self.docker_client = docker.DockerClient(base_url='unix://var/run/docker.sock')
        self.storage_monitor = StorageMonitor(STORAGE_THRESHOLD)
        self.db_path = DB_PATH
        self.discovery: AgentDiscovery = get_discovery()
        self.init_database()
        logger.info("Agent Container Manager initialized")
        logger.info(f"Discovered {len(self.discovery.list_agents())} agents")

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections with WAL mode"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_database(self):
        """Initialize SQLite database with WAL mode"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self.get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_uuid TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    container_id TEXT NOT NULL,
                    container_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    status TEXT DEFAULT 'running',
                    retention_days INTEGER DEFAULT 7
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT,
                    FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_uuid ON agent_sessions(user_uuid)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_type ON agent_sessions(agent_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_activity ON agent_sessions(last_activity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON agent_sessions(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_activity_session ON activity_log(session_id)")

            logger.info("Database initialized with WAL mode")

    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        # Format: timestamp_random
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        random_part = secrets.token_hex(6)
        return f"sess_{timestamp}_{random_part}"

    def hash_session(self, session_id: str) -> str:
        """Generate short hash for session ID (12 chars for container names)"""
        return hashlib.sha256(session_id.encode()).hexdigest()[:12]

    def get_or_create_user_network(self, user_uuid: str) -> str:
        """
        Get or create an isolated network for a user.
        Separate from main CLI-agent networks.
        """
        uuid_short = hashlib.sha256(user_uuid.encode()).hexdigest()[:12]
        network_name = f"agent-saas-user-{uuid_short}"

        try:
            self.docker_client.networks.get(network_name)
            logger.debug(f"Using existing network: {network_name}")
            return network_name
        except docker.errors.NotFound:
            logger.info(f"Creating isolated network for user {user_uuid[:8]}...: {network_name}")

            self.docker_client.networks.create(
                network_name,
                driver="bridge",
                internal=False,  # Allow external connections (for AI APIs)
                labels={
                    "agent-saas.managed": "true",
                    "agent-saas.user-uuid": user_uuid,
                    "agent-saas.network-type": "user-isolated"
                }
            )

            return network_name

    async def create_session(self, user_uuid: str, agent_type: str) -> AgentSession:
        """Create new agent session"""
        # Validate UUID
        validate_uuid(user_uuid)

        # Validate agent type
        agent_def = self.discovery.get_agent(agent_type)
        if not agent_def:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown agent type: {agent_type}. Available: {[a.name for a in self.discovery.list_agents()]}"
            )

        # Check session limit
        user_sessions = self.list_user_sessions(user_uuid, status_filter="running")
        if len(user_sessions) >= MAX_SESSIONS_PER_USER:
            raise HTTPException(
                status_code=429,
                detail=f"Maximum sessions per user reached ({MAX_SESSIONS_PER_USER}). Please terminate existing sessions."
            )

        # Check storage
        if self.storage_monitor.is_critical():
            logger.warning(f"Storage critical: {self.storage_monitor.get_usage_percent():.1f}%")
            await self.aggressive_cleanup()

        # Generate session ID
        session_id = self.generate_session_id()
        session_hash = self.hash_session(session_id)
        container_name = f"agent-{agent_type}-{session_hash}"

        logger.info(f"Creating session {session_id} for agent {agent_type}")

        # Labels for resource tracking
        labels = {
            "agent-saas.managed": "true",
            "agent-saas.session-id": session_id,
            "agent-saas.user-uuid": user_uuid,
            "agent-saas.agent-type": agent_type,
            "agent-saas.created": datetime.utcnow().isoformat()
        }

        try:
            # Create volumes
            for suffix in ['data', 'logs', 'config']:
                vol_name = f"agent-{session_hash}-{suffix}"
                try:
                    self.docker_client.volumes.create(name=vol_name, labels=labels)
                    logger.debug(f"Created volume: {vol_name}")
                except docker.errors.APIError as e:
                    if "already exists" not in str(e):
                        raise

            # Get or create isolated network
            user_network = self.get_or_create_user_network(user_uuid)

            # Prepare environment
            env_vars = {
                "AGENT_MODE": "saas",
                "AGENT_SESSION_ID": session_id,
                "AGENT_DATA_DIR": "/workspace/data",
                "AGENT_LOG_DIR": "/workspace/logs",
                "AGENT_CONFIG_DIR": "/workspace/config",
                "PYTHONUNBUFFERED": "1"
            }

            # Pass LLM API keys from host environment
            api_key_vars = [
                "GOOGLE_API_KEY", "GEMINI_API_KEY",
                "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                "GROQ_API_KEY", "NVIDIA_API_KEY",
                "OLLAMA_HOST", "BRAVE_API_KEY"
            ]
            for key in api_key_vars:
                value = os.environ.get(key)
                if value:
                    env_vars[key] = value

            # Create container
            container = self.docker_client.containers.run(
                image=agent_def.docker_image,
                name=container_name,
                ports={'8000/tcp': None},  # Agent server port (dynamic)
                environment=env_vars,
                volumes={
                    f"agent-{session_hash}-data": {"bind": "/workspace/data", "mode": "rw"},
                    f"agent-{session_hash}-logs": {"bind": "/workspace/logs", "mode": "rw"},
                    f"agent-{session_hash}-config": {"bind": "/workspace/config", "mode": "rw"}
                },
                network=user_network,
                detach=True,
                restart_policy={"Name": "unless-stopped"},

                # Resource limits
                cpu_quota=100000,
                cpu_period=100000,
                mem_limit="1g",  # Smaller than main CLI-agent
                pids_limit=50,   # Fewer processes

                # Security
                security_opt=[
                    "no-new-privileges:true",
                    "apparmor=docker-default"
                ],
                cap_drop=["ALL"],
                cap_add=["CHOWN", "SETUID", "SETGID"],

                labels=labels
            )

            # Register session
            session = AgentSession(
                session_id=session_id,
                user_uuid=user_uuid,
                agent_type=agent_type,
                container_id=container.id,
                container_name=container_name,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                status="running",
                retention_days=DEFAULT_RETENTION_DAYS
            )

            # Connect to shared network for proxy access
            try:
                shared_network = self.docker_client.networks.get("agent-saas-network")
                shared_network.connect(container)
                logger.debug(f"Connected {container_name} to shared agent-saas-network")
            except Exception as e:
                logger.warning(f"Could not connect to agent-saas-network: {e}")

            self.register_session(session)
            self.log_activity(session_id, "session_created")

            logger.info(f"Session created: {session_id} (Container: {container_name})")

            # Wait for container to be ready
            await self._wait_for_container_ready(container, timeout=30)

            return session

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            try:
                await self.cleanup_session_resources(session_hash)
            except Exception as cleanup_err:
                logger.error(f"Cleanup after failed creation also failed: {cleanup_err}")
            raise HTTPException(status_code=500, detail=f"Session creation failed: {e}")

    async def _wait_for_container_ready(self, container, timeout: int = 30):
        """Wait for container to become ready"""
        import time
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                container.reload()
                if container.status == 'running':
                    await asyncio.sleep(2)  # Wait for services to start
                    logger.info(f"Container {container.name} is ready")
                    return True
            except Exception as e:
                logger.debug(f"Readiness check error: {e}")

            await asyncio.sleep(1)

        logger.warning(f"Container {container.name} readiness timeout after {timeout}s")
        return False

    async def cleanup_session_resources(self, session_hash: str):
        """Cleanup container and volumes by session hash"""
        try:
            container = self.docker_client.containers.get(f"agent-*-{session_hash}")
            container.stop(timeout=10)
            container.remove()
        except docker.errors.NotFound:
            pass

        for suffix in ['data', 'logs', 'config']:
            try:
                volume = self.docker_client.volumes.get(f"agent-{session_hash}-{suffix}")
                volume.remove()
            except docker.errors.NotFound:
                pass

    def register_session(self, session: AgentSession):
        """Register session in database"""
        with self.get_db_connection() as conn:
            conn.execute(
                """INSERT INTO agent_sessions
                   (session_id, user_uuid, agent_type, container_id, container_name,
                    created_at, last_activity, status, retention_days)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session.session_id, session.user_uuid, session.agent_type,
                 session.container_id, session.container_name,
                 session.created_at.isoformat(), session.last_activity.isoformat(),
                 session.status, session.retention_days)
            )

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get session by ID"""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                """SELECT session_id, user_uuid, agent_type, container_id, container_name,
                          created_at, last_activity, status, retention_days
                   FROM agent_sessions WHERE session_id = ?""",
                (session_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return AgentSession(
                session_id=row[0],
                user_uuid=row[1],
                agent_type=row[2],
                container_id=row[3],
                container_name=row[4],
                created_at=datetime.fromisoformat(row[5]),
                last_activity=datetime.fromisoformat(row[6]),
                status=row[7],
                retention_days=row[8]
            )

    def list_user_sessions(self, user_uuid: str, status_filter: str = None) -> list[AgentSession]:
        """List all sessions for a user"""
        with self.get_db_connection() as conn:
            if status_filter:
                cursor = conn.execute(
                    """SELECT session_id, user_uuid, agent_type, container_id, container_name,
                              created_at, last_activity, status, retention_days
                       FROM agent_sessions WHERE user_uuid = ? AND status = ?
                       ORDER BY created_at DESC""",
                    (user_uuid, status_filter)
                )
            else:
                cursor = conn.execute(
                    """SELECT session_id, user_uuid, agent_type, container_id, container_name,
                              created_at, last_activity, status, retention_days
                       FROM agent_sessions WHERE user_uuid = ?
                       ORDER BY created_at DESC""",
                    (user_uuid,)
                )

            return [
                AgentSession(
                    session_id=row[0],
                    user_uuid=row[1],
                    agent_type=row[2],
                    container_id=row[3],
                    container_name=row[4],
                    created_at=datetime.fromisoformat(row[5]),
                    last_activity=datetime.fromisoformat(row[6]),
                    status=row[7],
                    retention_days=row[8]
                )
                for row in cursor.fetchall()
            ]

    def update_activity(self, session_id: str):
        """Update last activity timestamp"""
        with self.get_db_connection() as conn:
            conn.execute(
                "UPDATE agent_sessions SET last_activity = ? WHERE session_id = ?",
                (datetime.utcnow().isoformat(), session_id)
            )

    def log_activity(self, session_id: str, event_type: str):
        """Log activity event"""
        with self.get_db_connection() as conn:
            conn.execute(
                "INSERT INTO activity_log (session_id, timestamp, event_type) VALUES (?, ?, ?)",
                (session_id, datetime.utcnow().isoformat(), event_type)
            )

    async def terminate_session(self, session_id: str):
        """Terminate session and cleanup resources"""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"No session found: {session_id}")
            return

        logger.info(f"Terminating session {session_id}")

        # Stop container
        try:
            container = self.docker_client.containers.get(session.container_id)
            container.stop(timeout=10)
            container.remove()
            logger.info(f"Container removed: {session.container_name}")
        except docker.errors.NotFound:
            logger.warning(f"Container already removed: {session.container_id}")
        except Exception as e:
            logger.error(f"Failed to remove container: {e}")

        # Remove volumes
        session_hash = self.hash_session(session_id)
        volumes = self.docker_client.volumes.list(
            filters={"label": f"agent-saas.session-id={session_id}"}
        )

        for volume in volumes:
            try:
                volume.remove()
                logger.info(f"Removed volume: {volume.name}")
            except Exception as e:
                logger.error(f"Failed to remove volume: {e}")

        # Update status
        with self.get_db_connection() as conn:
            conn.execute(
                "UPDATE agent_sessions SET status = 'terminated' WHERE session_id = ?",
                (session_id,)
            )

        self.log_activity(session_id, "session_terminated")
        logger.info(f"Session terminated: {session_id}")

    async def cleanup_expired_sessions(self):
        """Clean up sessions that have exceeded retention period"""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                """SELECT session_id, last_activity, retention_days
                   FROM agent_sessions WHERE status = 'running'"""
            )

            now = datetime.utcnow()
            for row in cursor.fetchall():
                session_id = row[0]
                last_activity = datetime.fromisoformat(row[1])
                retention_days = row[2]

                cutoff = now - timedelta(days=retention_days)
                if last_activity < cutoff:
                    logger.info(f"Cleaning up expired session {session_id}")
                    await self.terminate_session(session_id)

    async def aggressive_cleanup(self):
        """Aggressively clean up oldest sessions when storage is low"""
        logger.warning("Starting aggressive cleanup due to low storage")

        with self.get_db_connection() as conn:
            cursor = conn.execute(
                """SELECT session_id FROM agent_sessions
                   WHERE status = 'running'
                   ORDER BY created_at ASC"""
            )

            session_ids = [row[0] for row in cursor.fetchall()]

        cleaned = 0
        for session_id in session_ids:
            await self.terminate_session(session_id)
            cleaned += 1

            if self.storage_monitor.get_usage_percent() < 85:
                logger.info(f"Aggressive cleanup complete. Removed {cleaned} sessions.")
                break


# Global manager instance
manager = AgentContainerManager()


# === API Endpoints ===

@app.get("/api/agents/list")
async def list_agents():
    """List all available agents"""
    agents = manager.discovery.list_agents()
    return JSONResponse({
        "status": "success",
        "count": len(agents),
        "agents": [
            {
                "name": agent.name,
                "display_name": agent.display_name,
                "description": agent.description,
                "icon": agent.icon,
                "category": agent.category,
                "tool_count": agent.tool_count,
                "tools": agent.tools,
                "has_write_access": agent.has_write_access,
                "docker_image": agent.docker_image
            }
            for agent in agents
        ]
    })


@app.post("/api/agents/sessions/create")
async def create_session(request: Request):
    """Create new agent session"""
    try:
        data = await request.json()
        user_uuid = data.get('user_uuid')
        agent_type = data.get('agent_type')

        if not user_uuid or not agent_type:
            raise HTTPException(status_code=400, detail="Missing user_uuid or agent_type")

        session = await manager.create_session(user_uuid, agent_type)

        return JSONResponse({
            "status": "success",
            "session": {
                "session_id": session.session_id,
                "agent_type": session.agent_type,
                "container_name": session.container_name,
                "created_at": session.created_at.isoformat(),
                "status": session.status
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents/sessions/list")
async def list_sessions(user_uuid: str, status: str = None):
    """List user's sessions"""
    try:
        validate_uuid(user_uuid)
        sessions = manager.list_user_sessions(user_uuid, status_filter=status)

        return JSONResponse({
            "status": "success",
            "count": len(sessions),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "agent_type": s.agent_type,
                    "created_at": s.created_at.isoformat(),
                    "last_activity": s.last_activity.isoformat(),
                    "status": s.status
                }
                for s in sessions
            ]
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List sessions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/agents/sessions/{session_id}")
async def delete_session(session_id: str):
    """Terminate session"""
    try:
        await manager.terminate_session(session_id)
        return JSONResponse({"status": "success", "session_id": session_id})
    except Exception as e:
        logger.error(f"Session termination failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/agents/sessions/{session_id}/ws")
async def websocket_proxy(websocket: WebSocket, session_id: str):
    """WebSocket proxy to agent container"""
    await websocket.accept()

    logger.info(f"WebSocket connection for session {session_id}")

    session = manager.get_session(session_id)
    if not session:
        await websocket.close(code=1011, reason="Session not found")
        return

    manager.update_activity(session_id)

    # Agent container's WebSocket URL
    backend_ws_url = f"ws://{session.container_name}:8000/ws"
    logger.debug(f"Connecting to agent WebSocket: {backend_ws_url}")

    try:
        async with websockets.connect(backend_ws_url, ping_interval=30, ping_timeout=10) as backend_ws:
            logger.info(f"Connected to agent WebSocket for session {session_id}")

            async def forward_to_backend():
                try:
                    while True:
                        data = await websocket.receive_text()
                        await backend_ws.send(data)
                        logger.debug(f"Forwarded message to agent (session {session_id})")
                except WebSocketDisconnect:
                    logger.info(f"Client disconnected (session {session_id})")

            async def forward_to_client():
                try:
                    while True:
                        data = await backend_ws.recv()
                        await websocket.send_text(data)
                        logger.debug(f"Forwarded message to client (session {session_id})")
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"Agent disconnected (session {session_id})")

            await asyncio.gather(forward_to_client(), forward_to_backend())

    except Exception as e:
        logger.error(f"WebSocket proxy error for session {session_id}: {e}")
        await websocket.close(code=1011, reason="Proxy error")


@app.api_route("/api/agents/sessions/{session_id}/http/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def http_proxy(request: Request, session_id: str, path: str):
    """HTTP proxy to agent container"""
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    manager.update_activity(session_id)

    backend_url = f"http://{session.container_name}:8000/{path}"
    if request.url.query:
        backend_url += f"?{request.url.query}"

    logger.debug(f"Proxying HTTP {request.method} to {backend_url}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=request.method,
                url=backend_url,
                headers=dict(request.headers),
                content=await request.body()
            )

            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=dict(response.headers)
            )
    except Exception as e:
        logger.error(f"HTTP proxy error for session {session_id}: {e}")
        raise HTTPException(status_code=502, detail=f"Backend error: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    sessions_count = len(manager.list_user_sessions("", status_filter="running"))
    return JSONResponse({
        "status": "healthy",
        "active_sessions": sessions_count,
        "storage_usage": manager.storage_monitor.get_usage_percent(),
        "storage_available_gb": manager.storage_monitor.get_available_gb(),
        "agents_available": len(manager.discovery.list_agents())
    })


# === Background Tasks ===

@app.on_event("startup")
async def startup():
    """Start background tasks"""
    logger.info("Starting background tasks...")
    asyncio.create_task(cleanup_loop())
    asyncio.create_task(storage_monitoring_loop())
    logger.info("Agent Container Manager started successfully")


async def cleanup_loop():
    """Background task for session cleanup"""
    while True:
        await asyncio.sleep(3600)  # Every hour
        try:
            logger.info("Running cleanup loop...")
            await manager.cleanup_expired_sessions()
            logger.info("Cleanup loop complete")
        except Exception as e:
            logger.error(f"Cleanup loop error: {e}")


async def storage_monitoring_loop():
    """Background task for storage monitoring"""
    while True:
        await asyncio.sleep(600)  # Every 10 minutes
        try:
            if manager.storage_monitor.is_critical():
                logger.warning("Storage critical, triggering aggressive cleanup")
                await manager.aggressive_cleanup()
        except Exception as e:
            logger.error(f"Storage monitoring loop error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
