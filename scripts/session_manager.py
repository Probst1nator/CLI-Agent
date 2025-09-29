"""
Session Manager for CLI-Agent Multi-Instance Setup

This service manages session assignment and container orchestration for
per-browser-tab CLI-Agent instances.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional

import docker
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("session_manager")

app = FastAPI(title="CLI-Agent Session Manager")

# Docker client for container management
docker_client = docker.from_env()

# Session tracking
sessions: Dict[str, Dict] = {}
instance_health: Dict[str, Dict] = {}

# Configuration
MAX_INSTANCES = int(os.getenv("MAX_INSTANCES", "5"))
INSTANCE_TIMEOUT = int(os.getenv("INSTANCE_TIMEOUT", "1800"))  # 30 minutes
BASE_PORT = 7861

class InstanceManager:
    """Manages CLI-Agent container instances."""
    
    @staticmethod
    def get_available_instance() -> Optional[str]:
        """Find an available instance or create a new one."""
        # Check existing instances
        for instance_id, health in instance_health.items():
            if health["connections"] == 0 and health["status"] == "healthy":
                return instance_id
                
        # Create new instance if under limit
        active_instances = len([h for h in instance_health.values() if h["status"] == "running"])
        if active_instances < MAX_INSTANCES:
            return InstanceManager.create_new_instance()
            
        # Use least loaded instance as fallback
        if instance_health:
            return min(instance_health.keys(), key=lambda x: instance_health[x]["connections"])
            
        return None
        
    @staticmethod
    def create_new_instance() -> str:
        """Create a new CLI-Agent container instance."""
        instance_id = f"instance-{len(instance_health) + 1}"
        port = BASE_PORT + len(instance_health)
        
        try:
            # Create container
            container = docker_client.containers.run(
                image="cli-agent:isolated",
                name=f"cli-agent-{instance_id}",
                ports={7860: port},
                environment=[
                    "CLI_AGENT_GUI_MODE=true",
                    "CLI_AGENT_HOST=0.0.0.0",
                    f"INSTANCE_ID={instance_id}",
                    f"IDLE_TIMEOUT={INSTANCE_TIMEOUT}"
                ],
                volumes={
                    f"cli-agent-data-{instance_id}": {"bind": "/workspace/data", "mode": "rw"},
                    f"cli-agent-logs-{instance_id}": {"bind": "/workspace/logs", "mode": "rw"},
                    f"cli-agent-config-{instance_id}": {"bind": "/home/cliagent/.config/cli-agent", "mode": "rw"}
                },
                network="cli-agent-multi-net",
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            instance_health[instance_id] = {
                "container_id": container.id,
                "port": port,
                "status": "starting",
                "connections": 0,
                "created": datetime.now(),
                "last_activity": datetime.now()
            }
            
            logger.info(f"Created new instance: {instance_id} on port {port}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to create instance {instance_id}: {e}")
            return None
            
    @staticmethod
    def cleanup_idle_instances():
        """Remove instances that have been idle for too long."""
        cutoff_time = datetime.now() - timedelta(seconds=INSTANCE_TIMEOUT)
        
        for instance_id, health in list(instance_health.items()):
            if (health["connections"] == 0 and 
                health["last_activity"] < cutoff_time and
                health["status"] != "terminating"):
                
                logger.info(f"Terminating idle instance: {instance_id}")
                InstanceManager.terminate_instance(instance_id)
                
    @staticmethod
    def terminate_instance(instance_id: str):
        """Terminate a specific instance."""
        if instance_id not in instance_health:
            return
            
        try:
            container_id = instance_health[instance_id]["container_id"]
            container = docker_client.containers.get(container_id)
            container.stop()
            container.remove()
            
            instance_health[instance_id]["status"] = "terminated"
            logger.info(f"Terminated instance: {instance_id}")
            
        except Exception as e:
            logger.error(f"Failed to terminate instance {instance_id}: {e}")

@app.get("/")
async def root(request: Request, force_new: bool = False):
    """Root endpoint - assign session to available instance."""
    
    # Check if client already has a valid session cookie (unless forcing new)
    existing_session = request.cookies.get("session_id")
    existing_instance = request.cookies.get("instance_id")
    
    if not force_new and existing_session and existing_instance and existing_session in sessions:
        # Redirect to existing instance using the same host as the request
        port = instance_health.get(existing_instance, {}).get("port", 7861)
        host_header = request.headers.get("host", "localhost")
        host = host_header.split(":")[0]
        logger.info(f"Request host header: {host_header}, extracted host: {host}")
        logger.info(f"Redirecting existing session {existing_session} to instance {existing_instance} (port {port})")
        return RedirectResponse(f"http://{host}:{port}/", status_code=302)
    
    session_id = str(uuid.uuid4())
    
    # Simple round-robin assignment to existing instances
    available_instances = [
        {"id": "1", "port": 7861, "name": "cli-agent-instance-1"},
        {"id": "2", "port": 7862, "name": "cli-agent-instance-2"}, 
        {"id": "3", "port": 7863, "name": "cli-agent-instance-3"}
    ]
    
    # Find least loaded instance
    min_connections = float('inf')
    selected_instance = available_instances[0]  # Default fallback
    
    for instance in available_instances:
        instance_id = instance["id"]
        if instance_id not in instance_health:
            # Initialize instance health tracking
            instance_health[instance_id] = {
                "port": instance["port"],
                "status": "running", 
                "connections": 0,
                "created": datetime.now(),
                "last_activity": datetime.now()
            }
        
        connections = instance_health[instance_id]["connections"]
        if connections < min_connections:
            min_connections = connections
            selected_instance = instance
    
    instance_id = selected_instance["id"]
    port = selected_instance["port"]
    
    # Register session
    sessions[session_id] = {
        "instance_id": instance_id,
        "created": datetime.now(),
        "last_activity": datetime.now()
    }
    
    # Update instance connection count
    instance_health[instance_id]["connections"] += 1
    instance_health[instance_id]["last_activity"] = datetime.now()
    
    # Create response with session cookie and redirect using the same host as the request
    host_header = request.headers.get("host", "localhost")
    host = host_header.split(":")[0]
    logger.info(f"Request host header: {host_header}, extracted host: {host}")
    response = RedirectResponse(f"http://{host}:{port}/", status_code=302)
    response.set_cookie(
        "session_id", 
        session_id, 
        max_age=INSTANCE_TIMEOUT,
        httponly=True
    )
    response.set_cookie(
        "instance_id", 
        instance_id, 
        max_age=INSTANCE_TIMEOUT
    )
    
    logger.info(f"Assigned session {session_id} to instance {instance_id} (port {port}) - {min_connections} connections")
    return response

@app.get("/new")
async def new_session(request: Request):
    """Force create a new session, ignoring existing cookies."""
    return await root(request, force_new=True)

@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = sessions[session_id]
    instance_id = session["instance_id"]
    
    return {
        "session_id": session_id,
        "instance_id": instance_id,
        "port": instance_health.get(instance_id, {}).get("port"),
        "status": instance_health.get(instance_id, {}).get("status"),
        "created": session["created"],
        "last_activity": session["last_activity"]
    }

@app.delete("/api/session/{session_id}")
async def end_session(session_id: str):
    """End a specific session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = sessions[session_id]
    instance_id = session["instance_id"]
    
    # Decrease connection count
    if instance_id in instance_health:
        instance_health[instance_id]["connections"] = max(0, instance_health[instance_id]["connections"] - 1)
        
    # Remove session
    del sessions[session_id]
    
    logger.info(f"Ended session {session_id} for instance {instance_id}")
    return {"message": "Session ended"}

@app.get("/api/instances")
async def get_instances():
    """Get status of all instances."""
    return {
        "instances": instance_health,
        "sessions": len(sessions),
        "max_instances": MAX_INSTANCES
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "instances": len(instance_health)}

# Background task for cleanup
@app.on_event("startup")
async def startup():
    """Start background cleanup task."""
    asyncio.create_task(cleanup_task())

async def cleanup_task():
    """Background task to cleanup idle instances and sessions."""
    while True:
        try:
            # Cleanup idle instances
            InstanceManager.cleanup_idle_instances()
            
            # Cleanup expired sessions
            cutoff_time = datetime.now() - timedelta(seconds=INSTANCE_TIMEOUT)
            expired_sessions = [
                sid for sid, session in sessions.items() 
                if session["last_activity"] < cutoff_time
            ]
            
            for session_id in expired_sessions:
                await end_session(session_id)
                
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            
        await asyncio.sleep(60)  # Run every minute