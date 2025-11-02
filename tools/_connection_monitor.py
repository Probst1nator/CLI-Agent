"""
Connection monitoring tool for CLI-Agent.
Monitors WebSocket connections and triggers process restart when no clients are connected.
"""

import asyncio
import logging
import os
import signal


class ConnectionMonitor:
    """Monitors active connections and handles auto-restart logic."""
    
    def __init__(self, idle_timeout: int = 300):  # 5 minutes default
        self.idle_timeout = idle_timeout
        self.last_activity = asyncio.get_event_loop().time()
        self.monitoring_task = None
        self.clients: set = set()
        self.logger = logging.getLogger("connection_monitor")
        
    def register_client(self, client):
        """Register a new WebSocket client."""
        self.clients.add(client)
        self.last_activity = asyncio.get_event_loop().time()
        self.logger.info(f"Client connected. Active connections: {len(self.clients)}")
        
    def unregister_client(self, client):
        """Unregister a WebSocket client."""
        self.clients.discard(client)
        self.last_activity = asyncio.get_event_loop().time()
        self.logger.info(f"Client disconnected. Active connections: {len(self.clients)}")
        
    def has_active_connections(self) -> bool:
        """Check if there are any active connections."""
        # Clean up closed connections
        self.clients = {client for client in self.clients if not client.closed}
        return len(self.clients) > 0
        
    async def start_monitoring(self):
        """Start the connection monitoring loop."""
        self.logger.info(f"Starting connection monitor (idle timeout: {self.idle_timeout}s)")
        
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            current_time = asyncio.get_event_loop().time()
            idle_time = current_time - self.last_activity
            
            if not self.has_active_connections():
                self.logger.info(f"No active connections. Idle time: {idle_time:.1f}s")
                
                if idle_time >= self.idle_timeout:
                    self.logger.warning("Idle timeout reached with no connections. Restarting process...")
                    await self._restart_process()
                    break
            else:
                # Reset idle timer when we have connections
                self.last_activity = current_time
                
    async def _restart_process(self):
        """Restart the CLI-Agent process."""
        self.logger.info("Initiating process restart...")
        
        # Gracefully close all connections
        for client in self.clients:
            try:
                await client.close(code=1000, message=b'Server restarting - no active sessions')
            except Exception:
                pass
                
        # Signal the main container process (PID 1) to restart
        # In Docker, this will cause the container to restart if restart policy is set
        # We need to kill PID 1, not the current subprocess
        try:
            os.kill(1, signal.SIGTERM)
            self.logger.info("Sent SIGTERM to main container process (PID 1)")
        except Exception as e:
            self.logger.error(f"Failed to signal main process: {e}")
            # Fallback: kill current process (may not trigger container restart)
            os.kill(os.getpid(), signal.SIGTERM)

# Global instance
connection_monitor = ConnectionMonitor()