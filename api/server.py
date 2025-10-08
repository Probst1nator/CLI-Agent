"""
Web server for CLI-Agent GUI mode.

Provides aiohttp server with WebSocket support, session management,
and audit logging. Auto-starts when visualresponse tool is called,
auto-stops when last client disconnects (unless --gui flag is set).
"""

import asyncio
import hashlib
import json
import logging
import socket
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from aiohttp import web
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    web = None
    aiohttp = None
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Server configuration
SERVER_HOST = "0.0.0.0"
PORT_RANGE_START = 7860
PORT_RANGE_END = 7870
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


class SessionAuditLogger:
    """
    Audit logging for all client sessions.
    Complies with professional logging standards (GDPR-aware, structured).
    """

    def __init__(self, base_path: Path):
        self.base_path = base_path / "audit_logs" / "sessions"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_path / "index.json"
        self.active_sessions: Dict[str, 'SessionLog'] = {}

    def start_session(self, client_ip: str, user_agent: str,
                      forwarded_for: Optional[str] = None) -> str:
        """Create new session log"""
        session_id = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{self._gen_hash()}"

        session = SessionLog(
            session_id=session_id,
            client_ip=client_ip,
            user_agent=user_agent,
            forwarded_for=forwarded_for,
            log_path=self.base_path / f"session_{session_id}.json"
        )

        self.active_sessions[session_id] = session
        session.log_event("connection", "websocket_connected")

        self._update_index(session_id, "started")
        logger.info(f"Started session: {session_id} from {client_ip}")
        return session_id

    def end_session(self, session_id: str, chat_archive_path: Optional[str] = None):
        """Finalize and write session log"""
        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]
        session.end_time = datetime.utcnow()
        session.log_event("connection", "websocket_disconnected")

        if chat_archive_path:
            session.chat_snapshot = {
                "archived_at": datetime.utcnow().isoformat(),
                "archive_path": chat_archive_path
            }

        session.write_to_disk()
        del self.active_sessions[session_id]

        self._update_index(session_id, "completed")
        logger.info(f"Ended session: {session_id}")

    def log_http_request(self, session_id: str, method: str, path: str, status: int):
        """Log HTTP request"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].log_event("http_request", {
                "method": method,
                "path": path,
                "status": status
            })

    def log_chat_message(self, session_id: str, role: str,
                         content: str, message_id: str, model: Optional[str] = None):
        """Log chat interaction with content hash (not full content for privacy)"""
        if session_id in self.active_sessions:
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            event_data = {
                "action": f"{role}_message",
                "message_id": message_id,
                "content_hash": content_hash
            }
            if model:
                event_data["model"] = model

            self.active_sessions[session_id].log_event("chat_interaction", event_data)

    def _gen_hash(self) -> str:
        """Generate 4-char hash for session ID"""
        return hashlib.md5(str(datetime.utcnow().timestamp()).encode()).hexdigest()[:4]

    def _update_index(self, session_id: str, status: str):
        """Update session index"""
        index = self._load_index()
        index[session_id] = {
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.index_path.write_text(json.dumps(index, indent=2))

    def _load_index(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {}

    def cleanup_old_logs(self, retention_days: int = 90):
        """Delete audit logs older than retention period"""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        for log_file in self.base_path.glob("session_*.json"):
            try:
                log_data = json.loads(log_file.read_text())
                log_time = datetime.fromisoformat(log_data["start_time"])

                if log_time < cutoff:
                    log_file.unlink()
                    logger.info(f"Deleted old audit log: {log_file.name}")
            except Exception as e:
                logger.warning(f"Failed to clean up log {log_file}: {e}")


class SessionLog:
    """Individual session log"""

    def __init__(self, session_id: str, client_ip: str,
                 user_agent: str, forwarded_for: Optional[str], log_path: Path):
        self.session_id = session_id
        self.start_time = datetime.utcnow()
        self.end_time = None
        self.client_info = {
            "ip": client_ip,
            "user_agent": user_agent,
            "forwarded_for": forwarded_for
        }
        self.events: List[Dict[str, Any]] = []
        self.chat_snapshot: Dict[str, Any] = {}
        self.metrics = {
            "api_calls": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0
        }
        self.log_path = log_path

    def log_event(self, event_type: str, event_data: Any):
        """Add event to log"""
        self.events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "event": event_data
        })

        if event_type == "http_request":
            self.metrics["api_calls"] += 1
        elif event_type == "chat_interaction":
            if "user" in str(event_data):
                self.metrics["messages_sent"] += 1
            else:
                self.metrics["messages_received"] += 1

    def write_to_disk(self):
        """Write final log to disk"""
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        self.metrics["duration_seconds"] = duration

        log_data = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "client_info": self.client_info,
            "events": self.events,
            "chat_snapshot": self.chat_snapshot,
            "metrics": self.metrics
        }

        self.log_path.write_text(json.dumps(log_data, indent=2))


class VisualServer:
    """
    Web server for CLI-Agent GUI mode.

    Auto-starts when visualresponse tool is called.
    Auto-stops when last WebSocket disconnects (unless --gui flag set).
    """

    def __init__(self, persist_mode: bool = False):
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available - install with: pip install aiohttp")

        self.app = web.Application()
        self.persist_mode = persist_mode  # True when started with --gui
        self.websocket_clients: Set[web.WebSocketResponse] = set()
        self.message_history: List[dict] = []
        self.session_map: Dict[web.WebSocketResponse, str] = {}  # WebSocket -> session_id
        self.port: Optional[int] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        # Session audit logging
        from core.globals import g
        audit_base = Path(g.CONFIG_DIR if hasattr(g, 'CONFIG_DIR') else Path.home() / ".cli-agent")
        self.audit_logger = SessionAuditLogger(audit_base)

        # Setup routes and middleware
        self.setup_routes()
        self.setup_middleware()

    def setup_routes(self):
        """Register all routes"""
        # WebSocket
        self.app.router.add_get('/ws', self.websocket_handler)

        # Static files
        self.app.router.add_static('/css/', path=FRONTEND_DIR / 'css', name='css')
        self.app.router.add_static('/js/', path=FRONTEND_DIR / 'js', name='js')
        self.app.router.add_static('/libs/', path=FRONTEND_DIR / 'libs', name='libs')

        # Serve files from working directory
        working_dir = Path.cwd()
        self.app.router.add_static('/files/', path=working_dir, name='working_dir', show_index=False)

        # Main HTML page
        self.app.router.add_get('/', self.index_handler)

        # Register API routes
        from api import chat_api, audio_api, tools_api, profile_api, agent_state_api
        chat_api.setup_routes(self.app)
        audio_api.setup_routes(self.app)
        tools_api.setup_routes(self.app)
        profile_api.setup_routes(self.app)
        agent_state_api.setup_routes(self.app)

        logger.info("All routes registered")

    def setup_middleware(self):
        """Setup middleware for logging and session tracking"""
        @web.middleware
        async def session_logging_middleware(request, handler):
            """Log HTTP requests with session ID"""
            # Extract session ID from header (if available)
            session_id = request.headers.get('X-Session-ID')

            response = await handler(request)

            # Log request if we have a session ID
            if session_id and session_id != 'null':
                self.audit_logger.log_http_request(
                    session_id,
                    request.method,
                    request.path,
                    response.status
                )

            return response

        self.app.middlewares.append(session_logging_middleware)

    async def index_handler(self, request):
        """Serve main HTML page with cache-busting headers"""
        index_path = FRONTEND_DIR / 'index.html'
        response = web.FileResponse(index_path)
        # Prevent caching to ensure latest version is always loaded
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    async def websocket_handler(self, request):
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Extract client info
        client_ip = request.remote
        user_agent = request.headers.get('User-Agent', 'Unknown')
        forwarded_for = request.headers.get('X-Forwarded-For')

        # Start session logging
        session_id = self.audit_logger.start_session(client_ip, user_agent, forwarded_for)
        self.session_map[ws] = session_id

        # Send session init message
        await ws.send_json({
            'type': 'session_init',
            'session_id': session_id,
            'timestamp': time.time()
        })

        # Add to active clients
        self.websocket_clients.add(ws)
        logger.info(f"WebSocket client connected (session: {session_id}). Total clients: {len(self.websocket_clients)}")

        # Send current agent state to new client
        try:
            from core.globals import g
            state = {
                'voice_mode': getattr(g, 'voice_mode', False),
                'tts_only_mode': getattr(g, 'tts_only_mode', False),
                'auto_execute': getattr(g, 'auto_execute', False),
                'online_mode': getattr(g, 'online_mode', False),
                'local_mode': getattr(g, 'local_mode', False)
            }
            await ws.send_json({
                'type': 'state_changed',
                'state': state,
                'timestamp': time.time()
            })
        except Exception as e:
            logger.warning(f"Failed to send initial state to client: {e}")

        # Send message history to new client
        for message in self.message_history:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send history message: {e}")
                break

        try:
            # Handle incoming messages
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self.handle_websocket_message(ws, session_id, msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')

        except Exception as e:
            logger.error(f"WebSocket error: {e}")

        finally:
            # Client disconnected
            self.websocket_clients.discard(ws)
            logger.info(f"WebSocket client disconnected (session: {session_id}). Remaining clients: {len(self.websocket_clients)}")

            # End session and archive chat
            from core.globals import g
            chat_path = None
            if hasattr(g, 'chat') and g.chat:
                try:
                    g.chat.save_to_json(also_archive=True)
                    chat_path = f"chat_{session_id}.json"
                except Exception as e:
                    logger.warning(f"Failed to archive chat: {e}")

            self.audit_logger.end_session(session_id, chat_path)

            # Auto-stop server if no clients remain (unless persist mode)
            if not self.persist_mode and len(self.websocket_clients) == 0:
                logger.info("Last client disconnected and not in persist mode - stopping server")
                asyncio.create_task(self.stop())

        return ws

    async def handle_websocket_message(self, ws: web.WebSocketResponse, session_id: str, data: str):
        """Handle incoming WebSocket messages"""
        try:
            message_data = json.loads(data) if data else {}

            # Handle different message types
            if isinstance(message_data, dict):
                message_type = message_data.get('type')

                if message_type == 'permission_response':
                    # Permission response - forward to input queue
                    from core.globals import g
                    if hasattr(g, 'visual_input_queue'):
                        g.visual_input_queue.put_nowait(message_data)

                elif message_type == 'interrupt_signal':
                    # Interrupt signal
                    logger.info(f"Interrupt signal received from session {session_id}")
                    from core.globals import g
                    if hasattr(g, 'visual_input_queue'):
                        g.visual_input_queue.put_nowait({
                            'type': 'interrupt_command',
                            'source': 'web_ui',
                            'timestamp': time.time()
                        })

                elif message_type == 'image_upload':
                    # Screen capture image upload
                    logger.info(f"Image upload received from session {session_id}")
                    from core.globals import g

                    base64_image = message_data.get('image')
                    if base64_image:
                        # Strip data URI prefix if present (data:image/png;base64,)
                        if base64_image.startswith('data:image'):
                            base64_image = base64_image.split(',', 1)[1]
                            logger.debug(f"Stripped data URI prefix, now {len(base64_image)} chars")

                        # Validate base64 length (must be multiple of 4)
                        if len(base64_image) % 4 != 0:
                            # Pad with = to make it valid base64
                            padding_needed = 4 - (len(base64_image) % 4)
                            base64_image = base64_image + ('=' * padding_needed)
                            logger.debug(f"Padded base64 image with {padding_needed} = characters")

                        # Add to chat's base64_images for archival
                        if hasattr(g, 'chat') and g.chat:
                            if not hasattr(g.chat, 'base64_images'):
                                g.chat.base64_images = []
                            g.chat.base64_images.append(base64_image)
                            logger.info(f"Added base64 image to chat archive ({len(base64_image)} chars)")

                        # Add to current_images for immediate LLM use
                        if not hasattr(g, 'current_images'):
                            g.current_images = []
                        g.current_images.append(base64_image)

                        # Send confirmation back to client
                        await ws.send_json({
                            'type': 'image_upload_ack',
                            'status': 'success',
                            'timestamp': time.time()
                        })
                        logger.info("Image upload acknowledged")
                    else:
                        logger.warning("Image upload message missing 'image' field")

                elif isinstance(message_data, dict) and 'message' in message_data:
                    # Regular user message with optional images
                    user_message = message_data.get('message', '').strip()
                    images = message_data.get('images', [])

                    if user_message:
                        logger.info(f"Received user message via WebSocket: {user_message[:50]}... with {len(images)} images")
                        from core.globals import g

                        if images:
                            if not hasattr(g, 'current_images'):
                                g.current_images = []
                            for image in images:
                                if image.startswith('data:image'):
                                    image = image.split(',', 1)[1]
                                if len(image) % 4 != 0:
                                    padding_needed = 4 - (len(image) % 4)
                                    image = image + ('=' * padding_needed)
                                g.current_images.append(image)
                            logger.info(f"Added {len(images)} images to current_images")

                        if hasattr(g, 'visual_input_queue'):
                            g.visual_input_queue.put_nowait(user_message)
                            logger.info("Message added to visual_input_queue")

                        # Log chat message
                        self.audit_logger.log_chat_message(
                            session_id,
                            'user',
                            user_message,
                            f"msg_{int(time.time())}"
                        )
                else:
                    # Unknown message - treat as text input
                    logger.info(f"Received unknown message type, treating as text: {str(data)[:50]}...")
                    from core.globals import g
                    if hasattr(g, 'visual_input_queue'):
                        g.visual_input_queue.put_nowait(data)

        except json.JSONDecodeError:
            # Not JSON - treat as plain text
            logger.info(f"Received non-JSON message, treating as plain text: {str(data)[:50]}...")
            from core.globals import g
            if hasattr(g, 'visual_input_queue'):
                g.visual_input_queue.put_nowait(data)
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not isinstance(message, dict):
            logger.warning(f"Attempted to broadcast non-dict message: {type(message)}")
            return

        # Add to message history (but skip ephemeral sync events like message_added)
        # These events are already in the chat state and will be loaded via /api/chat/current
        if message.get('type') not in ['message_added', 'chat_replaced', 'chat_cleared', 'message_deleted', 'message_edited', 'chat_truncated']:
            self.message_history.append(message)

            # Limit history size
            if len(self.message_history) > 1000:
                self.message_history = self.message_history[-1000:]

        # Broadcast to all clients
        dead_clients = set()
        for ws in self.websocket_clients:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                dead_clients.add(ws)

        # Clean up dead clients
        for ws in dead_clients:
            self.websocket_clients.discard(ws)

    async def broadcast_state_change(self):
        """Broadcast current agent state to all connected clients"""
        try:
            from core.globals import g

            state = {
                'voice_mode': getattr(g, 'voice_mode', False),
                'tts_only_mode': getattr(g, 'tts_only_mode', False),
                'auto_execute': getattr(g, 'auto_execute', False),
                'online_mode': getattr(g, 'online_mode', False),
                'local_mode': getattr(g, 'local_mode', False)
            }

            await self.broadcast({
                'type': 'state_changed',
                'state': state,
                'timestamp': time.time()
            })
        except Exception as e:
            logger.error(f"Failed to broadcast state change: {e}")

    async def start(self, open_browser: bool = True) -> str:
        """Start the server and return URL"""
        # Find available port
        self.port = self._find_available_port()
        if not self.port:
            raise RuntimeError(f"No available ports in range {PORT_RANGE_START}-{PORT_RANGE_END}")

        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, SERVER_HOST, self.port)

        await self.site.start()
        url = f"http://127.0.0.1:{self.port}"  # Use 127.0.0.1 for browser
        logger.info(f"Server started at {url}")

        # Open browser if requested
        if open_browser:
            try:
                webbrowser.open(url)
                logger.info(f"Opening browser at {url}")
            except Exception as e:
                logger.warning(f"Failed to open browser: {e}")

        # Cleanup old audit logs
        try:
            self.audit_logger.cleanup_old_logs()
        except Exception as e:
            logger.warning(f"Failed to cleanup old logs: {e}")

        return url

    async def stop(self):
        """Stop the server"""
        logger.info("Stopping server...")

        # Close all WebSocket connections
        for ws in list(self.websocket_clients):
            await ws.close()

        # Stop server
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

        logger.info("Server stopped")

    def _find_available_port(self) -> Optional[int]:
        """Find an available port in the configured range"""
        for port in range(PORT_RANGE_START, PORT_RANGE_END + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return None


# Global server instance
_server_instance: Optional[VisualServer] = None


async def get_or_create_server(persist_mode: bool = False, open_browser: bool = True) -> VisualServer:
    """Get existing server or create new one"""
    global _server_instance

    if _server_instance is None:
        _server_instance = VisualServer(persist_mode=persist_mode)
        await _server_instance.start(open_browser=open_browser)

    return _server_instance


def get_server() -> Optional[VisualServer]:
    """Get existing server instance (if running)"""
    return _server_instance


async def broadcast_message_added(message) -> None:
    """
    Broadcast message_added event to all WebSocket clients.
    Called when a message is added or updated in context_chat.

    The frontend uses simple logic: if last message is assistant and incoming is assistant,
    replace it (content/metadata has grown). Otherwise add new message.

    Args:
        message: Message object from Chat.messages list
    """
    server = get_server()
    if not server:
        return

    try:

        msg_dict = {
            'role': message.role.name.lower() if hasattr(message.role, 'name') else str(message.role),
            'content': message.content,
            'images': message.images if hasattr(message, 'images') else [],
            'metadata': message.metadata if hasattr(message, 'metadata') else {}
        }

        payload = {
            'type': 'message_added',
            'message': msg_dict
        }

        await server.broadcast(payload)
        logger.debug(f"Broadcasted message_added event: {message.role} with {len(msg_dict.get('metadata', {}).get('tool_info', []))} tool calls")

    except Exception as e:
        logger.error(f"Failed to broadcast message_added: {e}", exc_info=True)
