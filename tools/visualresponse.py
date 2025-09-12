# tools/visualresponse.py
"""
Implements the 'visualresponse' tool for rich, interactive, web-based visualizations.
Supports Markdown, HTML, 3Dmol.js for molecular structures, and Plotly.js for 2D/3D charts.

Requires: pip install aiohttp
"""

import multiprocessing
import logging
import asyncio
import json
import re
import webbrowser
import socket
from pathlib import Path
from typing import Optional, Set, Any, List
from termcolor import colored
from queue import Empty as QueueEmpty

try:
    from aiohttp import web
    import aiohttp
    AIOHTTP_VERSION = aiohttp.__version__
    AIOHTTP_AVAILABLE = True
except ImportError as e:
    web = None
    AIOHTTP_VERSION = None
    AIOHTTP_AVAILABLE = False
    AIOHTTP_IMPORT_ERROR = str(e)

# Import BeautifulSoup for HTML parsing and sanitization
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None
    BS4_AVAILABLE = False

from core.globals import g

# --- Server Configuration ---
SERVER_HOST = "127.0.0.1"
PORT_RANGE_START = 7860
PORT_RANGE_END = 7870
ASSETS_DIR = Path(__file__).parent / "visual_assets"

def find_open_port(start_port, end_port, host="127.0.0.1"):
    """Finds an available TCP port in a given range with detailed diagnostics."""
    tested_ports = []
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                logging.debug(f"Port {port} is available")
                return port
            except OSError as e:
                tested_ports.append((port, str(e)))
                logging.debug(f"Port {port} is unavailable: {e}")
                continue
    
    # Log all tested ports for debugging
    logging.warning(f"No available ports found in range {start_port}-{end_port}")
    for port, error in tested_ports[-5:]:  # Show last 5 attempts
        logging.debug(f"Port {port}: {error}")
    
    return None

def test_port_connectivity(host, port, timeout=5):
    """Test if a port is actually accessible for connections."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((host, port))
            return result == 0
    except Exception as e:
        logging.debug(f"Port connectivity test failed for {host}:{port}: {e}")
        return False

def _server_process_main_loop(output_queue: multiprocessing.Queue, input_queue: multiprocessing.Queue, port: int):
    """The entry point for the dedicated aiohttp server process."""
    
    # Import required modules first
    import os
    from pathlib import Path
    
    # Immediate console output to diagnose issues
    print(f"[DEBUG] Visual server subprocess starting, PID: {os.getpid()}")
    print(f"[DEBUG] Target port: {port}")
    
    try:
        # Set up proper logging for the subprocess
        # Create log directory - use a more accessible location for subprocess
        log_dir = Path("/tmp") / "cli-agent-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "visual_server.log"
        print(f"[DEBUG] Log directory created: {log_dir}")
        
        # Also try to create the standard location as backup
        try:
            standard_log_dir = Path.home() / ".cli-agent" / "logs"
            standard_log_dir.mkdir(parents=True, exist_ok=True)
            backup_log_file = standard_log_dir / "visual_server.log"
            print(f"[DEBUG] Backup log directory created: {standard_log_dir}")
        except Exception as e:
            print(f"[DEBUG] Failed to create backup log directory: {e}")
            backup_log_file = None
    except Exception as e:
        print(f"[DEBUG] Failed during log setup: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Configure logging for this subprocess
    import logging
    subprocess_logger = logging.getLogger('visualserver')
    subprocess_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers to avoid duplicates
    subprocess_logger.handlers.clear()
    
    # File handler with rotation
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter with timestamp and process info
    formatter = logging.Formatter(
        '%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    subprocess_logger.addHandler(file_handler)
    subprocess_logger.addHandler(console_handler)
    
    subprocess_logger.info(f"Visual server subprocess starting on port {port}")
    subprocess_logger.info(f"Log file: {log_file}")
    subprocess_logger.info(f"Assets directory: {ASSETS_DIR}")
    subprocess_logger.info(f"Assets directory exists: {ASSETS_DIR.exists()}")
    
    # Check aiohttp availability with detailed diagnostics
    if not AIOHTTP_AVAILABLE:
        error_msg = f"FATAL ERROR: aiohttp is not available. Import error: {AIOHTTP_IMPORT_ERROR}"
        subprocess_logger.error(error_msg)
        subprocess_logger.error("Please run 'pip install aiohttp' to install the required dependency")
        print(f"[VisualServer] {error_msg}")
        return
    else:
        subprocess_logger.info(f"aiohttp version {AIOHTTP_VERSION} is available")
    
    # Verify assets directory
    if not ASSETS_DIR.exists():
        error_msg = f"FATAL ERROR: Assets directory not found: {ASSETS_DIR}"
        subprocess_logger.error(error_msg)
        print(f"[VisualServer] {error_msg}")
        return
    
    # Check critical asset files
    required_files = ['index.html', 'favicon.ico']
    missing_files = []
    for file_name in required_files:
        file_path = ASSETS_DIR / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            subprocess_logger.debug(f"Found required asset: {file_path}")
    
    if missing_files:
        error_msg = f"Missing required asset files: {missing_files}"
        subprocess_logger.error(error_msg)
        print(f"[VisualServer] {error_msg}")
        return

    clients: Set[web.WebSocketResponse] = set()
    message_history: List[Any] = []

    async def broadcast_updates():
        """Checks queue for new messages and broadcasts them to all clients."""
        while True:
            try:
                message = output_queue.get_nowait()
                if message == "__EXIT__":
                    break
                
                message_history.append(message)
                
                if clients:
                    payload = json.dumps(message)
                    tasks = [asyncio.create_task(ws.send_str(payload)) for ws in clients]
                    await asyncio.gather(*tasks, return_exceptions=True)
            except QueueEmpty:
                await asyncio.sleep(0.1)
        
        for ws in clients:
            await ws.close(code=1000, message=b'Server shutdown')

    async def websocket_handler(request: web.Request):
        """Handles new WebSocket connections, sends history, and listens for user input."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clients.add(ws)
        subprocess_logger.info(f"Browser client connected: {request.remote}")
        
        # Send message history to new clients
        subprocess_logger.debug(f"Message history contains {len(message_history)} messages")
        if message_history:
            try:
                for i, message in enumerate(message_history):
                    subprocess_logger.debug(f"Sending message {i+1}/{len(message_history)} to new client")
                    await ws.send_str(json.dumps(message))
                subprocess_logger.info(f"Successfully sent all {len(message_history)} messages to new client")
            except ConnectionResetError:
                subprocess_logger.warning("Connection reset while sending history to new client")
                pass
        else:
            subprocess_logger.debug("No message history to send to new client")
        
        try:
            # Listen for incoming messages from the browser client
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    subprocess_logger.info(f"Received message from browser: '{msg.data}'")
                    # Put the received user input into the input_queue for the main process
                    input_queue.put(msg.data)
                elif msg.type == web.WSMsgType.ERROR:
                    subprocess_logger.error(f'WebSocket connection closed with exception {ws.exception()}')
        finally:
            subprocess_logger.info(f"Browser client disconnected: {request.remote}")
            clients.remove(ws)
        return ws

    async def index_handler(request: web.Request):
        """Serves the main index.html file."""
        return web.FileResponse(ASSETS_DIR / 'index.html')

    async def favicon_handler(request: web.Request):
        """Serves the favicon.ico file."""
        return web.FileResponse(ASSETS_DIR / 'favicon.ico')

    async def main():
        """Sets up and runs the aiohttp web server application."""
        app = web.Application()
        app.router.add_get('/', index_handler)
        app.router.add_get('/ws', websocket_handler)
        app.router.add_get('/favicon.ico', favicon_handler)
        
        # Add a route to serve static files from the 'libs' directory
        app.router.add_static('/libs/',
                              path=ASSETS_DIR / 'libs',
                              name='libs')

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, SERVER_HOST, port)
        
        try:
            await site.start()
            subprocess_logger.info(f"Server started successfully at http://{SERVER_HOST}:{port}")
            
            # Verify the server is actually listening
            import asyncio
            await asyncio.sleep(0.1)  # Give server time to bind
            
            # Test local connectivity
            try:
                import urllib.request
                test_url = f"http://{SERVER_HOST}:{port}/"
                req = urllib.request.Request(test_url, headers={'User-Agent': 'CLI-Agent-Test'})
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        subprocess_logger.info("Server connectivity verification passed")
                    else:
                        subprocess_logger.warning(f"Server responded with unexpected status: {response.status}")
            except Exception as connectivity_error:
                subprocess_logger.warning(f"Server connectivity verification failed: {connectivity_error}")
            
            subprocess_logger.info("Server is now accepting connections")
            await broadcast_updates()
        except OSError as e:
            if "Address already in use" in str(e):
                subprocess_logger.error(f"Port {port} is already in use by another process")
            else:
                subprocess_logger.error(f"Failed to bind to port {port}: {e}")
            raise
        except Exception as e:
            subprocess_logger.error(f"Failed to start server: {e}", exc_info=True)
            raise
        finally:
            await runner.cleanup()
            subprocess_logger.info("Server exited cleanly")

    try:
        subprocess_logger.info("Starting asyncio main loop")
        print("[DEBUG] About to start asyncio.run(main())")
        asyncio.run(main())
    except Exception as e:
        error_msg = f"FATAL ERROR in server process: {e}"
        subprocess_logger.error(error_msg, exc_info=True)
        print(f"[DEBUG] {error_msg}")
        import traceback
        traceback.print_exc()
        # Exit with error code to signal failure
        import sys
        sys.exit(1)

class VisualServerSession:
    """Manages the lifecycle of the persistent visual response server process."""
    def __init__(self, register_cleanup: bool = True):
        self.ctx = multiprocessing.get_context()
        self.output_queue = self.ctx.Queue()
        self.input_queue = self.ctx.Queue()
        self.process: Optional[multiprocessing.Process] = None
        self.port: Optional[int] = None
        
        # Make the input_queue globally accessible for the main loop.
        # The main application loop should check this queue for user input. For example:
        # if hasattr(g, 'visual_input_queue') and not g.visual_input_queue.empty():
        #     user_message = g.visual_input_queue.get()
        #     args.m.push(role='user', content=user_message)
        g.visual_input_queue = self.input_queue

        self.start()
        
        # CRITICAL DEBUG: Log cleanup registration decision
        if register_cleanup:
            logging.debug("  - Visual server registering cleanup callback (register_cleanup=True)")
            g.register_cleanup(self.close)
        else:
            logging.debug("  - Visual server SKIPPING cleanup registration (register_cleanup=False)")


    def start(self):
        if self.process and self.process.is_alive():
            logging.debug("Visual server process already running")
            return
        
        logging.info("Starting visual server process...")
        
        self.port = find_open_port(PORT_RANGE_START, PORT_RANGE_END, SERVER_HOST)
        if self.port is None:
            logging.error(colored(f"  - Could not find an open port in range {PORT_RANGE_START}-{PORT_RANGE_END}.", "red"))
            return

        logging.info(colored(f"  - Spawning visual response server on port {self.port}...", "cyan"))
        
        # Create the subprocess with better error handling
        try:
            self.process = self.ctx.Process(target=_server_process_main_loop, args=(self.output_queue, self.input_queue, self.port))
            self.process.start()
            logging.info(f"Visual server subprocess started with PID: {self.process.pid}")
            
            # Give the server a moment to start and check if it's still alive
            import time
            time.sleep(1.0)
            
            if not self.process.is_alive():
                logging.error(f"Visual server process died immediately after start. Exit code: {self.process.exitcode}")
                return
            else:
                logging.info("Visual server process is running successfully")
            
        except Exception as e:
            logging.error(f"Failed to start visual server process: {e}", exc_info=True)
            return
        
        server_url = f"http://{SERVER_HOST}:{self.port}"
        logging.info(colored(f"  - Visual response UI available at {server_url}", "cyan"))
        
        # Test server connectivity
        try:
            import urllib.request
            import time
            time.sleep(0.5)  # Give server time to bind
            
            # Quick connectivity test
            req = urllib.request.Request(server_url, headers={'User-Agent': 'CLI-Agent'})
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    logging.info("Server connectivity test passed")
                else:
                    logging.warning(f"Server responded with status {response.status}")
                    
        except Exception as e:
            logging.warning(f"Server connectivity test failed: {e}")
        
        # Try to open browser
        try:
            webbrowser.open(server_url)
            logging.info("Browser opened successfully")
        except Exception as e:
            logging.warning(colored(f"  - Could not automatically open browser: {e}", "yellow"))

    def send_update(self, data: dict):
        if not self.process or not self.process.is_alive():
            logging.warning("Visual server process not running, attempting to restart")
            self.start()
        
        if self.process and self.process.is_alive():
            try:
                self.output_queue.put(data)
                logging.debug(f"Sent update to visual server: {data.get('type', 'unknown')}")
            except Exception as e:
                logging.error(f"Failed to send update to visual server: {e}")
        else:
            logging.error("Cannot send update: visual server process failed to start")
    
    def is_healthy(self):
        """Check if the visual server is running and responsive."""
        if not self.process or not self.process.is_alive():
            return False
        
        if not self.port:
            return False
        
        # Test if the server is actually responding
        return test_port_connectivity(SERVER_HOST, self.port, timeout=2)
    
    def get_status(self):
        """Get detailed status information about the visual server."""
        status = {
            'process_alive': self.process and self.process.is_alive(),
            'process_pid': self.process.pid if self.process else None,
            'port': self.port,
            'server_responsive': False,
            'queue_size': None
        }
        
        if self.process:
            status['process_exitcode'] = self.process.exitcode
        
        if self.port:
            status['server_responsive'] = test_port_connectivity(SERVER_HOST, self.port, timeout=1)
        
        try:
            status['queue_size'] = self.output_queue.qsize()
        except:
            pass
        
        return status

    def close(self):
        if self.process and self.process.is_alive():
            logging.info(colored(f"  - Closing visual response server (PID: {self.process.pid})...", "cyan"))
            self.output_queue.put("__EXIT__")
            self.process.join(timeout=3)
            if self.process.is_alive():
                logging.warning(colored("  - Server process did not exit gracefully, terminating.", "yellow"))
                self.process.terminate()
            logging.info(colored("  - Visual response server closed.", "cyan"))
        self.process = None

        # Clean up the global input queue reference
        if hasattr(g, 'visual_input_queue'):
            delattr(g, 'visual_input_queue')

class visualresponse:
    _session: Optional[VisualServerSession] = None
    
    @staticmethod
    def close_session():
        """Manually close the visual server session - for explicit shutdown only."""
        if visualresponse._session:
            visualresponse._session.close()
            visualresponse._session = None

    @staticmethod
    def get_delim() -> str:
        return 'visualresponse'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "visualresponse",
            "description": (
                "Renders rich, interactive 2D and 3D visualizations in a live browser window. "
                "The UI also includes a chat input field to send messages back to the agent. "
                "Use the <type> tag to specify content type: "
                "'markdown', 'html', '3Dmol.js' (for PDB/SDF/XYZ data), or "
                "'plotly' (using Plotly.js for 2D/3D charts)."
            ),
            "example": (
                "<visualresponse><type>3Dmol.js</type><content>\n"
                '{"format": "pdb", "style": {"stick": {}}, "data": "PDB_DATA_HERE"}\n'
                "</content></visualresponse>"
            )
        }

    @staticmethod
    def _get_session() -> VisualServerSession:
        session_dead = (visualresponse._session is None or 
                        visualresponse._session.process is None or 
                        not visualresponse._session.process.is_alive())
        
        if session_dead:
            if visualresponse._session:
                logging.warning(colored("  - Visual server session was found dead. Restarting.", "yellow"))
                visualresponse._session.close()
            
            # This ensures the server persists after the tool call is complete,
            # allowing the user to view the content. The server will be cleaned up
            # when the main CLI-Agent process exits.
            logging.debug("  - Creating visual server session with register_cleanup=False")
            visualresponse._session = VisualServerSession(register_cleanup=False)
            logging.debug("  - Visual server session created successfully")

        return visualresponse._session

    @staticmethod
    def run(content: str) -> str:
        try:
            # Check if this is a diagnostic command
            if content.strip().lower() in ['status', 'test', 'diagnostic', 'debug']:
                return visualresponse._run_diagnostics()
            
            type_match = re.search(r'<type>(.*?)</type>', content, re.DOTALL)
            content_match = re.search(r'<content>(.*?)</content>', content, re.DOTALL)

            if not type_match or not content_match:
                return "Error: Invalid format. Must contain <type> and <content> tags."

            viz_type = type_match.group(1).strip().lower()
            viz_content_str = content_match.group(1).strip()
            
            supported_types = ['markdown', 'html', '3Dmol.js', 'plotly']
            if viz_type not in supported_types:
                return f"Error: Unsupported visual type '{viz_type}'. Supported types are: {', '.join(supported_types)}"
            
            # Process content based on type
            viz_content: Any  # Define the type for clarity
            
            # For types expecting JSON, parse the content string
            if viz_type in ['3Dmol.js', 'plotly']:
                try:
                    viz_content = json.loads(viz_content_str)
                except json.JSONDecodeError as e:
                    return f"Error: Content for type '{viz_type}' must be a valid JSON object. Details: {e}"
            
            # --- HTML SANITIZATION ---
            # If the type is HTML, parse it and extract only the body's content
            elif viz_type == 'html':
                if BS4_AVAILABLE:
                    try:
                        soup = BeautifulSoup(viz_content_str, 'html.parser')
                        # Check if a body tag exists
                        if soup.body:
                            # Extract the inner HTML of the body tag
                            viz_content = ''.join(str(c) for c in soup.body.contents).strip()
                            logging.debug(f"HTML sanitization: Extracted body content ({len(viz_content)} chars)")
                        else:
                            # If no body tag, assume the content is already a fragment
                            viz_content = viz_content_str
                            logging.debug("HTML sanitization: No body tag found, using content as-is")
                    except Exception as e:
                        logging.error(f"Error parsing HTML content: {e}")
                        viz_content = f"<pre>Error parsing HTML: {e}</pre>"
                else:
                    logging.warning("BeautifulSoup not available, HTML content not sanitized")
                    viz_content = viz_content_str
            
            else:  # This handles 'markdown' and other types
                viz_content = viz_content_str
            
            payload = {"type": viz_type, "content": viz_content}
            
            session = visualresponse._get_session()
            
            # Check server health before sending
            if not session.is_healthy():
                logging.warning("Visual server appears unhealthy, attempting restart")
                session.close()
                session = visualresponse._get_session()
            
            session.send_update(payload)
            
            # Verify the update was successful
            status = session.get_status()
            if status['server_responsive']:
                return f"Visual response of type '{viz_type}' sent to the live UI on port {session.port}."
            else:
                return "Visual response sent but server may not be responding. Check logs at ~/.cli-agent/logs/visual_server.log"
                
        except Exception as e:
            logging.error(f"Fatal error in visualresponse tool: {e}", exc_info=True)
            return f"Fatal error in visualresponse tool: {e}"
    
    @staticmethod
    def _run_diagnostics() -> str:
        """Run comprehensive diagnostics on the visual server."""
        diagnostics = []
        diagnostics.append("🔍 Visual Server Diagnostics")
        diagnostics.append("=" * 40)
        
        # Check aiohttp availability
        if AIOHTTP_AVAILABLE:
            diagnostics.append(f"✅ aiohttp {AIOHTTP_VERSION} is available")
        else:
            diagnostics.append(f"❌ aiohttp not available: {AIOHTTP_IMPORT_ERROR}")
        
        # Check assets directory
        if ASSETS_DIR.exists():
            diagnostics.append(f"✅ Assets directory exists: {ASSETS_DIR}")
            
            # Check required files
            required_files = ['index.html', 'favicon.ico']
            for file_name in required_files:
                file_path = ASSETS_DIR / file_name
                if file_path.exists():
                    size = file_path.stat().st_size
                    diagnostics.append(f"  ✅ {file_name} ({size} bytes)")
                else:
                    diagnostics.append(f"  ❌ {file_name} missing")
        else:
            diagnostics.append(f"❌ Assets directory missing: {ASSETS_DIR}")
        
        # Check current session
        if visualresponse._session:
            status = visualresponse._session.get_status()
            diagnostics.append("\n📊 Current Session Status:")
            diagnostics.append(f"  Process alive: {status['process_alive']}")
            diagnostics.append(f"  Process PID: {status['process_pid']}")
            diagnostics.append(f"  Port: {status['port']}")
            diagnostics.append(f"  Server responsive: {status['server_responsive']}")
            diagnostics.append(f"  Queue size: {status['queue_size']}")
            
            if status['process_exitcode'] is not None:
                diagnostics.append(f"  Exit code: {status['process_exitcode']}")
        else:
            diagnostics.append("\n📊 No active session")
        
        # Check port availability
        available_port = find_open_port(PORT_RANGE_START, PORT_RANGE_END)
        if available_port:
            diagnostics.append(f"\n🔌 Available port found: {available_port}")
        else:
            diagnostics.append(f"\n❌ No available ports in range {PORT_RANGE_START}-{PORT_RANGE_END}")
        
        # Check log files
        log_locations = [
            Path("/tmp") / "cli-agent-logs" / "visual_server.log",
            Path.home() / ".cli-agent" / "logs" / "visual_server.log"
        ]
        
        log_found = False
        for log_file in log_locations:
            if log_file.exists():
                size = log_file.stat().st_size
                diagnostics.append(f"\n📝 Log file: {log_file} ({size} bytes)")
                
                # Show recent log entries if small enough
                if size < 10000:  # Less than 10KB
                    try:
                        with open(log_file, 'r') as f:
                            recent_lines = f.readlines()[-10:]  # Last 10 lines
                        diagnostics.append("\n📋 Recent log entries:")
                        for line in recent_lines:
                            diagnostics.append(f"  {line.strip()}")
                    except:
                        pass
                log_found = True
                break
        
        if not log_found:
            diagnostics.append(f"\n📝 No log files found in: {[str(loc) for loc in log_locations]}")
        
        return "\n".join(diagnostics)