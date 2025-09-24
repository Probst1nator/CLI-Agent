# tools/visualresponse.py
"""
Implements the 'visualresponse' tool for rich, interactive, web-based visualizations.
Supports Markdown and fully self-contained HTML for maximum flexibility.

Requires: pip install aiohttp
"""

import multiprocessing
import asyncio
import json
import os
import sys
import time
import webbrowser
import socket
import re # Import the regular expression module
from pathlib import Path
from typing import Optional, Set, Any, List, Dict
from termcolor import colored
from queue import Empty as QueueEmpty

# Set up tool-specific logging
from shared.utils.tool_logging import get_logger_for_tool_module
logger = get_logger_for_tool_module(__name__)

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

from core.globals import g
from core.tool_context import tool_context

# --- Server Configuration ---
SERVER_HOST = "0.0.0.0"  # Changed from 127.0.0.1 for Docker compatibility
PORT_RANGE_START = 7860
PORT_RANGE_END = 7870
ASSETS_DIR = Path(__file__).parent / "visual_assets"

# --- All code from find_open_port to the VisualServerSession class remains unchanged ---
def find_open_port(start_port, end_port, host="127.0.0.1"):
    tested_ports = []
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                logger.debug(f"Port {port} is available")
                return port
            except OSError as e:
                tested_ports.append((port, str(e)))
                logger.debug(f"Port {port} is unavailable: {e}")
                continue
    logger.warning(f"No available ports found in range {start_port}-{end_port}")
    for port, error in tested_ports[-5:]:
        logger.debug(f"Port {port}: {error}")
    return None

def test_port_connectivity(host, port, timeout=5):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            result = s.connect_ex((host, port))
            return result == 0
    except Exception as e:
        logger.debug(f"Port connectivity test failed for {host}:{port}: {e}")
        return False

def _server_process_main_loop(output_queue: multiprocessing.Queue, input_queue: multiprocessing.Queue, port: int):
    # Signal handling for graceful subprocess interruption
    import os
    import signal
    from pathlib import Path

    # Setup signal handlers to prevent subprocess crash on interrupt
    def signal_handler(signum, frame):
        """Handle SIGINT/SIGTERM gracefully in subprocess."""
        print(f"[DEBUG] Visual server subprocess received interrupt signal {signum}")
        # Don't raise KeyboardInterrupt, just log and let cleanup happen naturally
        # The main loop will handle cleanup when asyncio.run exits
        return

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    print(f"[DEBUG] Visual server subprocess starting, PID: {os.getpid()}")
    print(f"[DEBUG] Target port: {port}")
    try:
        log_dir = Path("/tmp") / "cli-agent-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "visual_server.log"
        print(f"[DEBUG] Log directory created: {log_dir}")
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
    # Use centralized tool logging
    from shared.utils.tool_logging import setup_tool_logger
    subprocess_logger = setup_tool_logger("visualserver", debug_mode=True)
    subprocess_logger.info(f"Visual server subprocess starting on port {port}")
    subprocess_logger.info(f"Log file: {log_file}")
    subprocess_logger.info(f"Assets directory: {ASSETS_DIR}")
    subprocess_logger.info(f"Assets directory exists: {ASSETS_DIR.exists()}")
    if not AIOHTTP_AVAILABLE:
        error_msg = f"FATAL ERROR: aiohttp is not available. Import error: {AIOHTTP_IMPORT_ERROR}"
        subprocess_logger.error(error_msg)
        subprocess_logger.error("Please run 'pip install aiohttp' to install the required dependency")
        print(f"[VisualServer] {error_msg}")
        return
    else:
        subprocess_logger.info(f"aiohttp version {AIOHTTP_VERSION} is available")
    if not ASSETS_DIR.exists():
        error_msg = f"FATAL ERROR: Assets directory not found: {ASSETS_DIR}"
        subprocess_logger.error(error_msg)
        print(f"[VisualServer] {error_msg}")
        return
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

    async def broadcast_console_log(line: str, level: str = 'info'):
        """Broadcast console log message to all connected WebSocket clients."""
        if not clients:
            return

        console_message = {
            "type": "console_log",
            "line": line,
            "level": level,
            "timestamp": time.time()
        }

        payload = json.dumps(console_message)
        tasks = [asyncio.create_task(ws.send_str(payload)) for ws in clients if not ws.closed]

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            subprocess_logger.debug(f"Error broadcasting console log: {e}")

    async def websocket_handler(request: web.Request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clients.add(ws)
        subprocess_logger.info(f"Browser client connected: {request.remote}")

        # Send continue mode configuration to new client
        try:
            from core.globals import g
            continue_mode = getattr(g, 'CONTINUE_CONVERSATION', False)
            config_message = {
                "type": "config",
                "action": "connection_init",
                "continue_mode": continue_mode,
                "id": f"config-{int(time.time() * 1000)}"
            }
            await ws.send_str(json.dumps(config_message))
            subprocess_logger.info(f"Sent continue_mode={continue_mode} to new client")
        except Exception as e:
            subprocess_logger.warning(f"Failed to send continue mode config: {e}")

        if message_history:
            try:
                for i, message in enumerate(message_history):
                    await ws.send_str(json.dumps(message))
                subprocess_logger.info(f"Successfully sent all {len(message_history)} messages to new client")
            except ConnectionResetError:
                subprocess_logger.warning("Connection reset while sending history to new client")
                pass
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    # Parse message to distinguish between permission responses, interrupt signals, and regular user input
                    try:
                        message_data = json.loads(msg.data) if msg.data else {}

                        # Check message type
                        if isinstance(message_data, dict):
                            message_type = message_data.get('type')

                            if message_type == 'permission_response':
                                # For permission responses, put the structured data directly
                                input_queue.put(message_data)



                            elif message_type == 'interrupt_signal':
                                # Handle interrupt signal - trigger the main interrupt system
                                subprocess_logger.info(f"Interrupt signal received from web UI: {message_data.get('source', 'unknown')}")

                                # Use the established IPC mechanism instead of signals
                                # Send a special interrupt command through the input queue
                                interrupt_command = {
                                    'type': 'interrupt_command',
                                    'source': 'web_ui',
                                    'timestamp': message_data.get('timestamp', time.time()),
                                    'reason': 'user_interrupt_button'
                                }

                                try:
                                    # Send to the visual input queue that the main process monitors
                                    input_queue.put(interrupt_command)
                                    subprocess_logger.info("Interrupt command sent via queue to main process")

                                    # Also try the signal approach as backup
                                    import signal
                                    import os
                                    main_pid = os.getppid()
                                    os.kill(main_pid, signal.SIGINT)
                                    subprocess_logger.info(f"Also sent SIGINT to main process (PID: {main_pid}) as backup")
                                except Exception as e:
                                    subprocess_logger.error(f"Failed to send interrupt via queue or signal: {e}")

                                # Also try setting the interrupt flag directly if we can access it
                                try:
                                    # This is a fallback - we'll rely on the SIGINT signal primarily
                                    pass
                                except Exception as e:
                                    subprocess_logger.debug(f"Could not access main interrupt flag: {e}")


                            else:
                                # For other JSON messages, put the raw string
                                input_queue.put(msg.data)
                                subprocess_logger.info(f"JSON message queued: {message_type}")
                        else:
                            # For regular user input, put the raw string
                            input_queue.put(msg.data)
                            subprocess_logger.info(f"User input queued: {msg.data[:50]}...")
                    except json.JSONDecodeError:
                        # If it's not valid JSON, treat it as regular user input
                        input_queue.put(msg.data)
                        subprocess_logger.info(f"Non-JSON user input queued: {msg.data[:50]}...")
                    except Exception as e:
                        subprocess_logger.error(f"Error processing WebSocket message: {e}")
                        # Default to treating as user input
                        input_queue.put(msg.data)
                elif msg.type == web.WSMsgType.ERROR:
                    subprocess_logger.error(f'WebSocket connection closed with exception {ws.exception()}')
        finally:
            subprocess_logger.info(f"Browser client disconnected: {request.remote}")
            clients.remove(ws)
        return ws
    async def index_handler(request: web.Request):
        response = web.FileResponse(ASSETS_DIR / 'index.html')
        response.headers['Content-Security-Policy'] = ("default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com https://cdn.plot.ly https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https://* http://localhost:* http://127.0.0.1:*; frame-src 'self' data: blob:; connect-src 'self' ws: wss:; object-src 'none'; base-uri 'self'; form-action 'self'")
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        return response
    async def favicon_handler(request: web.Request):
        response = web.FileResponse(ASSETS_DIR / 'favicon.ico')
        response.headers['Cache-Control'] = 'public, max-age=86400'
        return response
    async def llm_data_handler(request: web.Request):
        """Serve available LLM models data for the frontend selectboxes."""
        try:
            # Import here to avoid circular imports and ensure g is initialized
            from core.llm import Llm
            from core.globals import g

            # Get all available LLMs including dynamic discovery
            available_llms = Llm.get_available_llms(include_dynamic=True)

            # Convert to JSON-serializable format
            llm_data = []
            for llm in available_llms:
                # Get provider name and emoji from the provider class
                provider_name = llm.provider.__class__.__name__.replace('API', '')

                # Create a clean model entry
                model_entry = {
                    'id': llm.model_key,
                    'name': llm.model_key,
                    'provider': provider_name,
                    'context_window': llm.context_window,
                    'capabilities': [str(capability) for capability in llm.strengths] if llm.strengths else [],
                    'status': 'available'  # Could be enhanced to check actual availability
                }
                llm_data.append(model_entry)

            # Sort by provider and then by name for consistent ordering
            llm_data.sort(key=lambda x: (x['provider'], x['name']))

            # Get current system configuration from the most recent saved state
            # The web server subprocess should always read fresh config to get latest changes
            # Use unified LLM role configuration
            g.load_llm_roles()  # Refresh from disk
            llm_config = g.get_llm_config(force_reread=True)
            selected_llms = g.SELECTED_LLMS
            planner_llm = g.planner_llm
            router_llm = g.router_llm

            selected_configs = []
            for model_key in selected_llms:
                config = llm_config.get(model_key, {})
                config['model_key'] = model_key
                selected_configs.append(config)

            # All selected models are treated equally (no specialized roles)
            unique_models = []
            seen = set()
            for model in selected_llms:
                if model not in seen:
                    unique_models.append(model)
                    seen.add(model)

            logger.info(f"Served LLM data: {len(llm_data)} models from {len(set(m['provider'] for m in llm_data))} providers")
            logger.info(f"Current system state: {len(selected_llms)} selected models")

            # DEBUG: Log the final response being sent to UI
            response_config = {
                'selected_llms': selected_llms,
                'planner_llm': planner_llm,
                'router_llm': router_llm
            }
            subprocess_logger.info(f"🔍 FINAL API RESPONSE current_configuration: {response_config}")

            return web.json_response({
                'status': 'success',
                'models': llm_data,
                'total_count': len(llm_data),
                'current_configuration': {
                    'selected_llms': selected_llms,
                    'planner_llm': planner_llm,
                    'router_llm': router_llm
                }
            })

        except Exception as e:
            logger.error(f"Failed to get LLM data: {e}", exc_info=True)
            return web.json_response({
                'status': 'error',
                'message': f'Failed to retrieve LLM data: {str(e)}',
                'models': []
            }, status=500)

    async def upload_handler(request: web.Request):
        try:
            reader = await request.multipart()
            uploaded_files = []

            while True:
                field = await reader.next()
                if field is None:
                    break

                if field.name == 'file':
                    # Get the filename, sanitize it
                    filename = field.filename or 'uploaded_file'
                    safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
                    if not safe_filename:
                        safe_filename = f"uploaded_file_{int(time.time())}"

                    # Import g to get temp storage path
                    from core.globals import g
                    temp_dir = Path(g.CLIAGENT_TEMP_STORAGE_PATH)
                    temp_dir.mkdir(parents=True, exist_ok=True)

                    # Create unique filename if it already exists
                    counter = 1
                    base_name, ext = safe_filename.rsplit('.', 1) if '.' in safe_filename else (safe_filename, '')
                    final_filename = safe_filename
                    while (temp_dir / final_filename).exists():
                        if ext:
                            final_filename = f"{base_name}_{counter}.{ext}"
                        else:
                            final_filename = f"{safe_filename}_{counter}"
                        counter += 1

                    file_path = temp_dir / final_filename

                    # Save the file
                    with open(file_path, 'wb') as f:
                        while True:
                            chunk = await field.read_chunk()
                            if not chunk:
                                break
                            f.write(chunk)

                    uploaded_files.append(str(file_path.absolute()))
                    subprocess_logger.info(f"File uploaded: {final_filename} -> {file_path}")

            return web.json_response({
                'status': 'success',
                'files': uploaded_files,
                'message': f'Successfully uploaded {len(uploaded_files)} file(s)'
            })

        except Exception as e:
            subprocess_logger.error(f"File upload error: {e}")
            return web.json_response({
                'status': 'error',
                'message': str(e)
            }, status=500)
    async def main():
        app = web.Application()
        app.router.add_get('/', index_handler)
        app.router.add_get('/ws', websocket_handler)
        app.router.add_get('/favicon.ico', favicon_handler)
        app.router.add_get('/api/llms', llm_data_handler)
        app.router.add_post('/upload', upload_handler)
        app.router.add_static('/libs/', path=ASSETS_DIR / 'libs', name='libs')
        app.router.add_static('/css/', path=ASSETS_DIR / 'css', name='css')
        app.router.add_static('/js/', path=ASSETS_DIR / 'js', name='js')
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, SERVER_HOST, port)
        try:
            await site.start()
            subprocess_logger.info(f"Server started successfully at http://{SERVER_HOST}:{port}")
            await asyncio.sleep(0.1)
            try:
                import urllib.request
                test_url = f"http://{SERVER_HOST}:{port}/"
                req = urllib.request.Request(test_url, headers={'User-Agent': 'CLI-Agent-Test'})
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status != 200:
                        subprocess_logger.warning(f"Server responded with unexpected status: {response.status}")
            except Exception as connectivity_error:
                subprocess_logger.warning(f"Server connectivity verification failed: {connectivity_error}")
            subprocess_logger.info("Server is now accepting connections")

            # Start file watcher for console logging
            # Create file watcher for the standard log file paths
            log_paths = {
                'stdout': '/tmp/cli-agent-stdout.log',
                'stderr': '/tmp/cli-agent-stderr.log'
            }

            # Check if log files exist (indicates console capture is enabled)
            import os
            subprocess_logger.info(f"Checking for log files: stdout={os.path.exists(log_paths['stdout'])}, stderr={os.path.exists(log_paths['stderr'])}")
            if os.path.exists(log_paths['stdout']) or os.path.exists(log_paths['stderr']):
                subprocess_logger.info("Log files found - creating file watcher")
                global _file_watcher
                _file_watcher = LogFileWatcher(log_paths)
                loop = asyncio.get_running_loop()
                subprocess_logger.info(f"Created file watcher: {_file_watcher}")

                # Start file watching directly
                try:
                    _file_watcher.running = True
                    _file_watcher.watch_task = loop.create_task(_file_watcher._watch_files())
                    subprocess_logger.info(f"File watching task created: {_file_watcher.watch_task}")
                    subprocess_logger.info("File-based console watching started")
                except Exception as e:
                    subprocess_logger.error(f"Failed to start file watching task: {e}", exc_info=True)
            else:
                subprocess_logger.warning("No log files found - file watcher not started")

            await broadcast_updates()
        except Exception as e:
            subprocess_logger.error(f"Failed to start server: {e}", exc_info=True)
            raise
        finally:
            await runner.cleanup()
            subprocess_logger.info("Server exited cleanly")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle interrupt gracefully without traceback
        subprocess_logger.info("Visual server subprocess interrupted gracefully")
        print("[DEBUG] Visual server subprocess received interrupt signal")
    except Exception as e:
        error_msg = f"FATAL ERROR in server process: {e}"
        subprocess_logger.error(error_msg, exc_info=True)
        print(f"[DEBUG] {error_msg}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

class VisualServerSession:
    # This entire class remains the same.
    def __init__(self, register_cleanup: bool = True):
        self.ctx = multiprocessing.get_context()
        self.output_queue = self.ctx.Queue()
        self.input_queue = self.ctx.Queue()
        self.process: Optional[multiprocessing.Process] = None
        self.port: Optional[int] = None
        g.visual_input_queue = self.input_queue
        self.start()
        if register_cleanup:
            g.register_cleanup(self.close)
    def start(self):
        if self.process and self.process.is_alive(): return
        logger.info("Starting visual server process...")
        self.port = find_open_port(PORT_RANGE_START, PORT_RANGE_END, SERVER_HOST)
        if self.port is None:
            logger.error(colored(f"Could not find an open port in range {PORT_RANGE_START}-{PORT_RANGE_END}.", "red"))
            return
        logger.info(colored(f"Spawning visual response server on port {self.port}...", "cyan"))
        try:
            self.process = self.ctx.Process(target=_server_process_main_loop, args=(self.output_queue, self.input_queue, self.port))
            self.process.start()
            logger.info(f"Visual server subprocess started with PID: {self.process.pid}")
            time.sleep(1.0)
            if not self.process.is_alive():
                logger.error(f"Visual server process died immediately. Exit code: {self.process.exitcode}")
                return
            else:
                logger.info("Visual server process is running successfully")
        except Exception as e:
            logger.error(f"Failed to start visual server process: {e}", exc_info=True)
            return
        server_url = f"http://{SERVER_HOST}:{self.port}"
        logger.info(colored(f"Visual response UI available at {server_url}", "cyan"))
        try:
            time.sleep(0.5)
            import urllib.request
            req = urllib.request.Request(server_url, headers={'User-Agent': 'CLI-Agent'})
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200: logger.info("Server connectivity test passed")
                else: logger.warning(f"Server responded with status {response.status}")
        except Exception as e:
            logger.warning(f"Server connectivity test failed: {e}")
        try:
            webbrowser.open(server_url)
            logger.info("Browser opened successfully")
        except Exception as e:
            logger.warning(colored(f"Could not automatically open browser: {e}", "yellow"))
    def send_update(self, data: dict):
        if not self.process or not self.process.is_alive():
            logger.warning("Visual server process not running, attempting to restart")
            self.start()
        if self.process and self.process.is_alive():
            try:
                self.output_queue.put(data)
            except Exception as e:
                logger.error(f"Failed to send update to visual server: {e}")
        else:
            logger.error("Cannot send update: visual server process failed to start")
    def is_healthy(self):
        if not self.process or not self.process.is_alive(): return False
        if not self.port: return False
        return test_port_connectivity(SERVER_HOST, self.port, timeout=2)
    def get_status(self):
        status = {'process_alive': self.process and self.process.is_alive(),'process_pid': self.process.pid if self.process else None,'port': self.port,'server_responsive': False,'queue_size': None}
        if self.process: status['process_exitcode'] = self.process.exitcode
        if self.port: status['server_responsive'] = test_port_connectivity(SERVER_HOST, self.port, timeout=1)
        try: status['queue_size'] = self.output_queue.qsize()
        except: pass
        return status
    def close(self):
        if self.process and self.process.is_alive():
            logger.info(colored(f"Closing visual response server (PID: {self.process.pid})...", "cyan"))
            self.output_queue.put("__EXIT__")
            self.process.join(timeout=3)
            if self.process.is_alive():
                logger.warning(colored("Server process did not exit gracefully, terminating.", "yellow"))
                self.process.terminate()
            logger.info(colored("Visual response server closed.", "cyan"))
        self.process = None
        if hasattr(g, 'visual_input_queue'):
            delattr(g, 'visual_input_queue')


class visualresponse:
    _session: Optional[VisualServerSession] = None

    @staticmethod
    def close_session():
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
                "Create content in a browser UI. Use `<type>html</type>` for any interactive or complex content by providing a complete, self-contained HTML document. Use `<type>markdown</type>` for simple text and diagrams. IDs are auto-generated.\n"
                "PARAMETERS:\n"
                "- `<type>`: Must be either `html` or `markdown`.\n"
                "- `<content>`: The full HTML document or Markdown text. For HTML, you can wrap content in <![CDATA[...]]> to ensure correct parsing."
            ),
            "example": (
                "// Create an interactive Plotly chart using the 'html' type\n"
                "<visualresponse><type>html</type><content><![CDATA[\n"
                "<!DOCTYPE html><html><head>\n"
                "  <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>\n"
                "</head><body>\n"
                "  <div id='plotDiv' style='width:100%;height:100%;'></div>\n"
                "  <script>\n"
                "    const data = [{'x': [1,2,3], 'y': [4,2,5], 'type': 'bar'}];\n"
                "    Plotly.newPlot('plotDiv', data, {title: 'My Bar Chart'});\n"
                "  </script>\n"
                "</body></html>\n"
                "]]></content></visualresponse>\n\n"
                "// Create a simple markdown panel\n"
                "<visualresponse><type>markdown</type><content># Analysis Complete</content></visualresponse>"
            )
        }

    @staticmethod
    def _get_session() -> VisualServerSession:
        session_dead = (visualresponse._session is None or
                        visualresponse._session.process is None or
                        not visualresponse._session.process.is_alive())
        if session_dead:
            if visualresponse._session:
                visualresponse._session.close()
            visualresponse._session = VisualServerSession(register_cleanup=False)
        return visualresponse._session

    @staticmethod
    def run(content: str) -> str:
        try:
            if content.strip().lower() in ['status', 'test', 'diagnostic', 'debug']:
                return visualresponse._run_diagnostics()

            # --- Simplified parsing: always create, auto-generate ID ---
            args = {}
            # Use re.DOTALL to make '.' match newlines, crucial for multi-line content
            type_match = re.search(r'<type>(.*?)</type>', content, re.IGNORECASE | re.DOTALL)
            if type_match:
                args['type'] = type_match.group(1).strip()

            content_match = re.search(r'<content>(.*?)</content>', content, re.IGNORECASE | re.DOTALL)
            if content_match:
                # Strip CDATA wrapper if present, then strip whitespace
                c = content_match.group(1)
                if c.strip().startswith('<![CDATA['):
                    c = c.strip()[9:-3] # Remove <![CDATA[ and ]]>
                args['content'] = c.strip()

            # Always default to 'create' action
            action = 'create'
            # Auto-generate unique ID using precise timestamp
            unique_timestamp = int(time.time() * 1000000)  # Microsecond precision
            viz_id = f"viz-{unique_timestamp}"

            payload = {"id": viz_id, "action": action}

            # Add tool execution info if available
            if hasattr(g, 'last_executed_tools_info') and g.last_executed_tools_info:
                payload['tool_info'] = g.last_executed_tools_info
                # Clear the info after using it to avoid reusing stale data
                g.last_executed_tools_info = None

            # Always create content
            viz_type = args.get('type')
            viz_content_str = args.get('content')

            if not viz_type or viz_content_str is None:
                return "Error: visualresponse requires <type> and <content> tags."

            viz_type = viz_type.lower()
            supported_types = ['markdown', 'html']
            if viz_type not in supported_types:
                return f"Error: Unsupported type '{viz_type}'. Supported types are: {', '.join(supported_types)}"

            payload['type'] = viz_type
            payload['content'] = viz_content_str

            session = visualresponse._get_session()
            if not session.is_healthy():
                session.close()
                session = visualresponse._get_session()

            session.send_update(payload)

            # Set current viz_id in tool context for live tool updates
            tool_context.set_current_viz_id(viz_id)

            return f"Visual response created with ID '{viz_id}' and sent to the live UI on port {session.port}."

        except Exception as e:
            logger.error(f"Fatal error in visualresponse tool: {e}", exc_info=True)
            return f"Fatal error in visualresponse tool: {e}"

    @staticmethod
    def _run_diagnostics() -> str:
        # This function remains the same
        diagnostics = ["🔍 Visual Server Diagnostics", "=" * 40]
        if AIOHTTP_AVAILABLE: diagnostics.append(f"✅ aiohttp {AIOHTTP_VERSION} is available")
        else: diagnostics.append(f"❌ aiohttp not available: {AIOHTTP_IMPORT_ERROR}")
        if ASSETS_DIR.exists():
            diagnostics.append(f"✅ Assets directory exists: {ASSETS_DIR}")
            for file_name in ['index.html', 'favicon.ico']:
                file_path = ASSETS_DIR / file_name
                if file_path.exists(): diagnostics.append(f"  ✅ {file_name} ({file_path.stat().st_size} bytes)")
                else: diagnostics.append(f"  ❌ {file_name} missing")
        else: diagnostics.append(f"❌ Assets directory missing: {ASSETS_DIR}")
        if visualresponse._session:
            status = visualresponse._session.get_status()
            diagnostics.append("\n📊 Current Session Status:")
            for key, value in status.items(): diagnostics.append(f"  {key.replace('_', ' ').title()}: {value}")
        else: diagnostics.append("\n📊 No active session")
        available_port = find_open_port(PORT_RANGE_START, PORT_RANGE_END)
        if available_port: diagnostics.append(f"\n🔌 Available port found: {available_port}")
        else: diagnostics.append(f"\n❌ No available ports in range {PORT_RANGE_START}-{PORT_RANGE_END}")
        return "\n".join(diagnostics)

    @staticmethod
    def send_user_message(message: str) -> bool:
        """
        Send a user message to the visual interface for display.
        Used when user messages come from command line arguments instead of chat interface.

        Args:
            message: The user message to display

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            session = visualresponse._get_session()
            if not session or not session.is_healthy():
                logger.debug("Visual session not available for user message")
                return False

            payload = {
                "type": "user_message",
                "action": "display",
                "message": message,
                "timestamp": time.time(),
                "id": f"user-msg-{int(time.time() * 1000000)}"
            }

            print(f"[DEBUG] About to send user message payload: {payload}")
            session.send_update(payload)
            print(f"[DEBUG] ✅ Sent user message to visual interface: {message[:50]}...")
            return True

        except Exception as e:
            print(f"[DEBUG] ❌ Failed to send user message to visual interface: {e}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def send_console_log(line: str, level: str = 'info') -> bool:
        """
        Send a console log message to the visual interface.

        Args:
            line: The console line to send
            level: The log level (info, warning, error, debug)

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            session = visualresponse._get_session()
            if not session or not session.is_healthy():
                return False

            # Use the broadcast_console_log function from the server process
            # by sending it through the update queue
            payload = {
                "type": "console_log",
                "line": line.strip(),
                "level": level,
                "timestamp": time.time(),
                "id": f"console-log-{int(time.time() * 1000000)}"
            }

            session.send_update(payload)
            return True

        except Exception as e:
            logger.debug(f"Failed to send console log to visual interface: {e}")
            return False


class FileBasedConsoleCapture:
    """
    File-based console output capture system for browser console display.
    Redirects stdout/stderr to log files for real-time streaming to web interface.
    """

    def __init__(self, log_dir: str = None):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.enabled = False

        # Set up log file paths using XDG cache directory specification
        if log_dir is None:
            # Follow XDG Base Directory specification for cache files
            xdg_cache_home = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
            log_dir = os.path.join(xdg_cache_home, 'cli-agent', 'logs')

        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        self.stdout_log_path = os.path.join(log_dir, 'console-stdout.log')
        self.stderr_log_path = os.path.join(log_dir, 'console-stderr.log')

        self.stdout_stream = None
        self.stderr_stream = None

    def enable(self):
        """Enable file-based console capture."""
        if self.enabled:
            return

        # Create file streams that redirect to log files
        self.stdout_stream = LogFileStream(self.original_stdout, self.stdout_log_path)
        self.stderr_stream = LogFileStream(self.original_stderr, self.stderr_log_path)

        # Redirect stdout/stderr to file streams
        sys.stdout = self.stdout_stream
        sys.stderr = self.stderr_stream
        self.enabled = True
        logger.debug(f"File-based console capture enabled: stdout={self.stdout_log_path}, stderr={self.stderr_log_path}")

    def disable(self):
        """Disable console capture and restore original streams."""
        if not self.enabled:
            return

        # Close file streams
        if self.stdout_stream:
            self.stdout_stream.close()
        if self.stderr_stream:
            self.stderr_stream.close()

        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.enabled = False
        logger.debug("File-based console capture disabled")

    def get_log_paths(self):
        """Get the paths to the log files."""
        return {
            'stdout': self.stdout_log_path,
            'stderr': self.stderr_log_path
        }

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()


class ConsoleCapture:
    """
    LEGACY: Console output capture system for forwarding stdout/stderr to visual interface.
    This class is kept for backward compatibility but should be replaced by FileBasedConsoleCapture.
    """

    def __init__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.enabled = False

    def enable(self):
        """Enable console capture and forwarding."""
        if self.enabled:
            return

        # Create wrapper streams that forward to visual interface
        sys.stdout = ConsoleWrapper(self.original_stdout, 'info')
        sys.stderr = ConsoleWrapper(self.original_stderr, 'error')
        self.enabled = True
        logger.debug("Console capture enabled for visual interface")

    def disable(self):
        """Disable console capture and restore original streams."""
        if not self.enabled:
            return

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.enabled = False
        logger.debug("Console capture disabled")

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()


class LogFileStream:
    """
    File-based stream that redirects stdout/stderr to log files for browser console.
    """

    def __init__(self, original_stream, log_file_path: str):
        self.original_stream = original_stream
        self.log_file_path = log_file_path

        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Open log file in append mode with line buffering for real-time updates
        self.log_file = open(log_file_path, 'a', buffering=1, encoding='utf-8')

    def write(self, text):
        # Write to original stream first (for terminal display)
        self.original_stream.write(text)

        # Write raw text directly to log file (no filtering)
        self.log_file.write(text)

        # Directly send to WebSocket clients if we have non-empty content
        if text.strip():
            self._broadcast_to_console(text.rstrip('\n\r'))

    def _broadcast_to_console(self, content: str):
        """Send content directly to browser console via WebSocket."""
        try:
            # Strip ANSI escape sequences for clean console display
            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            clean_content = ansi_escape.sub('', content)

            # Use the existing visualresponse.send_console_log function
            # which properly handles the WebSocket communication
            level = 'error' if 'stderr' in self.log_file_path.lower() else 'info'

            # Import and use the visualresponse class's console log method
            from tools.visualresponse import visualresponse
            visualresponse.send_console_log(clean_content, level)
        except Exception:
            # Don't let broadcast errors break stdout/stderr
            pass

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()

    def close(self):
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


class ConsoleWrapper:
    """
    LEGACY: Wrapper for stdout/stderr that forwards output to visual interface.
    This class is kept for backward compatibility but should be replaced by LogFileStream.
    """

    def __init__(self, original_stream, level: str):
        self.original_stream = original_stream
        self.level = level

    def write(self, text):
        # Write to original stream first
        self.original_stream.write(text)

        # Forward to visual interface (only non-empty lines)
        if text.strip() and not text.isspace():
            # Clean text of ANSI escape sequences (comprehensive pattern)
            import re
            # Remove all ANSI escape sequences: colors, cursor movements, etc.
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[.*?[@-~]')
            clean_text = ansi_escape.sub('', text.strip())

            # Also remove any remaining escape sequences
            clean_text = re.sub(r'\x1b.*?[a-zA-Z]', '', clean_text)

            # Only skip completely empty lines after ANSI cleaning
            if not clean_text:
                return

            # Send everything to console log for maximum verbosity
            visualresponse.send_console_log(clean_text, self.level)

    def flush(self):
        self.original_stream.flush()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


class LogFileWatcher:
    """
    Monitors log files for changes and streams content to browser console.
    """

    def __init__(self, log_file_paths: Dict[str, str]):
        self.log_file_paths = log_file_paths  # {'stdout': path, 'stderr': path}
        self.file_positions = {}  # Track last read position for each file
        self.running = False
        self.watch_task = None

        # Initialize file positions
        for stream_type, log_path in log_file_paths.items():
            try:
                # Start from end of existing file or 0 for new file
                if os.path.exists(log_path):
                    with open(log_path, 'r', encoding='utf-8') as f:
                        f.seek(0, 2)  # Seek to end
                        self.file_positions[log_path] = f.tell()
                else:
                    self.file_positions[log_path] = 0
            except Exception as e:
                logger.warning(f"Could not initialize position for {log_path}: {e}")
                self.file_positions[log_path] = 0

    def start_watching(self, loop=None):
        """Start monitoring log files for changes."""
        if self.running:
            return

        self.running = True

        # Get or create event loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, we'll start the task later when server starts
                logger.debug("No running event loop - file watching will start with server")
                return

        self.watch_task = loop.create_task(self._watch_files())
        logger.debug("Log file watching started")

    def stop_watching(self):
        """Stop monitoring log files."""
        if not self.running:
            return

        self.running = False
        if self.watch_task:
            self.watch_task.cancel()
        logger.debug("Log file watching stopped")

    def ensure_watching_started(self, loop):
        """Ensure file watching is started with the given event loop."""
        logger.info(f"ensure_watching_started: running={self.running}, watch_task={self.watch_task}")
        if self.running and not self.watch_task:
            self.watch_task = loop.create_task(self._watch_files())
            logger.info("Log file watching started with server event loop")
        else:
            logger.warning(f"File watching not started: running={self.running}, watch_task={self.watch_task}")

    async def _watch_files(self):
        """Main file watching loop."""
        logger.info("File watcher started - monitoring log files")
        loop_count = 0
        while self.running:
            try:
                loop_count += 1
                if loop_count % 50 == 0:  # Log every 5 seconds (50 * 0.1s)
                    logger.info(f"File watcher alive - loop {loop_count}")

                for stream_type, log_path in self.log_file_paths.items():
                    await self._check_file_for_updates(log_path, stream_type)

                # Check files every 0.1 seconds for real-time updates
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                logger.info("File watcher cancelled")
                break
            except Exception as e:
                logger.error(f"Error in file watching loop: {e}")
                await asyncio.sleep(1)  # Back off on errors

    async def _check_file_for_updates(self, log_path: str, stream_type: str):
        """Check a single log file for new content."""
        try:
            if not os.path.exists(log_path):
                return

            # Get current file size
            current_size = os.path.getsize(log_path)
            last_position = self.file_positions.get(log_path, 0)

            # Check if file has new content
            if current_size > last_position:
                logger.info(f"New content detected in {log_path}: {current_size - last_position} bytes")
                # Read new content
                with open(log_path, 'r', encoding='utf-8') as f:
                    f.seek(last_position)
                    new_content = f.read()

                    if new_content:
                        logger.info(f"Sending {len(new_content)} characters to browser console")
                        # Send new content to browser via WebSocket
                        await self._send_log_content(new_content, stream_type)

                        # Update position
                        self.file_positions[log_path] = f.tell()

            elif current_size < last_position:
                # File was truncated or rotated, start from beginning
                self.file_positions[log_path] = 0

        except Exception as e:
            logger.error(f"Error reading log file {log_path}: {e}")

    async def _send_log_content(self, content: str, stream_type: str):
        """Send log content to browser console via WebSocket."""
        try:
            # Split content into lines for better browser handling
            lines = content.splitlines(keepends=True)

            for line in lines:
                if line.strip():  # Only send non-empty lines
                    # Determine log level based on stream type
                    level = 'error' if stream_type == 'stderr' else 'info'

                    # Use existing broadcast system
                    await broadcast_console_log(line.rstrip('\n\r'), level)

        except Exception as e:
            logger.error(f"Error sending log content to browser: {e}")


# Global console capture instance
_console_capture = None
_file_watcher = None


def enable_file_based_console_capture():
    """Enable file-based console output capture for visual interface."""
    global _console_capture, _file_watcher

    # Disable old console capture if active
    if _console_capture is not None:
        _console_capture.disable()

    # Create file-based console capture
    _console_capture = FileBasedConsoleCapture()
    _console_capture.enable()

    # Start file monitoring
    log_paths = _console_capture.get_log_paths()
    _file_watcher = LogFileWatcher(log_paths)
    _file_watcher.start_watching()

    logger.info("File-based console capture enabled with real-time file monitoring")


def disable_file_based_console_capture():
    """Disable file-based console output capture for visual interface."""
    global _console_capture, _file_watcher

    # Stop file monitoring
    if _file_watcher is not None:
        _file_watcher.stop_watching()
        _file_watcher = None

    # Disable console capture
    if _console_capture is not None:
        _console_capture.disable()
        _console_capture = None

    logger.info("File-based console capture disabled")


def enable_console_capture():
    """Enable console output capture for visual interface (legacy method)."""
    # Use the new file-based system by default
    enable_file_based_console_capture()


def disable_console_capture():
    """Disable console output capture (legacy method)."""
    # Use the new file-based system by default
    disable_file_based_console_capture()
