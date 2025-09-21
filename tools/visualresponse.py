# tools/visualresponse.py
"""
Implements the 'visualresponse' tool for rich, interactive, web-based visualizations.
Supports Markdown and fully self-contained HTML for maximum flexibility.

Requires: pip install aiohttp
"""

import multiprocessing
import asyncio
import json
import time
import webbrowser
import socket
import re # Import the regular expression module
from pathlib import Path
from typing import Optional, Set, Any, List
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
import sys

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
            llm_config = g.get_llm_config(force_reread=True)
            # Get selected models and sort them by their saved order
            selected_models_with_order = []
            for key, data in llm_config.items():
                if key != '__planner_llm__' and isinstance(data, dict) and data.get('selected'):
                    order = data.get('order', 999)  # Default high order for models without order
                    selected_models_with_order.append((order, key))

            # Sort by order and extract just the model keys
            selected_models_with_order.sort(key=lambda x: x[0])
            selected_llms = [key for order, key in selected_models_with_order]

            # Get planner LLM and router LLM from the fresh config
            planner_llm = llm_config.get('__planner_llm__')
            router_llm = llm_config.get('__router_llm__')

            selected_configs = []
            for model_key in selected_llms:
                config = llm_config.get(model_key, {})
                config['model_key'] = model_key
                selected_configs.append(config)

            # Categorize models by their roles
            branch_models = [config['model_key'] for config in selected_configs if config.get('beams', 0) > 0 or config.get('branches', 0) > 0]
            eval_models = [config['model_key'] for config in selected_configs if config.get('eval', 0) > 0]
            guard_models = [config['model_key'] for config in selected_configs if config.get('guard', 0) > 0]

            # Calculate fallback models (selected but not specialized)
            unique_models = []
            seen = set()
            for model in selected_llms:
                if model not in seen:
                    unique_models.append(model)
                    seen.add(model)

            specialized_models = set(branch_models + eval_models + guard_models)
            fallback_models = [model for model in unique_models if model not in specialized_models]

            logger.info(f"Served LLM data: {len(llm_data)} models from {len(set(m['provider'] for m in llm_data))} providers")
            logger.info(f"Current system state: {len(selected_llms)} selected, {len(fallback_models)} fallback models")

            # DEBUG: Log the final response being sent to UI
            response_config = {
                'selected_llms': selected_llms,
                'fallback_models': fallback_models,
                'branch_models': branch_models,
                'eval_models': eval_models,
                'guard_models': guard_models,
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
                    'fallback_models': fallback_models,
                    'branch_models': branch_models,
                    'eval_models': eval_models,
                    'guard_models': guard_models,
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
                "Respond using a browser UI. Use `<type>html</type>` for any interactive or complex content by providing a complete, self-contained HTML document. Use `<type>markdown</type>` for simple text and diagrams. An `<id>` is recommended but optional for creation.\n"
                "PARAMETERS:\n"
                "- `<action>`: `create` (default), `delete`, `clear`, `agent_reset`.\n"
                "- `<id>`: A unique ID for the element. If omitted on create, one is generated.\n"
                "- `<type>`: Must be either `html` or `markdown`.\n"
                "- `<content>`: The full HTML document or Markdown text. For HTML, you can wrap content in <![CDATA[...]]> to ensure correct parsing."
            ),
            "example": (
                "// Create an interactive Plotly chart using the 'html' type\n"
                "<visualresponse><id>my-chart</id><type>html</type><content><![CDATA[\n"
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
                "// Create a simple markdown panel (ID will be auto-generated)\n"
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

            # --- FIX STARTS HERE: Robust parsing with Regular Expressions ---
            args = {}
            # Use re.DOTALL to make '.' match newlines, crucial for multi-line content
            action_match = re.search(r'<action>(.*?)</action>', content, re.IGNORECASE | re.DOTALL)
            if action_match:
                args['action'] = action_match.group(1).strip()

            id_match = re.search(r'<id>(.*?)</id>', content, re.IGNORECASE | re.DOTALL)
            if id_match:
                args['id'] = id_match.group(1).strip()

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
            # --- END OF FIX ---

            action = args.get('action', 'create').lower()
            # Generate unique ID to prevent collisions using higher precision timestamp + random component
            import random
            unique_timestamp = int(time.time() * 1000000)  # Microsecond precision
            random_suffix = random.randint(100, 999)
            viz_id = args.get('id', f"viz-{unique_timestamp}-{random_suffix}")

            payload = {"id": viz_id, "action": action}

            # Add tool execution info if available (for create actions)
            if action == 'create' and hasattr(g, 'last_executed_tools_info') and g.last_executed_tools_info:
                payload['tool_info'] = g.last_executed_tools_info
                # Clear the info after using it to avoid reusing stale data
                g.last_executed_tools_info = None

            if action == 'create':
                viz_type = args.get('type')
                viz_content_str = args.get('content')

                if not viz_type or viz_content_str is None:
                    return "Error: 'create' action requires <type> and <content> tags."

                viz_type = viz_type.lower()
                supported_types = ['markdown', 'html']
                if viz_type not in supported_types:
                    return f"Error: Unsupported type '{viz_type}'. Supported types are: {', '.join(supported_types)}"

                payload['type'] = viz_type
                payload['content'] = viz_content_str

            elif action == 'delete':
                if 'id' not in args:
                    return "Error: 'delete' action requires an <id> tag."
                payload['type'] = 'system'

            elif action == 'clear':
                payload['type'] = 'system'

            elif action == 'agent_reset':
                payload['type'] = 'system'
                payload['clear'] = True  # Use same logic as /clear command

            else:
                return f"Error: Unknown action '{action}'. Supported actions: create, delete, clear, agent_reset."

            session = visualresponse._get_session()
            if not session.is_healthy():
                session.close()
                session = visualresponse._get_session()

            session.send_update(payload)

            # Set current viz_id in tool context for live tool updates
            if action == 'create':
                tool_context.set_current_viz_id(viz_id)

            return f"Visual response command '{action}' for ID '{viz_id}' sent to the live UI on port {session.port}."

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


class ConsoleCapture:
    """
    Console output capture system for forwarding stdout/stderr to visual interface.
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


class ConsoleWrapper:
    """
    Wrapper for stdout/stderr that forwards output to visual interface.
    """

    def __init__(self, original_stream, level: str):
        self.original_stream = original_stream
        self.level = level

    def write(self, text):
        # Write to original stream first
        self.original_stream.write(text)

        # Forward to visual interface (only non-empty, meaningful lines)
        if text.strip() and not text.isspace():
            # Clean text of ANSI escape sequences (comprehensive pattern)
            import re
            # Remove all ANSI escape sequences: colors, cursor movements, etc.
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[.*?[@-~]')
            clean_text = ansi_escape.sub('', text.strip())

            # Also remove any remaining escape sequences
            clean_text = re.sub(r'\x1b.*?[a-zA-Z]', '', clean_text)

            # Skip very short fragments (likely streaming artifacts)
            if len(clean_text) < 3:
                return

            # Skip some noisy patterns
            skip_patterns = [
                'debug:', 'loading', 'initializing', 'connection',
                'browser client', 'sent continue_mode', 'successfully sent',
                'get /', 'http/1.1', 'mozilla/', 'webkit/',
                # Skip streaming fragments
                '<think>', '</think>', 'thinking', 'i should', 'i need to',
                'the user', 'let me', 'i will', 'i can',
                # Skip ANSI escape sequence fragments
                '[97m', '[0m', '[91m', '[92m', '[93m', '[94m', '[95m', '[96m'
            ]

            if not any(pattern in clean_text.lower() for pattern in skip_patterns):
                # Only send if it's a substantial line (not streaming fragments)
                if len(clean_text) > 10 or any(important in clean_text.lower() for important in [
                    'error', 'warning', 'failed', 'success', 'complete', 'started', 'ready'
                ]):
                    visualresponse.send_console_log(clean_text, self.level)

    def flush(self):
        self.original_stream.flush()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


# Global console capture instance
_console_capture = None


def enable_console_capture():
    """Enable console output capture for visual interface."""
    global _console_capture
    if _console_capture is None:
        _console_capture = ConsoleCapture()
    _console_capture.enable()


def disable_console_capture():
    """Disable console output capture."""
    global _console_capture
    if _console_capture is not None:
        _console_capture.disable()