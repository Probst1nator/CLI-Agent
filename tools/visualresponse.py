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
except ImportError:
    web = None

from core.globals import g

# --- Server Configuration ---
SERVER_HOST = "127.0.0.1"
PORT_RANGE_START = 7860
PORT_RANGE_END = 7870
ASSETS_DIR = Path(__file__).parent / "visual_assets"

def find_open_port(start_port, end_port, host="127.0.0.1"):
    """Finds an available TCP port in a given range."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return None

def _server_process_main_loop(output_queue: multiprocessing.Queue, input_queue: multiprocessing.Queue, port: int):
    """The entry point for the dedicated aiohttp server process."""
    if not web:
        print("[VisualServer] FATAL ERROR: 'aiohttp' is not installed. Please run 'pip install aiohttp'.")
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
        print(f"[VisualServer] Browser client connected: {request.remote}")
        
        # Send message history to new clients
        print(f"[VisualServer] Message history contains {len(message_history)} messages")
        if message_history:
            try:
                for i, message in enumerate(message_history):
                    print(f"[VisualServer] Sending message {i+1}/{len(message_history)} to new client")
                    await ws.send_str(json.dumps(message))
                print(f"[VisualServer] Successfully sent all {len(message_history)} messages to new client")
            except ConnectionResetError:
                print("[VisualServer] Connection reset while sending history to new client")
                pass
        else:
            print("[VisualServer] No message history to send to new client")
        
        try:
            # Listen for incoming messages from the browser client
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    print(f"[VisualServer] Received message from browser: '{msg.data}'")
                    # Put the received user input into the input_queue for the main process
                    input_queue.put(msg.data)
                elif msg.type == web.WSMsgType.ERROR:
                    print(f'[VisualServer] WebSocket connection closed with exception {ws.exception()}')
        finally:
            print(f"[VisualServer] Browser client disconnected: {request.remote}")
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
            print(f"[VisualServer] Started successfully at http://{SERVER_HOST}:{port}")
            await broadcast_updates()
        finally:
            await runner.cleanup()
            print("[VisualServer] Exited cleanly.")

    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[VisualServer] FATAL ERROR: {e}")

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
        
        if register_cleanup:
            g.register_cleanup(self.close)


    def start(self):
        if self.process and self.process.is_alive():
            return
        
        self.port = find_open_port(PORT_RANGE_START, PORT_RANGE_END, SERVER_HOST)
        if self.port is None:
            logging.error(colored(f"  - Could not find an open port in range {PORT_RANGE_START}-{PORT_RANGE_END}.", "red"))
            return

        logging.info(colored(f"  - Spawning visual response server on port {self.port}...", "cyan"))
        self.process = self.ctx.Process(target=_server_process_main_loop, args=(self.output_queue, self.input_queue, self.port))
        self.process.start()
        
        server_url = f"http://{SERVER_HOST}:{self.port}"
        logging.info(colored(f"  - Visual response UI available at {server_url}", "cyan"))
        try:
            import time
            time.sleep(0.5)
            webbrowser.open(server_url)
        except Exception as e:
            logging.warning(colored(f"  - Could not automatically open browser: {e}", "yellow"))

    def send_update(self, data: dict):
        if not self.process or not self.process.is_alive():
            self.start()
        if self.process:
            self.output_queue.put(data)

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
            visualresponse._session = VisualServerSession(register_cleanup=False)

        return visualresponse._session

    @staticmethod
    def run(content: str) -> str:
        try:
            type_match = re.search(r'<type>(.*?)</type>', content, re.DOTALL)
            content_match = re.search(r'<content>(.*?)</content>', content, re.DOTALL)

            if not type_match or not content_match:
                return "Error: Invalid format. Must contain <type> and <content> tags."

            viz_type = type_match.group(1).strip().lower()
            viz_content_str = content_match.group(1).strip()
            
            supported_types = ['markdown', 'html', '3Dmol.js', 'plotly']
            if viz_type not in supported_types:
                return f"Error: Unsupported visual type '{viz_type}'. Supported types are: {', '.join(supported_types)}"
            
            # For types expecting JSON, parse the content string
            if viz_type in ['3Dmol.js', 'plotly']:
                try:
                    viz_content = json.loads(viz_content_str)
                except json.JSONDecodeError as e:
                    return f"Error: Content for type '{viz_type}' must be a valid JSON object. Details: {e}"
            else:
                viz_content = viz_content_str
            
            payload = {"type": viz_type, "content": viz_content}
            
            session = visualresponse._get_session()
            session.send_update(payload)
            
            return f"Visual response of type '{viz_type}' sent to the live UI on port {session.port}."
        except Exception as e:
            logging.error(f"Fatal error in visualresponse tool: {e}", exc_info=True)
            return f"Fatal error in visualresponse tool: {e}"