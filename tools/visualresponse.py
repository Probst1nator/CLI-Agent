# tools/visualresponse.py
"""
Implements the 'visualresponse' tool for rich, interactive, web-based visualizations.
Supports Markdown and fully self-contained HTML for maximum flexibility.

Requires: pip install aiohttp
"""

import asyncio
import json
import multiprocessing
import os
import re  # Import the regular expression module
import socket
import sys
import time
import webbrowser
from pathlib import Path
from queue import Empty as QueueEmpty
from typing import TYPE_CHECKING, Any, Optional

from termcolor import colored

if TYPE_CHECKING:
    from aiohttp import web as web_types
else:
    web_types = None

# Set up tool-specific logging
from shared.utils.tool_logging import get_logger_for_tool_module

logger = get_logger_for_tool_module(__name__)

try:
    import aiohttp
    from aiohttp import web

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
ASSETS_DIR = Path(__file__).parent.parent / "frontend"


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


def _server_process_main_loop(
    output_queue: multiprocessing.Queue, input_queue: multiprocessing.Queue, port: int
):
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
        subprocess_logger.error(
            "Please run 'pip install aiohttp' to install the required dependency"
        )
        print(f"[VisualServer] {error_msg}")
        return
    else:
        subprocess_logger.info(f"aiohttp version {AIOHTTP_VERSION} is available")
    if not ASSETS_DIR.exists():
        error_msg = f"FATAL ERROR: Assets directory not found: {ASSETS_DIR}"
        subprocess_logger.error(error_msg)
        print(f"[VisualServer] {error_msg}")
        return
    required_files = ["index.html", "favicon.ico"]
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
    clients: set[Any] = set()
    message_history: list[Any] = []

    async def broadcast_updates():
        while True:
            try:
                message = output_queue.get_nowait()
                if message == "__EXIT__":
                    break

                message_history.append(message)
                msg_type = message.get("type", "unknown")
                subprocess_logger.debug(
                    f"📨 Added message to history: type={msg_type}, clients={len(clients)}, total_history={len(message_history)}"
                )

                if clients:
                    payload = json.dumps(message)
                    tasks = [
                        asyncio.create_task(ws.send_str(payload)) for ws in clients
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    subprocess_logger.debug(
                        f"✅ Broadcasted {msg_type} to {len(clients)} client(s)"
                    )
                else:
                    subprocess_logger.debug(
                        f"⏸️  No clients connected, message queued (type={msg_type})"
                    )
            except QueueEmpty:
                await asyncio.sleep(0.01)  # 10ms polling for near-instant updates
        for ws in clients:
            await ws.close(code=1000, message=b"Server shutdown")

    async def broadcast_console_log(line: str, level: str = "info"):
        """Broadcast console log message to all connected WebSocket clients."""
        if not clients:
            return

        console_message = {
            "type": "console_log",
            "line": line,
            "level": level,
            "timestamp": time.time(),
        }

        payload = json.dumps(console_message)
        tasks = [
            asyncio.create_task(ws.send_str(payload)) for ws in clients if not ws.closed
        ]

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            subprocess_logger.debug(f"Error broadcasting console log: {e}")

    async def websocket_handler(request: Any) -> Any:
        if web is None:
            raise RuntimeError("aiohttp not available")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clients.add(ws)

        # Register with connection monitor
        try:
            from tools._connection_monitor import connection_monitor

            connection_monitor.register_client(ws)
        except Exception as e:
            subprocess_logger.warning(
                f"Failed to register client with connection monitor: {e}"
            )

        subprocess_logger.info(f"Browser client connected: {request.remote}")

        # Send continue mode configuration to new client
        try:
            from core.globals import g

            continue_mode = getattr(g, "CONTINUE_CONVERSATION", False)

            # Check if this is a fresh chat (empty or only has system message)
            is_fresh_chat = True
            if hasattr(g, "chat") and g.chat and hasattr(g.chat, "messages"):
                # Consider it fresh only if it has 0-1 messages (just system prompt)
                is_fresh_chat = len(g.chat.messages) <= 1

            # Clear server-side message history ONLY if we're loading from a saved chat (-c flag)
            # Don't clear if it's a fresh session with queued messages from current run
            if is_fresh_chat and not continue_mode and len(message_history) > 0:
                # Check if messages are from a previous session (have old timestamps)
                # Only clear if first message is older than 30 seconds
                first_msg_time = (
                    message_history[0].get("timestamp", time.time())
                    if message_history
                    else time.time()
                )
                age_seconds = time.time() - first_msg_time
                if age_seconds > 30:
                    msg_count = len(message_history)
                    message_history.clear()
                    subprocess_logger.info(
                        f"🗑️  Cleared {msg_count} old messages (age: {age_seconds:.1f}s)"
                    )
                else:
                    subprocess_logger.info(
                        f"💾 Preserving {len(message_history)} recent messages (age: {age_seconds:.1f}s)"
                    )

            config_message = {
                "type": "config",
                "action": "connection_init",
                "continue_mode": continue_mode
                and not is_fresh_chat,  # Don't continue if fresh
                "clear_history": is_fresh_chat,  # Tell frontend to clear localStorage
                "id": f"config-{int(time.time() * 1000)}",
            }
            await ws.send_str(json.dumps(config_message))
            subprocess_logger.info(
                f"Sent continue_mode={continue_mode and not is_fresh_chat}, clear_history={is_fresh_chat} to new client"
            )
        except Exception as e:
            subprocess_logger.warning(f"Failed to send continue mode config: {e}")

        if message_history:
            try:
                subprocess_logger.info(
                    f"📤 Sending {len(message_history)} historical messages to new client"
                )
                for i, message in enumerate(message_history):
                    msg_type = message.get("type", "unknown")
                    subprocess_logger.debug(
                        f"  [{i+1}/{len(message_history)}] Sending {msg_type}"
                    )
                    await ws.send_str(json.dumps(message))
                subprocess_logger.info(
                    f"✅ Successfully sent all {len(message_history)} messages to new client"
                )
            except ConnectionResetError:
                subprocess_logger.warning(
                    "Connection reset while sending history to new client"
                )
                pass
        else:
            subprocess_logger.info("📭 No message history to send to new client")
        try:
            async for msg in ws:
                if web and msg.type == web.WSMsgType.TEXT:
                    # Parse message to distinguish between permission responses, interrupt signals, and regular user input
                    try:
                        message_data = json.loads(msg.data) if msg.data else {}

                        # Check message type
                        if isinstance(message_data, dict):
                            message_type = message_data.get("type")

                            if message_type == "permission_response":
                                # For permission responses, put the structured data directly
                                input_queue.put(message_data)

                            elif message_type == "interrupt_signal":
                                # Handle interrupt signal - trigger the main interrupt system
                                subprocess_logger.info(
                                    f"Interrupt signal received from web UI: {message_data.get('source', 'unknown')}"
                                )

                                # Use the established IPC mechanism instead of signals
                                # Send a special interrupt command through the input queue
                                interrupt_command = {
                                    "type": "interrupt_command",
                                    "source": "web_ui",
                                    "timestamp": message_data.get(
                                        "timestamp", time.time()
                                    ),
                                    "reason": "user_interrupt_button",
                                }

                                try:
                                    # Send to the visual input queue that the main process monitors
                                    input_queue.put(interrupt_command)
                                    subprocess_logger.info(
                                        "Interrupt command sent via queue to main process"
                                    )

                                    # Also try the signal approach as backup
                                    import os
                                    import signal

                                    main_pid = os.getppid()
                                    os.kill(main_pid, signal.SIGINT)
                                    subprocess_logger.info(
                                        f"Also sent SIGINT to main process (PID: {main_pid}) as backup"
                                    )
                                except Exception as e:
                                    subprocess_logger.error(
                                        f"Failed to send interrupt via queue or signal: {e}"
                                    )

                                # Also try setting the interrupt flag directly if we can access it
                                try:
                                    # This is a fallback - we'll rely on the SIGINT signal primarily
                                    pass
                                except Exception as e:
                                    subprocess_logger.debug(
                                        f"Could not access main interrupt flag: {e}"
                                    )

                            elif (
                                "message" in message_data
                                and "toolConfig" in message_data
                            ):
                                # Handle user message with tool configuration
                                user_message = message_data["message"]
                                tool_config = message_data.get("toolConfig", {})

                                # Log tool configuration
                                disabled_tools = tool_config.get("disabledTools", [])
                                if disabled_tools:
                                    subprocess_logger.info(
                                        f"User message with disabled tools: {disabled_tools}"
                                    )
                                else:
                                    subprocess_logger.debug(
                                        "User message with all tools enabled"
                                    )

                                # Store tool configuration globally for tool discovery to access
                                # This will be picked up by the agent's tool discovery system
                                if hasattr(g, "disabled_tools"):
                                    g.disabled_tools = disabled_tools

                                input_queue.put(user_message)
                                subprocess_logger.info(
                                    f"User message queued with tool restrictions: {user_message[:50]}..."
                                )

                            else:
                                # For other JSON messages, put the raw string
                                input_queue.put(msg.data)
                                subprocess_logger.info(
                                    f"JSON message queued: {message_type}"
                                )
                        else:
                            # For regular user input, put the raw string
                            input_queue.put(msg.data)
                            subprocess_logger.info(
                                f"User input queued: {msg.data[:50]}..."
                            )
                    except json.JSONDecodeError:
                        # If it's not valid JSON, treat it as regular user input
                        input_queue.put(msg.data)
                        subprocess_logger.info(
                            f"Non-JSON user input queued: {msg.data[:50]}..."
                        )
                    except Exception as e:
                        subprocess_logger.error(
                            f"Error processing WebSocket message: {e}"
                        )
                        # Default to treating as user input
                        input_queue.put(msg.data)
                elif web and msg.type == web.WSMsgType.ERROR:
                    subprocess_logger.error(
                        f"WebSocket connection closed with exception {ws.exception()}"
                    )
        finally:
            subprocess_logger.info(f"Browser client disconnected: {request.remote}")
            clients.remove(ws)

            # Unregister from connection monitor
            try:
                from tools._connection_monitor import connection_monitor

                connection_monitor.unregister_client(ws)
            except Exception as e:
                subprocess_logger.warning(
                    f"Failed to unregister client from connection monitor: {e}"
                )
        return ws

    async def index_handler(request: Any) -> Any:
        if web is None:
            raise RuntimeError("aiohttp not available")
        response = web.FileResponse(ASSETS_DIR / "index.html")
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com https://cdn.plot.ly https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https://* http://localhost:* http://127.0.0.1:*; frame-src 'self' data: blob:; connect-src 'self' ws: wss:; object-src 'none'; base-uri 'self'; form-action 'self'"
        )
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    async def favicon_handler(request: Any) -> Any:
        if web is None:
            raise RuntimeError("aiohttp not available")
        response = web.FileResponse(ASSETS_DIR / "favicon.ico")
        response.headers["Cache-Control"] = "public, max-age=86400"
        return response

    async def llm_data_handler(request: Any) -> Any:
        """Serve available LLM models data for the frontend selectboxes."""
        if web is None:
            raise RuntimeError("aiohttp not available")
        try:
            # Import here to avoid circular imports and ensure g is initialized
            from core.globals import g
            from core.llm import Llm

            # Get all available LLMs including dynamic discovery
            available_llms = Llm.get_available_llms(include_dynamic=True)

            # Convert to JSON-serializable format
            llm_data = []
            for llm in available_llms:
                # Get provider name and emoji from the provider class
                provider_name = llm.provider.__class__.__name__.replace("API", "")

                # Create a clean model entry
                model_entry = {
                    "id": llm.model_key,
                    "name": llm.model_key,
                    "provider": provider_name,
                    "context_window": llm.context_window,
                    "capabilities": [str(capability) for capability in llm.capabilities]
                    if llm.capabilities
                    else [],
                    "status": "available",  # Could be enhanced to check actual availability
                }
                llm_data.append(model_entry)

            # Sort by provider and then by name for consistent ordering
            llm_data.sort(key=lambda x: (x["provider"], x["name"]))

            # Get current system configuration from the most recent saved state
            # The web server subprocess should always read fresh config to get latest changes
            # Use unified LLM role configuration
            g.load_llm_roles()  # Refresh from disk
            llm_config = g.get_llm_config(force_reread=True)
            selected_llms = g.SELECTED_LLMS
            planner_llm = g.planner_llm
            router_llm = g.router_llm
            override_llm = g.override_llm

            selected_configs = []
            for model_key in selected_llms:
                config = llm_config.get(model_key, {})
                config["model_key"] = model_key
                selected_configs.append(config)

            # All selected models are treated equally (no specialized roles)
            unique_models = []
            seen = set()
            for model in selected_llms:
                if model not in seen:
                    unique_models.append(model)
                    seen.add(model)

            logger.info(
                f"Served LLM data: {len(llm_data)} models from {len(set(m['provider'] for m in llm_data))} providers"
            )
            logger.info(f"Current system state: {len(selected_llms)} selected models")

            # DEBUG: Log the final response being sent to UI
            response_config = {
                "selected_llms": selected_llms,
                "planner_llm": planner_llm,
                "router_llm": router_llm,
                "override_llm": override_llm,
            }
            subprocess_logger.info(
                f"🔍 FINAL API RESPONSE current_configuration: {response_config}"
            )

            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response(
                {
                    "status": "success",
                    "models": llm_data,
                    "total_count": len(llm_data),
                    "current_configuration": {
                        "selected_llms": selected_llms,
                        "planner_llm": planner_llm,
                        "router_llm": router_llm,
                        "override_llm": override_llm,
                    },
                }
            )

        except Exception as e:
            logger.error(f"Failed to get LLM data: {e}", exc_info=True)
            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response(
                {
                    "status": "error",
                    "message": f"Failed to retrieve LLM data: {str(e)}",
                    "models": [],
                },
                status=500,
            )

    async def tools_handler(request: Any) -> Any:
        """Serve available tools data for the frontend tool manager."""
        if web is None:
            raise RuntimeError("aiohttp not available")
        try:
            # Import here to avoid circular imports
            import importlib
            from pathlib import Path

            # Discover tools from the tools directory
            tools_dir = Path(__file__).parent.parent / "tools"
            tools_data = []

            # Scan both main tools directory and optional subdirectory
            tool_directories = [
                (tools_dir, "tools"),
                (tools_dir / "optional", "tools.optional"),
            ]

            for directory, module_prefix in tool_directories:
                if not directory.exists():
                    continue

                for file in directory.glob("*.py"):
                    if file.name.startswith("_") or file.stem == "__init__":
                        continue

                    try:
                        # Import the tool module with appropriate prefix
                        module_name = f"{module_prefix}.{file.stem}"
                        module = importlib.import_module(module_name)

                        # Find the tool class (should match the filename)
                        tool_class_name = file.stem
                        if hasattr(module, tool_class_name):
                            tool_class = getattr(module, tool_class_name)
                            if hasattr(tool_class, "get_delim") and hasattr(
                                tool_class, "get_tool_info"
                            ):
                                try:
                                    tool_info = tool_class.get_tool_info()
                                    tool_delim = tool_class.get_delim()

                                    tools_data.append(
                                        {
                                            "name": tool_class_name,
                                            "delim": tool_delim,
                                            "info": tool_info,
                                            "displayName": tool_info.get(
                                                "name", tool_class_name
                                            ),
                                            "description": tool_info.get(
                                                "description", ""
                                            ),
                                            "example": tool_info.get("example", ""),
                                        }
                                    )
                                except Exception as tool_error:
                                    subprocess_logger.debug(
                                        f"Could not get info for tool {tool_class_name}: {tool_error}"
                                    )
                    except Exception as import_error:
                        subprocess_logger.debug(
                            f"Could not import tool from {file}: {import_error}"
                        )

            # Sort tools alphabetically by name
            tools_data.sort(key=lambda x: x["name"])

            subprocess_logger.info(
                f"Served tools data: {len(tools_data)} tools discovered"
            )

            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response(
                {
                    "status": "success",
                    "tools": tools_data,
                    "total_count": len(tools_data),
                }
            )

        except Exception as e:
            subprocess_logger.error(f"Failed to get tools data: {e}", exc_info=True)
            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response(
                {
                    "status": "error",
                    "message": f"Failed to retrieve tools data: {str(e)}",
                    "tools": [],
                },
                status=500,
            )

    async def upload_handler(request: Any) -> Any:
        if web is None:
            raise RuntimeError("aiohttp not available")
        try:
            reader = await request.multipart()
            uploaded_files = []

            while True:
                field = await reader.next()
                if field is None:
                    break

                if field.name == "file":
                    # Get the filename, sanitize it
                    filename = field.filename or "uploaded_file"
                    safe_filename = "".join(
                        c for c in filename if c.isalnum() or c in (" ", ".", "_", "-")
                    ).rstrip()
                    if not safe_filename:
                        safe_filename = f"uploaded_file_{int(time.time())}"

                    # Import g to get temp storage path
                    from core.globals import g

                    temp_dir = Path(g.CLIAGENT_TEMP_STORAGE_PATH)
                    temp_dir.mkdir(parents=True, exist_ok=True)

                    # Create unique filename if it already exists
                    counter = 1
                    base_name, ext = (
                        safe_filename.rsplit(".", 1)
                        if "." in safe_filename
                        else (safe_filename, "")
                    )
                    final_filename = safe_filename
                    while (temp_dir / final_filename).exists():
                        if ext:
                            final_filename = f"{base_name}_{counter}.{ext}"
                        else:
                            final_filename = f"{safe_filename}_{counter}"
                        counter += 1

                    file_path = temp_dir / final_filename

                    # Save the file
                    with open(file_path, "wb") as f:
                        while True:
                            chunk = await field.read_chunk()
                            if not chunk:
                                break
                            f.write(chunk)

                    uploaded_files.append(str(file_path.absolute()))
                    subprocess_logger.info(
                        f"File uploaded: {final_filename} -> {file_path}"
                    )

            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response(
                {
                    "status": "success",
                    "files": uploaded_files,
                    "message": f"Successfully uploaded {len(uploaded_files)} file(s)",
                }
            )

        except Exception as e:
            subprocess_logger.error(f"File upload error: {e}")
            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def profiles_handler(request: Any) -> Any:
        if web is None:
            raise RuntimeError("aiohttp not available")
        """Serve LLM profiles data for the frontend."""
        try:
            from core.llm_profiles import get_profile_manager

            manager = get_profile_manager()
            profiles = manager.list_profiles()

            profiles_data = [
                {
                    "id": profile.id,
                    "override": profile.override,
                    "router": profile.router,
                    "planner": profile.planner,
                    "fallbacks": profile.fallbacks,
                }
                for profile in profiles
            ]

            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response({"status": "success", "profiles": profiles_data})

        except Exception as e:
            subprocess_logger.error(f"Failed to get profiles data: {e}", exc_info=True)
            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response(
                {
                    "status": "error",
                    "message": f"Failed to retrieve profiles data: {str(e)}",
                    "profiles": [],
                },
                status=500,
            )

    async def chat_history_list_handler(request: Any) -> Any:
        if web is None:
            raise RuntimeError("aiohttp not available")
        """List all archived chats."""
        try:
            from core.chat import Chat

            chats = Chat.list_archived_chats()

            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response({"status": "success", "chats": chats})

        except Exception as e:
            subprocess_logger.error(f"Failed to list chat history: {e}", exc_info=True)
            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response(
                {
                    "status": "error",
                    "message": f"Failed to retrieve chat history: {str(e)}",
                    "chats": [],
                },
                status=500,
            )

    async def chat_history_load_handler(request: Any) -> Any:
        if web is None:
            raise RuntimeError("aiohttp not available")
        """Load a specific archived chat by ID."""
        try:
            from core.chat import Chat

            chat_id = request.match_info.get("id")
            if not chat_id:
                if web is None:
                    raise RuntimeError("aiohttp not available")
                return web.json_response(
                    {"status": "error", "message": "Chat ID is required"}, status=400
                )

            chat = Chat.load_from_archive(chat_id)
            chat_dict = chat._to_dict()

            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response({"status": "success", "chat": chat_dict})

        except FileNotFoundError as e:
            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response({"status": "error", "message": str(e)}, status=404)
        except Exception as e:
            subprocess_logger.error(f"Failed to load chat: {e}", exc_info=True)
            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response(
                {"status": "error", "message": f"Failed to load chat: {str(e)}"},
                status=500,
            )

    async def chat_history_delete_handler(request: Any) -> Any:
        if web is None:
            raise RuntimeError("aiohttp not available")
        """Delete a specific archived chat by ID."""
        try:
            from core.chat import Chat

            chat_id = request.match_info.get("id")
            if not chat_id:
                if web is None:
                    raise RuntimeError("aiohttp not available")
                return web.json_response(
                    {"status": "error", "message": "Chat ID is required"}, status=400
                )

            success = Chat.delete_from_archive(chat_id)

            if success:
                if web is None:
                    raise RuntimeError("aiohttp not available")
                return web.json_response(
                    {
                        "status": "success",
                        "message": f"Chat {chat_id} deleted successfully",
                    }
                )
            else:
                if web is None:
                    raise RuntimeError("aiohttp not available")
                return web.json_response(
                    {"status": "error", "message": f"Chat {chat_id} not found"},
                    status=404,
                )

        except Exception as e:
            subprocess_logger.error(f"Failed to delete chat: {e}", exc_info=True)
            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response(
                {"status": "error", "message": f"Failed to delete chat: {str(e)}"},
                status=500,
            )

    async def chat_history_search_handler(request: Any) -> Any:
        if web is None:
            raise RuntimeError("aiohttp not available")
        """Search archived chats by query."""
        try:
            from core.chat import Chat

            query = request.query.get("q", "").lower()
            if not query:
                if web is None:
                    raise RuntimeError("aiohttp not available")
                return web.json_response(
                    {"status": "error", "message": "Search query is required"},
                    status=400,
                )

            chats = Chat.list_archived_chats()

            # Filter chats by query in preview
            filtered_chats = [
                chat for chat in chats if query in chat.get("preview", "").lower()
            ]

            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response(
                {"status": "success", "chats": filtered_chats, "query": query}
            )

        except Exception as e:
            subprocess_logger.error(f"Failed to search chats: {e}", exc_info=True)
            if web is None:
                raise RuntimeError("aiohttp not available")
            return web.json_response(
                {
                    "status": "error",
                    "message": f"Failed to search chats: {str(e)}",
                    "chats": [],
                },
                status=500,
            )

    async def main():
        if web is None:
            raise RuntimeError("aiohttp not available")
        app = web.Application()
        app.router.add_get("/", index_handler)
        app.router.add_get("/ws", websocket_handler)
        app.router.add_get("/favicon.ico", favicon_handler)
        app.router.add_get("/api/llms", llm_data_handler)
        app.router.add_get("/api/tools", tools_handler)
        app.router.add_get("/api/profiles", profiles_handler)
        app.router.add_get("/api/chat_history", chat_history_list_handler)
        app.router.add_get("/api/chat_history/{id}", chat_history_load_handler)
        app.router.add_delete("/api/chat_history/{id}", chat_history_delete_handler)
        app.router.add_get("/api/chat_history/search", chat_history_search_handler)
        app.router.add_post("/upload", upload_handler)
        app.router.add_static("/libs/", path=ASSETS_DIR / "libs", name="libs")
        app.router.add_static("/css/", path=ASSETS_DIR / "css", name="css")
        app.router.add_static("/js/", path=ASSETS_DIR / "js", name="js")

        # Serve files from the working directory (where main.py was started)
        working_dir = Path.cwd()
        app.router.add_static(
            "/files/", path=working_dir, name="working_dir", show_index=False
        )

        # Also serve .xyz files from root for convenience
        async def xyz_handler(request: Any) -> Any:
            if web is None:
                raise RuntimeError("aiohttp not available")
            filename = request.match_info["filename"]
            if not filename.endswith(".xyz"):
                return web.Response(status=404, text="Not found")
            filepath = working_dir / filename
            if not filepath.exists() or not filepath.is_file():
                return web.Response(status=404, text="Not found")
            return web.FileResponse(filepath)

        app.router.add_get("/{filename}.xyz", xyz_handler)
        if web is None:
            raise RuntimeError("aiohttp not available")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, SERVER_HOST, port)
        try:
            await site.start()
            subprocess_logger.info(
                f"Server started successfully at http://{SERVER_HOST}:{port}"
            )

            # Start connection monitoring
            try:
                from tools._connection_monitor import connection_monitor

                monitoring_task = asyncio.create_task(
                    connection_monitor.start_monitoring()
                )
                subprocess_logger.info("Connection monitoring started")
            except Exception as e:
                subprocess_logger.warning(f"Failed to start connection monitoring: {e}")
                monitoring_task = None

            await asyncio.sleep(0.1)
            try:
                import urllib.request

                test_url = f"http://{SERVER_HOST}:{port}/"
                req = urllib.request.Request(
                    test_url, headers={"User-Agent": "CLI-Agent-Test"}
                )
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status != 200:
                        subprocess_logger.warning(
                            f"Server responded with unexpected status: {response.status}"
                        )
            except Exception as connectivity_error:
                subprocess_logger.warning(
                    f"Server connectivity verification failed: {connectivity_error}"
                )
            subprocess_logger.info("Server is now accepting connections")

            # Start file watcher for console logging
            # Create file watcher for the standard log file paths
            log_paths = {
                "stdout": "/tmp/cli-agent-stdout.log",
                "stderr": "/tmp/cli-agent-stderr.log",
            }

            # Check if log files exist (indicates console capture is enabled)
            import os

            subprocess_logger.info(
                f"Checking for log files: stdout={os.path.exists(log_paths['stdout'])}, stderr={os.path.exists(log_paths['stderr'])}"
            )
            if os.path.exists(log_paths["stdout"]) or os.path.exists(
                log_paths["stderr"]
            ):
                subprocess_logger.info("Log files found - creating file watcher")
                global _file_watcher
                _file_watcher = LogFileWatcher(log_paths)
                loop = asyncio.get_running_loop()
                subprocess_logger.info(f"Created file watcher: {_file_watcher}")

                # Start file watching directly
                try:
                    _file_watcher.running = True
                    _file_watcher.watch_task = loop.create_task(
                        _file_watcher._watch_files()
                    )
                    subprocess_logger.info(
                        f"File watching task created: {_file_watcher.watch_task}"
                    )
                    subprocess_logger.info("File-based console watching started")
                except Exception as e:
                    subprocess_logger.error(
                        f"Failed to start file watching task: {e}", exc_info=True
                    )
            else:
                subprocess_logger.warning(
                    "No log files found - file watcher not started"
                )

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
        if self.process and self.process.is_alive():
            return
        logger.info("Starting visual server process...")
        self.port = find_open_port(PORT_RANGE_START, PORT_RANGE_END, SERVER_HOST)
        if self.port is None:
            logger.error(
                colored(
                    f"Could not find an open port in range {PORT_RANGE_START}-{PORT_RANGE_END}.",
                    "red",
                )
            )
            return
        logger.info(
            colored(f"Spawning visual response server on port {self.port}...", "cyan")
        )
        try:
            self.process = self.ctx.Process(
                target=_server_process_main_loop,
                args=(self.output_queue, self.input_queue, self.port),
            )
            self.process.start()
            logger.info(
                f"Visual server subprocess started with PID: {self.process.pid}"
            )
            time.sleep(1.0)
            if not self.process.is_alive():
                logger.error(
                    f"Visual server process died immediately. Exit code: {self.process.exitcode}"
                )
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

            req = urllib.request.Request(
                server_url, headers={"User-Agent": "CLI-Agent"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    logger.info("Server connectivity test passed")
                else:
                    logger.warning(f"Server responded with status {response.status}")
        except Exception as e:
            logger.warning(f"Server connectivity test failed: {e}")
        try:
            webbrowser.open(server_url)
            logger.info("Browser opened successfully")
        except Exception as e:
            logger.warning(
                colored(f"Could not automatically open browser: {e}", "yellow")
            )

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
        if not self.process or not self.process.is_alive():
            return False
        if not self.port:
            return False
        return test_port_connectivity(SERVER_HOST, self.port, timeout=2)

    def get_status(self):
        status = {
            "process_alive": self.process and self.process.is_alive(),
            "process_pid": self.process.pid if self.process else None,
            "port": self.port,
            "server_responsive": False,
            "queue_size": None,
        }
        if self.process:
            status["process_exitcode"] = self.process.exitcode
        if self.port:
            status["server_responsive"] = test_port_connectivity(
                SERVER_HOST, self.port, timeout=1
            )
        try:
            status["queue_size"] = self.output_queue.qsize()
        except:
            pass
        return status

    def close(self):
        if self.process and self.process.is_alive():
            logger.info(
                colored(
                    f"Closing visual response server (PID: {self.process.pid})...",
                    "cyan",
                )
            )
            self.output_queue.put("__EXIT__")
            self.process.join(timeout=3)
            if self.process.is_alive():
                logger.warning(
                    colored(
                        "Server process did not exit gracefully, terminating.", "yellow"
                    )
                )
                self.process.terminate()
            logger.info(colored("Visual response server closed.", "cyan"))
        self.process = None
        if hasattr(g, "visual_input_queue"):
            delattr(g, "visual_input_queue")


class visualresponse:
    _session: Optional[VisualServerSession] = None

    @staticmethod
    def close_session():
        if visualresponse._session:
            visualresponse._session.close()
            visualresponse._session = None

    @staticmethod
    def get_delim() -> str:
        return "visualresponse"

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "visualresponse",
            "description": (
                "Display HTML or Markdown in browser UI. Use <type>html</type> for interactive content "
                "(charts, forms, visualizations) with self-contained document. Use <type>markdown</type> "
                "for simple text. Prefer dark theme. Use <help>visualresponse</help> for CDATA wrapping and examples."
            ),
            "example": (
                "<visualresponse><type>html</type><content><![CDATA[<!DOCTYPE html>...full HTML...</content></visualresponse>\n"
                "<visualresponse><type>markdown</type><content># Title\nContent here</content></visualresponse>"
            ),
        }

    @staticmethod
    def get_detailed_tool_info() -> str:
        """Provides comprehensive help documentation for this tool."""
        return """VISUALRESPONSE - EXTENDED HELP

USAGE:
  <visualresponse><type>html</type><content><![CDATA[<!DOCTYPE html>...]]></content></visualresponse>
  <visualresponse><type>markdown</type><content># Title
Content</content></visualresponse>

PURPOSE:
Display rich content in browser UI. Supports interactive HTML with JavaScript
or simple Markdown formatting.

PARAMETERS:
  <type> - Content format: "html" or "markdown" (required)
  <content> - The HTML document or Markdown text (required, use CDATA for HTML)
  <end_turn> - End agent turn after display: true/false (optional, default true)

CONTENT TYPES:
  html:
    • Full HTML document (<!DOCTYPE html> required)
    • Self-contained (inline CSS/JS or CDN)
    • Interactive elements supported
    • Wrap in <![CDATA[...]]> to avoid XML parsing

  markdown:
    • GitHub-flavored Markdown
    • No JavaScript support
    • Faster rendering
    • Auto-styled with dark theme

EXAMPLES:
  Plotly chart:
    <visualresponse><type>html</type><content><![CDATA[
      <!DOCTYPE html>
      <html>
      <head><script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head>
      <body style="background:#1e1e1e;padding:20px;">
        <div id="chart"></div>
        <script>
          Plotly.newPlot('chart', [{x:[1,2,3], y:[2,4,6], type:'scatter'}], 
            {paper_bgcolor:'#1e1e1e', font:{color:'#e0e0e0'}});
        </script>
      </body></html>
    ]]></content></visualresponse>

  Markdown:
    <visualresponse><type>markdown</type><content>
# Analysis Results

- 45% increase in engagement
- 23% reduction in bounce rate
    </content></visualresponse>

CDATA WRAPPER:
  Why: XML parsers treat < > & as special chars
  When: ALWAYS for HTML (has <script>, <div>, etc.)
  Syntax: <content><![CDATA[...content...]]></content>

DARK THEME (HTML):
  • Background: #1e1e1e or #2d2d2d
  • Text: #ffffff or #e0e0e0
  • Borders: #444 or #555
  • Matches UI for consistency

MULTI-STEP:
  Use <end_turn>false</end_turn> to continue after display:
    <visualresponse><end_turn>false</end_turn><type>markdown</type><content># Step 1</content></visualresponse>
    <!-- Agent continues -->
    <visualresponse><type>html</type><content><![CDATA[...final viz...]]></content></visualresponse>

CDN LIBRARIES:
  • Plotly: https://cdn.plot.ly/plotly-latest.min.js
  • Chart.js: https://cdn.jsdelivr.net/npm/chart.js
  • D3.js: https://d3js.org/d3.v7.min.js
  • Mermaid (in Markdown): ```mermaid

TIPS:
  • ALWAYS use CDATA for HTML
  • Self-contained only (no local file references)
  • Dark theme for consistency
  • Test incrementally
  • Use markdown for simple text, HTML for interactive

COMMON ERRORS:
  Missing <!DOCTYPE html> → Include full HTML structure
  Not using CDATA → Wrap HTML in <![CDATA[...]]>
  Local file refs → Use CDN instead
  Syntax errors → Validate HTML/JS before sending

LIMITATIONS:
  • No file system access from HTML
  • No server-side execution
  • External resources must be CDN
  • JavaScript errors not reported to agent

RELATED: <help>readfile</help> <help>writefile</help> <help>bash</help>
"""

    @staticmethod
    async def _get_or_create_server(open_browser: bool = True):
        """Get or create the new API server"""
        print(
            f"[DEBUG _get_or_create_server] Called with open_browser={open_browser}",
            flush=True,
        )
        from api.server import get_or_create_server
        from core.globals import g

        persist_mode = hasattr(g, "GUI_MODE") and g.GUI_MODE
        print(
            f"[DEBUG _get_or_create_server] persist_mode={persist_mode}, about to call get_or_create_server",
            flush=True,
        )
        server = await get_or_create_server(
            persist_mode=persist_mode, open_browser=open_browser
        )
        print(f"[DEBUG _get_or_create_server] Server created: {server}", flush=True)
        return server

    @staticmethod
    async def _get_session_async(open_browser: bool = True) -> VisualServerSession:
        """Async version - creates/gets the API server"""
        server = await visualresponse._get_or_create_server(open_browser=open_browser)

        class ServerAdapter:
            """Adapter to make new server look like old session"""

            def __init__(self, srv):
                self.server = srv
                self.port = srv.port

            def is_healthy(self):
                return self.server is not None

            async def send_update_async(self, payload):
                await self.server.broadcast(payload)

            def send_update(self, payload):
                """Synchronous wrapper for send_update"""
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    asyncio.ensure_future(self.send_update_async(payload), loop=loop)
                except RuntimeError:
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.run_coroutine_threadsafe(
                                self.send_update_async(payload), loop
                            )
                        else:
                            loop.create_task(self.send_update_async(payload))
                    except Exception as e:
                        logger.error(f"Failed to send update: {e}")

            def close(self):
                pass

            def get_status(self):
                return {
                    "port": self.port,
                    "clients": len(self.server.websocket_clients),
                    "persist_mode": self.server.persist_mode,
                }

        if not hasattr(visualresponse, "_adapter") or visualresponse._adapter is None:
            visualresponse._adapter = ServerAdapter(server)

        return visualresponse._adapter

    @staticmethod
    def _get_session() -> VisualServerSession:
        """Legacy sync method - wraps async version"""
        import asyncio
        import threading

        # Check if we already have an adapter
        if hasattr(visualresponse, "_adapter") and visualresponse._adapter is not None:
            return visualresponse._adapter

        # If there's a running loop in this thread, offload to a new thread with its own loop
        try:
            running_loop = asyncio.get_running_loop()
            # We are in an async context; create a helper thread to obtain the session
            result_holder = {"adapter": None, "error": None}

            def _runner():
                try:
                    # Use a fresh event loop in this dedicated thread
                    adapter = asyncio.run(visualresponse._get_session_async())
                    result_holder["adapter"] = adapter
                except Exception as e:
                    result_holder["error"] = e

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            # Wait briefly for the adapter to be created
            t.join(timeout=5.0)
            if result_holder["adapter"] is not None:
                return result_holder["adapter"]
            if result_holder["error"] is not None:
                raise result_holder["error"]
            # If still not ready, return existing adapter if any, otherwise raise a clear error
            if (
                hasattr(visualresponse, "_adapter")
                and visualresponse._adapter is not None
            ):
                return visualresponse._adapter
            raise RuntimeError(
                "visualresponse server initialization is in progress; try again shortly"
            )
        except RuntimeError:
            # No running loop; safe to run the coroutine synchronously in this thread
            pass

        # Get or create event loop in this (non-async) thread and run the async method
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if getattr(loop, "is_running", lambda: False)():
            # Safety guard: if loop is somehow running, fall back to thread offload path
            return visualresponse._get_session()
        return loop.run_until_complete(visualresponse._get_session_async())

    @staticmethod
    def get_speech_text(content: str) -> Optional[str]:
        """
        Optional method to provide custom TTS text for this tool.

        Returns:
            - String to speak via TTS
            - None to skip TTS entirely
            - If not implemented, default XML filtering is used

        This is part of Tier 3 custom tool speech generation.
        """
        try:
            # Extract type from visualresponse content
            type_match = re.search(
                r"<type>(.*?)</type>", content, re.IGNORECASE | re.DOTALL
            )
            if type_match:
                viz_type = type_match.group(1).strip().lower()

                # Skip TTS for visual-only content
                if viz_type in ["html", "code", "javascript", "css"]:
                    return None

                # For markdown, extract and clean the text content
                elif viz_type == "markdown":
                    content_match = re.search(
                        r"<content>(.*?)</content>", content, re.IGNORECASE | re.DOTALL
                    )
                    if content_match:
                        markdown_text = content_match.group(1).strip()
                        # Remove CDATA wrapper if present
                        if markdown_text.startswith("<![CDATA["):
                            markdown_text = markdown_text[9:-3].strip()
                        # Remove markdown formatting
                        markdown_text = re.sub(r"[#*_`]", "", markdown_text)
                        return markdown_text

            # Default: use standard XML filtering
            return None
        except Exception as e:
            logger.debug(f"Error in get_speech_text: {e}")
            return None

    @staticmethod
    def run(content: str) -> str:
        try:
            if content.strip().lower() in ["status", "test", "diagnostic", "debug"]:
                return visualresponse._run_diagnostics()

            # Parse end_turn parameter (default: True for backward compatibility)
            end_turn = True
            end_turn_match = re.search(r'<end_turn>\s*(true|false)\s*</end_turn>', content, re.IGNORECASE)
            if end_turn_match:
                end_turn = end_turn_match.group(1).lower() == 'true'

            # Set the end_turn flag in tool_context for main loop to check
            tool_context.set_end_turn(end_turn)

            # --- Simplified parsing: always create, auto-generate ID ---
            args = {}
            # Use re.DOTALL to make '.' match newlines, crucial for multi-line content
            type_match = re.search(
                r"<type>(.*?)</type>", content, re.IGNORECASE | re.DOTALL
            )
            if type_match:
                args["type"] = type_match.group(1).strip()

            content_match = re.search(
                r"<content>(.*?)</content>", content, re.IGNORECASE | re.DOTALL
            )
            if content_match:
                # Strip CDATA wrapper if present, then strip whitespace
                c = content_match.group(1)
                if c.strip().startswith("<![CDATA["):
                    c = c.strip()[9:-3]  # Remove <![CDATA[ and ]]>
                args["content"] = c.strip()

            # Always default to 'create' action
            action = "create"
            # Auto-generate unique ID using precise timestamp
            unique_timestamp = int(time.time() * 1000000)  # Microsecond precision
            viz_id = f"viz-{unique_timestamp}"

            payload = {"id": viz_id, "action": action}

            # Add model key metadata
            try:
                from core.llm_router import LlmRouter

                used_model = LlmRouter.get_last_used_model()
                if used_model:
                    payload["model"] = used_model
            except Exception as e:
                logger.debug(f"Could not get model key for response: {e}")

            # Add tool execution info if available
            if hasattr(g, "last_executed_tools_info") and g.last_executed_tools_info:
                payload["tool_info"] = g.last_executed_tools_info
                # Clear the info after using it to avoid reusing stale data
                g.last_executed_tools_info = None

            # Always create content
            viz_type = args.get("type")
            viz_content_str = args.get("content")

            if not viz_type or viz_content_str is None:
                return "Error: visualresponse requires <type> and <content> tags."

            viz_type = viz_type.lower()
            supported_types = ["markdown", "html"]
            if viz_type not in supported_types:
                return f"Error: Unsupported type '{viz_type}'. Supported types are: {', '.join(supported_types)}"

            payload["type"] = viz_type
            payload["content"] = viz_content_str

            session = visualresponse._get_session()
            if not session.is_healthy():
                session.close()
                session = visualresponse._get_session()

            session.send_update(payload)

            # Set current viz_id in tool context for live tool updates
            tool_context.set_current_viz_id(viz_id)

            return "Visual response created successfully."
            # return f"Visual response created successfully with ID '{viz_id}' and sent to the live UI on port {session.port}."

        except Exception as e:
            logger.error(f"Fatal error in visualresponse tool: {e}", exc_info=True)
            return f"Fatal error in visualresponse tool: {e}"

    @staticmethod
    def _run_diagnostics() -> str:
        # This function remains the same
        diagnostics = ["🔍 Visual Server Diagnostics", "=" * 40]
        if AIOHTTP_AVAILABLE:
            diagnostics.append(f"✅ aiohttp {AIOHTTP_VERSION} is available")
        else:
            diagnostics.append(f"❌ aiohttp not available: {AIOHTTP_IMPORT_ERROR}")
        if ASSETS_DIR.exists():
            diagnostics.append(f"✅ Assets directory exists: {ASSETS_DIR}")
            for file_name in ["index.html", "favicon.ico"]:
                file_path = ASSETS_DIR / file_name
                if file_path.exists():
                    diagnostics.append(
                        f"  ✅ {file_name} ({file_path.stat().st_size} bytes)"
                    )
                else:
                    diagnostics.append(f"  ❌ {file_name} missing")
        else:
            diagnostics.append(f"❌ Assets directory missing: {ASSETS_DIR}")
        if hasattr(visualresponse, "_adapter") and visualresponse._adapter:
            status = visualresponse._adapter.get_status()
            diagnostics.append("\n📊 Current Session Status:")
            for key, value in status.items():
                diagnostics.append(f"  {key.replace('_', ' ').title()}: {value}")
        else:
            diagnostics.append("\n📊 No active session")
        available_port = find_open_port(PORT_RANGE_START, PORT_RANGE_END)
        if available_port:
            diagnostics.append(f"\n🔌 Available port found: {available_port}")
        else:
            diagnostics.append(
                f"\n❌ No available ports in range {PORT_RANGE_START}-{PORT_RANGE_END}"
            )
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
                "id": f"user-msg-{int(time.time() * 1000000)}",
            }

            # Add model key metadata for user messages too (to track context)
            try:
                from core.llm_router import LlmRouter

                used_model = LlmRouter.get_last_used_model()
                if used_model:
                    payload["model"] = used_model
            except Exception as e:
                logger.debug(f"Could not get model key for user message: {e}")

            print(f"[DEBUG] About to send user message payload: {payload}")
            session.send_update(payload)
            print(
                f"[DEBUG] ✅ Sent user message to visual interface: {message[:50]}..."
            )
            return True

        except Exception as e:
            print(f"[DEBUG] ❌ Failed to send user message to visual interface: {e}")
            import traceback

            traceback.print_exc()
            return False

    @staticmethod
    def send_console_log(line: str, level: str = "info") -> bool:
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
                "id": f"console-log-{int(time.time() * 1000000)}",
            }

            # Add model key metadata for console logs too
            try:
                from core.llm_router import LlmRouter

                used_model = LlmRouter.get_last_used_model()
                if used_model:
                    payload["model"] = used_model
            except Exception as e:
                logger.debug(f"Could not get model key for console log: {e}")

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

    def __init__(self, log_dir: Optional[str] = None):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.enabled = False

        # Set up log file paths using XDG cache directory specification
        if log_dir is None:
            # Follow XDG Base Directory specification for cache files
            xdg_cache_home = os.environ.get(
                "XDG_CACHE_HOME", os.path.expanduser("~/.cache")
            )
            log_dir = os.path.join(xdg_cache_home, "cli-agent", "logs")

        # Ensure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        self.stdout_log_path = os.path.join(log_dir, "console-stdout.log")
        self.stderr_log_path = os.path.join(log_dir, "console-stderr.log")

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
        logger.debug(
            f"File-based console capture enabled: stdout={self.stdout_log_path}, stderr={self.stderr_log_path}"
        )

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
        return {"stdout": self.stdout_log_path, "stderr": self.stderr_log_path}

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
        sys.stdout = ConsoleWrapper(self.original_stdout, "info")
        sys.stderr = ConsoleWrapper(self.original_stderr, "error")
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
        self.log_file = open(log_file_path, "a", buffering=1, encoding="utf-8")

    def write(self, text):
        # Write to original stream first (for terminal display)
        self.original_stream.write(text)

        # Write raw text directly to log file (no filtering)
        self.log_file.write(text)

        # Directly send to WebSocket clients if we have non-empty content
        if text.strip():
            self._broadcast_to_console(text.rstrip("\n\r"))

    def _broadcast_to_console(self, content: str):
        """Send content directly to browser console via WebSocket."""
        try:
            # Strip ANSI escape sequences for clean console display
            import re

            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            clean_content = ansi_escape.sub("", content)

            # Use the existing visualresponse.send_console_log function
            # which properly handles the WebSocket communication
            level = "error" if "stderr" in self.log_file_path.lower() else "info"

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
        if hasattr(self, "log_file") and self.log_file:
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
            ansi_escape = re.compile(
                r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[.*?[@-~]"
            )
            clean_text = ansi_escape.sub("", text.strip())

            # Also remove any remaining escape sequences
            clean_text = re.sub(r"\x1b.*?[a-zA-Z]", "", clean_text)

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

    def __init__(self, log_file_paths: dict[str, str]):
        self.log_file_paths = log_file_paths  # {'stdout': path, 'stderr': path}
        self.file_positions = {}  # Track last read position for each file
        self.running = False
        self.watch_task = None

        # Initialize file positions
        for stream_type, log_path in log_file_paths.items():
            try:
                # Start from end of existing file or 0 for new file
                if os.path.exists(log_path):
                    with open(log_path, encoding="utf-8") as f:
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
                logger.debug(
                    "No running event loop - file watching will start with server"
                )
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
        logger.info(
            f"ensure_watching_started: running={self.running}, watch_task={self.watch_task}"
        )
        if self.running and not self.watch_task:
            self.watch_task = loop.create_task(self._watch_files())
            logger.info("Log file watching started with server event loop")
        else:
            logger.warning(
                f"File watching not started: running={self.running}, watch_task={self.watch_task}"
            )

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
                logger.info(
                    f"New content detected in {log_path}: {current_size - last_position} bytes"
                )
                # Read new content
                with open(log_path, encoding="utf-8") as f:
                    f.seek(last_position)
                    new_content = f.read()

                    if new_content:
                        logger.info(
                            f"Sending {len(new_content)} characters to browser console"
                        )
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
                    level = "error" if stream_type == "stderr" else "info"

                    # Use existing broadcast system
                    # TODO: Fix scope issue - broadcast_console_log not accessible here
                    # await broadcast_console_log(line.rstrip('\n\r'), level)

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


_unified_stream_manager = None


def _console_broadcast_callback(line: str):
    """Callback for broadcasting console lines to WebSocket clients."""
    try:
        from tools.visualresponse import send_console_log

        level = "error" if "ERROR" in line or "Error" in line else "info"
        send_console_log(line, level)
    except Exception:
        pass


def enable_console_capture():
    """Enable unified console output capture and streaming."""
    global _unified_stream_manager

    try:
        from pathlib import Path

        from core.globals import g
        from shared.console_stream import init_console_streaming

        # Use same directory logic as logs API
        legacy_dir = Path(g.CLIAGENT_PERSISTENT_STORAGE_PATH) / "logs"
        if legacy_dir.exists():
            log_dir = legacy_dir
        else:
            log_dir = Path(g.CLIAGENT_CACHE_DIR) / "logs"

        log_dir.mkdir(parents=True, exist_ok=True)
        console_log_path = log_dir / "console-live.log"

        _unified_stream_manager = init_console_streaming(
            console_log_path,
            broadcast_callback=_console_broadcast_callback,
            strip_ansi=True,
        )

        logger.info(f"📺 Unified console streaming enabled -> {console_log_path}")

    except Exception as e:
        logger.warning(f"Could not enable unified console streaming: {e}")
        enable_file_based_console_capture()


def disable_console_capture():
    """Disable console output capture."""
    global _unified_stream_manager

    if _unified_stream_manager:
        try:
            from shared.console_stream import stop_console_streaming

            stop_console_streaming()
            _unified_stream_manager = None
            logger.info("Unified console streaming stopped")
        except Exception as e:
            logger.warning(f"Error stopping unified console streaming: {e}")
    else:
        disable_file_based_console_capture()
