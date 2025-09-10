# tools/opengl.py
"""
This file implements the 'opengl' tool. It allows the agent to create and
update a live OpenGL window on the user's screen by providing complete, 
self-contained Python scripts using the 'pyglet' library.

This tool runs a persistent pyglet window in a separate process and 'hot-reloads'
the entire script on each call, similar to how the <showuser> tool works.

Requires: pip install pyglet
"""
import multiprocessing
import logging
from typing import Optional
from termcolor import colored
from queue import Empty as QueueEmpty

# Assume g is initialized and available from the project's core
from core.globals import g

def _opengl_process_main_loop(queue: multiprocessing.Queue):
    """
    This is the entry point for the dedicated OpenGL rendering process.
    It creates a pyglet window and runs an event loop, listening for new
    scripts to execute from the main agent process.
    """
    try:
        import pyglet
        import traceback
    except ImportError:
        # This message will be visible in the console where the agent is running
        print("[OpenGL Process] FATAL: 'pyglet' library not found. Please run 'pip install pyglet' to use this tool.")
        return

    # --- State for the currently running script ---
    script_context = {}
    
    # --- Create the main window ---
    try:
        window = pyglet.window.Window(800, 600, "Agent's Live OpenGL Environment", resizable=True)
    except Exception as e:
        print(f"[OpenGL Process] Error creating window: {e}")
        return

    # This is the primary on_draw loop. It will call the agent's on_draw if defined.
    @window.event
    def on_draw():
        window.clear()
        # If the agent's script has defined a custom on_draw, execute it.
        if 'on_draw' in script_context and callable(script_context['on_draw']):
            try:
                script_context['on_draw']()
            except Exception as e:
                # If the script's on_draw fails, render an error message in the window
                window.clear()
                error_label = pyglet.text.Label(
                    f"RUNTIME ERROR in on_draw:\n{str(e)}",
                    font_name='monospace', font_size=12, x=10, y=window.height - 20,
                    multiline=True, width=window.width - 20, color=(255, 80, 80, 255)
                )
                error_label.draw()

    def update(dt):
        """
        Scheduled by pyglet's clock. This function checks the queue for a new script
        from the agent without blocking the render loop.
        """
        nonlocal script_context
        try:
            message = queue.get_nowait()

            if message == "__EXIT__":
                pyglet.app.exit()
                return

            if isinstance(message, str):
                new_code = message
                print("[OpenGL Process] Received new script. Hot-reloading...")
                
                # Prepare a temporary, clean context for the new script
                temp_context = {
                    'pyglet': pyglet,
                    'window': window,
                    # Provide access to common pyglet modules for convenience
                    'gl': pyglet.gl,
                    'shapes': pyglet.shapes,
                    'text': pyglet.text,
                    'image': pyglet.image,
                }
                
                try:
                    # Unregister all previous event handlers to prevent duplicates
                    window.pop_handlers()
                    
                    # Execute the new script in the temporary context
                    exec(new_code, temp_context)
                    
                    # If successful, replace the main context
                    script_context = temp_context
                    print("[OpenGL Process] Script executed successfully.")

                    # Re-register all event handlers found in the new script
                    # (including our main on_draw dispatcher)
                    window.push_handlers(on_draw) 
                    for name, value in script_context.items():
                        if name.startswith('on_') and callable(value) and name != 'on_draw':
                            window.push_handlers(value)
                            print(f"[OpenGL Process] Attached event handler: {name}")
                
                except Exception:
                    # If the script fails to compile/run, show the error in the window
                    error_str = traceback.format_exc()
                    print(f"[OpenGL Process] ERROR executing new script:\n{error_str}")
                    
                    # Create an error display context
                    def error_on_draw():
                        window.clear()
                        error_label = pyglet.text.Label(
                            f"SCRIPT ERROR:\n\n{error_str}",
                            font_name='monospace', font_size=10, x=10, y=window.height - 20,
                            multiline=True, width=window.width - 20, color=(255, 100, 100, 255)
                        )
                        error_label.draw()
                    
                    # Temporarily replace the script context to show the error
                    script_context = {'on_draw': error_on_draw}
                    window.pop_handlers()
                    window.push_handlers(on_draw)

        except QueueEmpty:
            pass # No new script, continue with the current one

    pyglet.clock.schedule_interval(update, 1/60.0)
    pyglet.app.run()
    window.close()
    print("[OpenGL Process] Exited cleanly.")


class OpenGLSession:
    """Manages the lifecycle of the persistent OpenGL process."""
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.process = None
        self.start()

    def start(self):
        if self.process and self.process.is_alive():
            return
        logging.info(colored("  - Spawning persistent OpenGL window process...", "cyan"))
        # Use 'spawn' context for compatibility across platforms (macOS, Windows)
        multiprocessing.set_start_method('spawn', force=True)
        self.process = multiprocessing.Process(target=_opengl_process_main_loop, args=(self.queue,))
        self.process.start()
        logging.info(colored("  - OpenGL process started. A window should appear on your screen.", "cyan"))

    def update_code(self, code: str):
        if not self.process or not self.process.is_alive():
            self.start()
        self.queue.put(code)

    def close(self):
        if self.process and self.process.is_alive():
            logging.info(colored("  - Closing OpenGL window process...", "cyan"))
            self.queue.put("__EXIT__")
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
            logging.info(colored("  - OpenGL process closed.", "cyan"))
        self.process = None

class opengl:
    _session: Optional[OpenGLSession] = None

    @staticmethod
    def get_delim() -> str:
        return 'opengl'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "opengl",
            "description": "Renders a live, interactive OpenGL window on the user's screen by executing a complete Python script using the 'pyglet' library. Each call replaces the entire previous script, allowing for full, iterative control of the visual output. The `window` and `pyglet` objects are pre-defined and available.",
            "example": "<opengl>\nimport pyglet\nfrom pyglet.gl import *\n\nlabel = pyglet.text.Label('Hello from the Agent!', font_size=24, x=window.width//2, y=window.height//2, anchor_x='center', anchor_y='center')\n\n@window.event\ndef on_draw():\n    window.clear()\n    label.draw()\n\n# You can also define other events like on_mouse_press, etc.\n</opengl>"
        }

    @staticmethod
    def _get_session() -> OpenGLSession:
        """Manages the singleton session for the OpenGL environment."""
        session_dead = (opengl._session is None or 
                        opengl._session.process is None or 
                        not opengl._session.process.is_alive())
        
        if session_dead:
            if opengl._session:
                logging.warning(colored("  - OpenGL session was found dead. Restarting.", "yellow"))
            opengl._session = OpenGLSession()
            g.register_cleanup(opengl._get_session().close)
        return opengl._session

    @staticmethod
    def run(content: str) -> str:
        """
        Receives a Python script from the agent and sends it to the
        OpenGL process to be rendered.
        """
        try:
            code_to_run = content.strip()
            if not code_to_run:
                return "Error: No Python code provided for the OpenGL tool."

            session = opengl._get_session()
            session.update_code(code_to_run)
            
            return "OpenGL script submitted for live rendering. The window on the user's screen should update."
        except Exception as e:
            logging.error(f"Fatal error in opengl tool: {e}", exc_info=True)
            return f"Fatal error executing opengl tool: {e}"