# tui_main.py
import asyncio
import sys
import traceback
import threading
import logging
import queue
import time
from typing import Optional, Dict, Any

from textual.app import App, ComposeResult
from textual.widgets import RichLog, Input, Static
from textual.worker import Worker, WorkerState
from textual.suggester import Suggester

# --- Project Imports ---
from main import main as run_agent_logic, parse_cli_args, setup_logging
from agent.core.command_system import _build_command_definitions, _build_command_descriptions

class CommandSuggester(Suggester):
    """A command suggester for Textual Input based on CLI-Agent's nested command dictionary."""

    def __init__(self, command_dict: Dict[str, Any]):
        super().__init__(use_cache=True)  # Enable caching for better performance
        self.command_dict = command_dict
        self.descriptions = _build_command_descriptions()

    async def get_suggestion(self, value: str) -> str:
        """
        Given the current input value, return the best matching suggestion.
        Note: Textual doesn't support tooltips, so we can only return the command itself.
        
        Args:
            value: Current input text
            
        Returns:
            Best matching command or empty string if no match
        """
        if not value or not value.startswith("/"):
            return ""

        # Handle exact matches first
        if value in self.command_dict:
            return value

        # Find the best partial match
        current_dict = self.command_dict
        parts = value.split()
        
        if len(parts) == 1:
            # Single command - find best match
            partial = parts[0]
            matches = [cmd for cmd in current_dict.keys() if cmd.startswith(partial)]
            if matches:
                # Return the shortest match (most likely intended command)
                best_match = sorted(matches, key=len)[0]
                return best_match
        
        elif len(parts) == 2:
            # Command with subcommand - check if main command exists and has subcommands
            main_cmd, sub_partial = parts[0], parts[1]
            if main_cmd in current_dict and isinstance(current_dict[main_cmd], dict):
                sub_matches = [sub for sub in current_dict[main_cmd].keys() if sub.startswith(sub_partial)]
                if sub_matches:
                    return f"{main_cmd} {sorted(sub_matches, key=len)[0]}"
        
        return ""

class StdoutRedirector:
    """A thread-safe object that redirects stdout to the TUI via the event loop."""
    def __init__(self, log_widget: RichLog, loop: asyncio.AbstractEventLoop, tui_app):
        self.log_widget = log_widget
        self.loop = loop
        self.tui_app = tui_app
        self.last_write_time = None
        self.time = time

    def write(self, s: str):
        # This is the thread-safe way to call an async-context method from a sync context
        def write_and_maybe_scroll():
            self.log_widget.write(s)
            # Store content for copying (strip ANSI codes for clean text)
            import re
            clean_text = re.sub(r'\x1b\[[0-9;]*[mGK]', '', s)
            self.tui_app.log_content.append(clean_text)
            if self.tui_app.auto_scroll:
                self.log_widget.scroll_end()
            
            # Track write time to detect when agent stops responding
            self.last_write_time = self.time.time()
            # Schedule a check to reset placeholder if no more writes come in
            if hasattr(self.tui_app, 'reset_placeholder_timer'):
                try:
                    self.tui_app.reset_placeholder_timer.cancel()
                except:
                    pass
            self.tui_app.reset_placeholder_timer = self.tui_app.call_later(2.0, self.tui_app.reset_input_placeholder)
        
        self.loop.call_soon_threadsafe(write_and_maybe_scroll)

    def flush(self):
        pass

class AgentTUI(App):
    """A TUI that hosts the CLI-Agent and acts as its terminal."""

    CSS_PATH = "agent_tui.css"

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_log", "Clear Log"),
        ("ctrl+s", "toggle_scroll", "Toggle Auto-scroll"),
        ("ctrl+e", "export_log", "Export Log"),
        ("ctrl+x", "copy_to_clipboard", "Copy All to Clipboard")
    ]

    def __init__(self, args):
        super().__init__()
        self.args = args
        # --- FIX: Use thread-safe communication objects ---
        self.input_event = asyncio.Event()
        self.input_lock = threading.Lock()
        self.shared_input: Dict[str, Any] = {"value": None}
        self.agent_worker: Optional[Worker] = None
        self.auto_scroll = True
        self.log_content = []  # Store all log content for copying
        self.reset_placeholder_timer = None  # For tracking when to reset input placeholder
        self.visual_queue_worker: Optional[Worker] = None  # For monitoring visual input queue
        
        # Setup separate debug logging for TUI internal operations
        self.logger = logging.getLogger("AgentTUI")
        self.logger.setLevel(logging.DEBUG)

        # Create a file handler to write logs to a file
        file_handler = logging.FileHandler('tui_debug.log', mode='w')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)

        # CRITICAL: Prevent this logger's messages from bubbling up to the root logger,
        # which would cause them to be printed in the RichLog.
        self.logger.propagate = False
        
        # Initialize command suggester for auto-completion
        self.command_suggester = CommandSuggester(_build_command_definitions())
        self.logger.info("Command suggester initialized")
        
        self.logger.info("=== TUI Application Starting ===")

    def compose(self) -> ComposeResult:
        """Create and arrange widgets."""
        yield Static("┌─── CLI-Agent TUI ─────────────────────────────────────────────┐", id="title_bar")
        yield RichLog(id="output_log", highlight=True, markup=True, auto_scroll=True)
        yield Input(placeholder="Agent is initializing...", id="command_input", disabled=True)

    def on_mount(self) -> None:
        """Set up stdout/stderr redirection and start the agent logic."""
        self.logger.info("TUI on_mount started")
        
        log_widget = self.query_one(RichLog)
        input_widget = self.query_one(Input)
        main_loop = asyncio.get_running_loop()

        # Show welcome messages FIRST, before any redirects
        log_widget.write("[bold cyan]🎉 CLI-Agent TUI Started[/bold cyan]")
        log_widget.write("[green]✅ Interface initialized successfully[/green]")
        log_widget.write("[dim]Agent is starting up... Please wait...[/dim]")
        log_widget.write("")
        
        # Test that the display is working
        log_widget.write("[yellow]📱 TUI Display Test:[/yellow]")
        log_widget.write("   - Title bar: ✅")
        log_widget.write("   - Output area: ✅") 
        log_widget.write("   - Input field: ✅")
        log_widget.write("")

        self.logger.info(f"Main loop: {main_loop}")
        self.logger.info(f"Input event: {self.input_event}")
        self.logger.info(f"Input lock: {self.input_lock}")
        self.logger.info(f"Shared input: {self.shared_input}")

        # Attach command suggester to the input widget
        input_widget.suggester = self.command_suggester
        self.logger.info("Command suggester attached to input widget")
        
        # Show visual integration status
        from core.globals import g
        if hasattr(g, 'visual_input_queue'):
            log_widget.write("[dim green]🌐 Visual input integration: Ready[/dim green]")
        else:
            log_widget.write("[dim yellow]🌐 Visual input integration: Not active[/dim yellow]")
        
        log_widget.write("[dim]Redirecting agent output to TUI...[/dim]")

        # IMPORTANT: Redirect stdout/stderr AFTER showing initial messages
        sys.stdout = StdoutRedirector(log_widget, main_loop, self)
        sys.stderr = StdoutRedirector(log_widget, main_loop, self)
        self.logger.info("Stdout/stderr redirected to TUI")

        # Start the main agent logic in a background thread
        self.logger.info("Starting agent worker thread...")
        self.agent_worker = self.run_worker(
            run_agent_logic(input_event=self.input_event, input_lock=self.input_lock, shared_input=self.shared_input, tui_callback=self.tui_message),
            thread=True,
            name="agent_logic"
        )
        self.logger.info("Agent worker thread started")
        
        # Start visual queue monitoring
        self.logger.info("Starting visual queue monitor...")
        self.visual_queue_worker = self.run_worker(
            self.monitor_visual_queue(),
            thread=True,
            name="visual_queue_monitor"
        )
        
        # Enable input after initialization
        self.set_timer(3.0, self.enable_input_after_delay)
        
    def tui_message(self, message: str, msg_type: str = "info") -> None:
        """Callback for agent to send messages to TUI."""
        log_widget = self.query_one(RichLog)
        if msg_type == "system":
            log_widget.write(f"[dim blue]🤖 {message}[/dim blue]")
        elif msg_type == "system_ok":
            log_widget.write(f"[green]✅ {message}[/green]")
        elif msg_type == "error":
            log_widget.write(f"[red]❌ {message}[/red]")
        else:
            log_widget.write(f"[dim]{message}[/dim]")
        
    def enable_input_after_delay(self) -> None:
        """Enable input field after agent initialization delay."""
        self.logger.info("Enabling input field after delay")
        input_widget = self.query_one(Input)
        input_widget.disabled = False
        input_widget.placeholder = "Enter your request..."
        input_widget.focus()
        # Also reset the placeholder when the agent becomes ready for input
        if input_widget.placeholder == "Agent is thinking...":
            input_widget.placeholder = "Enter your request..."
        
    async def on_input_submitted(self, message: Input.Submitted) -> None:
        """Sends user input from the TUI to the agent logic."""
        command = message.value
        self.logger.info(f"Input submitted: '{command}'") # This now goes to the debug file
        
        # Use unified input handler for CLI input
        await self.handle_input(command, source="cli")

    def reset_input_placeholder(self) -> None:
        """Reset the input placeholder when the agent finishes responding."""
        try:
            input_widget = self.query_one(Input)
            if input_widget.placeholder == "Agent is thinking...":
                input_widget.placeholder = "Enter your request..."
                self.logger.info("Reset input placeholder after agent response")
        except Exception as e:
            self.logger.error(f"Error resetting placeholder: {e}")
            
    async def monitor_visual_queue(self) -> None:
        """Monitor g.visual_input_queue for browser inputs and forward them to the TUI."""
        from core.globals import g
        
        while True:
            try:
                # Check if visual input queue exists and has messages
                if hasattr(g, 'visual_input_queue') and not g.visual_input_queue.empty():
                    try:
                        visual_message = g.visual_input_queue.get_nowait()
                        self.logger.info(f"Received visual input: '{visual_message}'")
                        
                        # Process visual input same as CLI input
                        await self.handle_input(visual_message, source="web")
                        
                    except queue.Empty:
                        pass  # Queue became empty between check and get
                    except Exception as e:
                        self.logger.error(f"Error processing visual input: {e}")
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in visual queue monitor: {e}")
                await asyncio.sleep(1)  # Longer delay on error
    
    async def handle_input(self, message: str, source: str = "cli") -> None:
        """Handle input from any source (CLI or visual)."""
        log_widget = self.query_one(RichLog)
        input_widget = self.query_one(Input)
        
        # Show input with source indicator
        if source == "web":
            log_widget.write(f"\n[bold bright_green]🌐 Visual Input:[/bold bright_green] {message}")
            # Update input placeholder to show web connection
            input_widget.placeholder = "Enter your request... (Web UI connected)"
        else:
            log_widget.write(f"\n[bold bright_blue]>[/bold bright_blue] {message}")
        
        # Update visual state
        input_widget.clear()
        input_widget.placeholder = "Agent is thinking..."
        
        # Send to agent logic
        with self.input_lock:
            self.shared_input["value"] = message
        self.input_event.set()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker completion to update the UI."""
        if event.worker is self.agent_worker:
            self.logger.info(f"Worker state changed: {event.state}")
            input_widget = self.query_one(Input)
            log = self.query_one(RichLog)
            
            # Enable input field when worker is running (after initialization)
            if event.state == WorkerState.RUNNING:
                self.logger.info("Worker is running, enabling input field")
                input_widget.disabled = False
                input_widget.placeholder = "Enter your request..."
                input_widget.focus()

            if event.state == WorkerState.ERROR:
                tb = traceback.format_exception(event.worker.error)
                log.write("\n\n[bold red]AGENT WORKER CRASHED[/bold red]\n")
                log.write("".join(tb))
                self.logger.error(f"Worker crashed: {event.worker.error}")
            elif event.state == WorkerState.SUCCESS:
                 log.write("\n[bold magenta]Agent worker finished.[/bold magenta]\n")
                 self.logger.info("Worker finished successfully")

    def action_clear_log(self) -> None:
        self.query_one(RichLog).clear()

    def action_toggle_scroll(self) -> None:
        """Toggle auto-scroll on/off."""
        self.auto_scroll = not self.auto_scroll
        log_widget = self.query_one(RichLog)
        status = "ON" if self.auto_scroll else "OFF"
        self.notify(f"Auto-scroll: {status}", timeout=2)
        if self.auto_scroll:
            log_widget.scroll_end()

    def action_export_log(self) -> None:
        """Export all log content to a file."""
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tui_export_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("".join(self.log_content))
            
            full_path = os.path.abspath(filename)
            self.notify(f"Log exported to: {full_path}", timeout=10)
        except Exception as e:
            self.notify(f"Export failed: {str(e)}", timeout=5)

    def action_copy_to_clipboard(self) -> None:
        """Copy all log content to clipboard."""
        try:
            import pyperclip
            content = "".join(self.log_content)
            pyperclip.copy(content)
            lines = len([line for line in content.split('\n') if line.strip()])
            self.notify(f"Copied {lines} lines to clipboard", timeout=3)
        except ImportError:
            # Fallback to writing to a temp file and showing instructions
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write("".join(self.log_content))
                temp_path = f.name
            
            self.notify(f"pyperclip not available. Content saved to: {temp_path}", timeout=10)
        except Exception as e:
            self.notify(f"Copy failed: {str(e)}", timeout=5)

if __name__ == "__main__":
    cli_args = parse_cli_args()
    setup_logging(cli_args.debug, cli_args.log_file, tui_mode=True)
    app = AgentTUI(args=cli_args)
    app.run()