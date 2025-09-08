# tui_main.py
import asyncio
import sys
import traceback
import threading
import logging
from typing import Optional, Dict, Any

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, RichLog, Input
from textual.worker import Worker, WorkerState

# --- Project Imports ---
from tools.main_cli_agent.main import main as run_agent_logic, parse_cli_args, setup_logging

class StdoutRedirector:
    """A thread-safe object that redirects stdout to the TUI via the event loop."""
    def __init__(self, log_widget: RichLog, loop: asyncio.AbstractEventLoop, tui_app):
        self.log_widget = log_widget
        self.loop = loop
        self.tui_app = tui_app
        self.last_write_time = None
        import time
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
        
        self.logger.info("=== TUI Application Starting ===")

    def compose(self) -> ComposeResult:
        """Create and arrange widgets."""
        yield Header(name="CLI-Agent TUI")
        yield RichLog(id="output_log", highlight=True, markup=True, auto_scroll=False)
        yield Footer()
        yield Input(placeholder="Agent is initializing...", id="command_input", disabled=True)

    def on_mount(self) -> None:
        """Set up stdout/stderr redirection and start the agent logic."""
        self.logger.info("TUI on_mount started")
        
        log_widget = self.query_one(RichLog)
        main_loop = asyncio.get_running_loop()

        self.logger.info(f"Main loop: {main_loop}")
        self.logger.info(f"Input event: {self.input_event}")
        self.logger.info(f"Input lock: {self.input_lock}")
        self.logger.info(f"Shared input: {self.shared_input}")

        # IMPORTANT: Redirect stdout/stderr BEFORE starting agent worker
        # so the agent's output goes to the TUI from the very beginning
        sys.stdout = StdoutRedirector(log_widget, main_loop, self)
        sys.stderr = StdoutRedirector(log_widget, main_loop, self)
        self.logger.info("Stdout/stderr redirected to TUI")

        # Start the main agent logic in a background thread
        self.logger.info("Starting agent worker thread...")
        self.agent_worker = self.run_worker(
            run_agent_logic(input_event=self.input_event, input_lock=self.input_lock, shared_input=self.shared_input),
            thread=True,
            name="agent_logic"
        )
        self.logger.info("Agent worker thread started")
        
        # Enable input after a reasonable initialization time
        self.set_timer(10.0, self.enable_input_after_delay)
        
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
        
        input_widget = self.query_one(Input)
        log_widget = self.query_one(RichLog)
        
        # 1. Echo the user's command into the log for clarity
        log_widget.write(f"\n[bold bright_blue]>[/bold bright_blue] {command}")
        
        # 2. Visually confirm the agent is working
        input_widget.clear()
        input_widget.placeholder = "Agent is thinking..."
        
        # 3. Send the command to the agent logic (this part is now silent in the UI)
        with self.input_lock:
            self.shared_input["value"] = command
        self.input_event.set()

    def reset_input_placeholder(self) -> None:
        """Reset the input placeholder when the agent finishes responding."""
        try:
            input_widget = self.query_one(Input)
            if input_widget.placeholder == "Agent is thinking...":
                input_widget.placeholder = "Enter your request..."
                self.logger.info("Reset input placeholder after agent response")
        except Exception as e:
            self.logger.error(f"Error resetting placeholder: {e}")

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