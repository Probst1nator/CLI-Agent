#!/usr/bin/env python3
"""
CLI-Agent GUI Entry Point - Enhanced prompt_toolkit GUI
Enhanced GUI using prompt_toolkit for conversational AI interface.

This provides a better alternative to the Textual TUI attempts by building
on the existing prompt_toolkit infrastructure already in use by CLI-Agent.
"""

import asyncio
import logging
import sys
import threading
import queue
import subprocess
import os
from pathlib import Path
from typing import Optional
import time

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import CLI-Agent core components
try:
    from core.globals import g
    from core.chat import Chat, Role
    from core.llm_router import LlmRouter
    from agent.core.command_system import _COMMAND_DEFINITIONS, _build_command_map
    from agent.gui.layouts import CLIAgentGUILayout
    from agent.gui.widgets import ChatDisplay, StatusPanel, ConsoleLogWidget

    # Build command map for command handling
    COMMAND_MAP = _build_command_map()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure to run this from the CLI-Agent root directory")
    sys.exit(1)

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.styles import Style


class CLIAgentGUI:
    """Enhanced CLI-Agent GUI using prompt_toolkit that streams main.py output."""

    def __init__(self):
        self.app: Optional[Application] = None
        self.layout = CLIAgentGUILayout()
        self.chat_display = ChatDisplay()
        self.status_panel = StatusPanel()
        self.console_log = ConsoleLogWidget()

        # CLI-Agent process and communication
        self.main_process: Optional[subprocess.Popen] = None
        self.input_queue = queue.Queue()

        # Monitoring flags
        self.running = False
        self.browser_monitor_active = False
        self.output_monitor_active = False

        # Setup initial display
        self.setup_initial_display()

    def setup_initial_display(self):
        """Setup initial display messages."""
        self.chat_display.add_message("🚀 CLI-Agent GUI",
            "Starting CLI-Agent process... Output will appear here.", "system")
        self.status_panel.add_status("GUI started")
        self.status_panel.add_status("Starting CLI-Agent process...")

    def create_key_bindings(self) -> KeyBindings:
        """Create key bindings for the GUI."""
        kb = KeyBindings()

        @kb.add('c-c')
        def exit_app(event):
            """Exit the application."""
            self.running = False
            event.app.exit()

        @kb.add('c-l')
        def clear_chat(event):
            """Clear the chat display."""
            self.chat_display.clear()
            self.status_panel.add_status("Chat cleared")

        @kb.add('f1')
        def show_help(event):
            """Show help information."""
            help_text = """
CLI-Agent GUI Help:
- Ctrl+C: Exit
- Ctrl+L: Clear chat
- F1: Show this help
- F2: Toggle console log sidebar
- Enter: Send message
- Up/Down: Navigate history
"""
            self.chat_display.add_message("📖 Help", help_text, "system")

        @kb.add('f2')
        def toggle_console_sidebar(event):
            """Toggle the console log sidebar."""
            sidebar_state = self.layout.toggle_sidebar()
            self.status_panel.add_status(f"Console sidebar: {'visible' if sidebar_state else 'hidden'}")

            # Rebuild the layout to reflect the change
            self._rebuild_layout()

            # Force refresh
            if self.app:
                self.app.invalidate()

        return kb

    def _rebuild_layout(self):
        """Rebuild the application layout (used when toggling sidebar)."""
        if self.app:
            # Create new layout
            new_layout = self.layout.create_layout(self.chat_display, self.status_panel, self.console_log)

            # Update the application layout
            self.app.layout = Layout(new_layout)

    def create_application(self) -> Application:
        """Create the prompt_toolkit application."""

        # Create layout
        layout = self.layout.create_layout(self.chat_display, self.status_panel, self.console_log)

        # Create style
        style = Style.from_dict({
            'chat-window': 'bg:#001122 #88ccff',
            'status-panel': 'bg:#1a1a1a #cccccc',
            'input-area': 'bg:#000044 #ffffff',
            'user-message': '#88ff88 bold',
            'assistant-message': '#88ccff bold',
            'system-message': '#ffaa44 bold',
            'error-message': '#ff8888 bold',
            'title': '#ffffff bold',
        })

        # Create application
        app = Application(
            layout=Layout(layout),
            key_bindings=self.create_key_bindings(),
            style=style,
            full_screen=True,
        )

        return app

    def start_main_process(self):
        """Start the main CLI-Agent process."""
        try:
            # Start main.py as subprocess with proper I/O handling
            self.main_process = subprocess.Popen(
                [sys.executable, "main.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr with stdout
                text=True,
                bufsize=0,  # Unbuffered for real-time output
                cwd=str(Path(__file__).parent),
                env=dict(os.environ, PYTHONUNBUFFERED="1")  # Force unbuffered output
            )

            self.status_panel.add_status("CLI-Agent process started")
            self.chat_display.add_message("✅ Process", f"CLI-Agent main.py started (PID: {self.main_process.pid})", "system")

            # Give process a moment to initialize
            time.sleep(0.5)

            # Check if process is still running
            if self.main_process.poll() is None:
                self.status_panel.add_status("✅ Process running")
                # Schedule an initial prompt to wake up CLI-Agent
                threading.Timer(3.0, self._send_initial_prompt).start()
            else:
                self.status_panel.add_status("❌ Process ended immediately")

        except Exception as e:
            error_msg = f"Failed to start CLI-Agent process: {e}"
            self.chat_display.add_message("❌ Error", error_msg, "error")
            self.status_panel.add_status("Process start failed")
            logging.error(error_msg)

    def _send_initial_prompt(self):
        """Send an initial empty line to wake up CLI-Agent's input loop."""
        try:
            if (self.main_process and
                self.main_process.stdin and
                self.main_process.poll() is None):

                # Send an empty line to trigger CLI-Agent's input prompt
                self.main_process.stdin.write('\n')
                self.main_process.stdin.flush()
                self.status_panel.add_status("🔧 Sent initial prompt wake-up")

        except Exception as e:
            self.status_panel.add_status(f"❌ Initial prompt failed: {e}")
            logging.error(f"Initial prompt error: {e}")

    def start_output_monitor(self):
        """Start monitoring CLI-Agent process output."""
        def monitor_output():
            """Monitor and display CLI-Agent output in real-time."""
            if not self.main_process:
                return

            self.output_monitor_active = True
            self.status_panel.add_status("Output monitor started")

            try:
                while self.running and self.output_monitor_active and self.main_process:
                    if self.main_process.poll() is not None:
                        # Process has ended
                        self.chat_display.add_message("⚠️ Process", "CLI-Agent process ended", "system")
                        break

                    # Read output line by line
                    line = self.main_process.stdout.readline()
                    if line:
                        # Clean the line and remove ANSI escape sequences if needed
                        clean_line = line.strip()
                        if clean_line:
                            # Add to console log (raw output)
                            self.console_log.add_log_line(clean_line)

                            # Add filtered output to chat display (skip some verbose messages)
                            should_show_in_chat = self._should_show_in_chat(clean_line)
                            if should_show_in_chat:
                                self.chat_display.add_message("📺 CLI-Agent", clean_line, "system")

                            # Force GUI refresh
                            if self.app:
                                self.app.invalidate()

                    time.sleep(0.01)  # Small delay to prevent overwhelming

            except Exception as e:
                self.chat_display.add_message("❌ Monitor", f"Output monitor error: {e}", "error")
                self.status_panel.add_status("Output monitor failed")
            finally:
                self.output_monitor_active = False
                self.status_panel.add_status("Output monitor stopped")

        # Start monitor thread
        monitor_thread = threading.Thread(target=monitor_output, daemon=True)
        monitor_thread.start()

    def _should_show_in_chat(self, line: str) -> bool:
        """Determine if a console line should be shown in the chat display."""
        line_lower = line.lower()

        # Skip very verbose debug lines
        skip_patterns = [
            "debug:",
            "reading",
            "writing",
            "loading",
            "initialized",
            "connecting to",
            "setting up",
            "[info]",
            "[debug]",
        ]

        for pattern in skip_patterns:
            if pattern in line_lower:
                return False

        # Show important messages
        show_patterns = [
            "error",
            "exception",
            "warning",
            "user:",
            "assistant:",
            "starting",
            "ready",
            "failed",
            "complete",
        ]

        for pattern in show_patterns:
            if pattern in line_lower:
                return True

        # Show short lines (likely important)
        if len(line) < 100:
            return True

        # Default: don't show in chat (but still in console log)
        return False

    def start_input_processor(self):
        """Start processing queued input to CLI-Agent."""
        def process_input():
            """Process input queue and send to CLI-Agent process."""
            self.status_panel.add_status("Input processor started")

            while self.running:
                try:
                    # Check if we have input and a valid process
                    if (not self.input_queue.empty() and
                        self.main_process and
                        self.main_process.stdin and
                        self.main_process.poll() is None):  # Process is still running

                        user_input = self.input_queue.get_nowait()

                        # Debug output
                        self.status_panel.add_status(f"Processing: {user_input[:20]}...")

                        # Send input to CLI-Agent process with proper encoding
                        try:
                            input_line = user_input + '\n'
                            self.main_process.stdin.write(input_line)
                            self.main_process.stdin.flush()

                        except BrokenPipeError:
                            self.status_panel.add_status("❌ Broken pipe - process may have ended")
                            break
                        except Exception as write_error:
                            self.status_panel.add_status(f"❌ Write error: {write_error}")
                            logging.error(f"Write error: {write_error}")

                except queue.Empty:
                    pass
                except Exception as e:
                    self.status_panel.add_status(f"❌ Input error: {e}")
                    logging.error(f"Input processing error: {e}")

                time.sleep(0.05)  # Faster processing

            self.status_panel.add_status("Input processor stopped")

        # Start input processor thread
        input_thread = threading.Thread(target=process_input, daemon=True)
        input_thread.start()

    async def handle_user_input(self, user_message: str) -> None:
        """Handle user input by queuing it for the CLI-Agent process."""
        if not user_message.strip():
            return

        try:
            # Display user input
            self.chat_display.add_message("👤 User Input", user_message, "user")

            # Debug: Check process status
            if self.main_process:
                if self.main_process.poll() is None:
                    self.status_panel.add_status("✅ Process alive, queuing input")
                else:
                    self.status_panel.add_status("❌ Process not running!")
                    return
            else:
                self.status_panel.add_status("❌ No process!")
                return

            # Queue input for CLI-Agent process
            self.input_queue.put(user_message)
            self.status_panel.add_status(f"Input queued (size: {self.input_queue.qsize()})")

        except Exception as e:
            error_msg = f"Error queuing input: {e}"
            self.chat_display.add_message("❌ Error", error_msg, "error")
            self.status_panel.add_status("Input queue error")
            logging.error(f"Input queue error: {e}")

    async def handle_command(self, command: str) -> None:
        """Handle CLI-Agent commands."""
        try:
            self.status_panel.add_status(f"Processing command: {command}")

            # Parse command
            parts = command.split()
            if not parts:
                return

            main_cmd = parts[0]
            args = parts[1:] if len(parts) > 1 else []

            # Check if command exists
            if main_cmd not in COMMAND_MAP:
                self.chat_display.add_message("❌ Command", f"Unknown command: {main_cmd}", "error")
                self.show_help()
                return

            canonical = COMMAND_MAP[main_cmd]

            # Handle specific commands
            if canonical == 'help':
                self.show_help()
            elif canonical == 'clear':
                self.chat_display.clear()
                self.chat = Chat(
                    instruction_message="You are Claude, a helpful AI assistant created by Anthropic.",
                    debug_title="GUI Chat"
                )
                self.chat_display.add_message("🤖 Assistant",
                    "Chat cleared. How can I help you?", "assistant")
                self.status_panel.add_status("Chat cleared")
            elif canonical == 'exit':
                self.status_panel.add_status("Exiting...")
                if self.app:
                    self.app.exit()
            elif canonical == 'screenshot':
                await self.handle_screenshot_command()
            elif canonical == 'regenerate':
                await self.handle_regenerate_command()
            else:
                self.chat_display.add_message("💡 Command",
                    f"Command '{canonical}' recognized but not implemented in GUI yet", "system")
                self.status_panel.add_status(f"Command {canonical} pending implementation")

        except Exception as e:
            error_msg = f"Command error: {e}"
            self.chat_display.add_message("❌ Command Error", error_msg, "error")
            self.status_panel.add_status("Command failed")
            logging.error(f"Command error: {e}")

    def show_help(self):
        """Show help information."""
        help_text = """Available Commands:

/help, /h - Show this help message
/clear - Clear chat history
/exit, /e - Exit the application
/screenshot, /img - Take a screenshot (coming soon)
/regenerate, /r - Regenerate last response (coming soon)

GUI Controls:
- Ctrl+C: Exit application
- Ctrl+L: Clear chat
- F1: Show help
- F2: Toggle console log sidebar (shows raw CLI-Agent output)
- Enter: Send message
- Shift+Enter: New line in input

Console Sidebar:
The console sidebar shows ALL raw output from the CLI-Agent process,
while the main chat area shows filtered important messages.
Use F2 to toggle the sidebar visibility for debugging and monitoring.

The GUI integrates with CLI-Agent's full command system.
More commands will be implemented in future updates.
"""
        self.chat_display.add_message("📖 Help", help_text, "system")
        self.status_panel.add_status("Help displayed")

    async def handle_screenshot_command(self):
        """Handle screenshot command."""
        self.chat_display.add_message("📷 Screenshot",
            "Screenshot functionality will be integrated in future updates.", "system")
        self.status_panel.add_status("Screenshot command called")

    async def handle_regenerate_command(self):
        """Handle regenerate command."""
        self.chat_display.add_message("🔄 Regenerate",
            "Regenerate functionality will be integrated in future updates.", "system")
        self.status_panel.add_status("Regenerate command called")

    def start_browser_monitor(self):
        """Start monitoring for browser input."""
        def monitor_browser():
            """Monitor browser input queue in background thread."""
            self.browser_monitor_active = True
            self.status_panel.add_status("Browser monitor started")

            while self.running and self.browser_monitor_active:
                try:
                    if (hasattr(g, 'visual_input_queue') and
                        not g.visual_input_queue.empty()):

                        # Get browser message
                        message = g.visual_input_queue.get_nowait()
                        self.status_panel.add_status(f"Browser input: {message[:30]}...")

                        # Process browser message async
                        if self.app:
                            asyncio.run_coroutine_threadsafe(
                                self.handle_browser_input(message),
                                asyncio.get_event_loop()
                            )

                except queue.Empty:
                    pass
                except Exception as e:
                    self.status_panel.add_status(f"Browser monitor error: {e}")
                    logging.error(f"Browser monitor error: {e}")

                time.sleep(0.2)  # Check every 200ms

            self.browser_monitor_active = False
            self.status_panel.add_status("Browser monitor stopped")

        # Start monitor thread
        monitor_thread = threading.Thread(target=monitor_browser, daemon=True)
        monitor_thread.start()

    async def handle_browser_input(self, message: str):
        """Handle input from browser interface."""
        # Mark as browser input and process
        browser_formatted = f"🌐 Browser: {message}"
        self.chat_display.add_message("🌐 Browser", message, "user")
        await self.handle_user_input(message)

    async def run_async(self):
        """Run the GUI asynchronously."""
        self.running = True

        # Create application
        self.app = self.create_application()

        # Start CLI-Agent process
        self.start_main_process()

        # Start monitoring and processing threads
        self.start_output_monitor()
        self.start_input_processor()
        self.start_browser_monitor()

        # Setup input handling
        self.layout.setup_input_handler(self.handle_user_input)

        # Initial status
        self.status_panel.add_status("All systems started")
        self.status_panel.add_status("Ready for input")

        # Run the application
        try:
            await self.app.run_async()
        finally:
            self.running = False
            # Clean up process
            if self.main_process:
                self.main_process.terminate()
                self.main_process.wait()

    def run(self):
        """Run the GUI."""
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            print("\n👋 CLI-Agent GUI closed.")
        except Exception as e:
            print(f"❌ GUI error: {e}")
            logging.error(f"GUI error: {e}")


def main():
    """Main entry point."""
    print("🚀 Starting CLI-Agent Enhanced GUI")
    print("Building on existing prompt_toolkit infrastructure...")
    print()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    gui = CLIAgentGUI()
    gui.run()


if __name__ == "__main__":
    main()