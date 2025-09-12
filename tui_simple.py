#!/usr/bin/env python3
"""
Simple, safe TUI for CLI-Agent that doesn't break terminal output.
"""

import asyncio
import threading
import queue
from typing import Dict, Any

from textual.app import App, ComposeResult
from textual.widgets import RichLog, Input, Static

class SafeAgentTUI(App):
    """A safe TUI that monitors visual input without breaking anything."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_event = asyncio.Event()
        self.input_lock = threading.Lock()
        self.shared_input: Dict[str, Any] = {"value": None}

    def compose(self) -> ComposeResult:
        yield Static("┌─── CLI-Agent TUI (Safe Mode) ─────────────────────────────────┐")
        yield RichLog(id="output_log", highlight=True, markup=True)
        yield Input(placeholder="Type your message here...", id="command_input")

    def on_mount(self) -> None:
        log = self.query_one(RichLog)
        log.write("[bold green]✅ Safe TUI Mode Active[/bold green]")
        log.write("[dim]This TUI shows input/output without breaking your terminal.[/dim]")
        log.write("")
        
        # Start monitoring visual queue
        self.run_worker(self.monitor_visual_queue(), name="visual_monitor")
        
        # Enable input immediately
        input_widget = self.query_one(Input)
        input_widget.focus()

    async def monitor_visual_queue(self) -> None:
        """Monitor the visual input queue safely."""
        try:
            from core.globals import g
            log = self.query_one(RichLog)
            
            log.write("[yellow]🔍 Monitoring for visual input...[/yellow]")
            
            while True:
                try:
                    if hasattr(g, 'visual_input_queue') and not g.visual_input_queue.empty():
                        message = g.visual_input_queue.get_nowait()
                        log.write(f"[bold green]🌐 Visual Input Received:[/bold green] {message}")
                        
                        # Forward to main input handler
                        with self.input_lock:
                            self.shared_input["value"] = message
                        self.input_event.set()
                        
                except queue.Empty:
                    pass
                except Exception as e:
                    log.write(f"[red]Error monitoring visual queue: {e}[/red]")
                
                await asyncio.sleep(0.5)  # Check every 500ms
                
        except Exception as e:
            log = self.query_one(RichLog)
            log.write(f"[red]Visual monitor error: {e}[/red]")

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        log = self.query_one(RichLog)
        log.write(f"[bold blue]> CLI Input:[/bold blue] {message.value}")
        
        # Forward to main input handler  
        with self.input_lock:
            self.shared_input["value"] = message.value
        self.input_event.set()
        
        # Clear input
        input_widget = self.query_one(Input)
        input_widget.clear()

if __name__ == "__main__":
    print("Starting Safe TUI...")
    app = SafeAgentTUI()
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nTUI closed safely.")
    except Exception as e:
        print(f"TUI error: {e}")
        print("Your terminal should still work normally.")