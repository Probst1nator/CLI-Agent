#!/usr/bin/env python3
"""Quick TUI display test"""

from textual.app import App, ComposeResult
from textual.widgets import RichLog, Input, Static

class SimpleTestTUI(App):
    """Minimal TUI to test display."""
    
    CSS_PATH = "agent_tui.css"
    
    def compose(self) -> ComposeResult:
        yield Static("┌─── CLI-Agent TUI Test ────────────────────────────────────────┐", id="title_bar")
        yield RichLog(id="output_log", highlight=True, markup=True, auto_scroll=True)
        yield Input(placeholder="Type something here...", id="command_input")

    def on_mount(self) -> None:
        log = self.query_one(RichLog)
        log.write("[bold cyan]🎉 TUI Display Test[/bold cyan]")
        log.write("[green]✅ Title bar visible[/green]")
        log.write("[green]✅ Output area visible[/green]")  
        log.write("[green]✅ Input field visible[/green]")
        log.write("[yellow]📱 Try typing in the input field below![/yellow]")
        log.write("")
        log.write("[dim]This test confirms the TUI layout is working.[/dim]")
        
        # Enable input immediately
        input_widget = self.query_one(Input)
        input_widget.disabled = False
        input_widget.focus()

    async def on_input_submitted(self, message: Input.Submitted) -> None:
        log = self.query_one(RichLog)
        log.write(f"[bold blue]>[/bold blue] {message.value}")
        log.write("[dim green]Input received! TUI is working correctly.[/dim green]")
        
        input_widget = self.query_one(Input)
        input_widget.clear()

if __name__ == "__main__":
    app = SimpleTestTUI()
    app.run()