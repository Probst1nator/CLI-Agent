"""
GUI Layouts for CLI-Agent

Layout definitions and management for the prompt_toolkit-based GUI.
Provides structured layouts for chat, input, and status areas.
"""

from typing import Callable, Optional
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import Frame
from prompt_toolkit.formatted_text import HTML

from .widgets import ChatDisplay, StatusPanel, InputArea


class CLIAgentGUILayout:
    """Main layout manager for CLI-Agent GUI."""
    
    def __init__(self):
        self.input_area: Optional[InputArea] = None
        self.input_handler: Optional[Callable] = None
        
    def create_header(self) -> Window:
        """Create the header area."""
        header_text = HTML(
            '<style fg="white" bg="blue"><b>  CLI-Agent Enhanced GUI - prompt_toolkit Interface  </b></style>'
        )
        
        return Window(
            content=FormattedTextControl(text=header_text),
            height=Dimension.exact(1),
            style='class:title'
        )
        
    def create_footer(self) -> Window:
        """Create the footer area with help text."""
        footer_text = HTML(
            '<style fg="white" bg="ansiblack">'
            ' Ctrl+C: Exit | Ctrl+L: Clear | F1: Help | Enter: Send | Shift+Enter: New Line '
            '</style>'
        )
        
        return Window(
            content=FormattedTextControl(text=footer_text),
            height=Dimension.exact(1)
        )
        
    def create_main_area(self, chat_display: ChatDisplay, status_panel: StatusPanel) -> VSplit:
        """Create the main content area with chat and status."""
        
        # Create framed chat area
        chat_frame = Frame(
            body=chat_display.get_widget(),
            title="Conversation"
        )
        
        # Create framed status area
        status_frame = Frame(
            body=status_panel.get_widget(),
            title="Status"
        )
        
        # Create horizontal split with chat taking most space
        return VSplit([
            chat_frame,  # Chat takes remaining space
            status_frame,  # Status has fixed width (defined in widget)
        ])
        
    def create_input_area(self, input_handler: Optional[Callable] = None) -> Frame:
        """Create the input area."""
        self.input_area = InputArea(on_submit=input_handler)
        self.input_handler = input_handler
        
        return Frame(
            body=self.input_area.get_widget(),
            title="Input (Enter to send, Shift+Enter for new line)"
        )
        
    def setup_input_handler(self, handler: Callable):
        """Setup the input handler after layout creation."""
        self.input_handler = handler
        if self.input_area:
            self.input_area.set_submit_handler(handler)
        
    def create_layout(self, chat_display: ChatDisplay, status_panel: StatusPanel) -> HSplit:
        """Create the complete layout structure."""
        
        # Create all components
        header = self.create_header()
        main_area = self.create_main_area(chat_display, status_panel)
        input_area = self.create_input_area()
        footer = self.create_footer()
        
        # Create vertical layout
        layout = HSplit([
            header,
            main_area,  # Takes most of the space
            input_area,
            footer,
        ])
        
        return layout
        
    def focus_input(self):
        """Focus the input area."""
        if self.input_area:
            self.input_area.focus()






