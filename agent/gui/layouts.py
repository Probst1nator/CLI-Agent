"""
GUI Layouts for CLI-Agent

Layout definitions and management for the prompt_toolkit-based GUI.
Provides structured layouts for chat, input, and status areas.
"""

from collections.abc import Callable
from typing import Optional

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.widgets import Frame

from .widgets import ChatDisplay, ConsoleLogWidget, InputArea, StatusPanel


class CLIAgentGUILayout:
    """Main layout manager for CLI-Agent GUI."""

    def __init__(self):
        self.input_area: Optional[InputArea] = None
        self.input_handler: Optional[Callable] = None
        self.console_log: Optional[ConsoleLogWidget] = None
        self.sidebar_visible: bool = False
        self._main_container: Optional[VSplit] = None

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
            ' Ctrl+C: Exit | Ctrl+L: Clear | F1: Help | F2: Toggle Console | Enter: Send | Shift+Enter: New Line '
            '</style>'
        )

        return Window(
            content=FormattedTextControl(text=footer_text),
            height=Dimension.exact(1)
        )

    def create_main_area(self, chat_display: ChatDisplay, status_panel: StatusPanel, console_log: ConsoleLogWidget) -> VSplit:
        """Create the main content area with chat, status, and optional console sidebar."""

        # Store reference to console log
        self.console_log = console_log

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

        # Create console sidebar frame
        console_frame = Frame(
            body=console_log.get_widget(),
            title="Console Log (F2 to toggle)"
        )

        # Build the main container based on sidebar visibility
        self._main_container = self._build_main_container(chat_frame, status_frame, console_frame)
        return self._main_container

    def _build_main_container(self, chat_frame: Frame, status_frame: Frame, console_frame: Frame) -> VSplit:
        """Build the main container with or without the console sidebar."""
        if self.sidebar_visible:
            # Layout: [Chat | Status | Console Sidebar]
            return VSplit([
                chat_frame,          # Chat takes remaining space
                status_frame,        # Status has fixed width
                console_frame,       # Console sidebar
            ])
        else:
            # Layout: [Chat | Status]
            return VSplit([
                chat_frame,          # Chat takes remaining space
                status_frame,        # Status has fixed width
            ])

    def toggle_sidebar(self):
        """Toggle the console sidebar visibility."""
        self.sidebar_visible = not self.sidebar_visible
        return self.sidebar_visible

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

    def create_layout(self, chat_display: ChatDisplay, status_panel: StatusPanel, console_log: ConsoleLogWidget) -> HSplit:
        """Create the complete layout structure."""

        # Create all components
        header = self.create_header()
        main_area = self.create_main_area(chat_display, status_panel, console_log)
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






