"""
GUI Widgets for CLI-Agent

Custom widgets for chat display, status panels, and other GUI components
using prompt_toolkit infrastructure with standardized event handling.
"""

import asyncio
from typing import List, Optional, Callable
from datetime import datetime
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

# Import standardized event system
from agent.gui.event_manager import GUIEvent, GUIEventType, GUISession


class ChatMessage:
    """Represents a single chat message."""
    
    def __init__(self, sender: str, content: str, msg_type: str = "user", timestamp: Optional[datetime] = None):
        self.sender = sender
        self.content = content
        self.msg_type = msg_type  # user, assistant, system, error
        self.timestamp = timestamp or datetime.now()
        
    def format_for_display(self) -> FormattedText:
        """Format message for display in the GUI."""
        timestamp_str = self.timestamp.strftime("%H:%M:%S")
        
        if self.msg_type == "user":
            return FormattedText([
                ('class:timestamp', f"[{timestamp_str}] "),
                ('class:user-message', f"{self.sender}: "),
                ('', self.content),
                ('', '\n')
            ])
        elif self.msg_type == "assistant":
            return FormattedText([
                ('class:timestamp', f"[{timestamp_str}] "),
                ('class:assistant-message', f"{self.sender}: "),
                ('', self.content),
                ('', '\n')
            ])
        elif self.msg_type == "system":
            return FormattedText([
                ('class:timestamp', f"[{timestamp_str}] "),
                ('class:system-message', f"{self.sender}: "),
                ('', self.content),
                ('', '\n')
            ])
        elif self.msg_type == "error":
            return FormattedText([
                ('class:timestamp', f"[{timestamp_str}] "),
                ('class:error-message', f"{self.sender}: "),
                ('', self.content),
                ('', '\n')
            ])
        else:
            return FormattedText([
                ('class:timestamp', f"[{timestamp_str}] "),
                ('', f"{self.sender}: {self.content}"),
                ('', '\n')
            ])


class ChatDisplay:
    """Chat display widget for showing conversation history with event support."""

    def __init__(self, session: Optional[GUISession] = None):
        self.session = session
        self.messages: List[ChatMessage] = []
        self.streaming_message: Optional[str] = None
        self.streaming_sender: Optional[str] = None
        self._control = FormattedTextControl(
            text=self._get_formatted_text,
            focusable=False,
            show_cursor=False
        )
        self._refresh_callbacks: List[Callable] = []
        
    def _get_formatted_text(self) -> FormattedText:
        """Get formatted text for all messages."""
        formatted = FormattedText([])
        
        # Add all complete messages
        for message in self.messages:
            formatted.extend(message.format_for_display())
            
        # Add streaming message if active
        if self.streaming_message is not None and self.streaming_sender:
            timestamp_str = datetime.now().strftime("%H:%M:%S")
            formatted.extend(FormattedText([
                ('class:timestamp', f"[{timestamp_str}] "),
                ('class:assistant-message', f"{self.streaming_sender}: "),
                ('', self.streaming_message),
                ('class:streaming-cursor', '▋'),
                ('', '\n')
            ]))
            
        return formatted
        
    def add_message(self, sender: str, content: str, msg_type: str = "user") -> None:
        """Add a complete message to the chat."""
        message = ChatMessage(sender, content, msg_type)
        self.messages.append(message)
        self._trigger_refresh_event()

    def start_streaming_message(self, sender: str) -> None:
        """Start a new streaming message."""
        self.streaming_sender = sender
        self.streaming_message = ""
        self._trigger_refresh_event()

    def update_streaming_message(self, chunk: str) -> None:
        """Update the streaming message with new content."""
        if self.streaming_message is not None:
            self.streaming_message += chunk
            self._trigger_refresh_event()

    def finish_streaming_message(self) -> None:
        """Finish the streaming message and add it to the chat."""
        if self.streaming_message is not None and self.streaming_sender:
            self.add_message(self.streaming_sender, self.streaming_message, "assistant")

        self.streaming_message = None
        self.streaming_sender = None
        self._trigger_refresh_event()

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.streaming_message = None
        self.streaming_sender = None
        self._trigger_refresh_event()

    def _trigger_refresh_event(self) -> None:
        """Trigger a refresh event for the chat display."""
        if self.session:
            event = GUIEvent(
                event_type=GUIEventType.REFRESH,
                source="chat_display",
                data={"widget": "chat_display", "message_count": len(self.messages)},
                timestamp=datetime.now()
            )
            asyncio.create_task(self.session.queue_event(event))

        # Also call any registered refresh callbacks
        for callback in self._refresh_callbacks:
            try:
                callback()
            except Exception as e:
                # Log error but don't break the refresh chain
                import logging
                logging.error(f"Error in refresh callback: {e}")

    def register_refresh_callback(self, callback: Callable) -> None:
        """Register a callback to be called when the display needs refreshing."""
        self._refresh_callbacks.append(callback)
        
    def get_widget(self) -> Window:
        """Get the prompt_toolkit widget for this chat display."""
        return Window(
            content=self._control,
            height=Dimension(min=3),
            wrap_lines=True
        )


class StatusPanel:
    """Status panel widget for showing system status and information."""
    
    def __init__(self):
        self.status_messages: List[str] = []
        self.max_messages = 50  # Keep last 50 status messages
        self._control = FormattedTextControl(
            text=self._get_formatted_text,
            focusable=False
        )
        
    def _get_formatted_text(self) -> FormattedText:
        """Get formatted text for status messages."""
        formatted = FormattedText([('class:title', 'Status Panel\n')])
        formatted.extend(FormattedText([('', '─' * 20 + '\n')]))
        
        # Show recent status messages
        for message in self.status_messages[-10:]:  # Show last 10
            formatted.extend(FormattedText([('', f"• {message}\n")]))
            
        return formatted
        
    def add_status(self, message: str) -> None:
        """Add a status message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        status_msg = f"[{timestamp}] {message}"
        
        self.status_messages.append(status_msg)
        
        # Keep only the most recent messages
        if len(self.status_messages) > self.max_messages:
            self.status_messages = self.status_messages[-self.max_messages:]
            
    def clear_status(self) -> None:
        """Clear all status messages."""
        self.status_messages.clear()
        
    def get_widget(self) -> Window:
        """Get the prompt_toolkit widget for this status panel."""
        return Window(
            content=self._control,
            width=Dimension(min=25, max=40),
            height=Dimension(min=3)
        )


class InputArea:
    """Input area widget for user message input."""
    
    def __init__(self, on_submit: Optional[Callable[[str], None]] = None):
        self.on_submit = on_submit
        self.text_area = TextArea(
            height=Dimension(min=3, max=8),
            prompt=">>> ",
            multiline=True,
            wrap_lines=True
        )
        self.buffer = self.text_area.buffer
        
        # Setup key bindings for the text area
        self._setup_key_bindings()
        
    def _setup_key_bindings(self):
        """Setup key bindings for input handling."""
        from prompt_toolkit.key_binding import KeyBindings
        
        kb = KeyBindings()
        
        @kb.add('enter')
        def submit_input(event):
            """Submit input on Enter (unless Shift+Enter for multiline)."""
            text = self.buffer.text.strip()
            if text and self.on_submit:
                # Submit the text
                asyncio.create_task(self._async_submit(text))
                # Clear the buffer
                self.buffer.reset()
                    
        @kb.add('c-j')  # Alternative for new line
        def new_line(event):
            """Add new line on Ctrl+J."""
            self.buffer.insert_text('\n')
            
        # Apply key bindings to the text area
        self.text_area.control.key_bindings = kb
            
    async def _async_submit(self, text: str):
        """Handle async submission of input."""
        if self.on_submit:
            if asyncio.iscoroutinefunction(self.on_submit):
                await self.on_submit(text)
            else:
                self.on_submit(text)
                
    def set_submit_handler(self, handler: Callable[[str], None]):
        """Set the submit handler for input."""
        self.on_submit = handler
        
    def get_widget(self) -> TextArea:
        """Get the prompt_toolkit widget for this input area."""
        return self.text_area
        
    def clear(self):
        """Clear the input area."""
        self.buffer.reset()
        
    def focus(self):
        """Focus the input area."""
        # This will be handled by the application layout
        pass