"""
Standardized GUI event handling and session management for CLI-Agent.

This module provides a unified event system for all GUI interfaces,
ensuring consistent behavior and proper session management.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agent.core.app_context import AppContext


class GUIEventType(Enum):
    """Types of GUI events."""
    INPUT = "input"
    COMMAND = "command"
    KEYPRESS = "keypress"
    MOUSE_CLICK = "mouse_click"
    FOCUS_CHANGE = "focus_change"
    RESIZE = "resize"
    CLOSE = "close"
    REFRESH = "refresh"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


@dataclass
class GUIEvent:
    """Represents a GUI event."""
    event_type: GUIEventType
    source: str  # Widget or component that generated the event
    data: dict[str, Any]
    timestamp: datetime
    handled: bool = False

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()


class GUIEventHandler(ABC):
    """Abstract base class for GUI event handlers."""

    @abstractmethod
    async def handle_event(self, event: GUIEvent) -> bool:
        """
        Handle a GUI event.

        Args:
            event: The event to handle

        Returns:
            bool: True if the event was handled, False otherwise
        """
        pass

    @abstractmethod
    def can_handle(self, event: GUIEvent) -> bool:
        """Check if this handler can handle the event."""
        pass


class GUISession:
    """Manages a GUI session with state and lifecycle."""

    def __init__(self, session_id: str, context: AppContext):
        self.session_id = session_id
        self.context = context
        self.logger = logging.getLogger(f"gui_session.{session_id}")

        # Session state
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.is_active = True

        # Input handling
        self.input_event: Optional[asyncio.Event] = None
        self.input_lock: Optional[threading.Lock] = None
        self.shared_input: Optional[dict[str, Any]] = None

        # Event handling
        self.event_handlers: list[GUIEventHandler] = []
        self.event_queue: asyncio.Queue = asyncio.Queue()

        # Session data
        self.session_data: dict[str, Any] = {}

        self.logger.info(f"GUI session {session_id} created")

    def setup_input_coordination(self) -> tuple[asyncio.Event, threading.Lock, dict[str, Any]]:
        """Set up input coordination for threaded GUI operations."""
        self.input_event = asyncio.Event()
        self.input_lock = threading.Lock()
        self.shared_input = {"value": ""}

        return self.input_event, self.input_lock, self.shared_input

    async def submit_input(self, input_value: str) -> None:
        """Submit input from GUI to the input coordination system."""
        if self.input_lock and self.shared_input and self.input_event:
            with self.input_lock:
                self.shared_input["value"] = input_value

            self.input_event.set()
            self.update_activity()
        else:
            self.logger.warning("Input coordination not set up")

    async def queue_event(self, event: GUIEvent) -> None:
        """Queue an event for processing."""
        await self.event_queue.put(event)
        self.update_activity()

    async def process_events(self) -> None:
        """Process queued events."""
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                await self._handle_event(event)
            except asyncio.QueueEmpty:
                break

    async def _handle_event(self, event: GUIEvent) -> None:
        """Handle a single event through registered handlers."""
        for handler in self.event_handlers:
            if handler.can_handle(event):
                try:
                    handled = await handler.handle_event(event)
                    if handled:
                        event.handled = True
                        break
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")

        if not event.handled:
            self.logger.warning(f"Unhandled event: {event.event_type.value}")

    def register_event_handler(self, handler: GUIEventHandler) -> None:
        """Register an event handler."""
        self.event_handlers.append(handler)

    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()

    def set_session_data(self, key: str, value: Any) -> None:
        """Set session-specific data."""
        self.session_data[key] = value

    def get_session_data(self, key: str, default: Any = None) -> Any:
        """Get session-specific data."""
        return self.session_data.get(key, default)

    def close_session(self) -> None:
        """Close the GUI session."""
        self.is_active = False
        self.logger.info(f"GUI session {self.session_id} closed")

    def get_session_info(self) -> dict[str, Any]:
        """Get information about the session."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active,
            "event_handlers": len(self.event_handlers),
            "session_data_keys": list(self.session_data.keys())
        }


class GUISessionManager:
    """Manages multiple GUI sessions."""

    def __init__(self, context: AppContext):
        self.context = context
        self.logger = logging.getLogger("gui_session_manager")
        self.sessions: dict[str, GUISession] = {}
        self._session_counter = 0

    def create_session(self, session_id: Optional[str] = None) -> GUISession:
        """Create a new GUI session."""
        if session_id is None:
            self._session_counter += 1
            session_id = f"gui_session_{self._session_counter}"

        if session_id in self.sessions:
            self.logger.warning(f"Session {session_id} already exists")
            return self.sessions[session_id]

        session = GUISession(session_id, self.context)
        self.sessions[session_id] = session

        self.logger.info(f"Created session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[GUISession]:
        """Get an existing session."""
        return self.sessions.get(session_id)

    def close_session(self, session_id: str) -> bool:
        """Close and remove a session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.close_session()
            del self.sessions[session_id]
            self.logger.info(f"Closed session: {session_id}")
            return True
        return False

    def cleanup_inactive_sessions(self, max_idle_seconds: int = 3600) -> int:
        """Clean up inactive sessions."""
        now = datetime.now()
        inactive_sessions = []

        for session_id, session in self.sessions.items():
            idle_time = (now - session.last_activity).total_seconds()
            if idle_time > max_idle_seconds or not session.is_active:
                inactive_sessions.append(session_id)

        cleaned_count = 0
        for session_id in inactive_sessions:
            if self.close_session(session_id):
                cleaned_count += 1

        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} inactive sessions")

        return cleaned_count

    def get_all_sessions(self) -> dict[str, dict[str, Any]]:
        """Get information about all sessions."""
        return {
            session_id: session.get_session_info()
            for session_id, session in self.sessions.items()
        }

    def get_active_session_count(self) -> int:
        """Get the number of active sessions."""
        return sum(1 for session in self.sessions.values() if session.is_active)


class StandardGUIEventHandlers:
    """Collection of standard GUI event handlers."""

    class InputEventHandler(GUIEventHandler):
        """Handles input events from GUI components."""

        def __init__(self, session: GUISession):
            self.session = session

        def can_handle(self, event: GUIEvent) -> bool:
            return event.event_type == GUIEventType.INPUT

        async def handle_event(self, event: GUIEvent) -> bool:
            input_text = event.data.get("text", "")
            await self.session.submit_input(input_text)
            return True

    class SessionEventHandler(GUIEventHandler):
        """Handles session lifecycle events."""

        def __init__(self, session_manager: GUISessionManager):
            self.session_manager = session_manager

        def can_handle(self, event: GUIEvent) -> bool:
            return event.event_type in [GUIEventType.SESSION_START, GUIEventType.SESSION_END]

        async def handle_event(self, event: GUIEvent) -> bool:
            if event.event_type == GUIEventType.SESSION_END:
                session_id = event.data.get("session_id")
                if session_id:
                    self.session_manager.close_session(session_id)
            return True

    class KeypressEventHandler(GUIEventHandler):
        """Handles keypress events."""

        def __init__(self, key_bindings: dict[str, Callable]):
            self.key_bindings = key_bindings

        def can_handle(self, event: GUIEvent) -> bool:
            return event.event_type == GUIEventType.KEYPRESS

        async def handle_event(self, event: GUIEvent) -> bool:
            key = event.data.get("key")
            if key in self.key_bindings:
                try:
                    result = self.key_bindings[key]()
                    if asyncio.iscoroutine(result):
                        await result
                    return True
                except Exception as e:
                    logging.error(f"Error in key binding for {key}: {e}")
            return False


def create_standard_gui_session(context: AppContext, session_id: Optional[str] = None) -> GUISession:
    """Create a GUI session with standard event handlers."""
    manager = GUISessionManager(context)
    session = manager.create_session(session_id)

    # Register standard event handlers
    session.register_event_handler(StandardGUIEventHandlers.InputEventHandler(session))
    session.register_event_handler(StandardGUIEventHandlers.SessionEventHandler(manager))

    return session


# Global session manager instance
_global_session_manager: Optional[GUISessionManager] = None


def get_global_session_manager(context: AppContext) -> GUISessionManager:
    """Get the global GUI session manager."""
    global _global_session_manager
    if _global_session_manager is None:
        _global_session_manager = GUISessionManager(context)
    return _global_session_manager