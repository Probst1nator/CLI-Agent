"""
Unified Input Management System for CLI-Agent.

This module provides a clean abstraction layer for all input sources
(CLI, GUI, Web, Voice) and standardizes input event handling across
the entire application.
"""

from __future__ import annotations
import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from agent.core.app_context import AppContext


class InputSource(Enum):
    """Enumeration of input sources."""
    CLI = "cli"
    GUI = "gui"
    WEB = "web"
    VOICE = "voice"
    QUEUE = "queue"  # For programmatic/queued inputs


class InputEventType(Enum):
    """Types of input events."""
    TEXT = "text"
    COMMAND = "command"
    IMAGE = "image"
    AUDIO = "audio"
    INTERRUPT = "interrupt"
    SYSTEM = "system"


@dataclass
class InputEvent:
    """Represents a unified input event from any source."""
    event_type: InputEventType
    source: InputSource
    content: str
    metadata: Dict[str, Any]
    timestamp: float
    images: List[str] = field(default_factory=list)  # Base64 encoded images


@dataclass
class InputResponse:
    """Response from input processing."""
    success: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    should_continue: bool = True


class InputSource_ABC(ABC):
    """Abstract base class for input sources."""

    def __init__(self, source_type: InputSource, context: AppContext):
        self.source_type = source_type
        self.context = context
        self.logger = logging.getLogger(f"input.{source_type.value}")
        self._enabled = True

    @abstractmethod
    async def get_input(
        self,
        prompt: str = "",
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> InputEvent:
        """Get input from this source."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources for this input source."""
        pass

    def is_enabled(self) -> bool:
        """Check if this input source is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable this input source."""
        self._enabled = True

    def disable(self) -> None:
        """Disable this input source."""
        self._enabled = False


class InputProcessor(ABC):
    """Abstract base class for input processors."""

    @abstractmethod
    async def process(self, event: InputEvent) -> InputResponse:
        """Process an input event."""
        pass

    @abstractmethod
    def can_handle(self, event: InputEvent) -> bool:
        """Check if this processor can handle the event."""
        pass


class InputManager:
    """
    Unified input manager that coordinates all input sources and processing.

    This class provides a single interface for managing input from multiple
    sources (CLI, GUI, Web, Voice) and routing events to appropriate processors.
    """

    def __init__(self, context: AppContext):
        self.context = context
        self.logger = logging.getLogger("input_manager")

        # Input sources registry
        self._sources: Dict[InputSource, InputSource_ABC] = {}

        # Input processors registry
        self._processors: List[InputProcessor] = []

        # Event handling
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._shutdown_event: asyncio.Event = asyncio.Event()

        # Thread safety
        self._lock = threading.Lock()

        # Configuration
        self._default_source = InputSource.CLI
        self._fallback_sources = [InputSource.CLI]

        self.logger.debug("InputManager initialized")

    def register_source(self, source: InputSource_ABC) -> None:
        """Register an input source."""
        with self._lock:
            self._sources[source.source_type] = source
            self.logger.debug(f"Registered input source: {source.source_type.value}")

    def unregister_source(self, source_type: InputSource) -> None:
        """Unregister an input source."""
        with self._lock:
            if source_type in self._sources:
                del self._sources[source_type]
                self.logger.info(f"Unregistered input source: {source_type.value}")

    def register_processor(self, processor: InputProcessor) -> None:
        """Register an input processor."""
        self._processors.append(processor)
        self.logger.debug(f"Registered input processor: {processor.__class__.__name__}")

    def set_default_source(self, source_type: InputSource) -> None:
        """Set the default input source."""
        self._default_source = source_type
        self.logger.debug(f"Default input source set to: {source_type.value}")

    def set_fallback_sources(self, sources: List[InputSource]) -> None:
        """Set fallback input sources in order of preference."""
        self._fallback_sources = sources
        self.logger.info(f"Fallback sources set to: {[s.value for s in sources]}")

    async def get_input(
        self,
        prompt: str = "",
        source: Optional[InputSource] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> InputEvent:
        """
        Get input from the specified source or default source.

        Args:
            prompt: Prompt to display to user
            source: Preferred input source, uses default if None
            timeout: Timeout in seconds
            **kwargs: Additional arguments passed to input source

        Returns:
            InputEvent: The input event received

        Raises:
            ValueError: If no input sources are available
            asyncio.TimeoutError: If timeout is exceeded
        """
        # Determine source to use
        target_source = source or self._default_source

        # PRIORITY: Check if queue source has messages first
        if InputSource.QUEUE in self._sources and self._sources[InputSource.QUEUE].has_messages():
            sources_to_try = [InputSource.QUEUE]
        else:
            # Try target source first
            sources_to_try = [target_source]

            # Add fallback sources if target source fails
            for fallback in self._fallback_sources:
                if fallback != target_source:
                    sources_to_try.append(fallback)

        last_error = None

        for source_type in sources_to_try:
            if source_type not in self._sources:
                continue

            source_obj = self._sources[source_type]

            if not source_obj.is_enabled():
                continue

            try:
                self.logger.debug(f"Attempting input from: {source_type.value}")
                event = await source_obj.get_input(prompt, timeout, **kwargs)
                self.logger.debug(f"Received input from: {source_type.value}")
                return event

            except Exception as e:
                last_error = e
                self.logger.warning(f"Input source {source_type.value} failed: {e}")
                continue

        # No sources worked
        if last_error:
            raise last_error
        else:
            raise ValueError("No available input sources")

    async def process_event(self, event: InputEvent) -> InputResponse:
        """
        Process an input event through registered processors.

        Args:
            event: The input event to process

        Returns:
            InputResponse: The processing result
        """
        self.logger.debug(f"Processing event: {event.event_type.value} from {event.source.value}")

        for processor in self._processors:
            if processor.can_handle(event):
                try:
                    response = await processor.process(event)
                    self.logger.debug(f"Event processed by: {processor.__class__.__name__}")
                    return response
                except Exception as e:
                    self.logger.error(f"Processor {processor.__class__.__name__} failed: {e}")
                    continue

        # No processor could handle the event
        return InputResponse(
            success=False,
            message=f"No processor available for event type: {event.event_type.value}"
        )

    async def start_event_loop(self) -> None:
        """Start the input event processing loop."""
        if self._processing_task is not None:
            self.logger.warning("Event loop already running")
            return

        self._processing_task = asyncio.create_task(self._event_processing_loop())
        self.logger.info("Input event loop started")

    async def stop_event_loop(self) -> None:
        """Stop the input event processing loop."""
        if self._processing_task is None:
            return

        self._shutdown_event.set()
        await self._processing_task
        self._processing_task = None
        self.logger.info("Input event loop stopped")

    async def _event_processing_loop(self) -> None:
        """Main event processing loop."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for event or shutdown signal
                event_task = asyncio.create_task(self._event_queue.get())
                shutdown_task = asyncio.create_task(self._shutdown_event.wait())

                done, pending = await asyncio.wait(
                    [event_task, shutdown_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()

                if shutdown_task in done:
                    break

                if event_task in done:
                    event = event_task.result()
                    await self.process_event(event)

            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(0.1)  # Prevent tight error loop

    async def queue_event(self, event: InputEvent) -> None:
        """Queue an event for processing."""
        await self._event_queue.put(event)

    async def cleanup(self) -> None:
        """Clean up the input manager and all sources."""
        self.logger.info("Cleaning up InputManager")

        # Stop event loop
        await self.stop_event_loop()

        # Clean up all sources
        for source in self._sources.values():
            try:
                await source.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up source {source.source_type.value}: {e}")

        self._sources.clear()
        self._processors.clear()

        self.logger.info("InputManager cleanup completed")

    def get_available_sources(self) -> List[InputSource]:
        """Get list of available input sources."""
        return [source_type for source_type, source in self._sources.items() if source.is_enabled()]

    def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all registered sources."""
        status = {}
        for source_type, source in self._sources.items():
            status[source_type.value] = {
                "enabled": source.is_enabled(),
                "type": source.__class__.__name__
            }
        return status