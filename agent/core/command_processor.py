"""
Command processing system for the unified input manager.

This module provides processors that integrate the new command system
with the unified input management framework.
"""

from __future__ import annotations
import asyncio
import time
from typing import Dict, Any, Optional, TYPE_CHECKING

from agent.core.input_manager import (
    InputProcessor, InputEvent, InputResponse, InputEventType, InputSource
)
from agent.commands.base_command import CommandResult
from agent.core.input_sources import QueueInputSource

if TYPE_CHECKING:
    from agent.core.app_context import AppContext
    from agent.commands.command_handler import CommandHandler
    import argparse


class CommandInputProcessor(InputProcessor):
    """Processor for command input events."""

    def __init__(self, context: AppContext, command_handler: CommandHandler):
        self.context = context
        self.command_handler = command_handler
        self.logger = context.logger

    def can_handle(self, event: InputEvent) -> bool:
        """Check if this processor can handle command events."""
        return event.event_type == InputEventType.COMMAND

    async def process(self, event: InputEvent) -> InputResponse:
        """Process a command input event."""
        try:
            self.logger.debug(f"Processing command: {event.content}")

            # Get args from context (this should be properly injected)
            args = getattr(self.context, 'args', None)
            if args is None:
                self.logger.warning("No args available in context")
                args = self._create_dummy_args()

            # Process the command
            response = await self.command_handler.handle_command(
                event.content,
                args,
                event=event
            )

            # Convert command response to input response
            return InputResponse(
                success=response.result != CommandResult.ERROR,
                message=response.message,
                data={
                    'command_result': response.result.value,
                    'should_regenerate': response.should_regenerate,
                    'images': response.base64_images
                },
                should_continue=response.result != CommandResult.EXIT
            )

        except Exception as e:
            self.logger.error(f"Error processing command: {e}")
            return InputResponse(
                success=False,
                message=f"Command processing failed: {str(e)}"
            )

    def _create_dummy_args(self) -> argparse.Namespace:
        """Create dummy args for command processing."""
        import argparse
        args = argparse.Namespace()

        # Set common defaults
        args.auto = False
        args.local = False
        args.voice = False
        args.online = True
        args.full_output_mode = False
        args.fast = False
        args.message = []
        args.private_remote_wake_detection = False
        args.online_voice = False

        return args


class TextInputProcessor(InputProcessor):
    """Processor for regular text input events."""

    def __init__(self, context: AppContext):
        self.context = context
        self.logger = context.logger

    def can_handle(self, event: InputEvent) -> bool:
        """Check if this processor can handle text events."""
        return event.event_type == InputEventType.TEXT

    async def process(self, event: InputEvent) -> InputResponse:
        """Process a text input event."""
        try:
            self.logger.debug(f"Processing text input: {event.content[:50]}...")

            # For text input, we typically just pass it through
            # The actual processing happens in the main agent loop
            return InputResponse(
                success=True,
                message="Text input processed",
                data={
                    'content': event.content,
                    'source': event.source.value,
                    'metadata': event.metadata
                }
            )

        except Exception as e:
            self.logger.error(f"Error processing text input: {e}")
            return InputResponse(
                success=False,
                message=f"Text processing failed: {str(e)}"
            )


class InterruptInputProcessor(InputProcessor):
    """Processor for interrupt events (Ctrl+C, etc.)."""

    def __init__(self, context: AppContext):
        self.context = context
        self.logger = context.logger

    def can_handle(self, event: InputEvent) -> bool:
        """Check if this processor can handle interrupt events."""
        return event.event_type == InputEventType.INTERRUPT

    async def process(self, event: InputEvent) -> InputResponse:
        """Process an interrupt event."""
        try:
            self.logger.info("Processing interrupt event")

            # Handle different types of interrupts
            reason = event.metadata.get('reason', 'unknown')

            if reason == 'keyboard_interrupt':
                self.logger.warning("User interrupt (Ctrl+C) detected")
                return InputResponse(
                    success=True,
                    message="User interrupt processed",
                    data={'reason': reason},
                    should_continue=False
                )

            return InputResponse(
                success=True,
                message=f"Interrupt processed: {reason}",
                data={'reason': reason}
            )

        except Exception as e:
            self.logger.error(f"Error processing interrupt: {e}")
            return InputResponse(
                success=False,
                message=f"Interrupt processing failed: {str(e)}"
            )


class ImageInputProcessor(InputProcessor):
    """Processor for image input events."""

    def __init__(self, context: AppContext):
        self.context = context
        self.logger = context.logger

    def can_handle(self, event: InputEvent) -> bool:
        """Check if this processor can handle image events."""
        return event.event_type == InputEventType.IMAGE

    async def process(self, event: InputEvent) -> InputResponse:
        """Process an image input event."""
        try:
            self.logger.debug(f"Processing image input with {len(event.images)} images")

            # Validate images
            if not event.images:
                return InputResponse(
                    success=False,
                    message="No images provided in image event"
                )

            return InputResponse(
                success=True,
                message=f"Processed {len(event.images)} images",
                data={
                    'image_count': len(event.images),
                    'images': event.images,
                    'metadata': event.metadata
                }
            )

        except Exception as e:
            self.logger.error(f"Error processing image input: {e}")
            return InputResponse(
                success=False,
                message=f"Image processing failed: {str(e)}"
            )


class UnifiedInputManager:
    """
    High-level manager that combines InputManager with processors.

    This class provides a simplified interface for the main application
    to handle all types of input through a single, unified system.
    """

    def __init__(self, context):
        from agent.core.input_manager import InputManager

        self.context = context

        # Handle different context types
        if hasattr(context, 'logger'):
            self.logger = context.logger
        else:
            import logging
            self.logger = logging.getLogger("unified_input_manager")

        # Core components
        self.input_manager = InputManager(context)

        # Try to create command handler with fallback
        try:
            from agent.commands.command_handler import CommandHandler
            self.command_handler = CommandHandler(context)
        except Exception as e:
            self.logger.warning(f"Could not create command handler: {e}")
            self.command_handler = None

        # Set up processors
        self._setup_processors()

        # Auto-setup basic input sources
        self._auto_setup_sources()

        self.logger.info("UnifiedInputManager initialized")

    def _setup_processors(self) -> None:
        """Set up all input processors."""
        processors = [
            TextInputProcessor(self.context),
            InterruptInputProcessor(self.context),
            ImageInputProcessor(self.context)
        ]

        # Only add command processor if command handler is available
        if self.command_handler is not None:
            processors.insert(0, CommandInputProcessor(self.context, self.command_handler))

        for processor in processors:
            self.input_manager.register_processor(processor)

    def _auto_setup_sources(self) -> None:
        """Auto-setup basic input sources during initialization."""
        try:
            from core.globals import g

            # Always enable CLI as the default
            from agent.core.input_sources import CLIInputSource
            cli_source = CLIInputSource(self.context)
            self.input_manager.register_source(cli_source)
            self.input_manager.set_default_source(cli_source.source_type)

            # Enable queue for programmatic inputs
            from agent.core.input_sources import QueueInputSource
            queue_source = QueueInputSource(self.context)
            self.input_manager.register_source(queue_source)

            # Auto-enable Web input source if GUI mode is active
            if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
                from agent.core.input_sources import WebInputSource
                web_source = WebInputSource(self.context)
                self.input_manager.register_source(web_source)
                # Set Web as default in GUI mode to prioritize GUI input
                self.input_manager.set_default_source(web_source.source_type)
                self.logger.info("Auto-configured CLI, Queue, and Web input sources (GUI mode)")
            else:
                self.logger.info("Auto-configured CLI and Queue input sources")

        except Exception as e:
            self.logger.error(f"Failed to auto-setup input sources: {e}")

    async def setup_sources(
        self,
        enable_cli: bool = True,
        enable_gui: bool = False,
        enable_web: bool = False,
        enable_voice: bool = False,
        enable_queue: bool = True
    ) -> None:
        """Set up input sources based on configuration."""
        from agent.core.input_sources import (
            CLIInputSource, GUIInputSource, WebInputSource,
            VoiceInputSource, QueueInputSource
        )

        from agent.core.input_manager import InputSource

        if enable_cli and InputSource.CLI not in self.input_manager._sources:
            cli_source = CLIInputSource(self.context)
            self.input_manager.register_source(cli_source)
            self.input_manager.set_default_source(InputSource.CLI)

        if enable_gui and InputSource.GUI not in self.input_manager._sources:
            gui_source = GUIInputSource(self.context)
            self.input_manager.register_source(gui_source)

        if enable_web and InputSource.WEB not in self.input_manager._sources:
            web_source = WebInputSource(self.context)
            self.input_manager.register_source(web_source)

        if enable_voice and InputSource.VOICE not in self.input_manager._sources:
            voice_source = VoiceInputSource(self.context)
            self.input_manager.register_source(voice_source)

        if enable_queue and InputSource.QUEUE not in self.input_manager._sources:
            queue_source = QueueInputSource(self.context)
            self.input_manager.register_source(queue_source)

        self.logger.info(f"Input sources configured: {self.input_manager.get_available_sources()}")

    async def get_user_input(
        self,
        prompt: str = "",
        args: Optional[argparse.Namespace] = None,
        context_chat = None,
        force_source: Optional[InputSource] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> tuple[str, list[str]]:
        """
        Get user input through the unified system.

        This method provides compatibility with the existing
        get_user_input_with_bindings interface.

        Returns:
            tuple: (user_input, base64_images)
        """
        try:
            # Store args in context for processors
            if args is not None:
                self.context.args = args

            # Check for queued messages first
            if hasattr(args, 'message') and args and args.message:
                queue_source = self.input_manager._sources.get(InputSource.QUEUE)
                if queue_source and isinstance(queue_source, QueueInputSource):
                    if not queue_source.has_messages():
                        queue_source.add_messages(list(args.message))
                        args.message.clear()

                    if queue_source.has_messages():
                        force_source = InputSource.QUEUE

            # Get input event
            event = await self.input_manager.get_input(
                prompt=prompt,
                source=force_source,
                timeout=timeout,
                context_chat=context_chat,
                args=args,
                **kwargs
            )

            # Process the event
            response = await self.input_manager.process_event(event)

            if not response.success:
                self.logger.error(f"Input processing failed: {response.message}")

            # Extract data for return
            images = []
            if response.data:
                images = response.data.get('images', [])

            # Handle special cases
            if event.event_type == InputEventType.INTERRUPT:
                raise KeyboardInterrupt()

            # Return in expected format
            return event.content, images

        except Exception as e:
            self.logger.error(f"Error in get_user_input: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up the unified input manager."""
        await self.input_manager.cleanup()
        self.logger.info("UnifiedInputManager cleanup completed")