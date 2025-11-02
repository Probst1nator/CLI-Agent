"""
Compatibility layer between legacy and new input systems.

This module provides backwards compatibility for existing code while
enabling migration to the new unified input management system.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import argparse

    from agent.core.app_context import AppContext
    from agent.core.command_processor import UnifiedInputManager
    from core.chat import Chat

# Global instance for compatibility
_unified_input_manager: Optional[UnifiedInputManager] = None


def initialize_unified_input(context) -> None:
    """Initialize the global unified input manager."""
    global _unified_input_manager

    if _unified_input_manager is None:
        try:
            from agent.core.command_processor import UnifiedInputManager
            _unified_input_manager = UnifiedInputManager(context)
        except (ImportError, AttributeError, TypeError) as e:
            # Log the error but don't fail completely
            import logging
            logging.warning(f"Could not initialize UnifiedInputManager: {e}")
            _unified_input_manager = None


async def get_unified_input_manager() -> UnifiedInputManager:
    """Get the global unified input manager."""
    if _unified_input_manager is None:
        # Auto-initialize with a minimal context if not already initialized
        import logging

        # Create minimal context for auto-initialization
        class MinimalContext:
            def __init__(self):
                self.logger = logging.getLogger("unified_input_auto_init")

        try:
            from agent.core.app_context import AppContext
            context = AppContext()
        except Exception as e:
            # Use minimal context if AppContext is not available
            logging.info(f"Using minimal context for unified input manager: {e}")
            context = MinimalContext()

        initialize_unified_input(context)

        if _unified_input_manager is None:
            raise RuntimeError("Failed to initialize unified input manager")

    return _unified_input_manager


async def get_user_input_with_bindings_unified(
    args: argparse.Namespace,
    context_chat: Optional[Chat],
    prompt: str = " Enter your request: ",
    input_override: Optional[str] = None,
    force_input: bool = False,
    input_event: Optional[asyncio.Event] = None,
    input_lock: Optional[threading.Lock] = None,
    shared_input: Optional[dict[str, Any]] = None
) -> tuple[str, list[str]]:
    """
    Unified replacement for get_user_input_with_bindings.

    This function provides the same interface as the original but uses
    the new unified input management system underneath.
    """
    manager = await get_unified_input_manager()

    # Handle input override (direct input provided)
    if input_override:
        return input_override, []

    # Handle GUI mode setup
    if input_event and input_lock and shared_input:
        # Set up GUI input source
        from agent.core.input_manager import InputSource
        from agent.core.input_sources import GUIInputSource

        gui_source = manager.input_manager._sources.get(InputSource.GUI)
        if gui_source and isinstance(gui_source, GUIInputSource):
            gui_source.set_gui_handlers(input_event, input_lock, shared_input)

    # Determine input source based on context
    source = None

    # Check for GUI mode
    try:
        from core.globals import g
        if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
            from agent.core.input_manager import InputSource
            source = InputSource.WEB  # Use WEB source for browser-based GUI interface
    except ImportError:
        # No GUI mode available
        pass

    # Check for voice mode - use voice input for main user prompts
    # Main prompt contains "Enter your request", internal prompts do not
    is_main_prompt = "Enter your request" in prompt

    # Check both command line args and runtime toggle state
    voice_mode_active = args.voice if hasattr(args, 'voice') else False
    tts_only_mode = False
    try:
        from core.globals import g
        if hasattr(g, 'voice_mode'):
            voice_mode_active = g.voice_mode
        if hasattr(g, 'tts_only_mode'):
            tts_only_mode = g.tts_only_mode
    except ImportError:
        pass

    # Only use voice input if not in GUI/API mode (frontend handles voice in those modes)
    gui_or_api_mode = False
    try:
        from core.globals import g
        if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
            gui_or_api_mode = True
    except ImportError:
        pass

    # Use backend voice input only in CLI mode with voice enabled
    if voice_mode_active and is_main_prompt and not tts_only_mode and not gui_or_api_mode:
        from agent.core.input_manager import InputSource
        source = InputSource.VOICE

    # Get input through unified system
    return await manager.get_user_input(
        prompt=prompt,
        args=args,
        context_chat=context_chat,
        force_source=source,
        force_input=force_input
    )


def handle_multiline_input_unified() -> str:
    """Unified multiline input handler."""
    from termcolor import colored
    print(colored("Enter your multiline input. Type '--f' on a new line when finished.", "blue"))
    lines = []
    while True:
        try:
            line = input()
            if line == "--f":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)


async def handle_screenshot_capture_unified() -> list[str]:
    """Unified screenshot capture handler."""
    import base64
    import datetime
    import logging
    import os
    import subprocess
    import tempfile

    import pyperclip
    from termcolor import colored

    from core.globals import g

    base64_images: list[str] = []
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        base64_images = []
        try:
            logging.info(colored("Attempting screenshot with Spectacle (region selection)...", "green"))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_filename = temp_file.name
            subprocess.run(['spectacle', '-rbno', temp_filename], check=True)

            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                with open(temp_filename, 'rb') as image_file:
                    png_data = image_file.read()
                base64_img = base64.b64encode(png_data).decode('utf-8')
                base64_images = [base64_img]
                pyperclip.copy(base64_img)
                logging.info(colored("Screenshot captured successfully. (+Copied to clipboard)", "green"))
            else:
                logging.warning(colored(f"No screenshot was captured (attempt {attempt}).", "yellow"))

            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

        except subprocess.CalledProcessError:
            logging.error(colored(f"Spectacle command failed on attempt {attempt}. Is it installed?", "red"))
        except Exception as e:
            logging.exception(f"Unexpected error during screenshot capture attempt {attempt}: {e}")

        if base64_images:
            break

        if attempt < max_attempts and not base64_images:
            logging.warning(colored("Retrying screenshot capture...", "yellow"))
            await asyncio.sleep(2)

    # Save screenshots to disk
    if base64_images:
        images_dir = os.path.join(g.AGENTS_SANDBOX_DIR, "screenshots")
        os.makedirs(images_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        for i, base64_img in enumerate(base64_images):
            try:
                img_data = base64.b64decode(base64_img)
                img_path = os.path.join(images_dir, f"screenshot_{timestamp}_{i+1}.png")
                with open(img_path, "wb") as img_file:
                    img_file.write(img_data)
                logging.info(colored(f"Screenshot saved to {img_path}", "green"))
            except Exception as e:
                logging.error(colored(f"Error saving screenshot {i+1}: {e}", "red"))
    else:
        logging.error(colored("No images were captured.", "red"))
        logging.warning(colored("# cli-agent: No screenshot was captured after multiple attempts.", "yellow"))

    return base64_images


class InputCompatibilityAdapter:
    """
    Adapter class to help migrate from legacy input system to unified system.

    This class provides methods to gradually migrate code while maintaining
    backwards compatibility.
    """

    def __init__(self, context: AppContext):
        self.context = context
        asyncio.create_task(self._setup_unified_manager())

    async def _setup_unified_manager(self) -> None:
        """Set up the unified input manager."""
        initialize_unified_input(self.context)
        manager = await get_unified_input_manager()

        # Set up input sources based on context
        await manager.setup_sources(
            enable_cli=True,
            enable_gui=True,  # Always enable GUI support
            enable_web=getattr(__import__('core.globals', fromlist=['g']).g, 'GUI_MODE', False),
            enable_voice=True,  # Enable voice support
            enable_queue=True
        )

    async def migrate_get_user_input(
        self,
        args: argparse.Namespace,
        context_chat: Optional[Chat],
        **kwargs: Any
    ) -> tuple[str, list[str]]:
        """
        Migration helper for get_user_input_with_bindings calls.

        This method can be used to gradually replace calls to the old
        function with the new unified system.
        """
        return await get_user_input_with_bindings_unified(
            args=args,
            context_chat=context_chat,
            **kwargs
        )

    def get_legacy_compatibility_shims(self) -> dict[str, Any]:
        """
        Get compatibility shims for legacy functions.

        Returns a dictionary of function names to their unified replacements.
        """
        return {
            'get_user_input_with_bindings': get_user_input_with_bindings_unified,
            'handle_multiline_input': handle_multiline_input_unified,
            'handle_screenshot_capture': handle_screenshot_capture_unified
        }


# Convenience functions for global access
async def setup_unified_input_system(context: AppContext) -> None:
    """Set up the unified input system globally."""
    initialize_unified_input(context)
    manager = await get_unified_input_manager()

    await manager.setup_sources(
        enable_cli=True,
        enable_gui=True,
        enable_web=True,
        enable_voice=True,
        enable_queue=True
    )


async def cleanup_unified_input_system() -> None:
    """Clean up the unified input system."""
    global _unified_input_manager
    if _unified_input_manager:
        await _unified_input_manager.cleanup()
        _unified_input_manager = None