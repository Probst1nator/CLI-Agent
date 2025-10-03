"""
Migration helper for transitioning from legacy main.py input handling to unified system.

This module provides utilities and integration points to help migrate
main.py gradually from the legacy input system to the new unified system.
"""

from __future__ import annotations
import asyncio
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.core.app_context import AppContext
    from core.chat import Chat
    import argparse

from agent.core.input_compatibility import (
    setup_unified_input_system,
    get_unified_input_manager,
    cleanup_unified_input_system
)
from core.globals import g


class MainPyMigrationHelper:
    """
    Helper class to gradually migrate main.py to the unified input system.

    This class provides drop-in replacements for legacy input functions
    while maintaining backwards compatibility with existing main.py code.
    """

    def __init__(self, context: AppContext):
        self.context = context
        self.logger = logging.getLogger("migration_helper")
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the unified input system."""
        if not self._initialized:
            await setup_unified_input_system(self.context)
            self._initialized = True
            self.logger.info("Migration helper initialized with unified input system")

    async def get_user_input_with_bindings_replacement(
        self,
        args: argparse.Namespace,
        context_chat: Optional[Chat],
        prompt: str = " Enter your request: ",
        input_override: str = None,
        force_input: bool = False,
        input_event: Optional[asyncio.Event] = None,
        input_lock: Optional[threading.Lock] = None,
        shared_input: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[str]]:
        """
        Drop-in replacement for get_user_input_with_bindings.

        This method provides the exact same interface as the original function
        but uses the new unified input management system underneath.
        """
        await self.initialize()

        try:
            # Use the unified input system
            from agent.core.input_compatibility import get_user_input_with_bindings_unified
            return await get_user_input_with_bindings_unified(
                args=args,
                context_chat=context_chat,
                prompt=prompt,
                input_override=input_override,
                force_input=force_input,
                input_event=input_event,
                input_lock=input_lock,
                shared_input=shared_input
            )

        except Exception as e:
            self.logger.error(f"Unified input system failed: {e}")
            raise RuntimeError(f"Input system failure: {e}")

    def handle_multiline_input_replacement(self) -> str:
        """Drop-in replacement for handle_multiline_input."""
        from agent.core.input_compatibility import handle_multiline_input_unified
        return handle_multiline_input_unified()

    async def handle_screenshot_capture_replacement(self) -> List[str]:
        """Drop-in replacement for handle_screenshot_capture."""
        from agent.core.input_compatibility import handle_screenshot_capture_unified
        return await handle_screenshot_capture_unified()

    async def cleanup(self) -> None:
        """Clean up the migration helper."""
        if self._initialized:
            await cleanup_unified_input_system()
            self._initialized = False
            self.logger.info("Migration helper cleaned up")


def create_migration_patches(context: AppContext) -> Dict[str, Any]:
    """
    Create migration patches for main.py functions.

    This function returns a dictionary of function replacements that can be
    used to monkey-patch main.py during the migration process.

    Args:
        context: Application context

    Returns:
        Dict of function name to replacement function mappings
    """
    helper = MainPyMigrationHelper(context)

    return {
        'get_user_input_with_bindings': helper.get_user_input_with_bindings_replacement,
        'handle_multiline_input': helper.handle_multiline_input_replacement,
        'handle_screenshot_capture': helper.handle_screenshot_capture_replacement,
    }


def apply_migration_patches(main_module, context: AppContext) -> MainPyMigrationHelper:
    """
    Apply migration patches to main.py module.

    This function monkey-patches the main.py module to use unified input
    system functions while maintaining full backwards compatibility.

    Args:
        main_module: The main.py module object
        context: Application context

    Returns:
        MainPyMigrationHelper: The helper instance for cleanup
    """
    helper = MainPyMigrationHelper(context)
    patches = create_migration_patches(context)

    # Apply patches to the module
    original_functions = {}
    for func_name, replacement_func in patches.items():
        if hasattr(main_module, func_name):
            # Store original function for potential rollback
            original_functions[func_name] = getattr(main_module, func_name)
            # Apply the patch
            setattr(main_module, func_name, replacement_func)
            logging.info(f"Applied migration patch for {func_name}")
        else:
            logging.warning(f"Function {func_name} not found in main module")

    # Store original functions in the helper for potential rollback
    helper._original_functions = original_functions
    helper._patched_module = main_module

    return helper


def rollback_migration_patches(helper: MainPyMigrationHelper) -> None:
    """
    Rollback migration patches to restore original functions.

    Args:
        helper: The migration helper that was used to apply patches
    """
    if hasattr(helper, '_original_functions') and hasattr(helper, '_patched_module'):
        for func_name, original_func in helper._original_functions.items():
            setattr(helper._patched_module, func_name, original_func)
            logging.info(f"Rolled back migration patch for {func_name}")


class LegacyInputIntegration:
    """
    Integration layer for legacy input functions that need special handling.

    This class provides integration points for main.py functions that require
    more complex migration logic or have special dependencies.
    """

    def __init__(self, context: AppContext):
        self.context = context
        self.logger = logging.getLogger("legacy_input_integration")

    async def integrate_with_agent_loop(self, args: argparse.Namespace) -> None:
        """
        Integrate unified input system with the main agent loop.

        This method sets up the unified input system to work seamlessly
        with the existing main.py agent loop.
        """
        try:
            # Initialize unified input system
            await setup_unified_input_system(self.context)

            # Set up input sources based on args
            manager = await get_unified_input_manager()

            # Configure input sources based on command line arguments
            await manager.setup_sources(
                enable_cli=True,
                enable_gui=getattr(args, 'gui', False) or hasattr(g, 'GUI_MODE'),
                enable_web=hasattr(g, 'GUI_MODE') and g.GUI_MODE,
                enable_voice=getattr(args, 'voice', False),
                enable_queue=True
            )

            self.logger.info("Unified input system integrated with agent loop")

        except Exception as e:
            self.logger.error(f"Failed to integrate with agent loop: {e}")
            raise

    def get_legacy_function_mapping(self) -> Dict[str, str]:
        """
        Get mapping of legacy function names to their replacements.

        Returns:
            Dict mapping old function names to new function names/paths
        """
        return {
            'get_user_input_with_bindings': 'agent.core.input_compatibility.get_user_input_with_bindings_unified',
            'handle_multiline_input': 'agent.core.input_compatibility.handle_multiline_input_unified',
            'handle_screenshot_capture': 'agent.core.input_compatibility.handle_screenshot_capture_unified',
        }

    def validate_migration_compatibility(self) -> List[str]:
        """
        Validate that the migration is compatible with the current codebase.

        Returns:
            List of compatibility issues found
        """
        issues = []

        # Check if required modules are available
        required_modules = [
            'agent.core.input_manager',
            'agent.core.input_sources',
            'agent.core.command_processor',
            'agent.core.input_compatibility'
        ]

        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                issues.append(f"Required module {module_name} not available: {e}")

        # Check if core globals are accessible
        try:
            from core.globals import g
        except ImportError:
            issues.append("Core globals module not accessible")

        return issues


# Global migration helper instance
_global_migration_helper: Optional[MainPyMigrationHelper] = None


def get_global_migration_helper(context: AppContext) -> MainPyMigrationHelper:
    """Get the global migration helper instance."""
    global _global_migration_helper
    if _global_migration_helper is None:
        _global_migration_helper = MainPyMigrationHelper(context)
    return _global_migration_helper