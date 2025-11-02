"""
Graceful shutdown utilities for CLI-Agent.

This module consolidates shutdown logic that was previously duplicated
in main.py and main_refactored.py.
"""

import logging
from typing import TYPE_CHECKING, Optional

from termcolor import colored

if TYPE_CHECKING:
    from agent.core.app_context import AppContext


def perform_graceful_shutdown(context: Optional["AppContext"] = None) -> None:
    """
    Perform graceful shutdown including log cleanup.

    This function handles both legacy (main.py) and modern (main_refactored.py)
    shutdown flows by accepting an optional AppContext parameter.

    Args:
        context: Optional AppContext for modern architecture (main_refactored.py).
                If None, uses legacy logging approach (main.py).
    """
    try:
        from shared.utils.log_manager import get_log_manager
        log_manager = get_log_manager()
        shutdown_stats = log_manager.shutdown()

        # Calculate total cleanup stats
        log_files_removed = shutdown_stats.get('log_cleanup', {}).get('files_removed', 0)
        audio_files_removed = shutdown_stats.get('audio_cleanup', {}).get('files_removed', 0)

        if log_files_removed > 0 or audio_files_removed > 0:
            total_files = log_files_removed + audio_files_removed
            total_bytes = (shutdown_stats.get('log_cleanup', {}).get('bytes_freed', 0) +
                         shutdown_stats.get('audio_cleanup', {}).get('bytes_freed', 0))

            cleanup_message = colored(
                f"🧹 Final cleanup: {total_files} files removed, {total_bytes / 1024:.1f} KB freed",
                "cyan"
            )

            # Use appropriate logging mechanism based on context
            if context and hasattr(context, 'logger'):
                # Modern architecture with AppContext
                context.logger.info(cleanup_message)
            else:
                # Legacy architecture using global logging
                logging.info(cleanup_message)

    except ImportError:
        # Log manager not available
        pass
    except Exception as e:
        error_message = f"Error during graceful shutdown: {e}"

        if context and hasattr(context, 'logger'):
            context.logger.debug(error_message)
        else:
            logging.debug(error_message)




