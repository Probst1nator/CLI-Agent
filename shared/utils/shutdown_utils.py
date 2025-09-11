"""
Graceful shutdown utilities for CLI-Agent.

This module consolidates shutdown logic that was previously duplicated
in main.py and main_refactored.py.
"""

import logging
from typing import Optional, TYPE_CHECKING
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


def perform_graceful_shutdown_legacy() -> None:
    """
    Legacy graceful shutdown for main.py compatibility.
    
    This is a wrapper around perform_graceful_shutdown() that maintains
    the exact same behavior as the original main.py implementation.
    """
    perform_graceful_shutdown(context=None)


def perform_graceful_shutdown_modern(context: "AppContext") -> None:
    """
    Modern graceful shutdown for main_refactored.py.
    
    Args:
        context: AppContext instance for proper cleanup and logging
    """
    try:
        # Cleanup temporary files first
        if hasattr(context, 'cleanup_temp_files'):
            context.cleanup_temp_files()
        
        # Then perform standard shutdown
        perform_graceful_shutdown(context=context)
        
    except Exception as e:
        print(colored(f"Error during graceful shutdown: {e}", "red"))