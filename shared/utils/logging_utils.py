"""
Logging utilities for CLI-Agent.

This module contains logging formatters and configuration functions extracted 
from main.py to provide reusable logging functionality across the codebase.
"""

import logging
import re
import sys
import time
from typing import Optional
from termcolor import colored

# Import the global start_time from main.py context
# This maintains compatibility with existing elapsed time calculations
try:
    from main import start_time
except ImportError:
    # Fallback if imported outside main.py context
    start_time = time.time()


class ConsoleFormatter(logging.Formatter):
    """A logging formatter that adds a colored elapsed time prefix for the console."""
    
    def __init__(self, fmt, **kwargs):
        super().__init__(fmt, **kwargs)
    
    def format(self, record):
        elapsed = time.time() - start_time
        # Create a prefix with the elapsed time
        prefix = colored(f"[{elapsed:6.2f}s] ", 'dark_grey')
        
        # The original message might already be colored
        formatted_message = super().format(record)
        
        # For debug messages, color the whole line for visibility
        if record.levelno == logging.DEBUG:
            return prefix + colored(formatted_message, 'light_blue')
        
        return prefix + formatted_message


class SimpleFormatter(logging.Formatter):
    """A simple formatter that just returns the message, used after startup or in TUI mode."""
    
    def format(self, record):
        return super().format(record)


class FileFormatter(logging.Formatter):
    """A logging formatter for files, including elapsed time and stripping ANSI color codes."""
    
    def __init__(self, fmt, **kwargs):
        super().__init__(fmt, **kwargs)

    def format(self, record):
        elapsed = time.time() - start_time
        record.elapsed = f"[{elapsed:6.2f}s]"
        
        # The original message might contain color codes, strip them for the file log.
        if isinstance(record.msg, str):
            record.msg = re.sub(r'\x1b\[[0-9;]*m', '', record.msg)
            
        return super().format(record)


def setup_logging(debug_mode: bool, log_file: Optional[str] = None, tui_mode: bool = False) -> None:
    """
    Configure the root logger for console and optional file output with automatic retention.
    
    Args:
        debug_mode: Enable debug logging level
        log_file: Optional path to log file
        tui_mode: Whether running in TUI mode (affects formatting)
    """
    logger = logging.getLogger() # Get root logger
    logger.setLevel(logging.DEBUG) # Capture all messages at the root

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_log_level = logging.DEBUG if debug_mode else logging.INFO
    console_handler.setLevel(console_log_level)
    
    # Use a simple formatter in TUI mode to avoid cluttering the UI with timestamps
    if tui_mode:
        console_formatter = SimpleFormatter("%(message)s")
    else:
        console_formatter = ConsoleFormatter("%(message)s")

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File Handler with Automatic Retention ---
    if log_file:
        try:
            # Import log manager for retention features
            from shared.utils.log_manager import get_log_manager
            
            log_manager = get_log_manager()
            
            # Determine policy based on file name and debug mode
            if 'debug' in log_file.lower() or debug_mode:
                policy_name = 'debug'
            elif 'tui' in log_file.lower() or tui_mode:
                policy_name = 'tui'
            else:
                policy_name = 'application'
            
            # Create rotating file handler with retention
            file_handler = log_manager.create_rotating_handler(log_file, policy_name)
            file_handler.setLevel(logging.DEBUG) # Always log everything to the file
            file_formatter = FileFormatter("%(elapsed)s [%(levelname)-8s] %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Start periodic cleanup in background
            log_manager.start_periodic_cleanup()
            
            # Perform immediate cleanup of old files on startup
            cleanup_stats = log_manager.cleanup_old_logs()
            if cleanup_stats['files_removed'] > 0:
                logging.info(colored(f"🧹 Cleaned up {cleanup_stats['files_removed']} old log files ({cleanup_stats['bytes_freed'] / 1024:.1f} KB freed)", "cyan"))
            
            logging.info(colored(f"📝 Logging to file with retention: {log_file} (policy: {policy_name})", "magenta"))
            
        except ImportError:
            # Fallback to basic file handler if log manager is not available
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = FileFormatter("%(elapsed)s [%(levelname)-8s] %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            logging.info(colored(f"Logging to file: {log_file}", "magenta"))

    # --- FIX: Silence Noisy Loggers (Always, even in debug mode) ---
    # These libraries produce networking/infrastructure logs we don't need
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("py_classes.ai_providers.cls_ollama_interface").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def swap_to_simple_logging() -> None:
    """
    Swaps the console handler's formatter to a simpler one without the timer.
    
    This is useful for switching to a cleaner output format after startup.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            # We found the console handler, now swap its formatter
            simple_formatter = SimpleFormatter("%(message)s")
            handler.setFormatter(simple_formatter)
            break
