"""
Logging utilities for CLI-Agent.

This module contains logging formatters and configuration functions extracted
from main.py to provide reusable logging functionality across the codebase.
"""

import logging
import re
import sys
import time
from pathlib import Path
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
    Configure the root logger for console and file output with automatic retention.

    Always logs to file at DEBUG level for production issue diagnosis.
    Uses default location ~/.cli-agent/logs/session.log if no log_file specified.

    Args:
        debug_mode: Enable DEBUG level console output (INFO by default)
        log_file: Optional custom path to log file (overrides default)
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

    # --- File Handler with Automatic Retention (ALWAYS ENABLED) ---
    # Use default location if not specified
    if not log_file:
        from pathlib import Path
        default_log_dir = Path.home() / ".cli-agent" / "logs"
        default_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(default_log_dir / "session.log")

    try:
        # Import log manager for retention features
        from shared.utils.log_manager import get_log_manager

        log_manager = get_log_manager()

        # Determine policy based on file name and mode
        if 'debug' in log_file.lower():
            policy_name = 'debug'
        elif 'tui' in log_file.lower() or tui_mode:
            policy_name = 'tui'
        else:
            policy_name = 'application'

        # Create rotating file handler with retention
        file_handler = log_manager.create_rotating_handler(log_file, policy_name)
        file_handler.setLevel(logging.DEBUG) # ALWAYS log everything to file, regardless of console level
        file_formatter = FileFormatter("%(elapsed)s [%(levelname)-8s] %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Start periodic cleanup in background
        log_manager.start_periodic_cleanup()

        # Perform immediate cleanup of old files on startup
        cleanup_stats = log_manager.cleanup_old_logs()
        if cleanup_stats['files_removed'] > 0:
            logging.info(colored(f"🧹 Cleaned up {cleanup_stats['files_removed']} old log files ({cleanup_stats['bytes_freed'] / 1024:.1f} KB freed)", "cyan"))

        if debug_mode:
            # Only show file location in debug mode to avoid clutter
            logging.info(colored(f"📝 Logging to file: {log_file} (policy: {policy_name})", "magenta"))
        else:
            # Silently log to file in production
            logging.debug(f"Logging to file: {log_file} (policy: {policy_name})")

    except ImportError:
        # Fallback to basic file handler if log manager is not available
        file_handler = logging.FileHandler(log_file, mode='a')  # Append mode
        file_handler.setLevel(logging.DEBUG)
        file_formatter = FileFormatter("%(elapsed)s [%(levelname)-8s] %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        if debug_mode:
            logging.info(colored(f"📝 Logging to file: {log_file} (basic mode)", "magenta"))
        else:
            logging.debug(f"Logging to file: {log_file} (basic mode)")

    # --- FIX: Silence Noisy Loggers (Always, even in debug mode) ---
    # These libraries produce networking/infrastructure logs we don't need
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("py_classes.ai_providers.cls_ollama_interface").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def setup_tool_logging(tool_name: str, debug_mode: bool = False) -> logging.Logger:
    """
    Configure a separate logger for tools with dedicated log file.

    Args:
        tool_name: Name of the tool (e.g., 'bash', 'visualresponse')
        debug_mode: Enable debug logging level

    Returns:
        Configured logger for the tool
    """
    logger_name = f"tools.{tool_name}"
    logger = logging.getLogger(logger_name)

    # Avoid duplicate setup
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    try:
        # Import log manager for retention features
        from shared.utils.log_manager import get_log_manager

        log_manager = get_log_manager()

        # Create tool-specific log file
        log_file = f"~/.cli-agent/logs/tools_{tool_name}.log"
        log_file_expanded = log_file.replace("~", str(Path.home()))

        # Use tool-specific retention policy
        file_handler = log_manager.create_rotating_handler(
            log_file_expanded,
            policy_name='application'  # Tools generate moderate logs
        )
        file_handler.setLevel(logging.DEBUG)

        # Tool-specific formatter with tool name prefix
        tool_formatter = FileFormatter(f"%(elapsed)s [{tool_name}] [%(levelname)-8s] %(message)s")
        file_handler.setFormatter(tool_formatter)
        logger.addHandler(file_handler)

        # Optional console handler for errors only (reduces noise)
        if debug_mode:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.WARNING)  # Only warnings/errors to console
            console_formatter = SimpleFormatter(f"[{tool_name}] %(levelname)s: %(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Don't propagate to root logger to avoid duplication
        logger.propagate = False

        logger.info(f"Tool logging initialized: {log_file}")

    except ImportError:
        # Fallback to basic file handler if log manager is not available
        log_file = f"/tmp/cli-agent-tool-{tool_name}.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        tool_formatter = logging.Formatter(f"%(asctime)s [{tool_name}] [%(levelname)-8s] %(message)s")
        file_handler.setFormatter(tool_formatter)
        logger.addHandler(file_handler)
        logger.propagate = False
        logger.info(f"Tool logging initialized (fallback): {log_file}")

    return logger


def get_tool_logger(tool_name: str, debug_mode: bool = False) -> logging.Logger:
    """
    Get or create a tool-specific logger.

    Args:
        tool_name: Name of the tool
        debug_mode: Enable debug logging level

    Returns:
        Logger instance for the tool
    """
    logger_name = f"tools.{tool_name}"
    logger = logging.getLogger(logger_name)

    # Setup if not already done
    if not logger.handlers:
        return setup_tool_logging(tool_name, debug_mode)

    return logger


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
