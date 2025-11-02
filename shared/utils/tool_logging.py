"""
Tool Logging Helper

Provides a simple interface for tools to set up consistent logging
with dedicated log files separated from the main application logs.
"""

import logging
import sys


def setup_tool_logger(tool_name: str, debug_mode: bool = None) -> logging.Logger:
    """
    Set up a logger for a specific tool with dedicated log file.

    This is the primary function tools should use to set up logging.
    It creates a separate log file for each tool to enable focused debugging.

    Args:
        tool_name: Name of the tool (automatically extracted from __name__ if not provided)
        debug_mode: Enable debug logging (auto-detected from sys.argv if None)

    Returns:
        Configured logger instance

    Example:
        # In tools/bash.py
        from shared.utils.tool_logging import setup_tool_logger
        logger = setup_tool_logger("bash")

        # Use the logger
        logger.info("Bash session started")
        logger.debug("Command executed: ls -la")
        logger.error("Failed to execute command")
    """
    # Auto-detect debug mode from command line
    if debug_mode is None:
        debug_mode = '--debug' in sys.argv

    try:
        from shared.utils.logging_utils import get_tool_logger
        return get_tool_logger(tool_name, debug_mode)
    except ImportError:
        # Fallback if logging_utils is not available
        return _fallback_tool_logger(tool_name, debug_mode)


def get_logger_for_tool_module(module_name: str, debug_mode: bool = None) -> logging.Logger:
    """
    Get a logger for a tool module using its __name__.

    Args:
        module_name: The __name__ of the tool module (e.g., 'tools.bash')
        debug_mode: Enable debug logging (auto-detected if None)

    Returns:
        Configured logger instance

    Example:
        # In tools/bash.py
        from shared.utils.tool_logging import get_logger_for_tool_module
        logger = get_logger_for_tool_module(__name__)
    """
    # Extract tool name from module name
    if '.' in module_name:
        tool_name = module_name.split('.')[-1]  # 'tools.bash' -> 'bash'
    else:
        tool_name = module_name

    return setup_tool_logger(tool_name, debug_mode)


def _fallback_tool_logger(tool_name: str, debug_mode: bool) -> logging.Logger:
    """Fallback logger creation if main logging system is unavailable."""
    logger_name = f"tools.{tool_name}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Create simple file handler
    log_file = f"/tmp/cli-agent-tool-{tool_name}.log"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        f"%(asctime)s [{tool_name}] [%(levelname)-8s] %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Don't propagate to avoid interference with main app logs
    logger.propagate = False

    print(f"[{tool_name}] Fallback logging to: {log_file}")
    return logger


# Convenience functions for common logging patterns in tools
def log_tool_startup(logger: logging.Logger, tool_name: str, **kwargs):
    """Log tool startup with common information."""
    logger.info(f"{tool_name} tool starting up")
    if kwargs:
        for key, value in kwargs.items():
            logger.debug(f"{tool_name} config: {key}={value}")


def log_tool_error(logger: logging.Logger, tool_name: str, error: Exception, context: str = ""):
    """Log tool errors with consistent format."""
    context_msg = f" ({context})" if context else ""
    logger.error(f"{tool_name} error{context_msg}: {error}", exc_info=True)


def log_tool_operation(logger: logging.Logger, operation: str, details: str = ""):
    """Log tool operations with optional details."""
    message = f"Operation: {operation}"
    if details:
        message += f" - {details}"
    logger.info(message)


# Example usage and testing
if __name__ == "__main__":
    # Test the tool logging system
    print("🧪 Testing Tool Logging System...")
    print("=" * 50)

    # Test basic setup
    test_logger = setup_tool_logger("test_tool", debug_mode=True)

    # Test logging functions
    log_tool_startup(test_logger, "test_tool", setting1="value1", setting2="value2")

    test_logger.info("This is an info message")
    test_logger.debug("This is a debug message")
    test_logger.warning("This is a warning message")

    log_tool_operation(test_logger, "test_operation", "with some details")

    try:
        raise ValueError("Test error")
    except Exception as e:
        log_tool_error(test_logger, "test_tool", e, "during testing")

    print("✅ Tool logging test completed!")
    print("Check log files for output.")