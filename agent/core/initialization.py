"""
Application initialization for CLI-Agent.

This module extracts the initialization logic from main.py into a
clean, reusable component with proper typing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from pyfiglet import figlet_format
from termcolor import colored

if TYPE_CHECKING:
    from agent.core.app_context import AppContext

def setup_project_path() -> Path:
    """Setup project path and ensure it's in sys.path."""
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

def setup_environment() -> None:
    """Setup environment variables and suppress noisy warnings."""
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(logging.FATAL)

    # Suppress other noisy warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")
    warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")

def setup_logging(debug_mode: bool, log_file: Optional[str] = None, tui_mode: bool = False) -> None:
    """
    Setup logging configuration (extracted from main.py).

    Args:
        debug_mode: Enable debug logging
        log_file: Optional custom log file path (uses default if None)
        tui_mode: Whether running in TUI mode
    """
    # Import here to avoid circular imports during startup
    from main import setup_logging as original_setup_logging
    # log_file parameter kept for backwards compatibility but is optional
    original_setup_logging(debug_mode, log_file, tui_mode)

def display_startup_header(tui_mode: bool = False) -> None:
    """Display the CLI-Agent startup header."""
    header_dashes = colored("# # # # # # # # # # # # # # # # # # # # # # # # # #", "blue")
    header = figlet_format("CLI-Agent", font="slant")

    print(header_dashes)
    print(colored(header, "cyan", attrs=["bold"]))
    print(header_dashes)

    if not tui_mode:
        logging.info("Starting CLI-Agent...")

def apply_default_configurations(context: AppContext) -> None:
    """Apply default LLM configuration."""
    try:
        # Import here to avoid startup overhead
        from agent.core.config_utils import apply_default_llm_config

        apply_default_llm_config()

        # Sync with context
        from core.globals import g
        context.llm_config.model_key = g.llm_config.get('model_key', '')
        context.llm_config.temperature = g.llm_config.get('temperature', 0.7)
        context.llm_config.provider = g.llm_config.get('provider', '')

        context.settings.debug_mode = g.DEBUG_MODE
        context.settings.auto_execute = g.auto_execute
        context.settings.voice_mode = hasattr(g, 'voice_mode') and g.voice_mode
        context.settings.local_mode = hasattr(g, 'local_mode') and g.local_mode
        context.settings.online_mode = hasattr(g, 'online_mode') and g.online_mode

    except Exception as e:
        logging.warning(f"Failed to apply default configurations: {e}")

def create_tui_args() -> argparse.Namespace:
    """Create default arguments for TUI mode."""
    return argparse.Namespace(
        debug=False,
        log_file=None,
        auto=False,
        local=False,
        message=[],
        regenerate=False,
        voice=False,
        online_voice=False,
        private_remote_wake_detection=False,
        gui=False,
        c=False,
        llm=None,
        fast=False,
        min=False,
        image=False,
        full_output_mode=False,
        exit=False,
        debug_chats=False,
        online=False
    )

def initialize_app(
    args: Optional[argparse.Namespace] = None,
    tui_mode: bool = False
) -> AppContext:
    """
    Initialize the CLI-Agent application.

    Args:
        args: Parsed command line arguments (None for TUI mode)
        tui_mode: Whether running in TUI mode

    Returns:
        Initialized AppContext
    """
    # Setup basic environment
    setup_environment()
    project_root = setup_project_path()

    # Create context
    from agent.core.app_context import AppContext
    context = AppContext()
    context.project_root = project_root

    # Handle TUI vs CLI mode
    if tui_mode:
        if args is None:
            args = create_tui_args()
        setup_logging(args.debug, tui_mode=True)
        display_startup_header(tui_mode=True)
    else:
        if args is None:
            from agent.core.cli_args import parse_cli_args
            args = parse_cli_args()
        setup_logging(args.debug, tui_mode=False)
        context.settings.debug_mode = args.debug
        display_startup_header(tui_mode=False)

    # Apply configurations
    apply_default_configurations(context)

    # Update context from args
    if hasattr(args, 'yolo'):
        context.settings.auto_execute = args.yolo
    if hasattr(args, 'voice') or hasattr(args, 'speak'):
        # Voice mode: full voice (mic + TTS) or TTS-only mode
        context.settings.voice_mode = args.voice if hasattr(args, 'voice') else False
        if hasattr(args, 'speak'):
            context.settings.voice_mode = context.settings.voice_mode or args.speak
    if hasattr(args, 'local'):
        context.settings.local_mode = args.local
    if hasattr(args, 'online'):
        context.settings.online_mode = args.online
    if hasattr(args, 'full_output_mode'):
        context.settings.full_output_mode = args.full_output_mode

    # Sync context settings back to global state to ensure consistency
    from core.globals import g
    g.auto_execute = context.settings.auto_execute
    g.voice_mode = context.settings.voice_mode
    # Set TTS-only mode flag (for -s/--speak without -v)
    if hasattr(args, 'speak') and hasattr(args, 'voice'):
        g.tts_only_mode = args.speak and not args.voice
    else:
        g.tts_only_mode = False
    g.local_mode = context.settings.local_mode
    g.online_mode = context.settings.online_mode

    logging.info("CLI-Agent initialization complete")
    return context