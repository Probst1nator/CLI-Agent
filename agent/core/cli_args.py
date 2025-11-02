"""
CLI argument parsing for CLI-Agent.

This module extracts the command-line argument parsing logic from main.py
into a reusable component that can be shared across different entry points.
"""

import argparse
import logging
import sys

from termcolor import colored


def parse_cli_args() -> argparse.Namespace:
    """
    Setup and parse CLI arguments, ensuring the script's functionality remains intact.

    This function is extracted from main.py to enable reuse across different
    entry points (main.py, main_refactored.py, etc.).

    Returns:
        Parsed command line arguments
    """

    parser = argparse.ArgumentParser(
        description="AI CLI-Agent with backend options and more.",
        add_help=False  # Disable automatic help to manually handle unrecognized args
    )

    parser.add_argument("-h", "--help", action="store_true", default=False,
                        help="Display this help")
    parser.add_argument("--yolo", action="store_true", default=False,
                        help="Automatically execute safe commands...")
    parser.add_argument("-c", action="store_true", default=False,
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-l", "--local", action="store_true", default=False,
                        help="Use the local Ollama backend for processing. Sets g.FORCE_LOCAL=True.")
    parser.add_argument("-m", "--message", type=str, default=[], nargs='*',
                        help="Enter one or more messages to process in sequence. Multiple messages can be passed like: -m 'first message' 'second message'. If no message is provided, multiline input mode will be activated.")
    parser.add_argument("-r", "--regenerate", action="store_true", default=False,
                        help="Regenerate the last response.")
    parser.add_argument("-v", "--voice", action="store_true", default=False,
                        help="Enable microphone input and text-to-speech output.")
    parser.add_argument("--online_voice", action="store_true", default=False,
                        help="Enable cloud-based speech transcription and TTS (Groq/OpenAI/ElevenLabs) with local fallback. By default, uses local engines only.")

    parser.add_argument("--full_output", dest='full_output_mode', action="store_true", default=True,
                        help="Full output mode. Shows live model output instead of hiding it.")
    parser.add_argument("-img", "--image", action="store_true", default=False,
                        help="Take a screenshot using Spectacle for region selection (with automatic fallbacks if not available).")
    parser.add_argument("-o", "--online", action="store_true", default=False,
                        help="Force use of cloud AI.")

    parser.add_argument("-llm", "--llm", type=str, nargs='?', const="__select__", default=None,
                        help="Specify the LLM model key to use (e.g., 'gemini-2.5-flash', 'gemma3n:e2b'). Use without value to open selection menu.")

    parser.add_argument("--gui", action="store_true", default=False,
                        help="Open a web interface for the chat")
    parser.add_argument("--debug-chats", action="store_true", default=False,
                        help="Enable debug windows for chat contexts without full debug logging. Sets g.DEBUG_CHATS=True.")
    parser.add_argument("--private_remote_wake_detection", action="store_true", default=False,
                        help="Use private remote wake detection")

    # --- Logging Arguments ---
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable detailed debug logging to the console.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to a file to write detailed logs.")
    # --- End Logging Arguments ---

    parser.add_argument("-e", "--exit", action="store_true", default=False,
                        help="Exit after all automatic messages have been parsed successfully")

    # CORRECTED: Parse directly from sys.argv, as the pre-processing logic was removed.
    args, unknown_args = parser.parse_known_args(sys.argv[1:])

    if unknown_args or args.help:
        if not args.help:
            # Need a temporary logger setup for this specific warning
            logging.basicConfig()
            logging.getLogger().warning(colored(f"Unrecognized arguments {' '.join(unknown_args)}.", "yellow"))
        parser.print_help()
        exit(1)

    return args


def create_tui_args() -> argparse.Namespace:
    """
    Create default arguments for TUI mode.

    This function provides the same default arguments that are used
    in main_refactored.py for TUI mode operation.

    Returns:
        Default argparse.Namespace for TUI mode
    """
    return argparse.Namespace(
        debug=False,
        log_file=None,
        yolo=False,
        local=False,
        message=[],
        regenerate=False,
        voice=False,
        online_voice=False,
        private_remote_wake_detection=False,
        gui=False,
        c=False,
        llm=None,
        min=False,
        image=False,
        full_output_mode=False,
        exit=False,
        debug_chats=False,
        online=False
    )