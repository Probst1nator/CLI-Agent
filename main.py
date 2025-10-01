# main.py
"""
CLI-Agent Main Entry Point - CURRENT/ACTIVE VERSION

This is the stable, production entry point for CLI-Agent (3,273 lines).
Contains all recent bug fixes and active development, including:
- Fixed ComputationalNotebook architecture (now uses direct tool calls)
- Full user interaction loop and tool invocation system
- All production features and stability improvements

For development and deployment, use this file.
Note: main_refactored.py is experimental and not ready for production use.
"""

# CRITICAL: Suppress warnings BEFORE any imports that might trigger them
import sys
import os
import warnings
import contextlib

# --- COMPREHENSIVE WARNING SUPPRESSION ---
# Must be set BEFORE importing ANY libraries that might generate warnings
# These environment variables affect the entire Python process lifecycle
if '--debug' not in sys.argv:
    # Suppress gRPC/protobuf/abseil warnings (affects runtime, not just imports)
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    os.environ['GRPC_TRACE'] = ''
    os.environ['GRPC_PYTHON_LOG_LEVEL'] = 'ERROR'

    # Suppress TensorFlow/ML library warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Suppress Python warnings system-wide
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')

    # Additional library-specific suppression
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Hugging Face warnings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Transformers library

    # Create comprehensive stderr suppression context
    @contextlib.contextmanager
    def suppress_stderr_completely():
        """Completely suppress stderr output during context"""
        devnull = open(os.devnull, "w")
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr
            devnull.close()

    @contextlib.contextmanager
    def suppress_warnings_runtime():
        """Suppress warnings during runtime operations"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old_stderr = sys.stderr
            stderr_buffer = None
            # Redirect stderr for known noisy operations
            if '--debug' not in sys.argv:
                stderr_buffer = open(os.devnull, 'w')
                sys.stderr = stderr_buffer
            try:
                yield
            finally:
                if '--debug' not in sys.argv and stderr_buffer is not None:
                    stderr_buffer.close()
                sys.stderr = old_stderr
else:
    # Debug mode: show ALL warnings and verbose output
    warnings.filterwarnings('default')
    os.environ['GRPC_VERBOSITY'] = 'DEBUG'

    # Dummy context managers for debug mode
    @contextlib.contextmanager
    def suppress_stderr_completely():
        yield

    @contextlib.contextmanager
    def suppress_warnings_runtime():
        yield

# NOW safe to import other modules
import time
import logging
import re
import asyncio
import threading
import json
import argparse
import signal
import queue
from pathlib import Path
from termcolor import colored
from typing import Dict, List, Optional, Callable, Any
from pyfiglet import figlet_format

# Import with suppression in non-debug mode
if '--debug' not in sys.argv:
    with suppress_stderr_completely():
        from tools.viewpermissions import viewpermissions
else:
    from tools.viewpermissions import viewpermissions

start_time = time.time() # Global timer for startup and operations

# --- Global Interrupt Flag ---
# Used for signal-based interruption instead of KeyboardInterrupt exceptions
_interrupt_flag = threading.Event()

# --- Prompt Toolkit for Command Completion ---

# --- Project Path Setup ---
# Ensure all project modules can be imported correctly.
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Tool and Core Imports ---
from core.tool_context import tool_context
from tools.primitives.todos import todos
from tools.readfile import readfile
from tools.bash import bash

# Import TextResourceManager for general resource creation
try:
    from shared.utils.text_resource_manager import TextResourceManager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False
from core.globals import g
from core.llm_router import LlmRouter, StreamInterruptedException, UserInterruptedException
from core.chat import Chat, Role
from core.permissions.enhanced_permissions import enhanced_permission_prompt
from shared.common_utils import extract_blocks
import traceback
import linecache

# --- Suppress Noisy Library Warnings ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.modeling_utils")
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")

# Import command system from extracted module

# Import display utilities from extracted module
from shared.utils.display_utils import colorize_model_list, print_startup_summary, show_unified_bindings_display

# Import logging utilities from extracted module
from shared.utils.logging_utils import setup_logging, swap_to_simple_logging

# Import tool discovery from extracted module
from agent.core.tool_discovery import _generate_tool_documentation

# Import input handlers from extracted module
from agent.core.input_handlers import (
    handle_multiline_input, get_user_input_with_bindings, handle_screenshot_capture
)


# Command system functions moved to agent.core.command_system

# Command completion and prompt session moved to agent.core.command_system


# --- Signal Handling for Graceful Interruption ---
def setup_signal_handlers():
    """Setup signal handlers for graceful interruption during AI generation."""
    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) by setting interrupt flag and attempting immediate cancellation."""
        _interrupt_flag.set()
        print(colored("\n🛑 INTERRUPT SIGNAL RECEIVED - Cancelling AI generation...", "red", attrs=['bold']))

        # Break out of autonomous loops immediately
        if hasattr(g, 'autonomous_iteration_count'):
            g.autonomous_iteration_count = 999  # Force exit from autonomous mode
        if hasattr(g, 'total_autonomous_count'):
            g.total_autonomous_count = 999  # Force exit from autonomous mode

        # Set a global interrupt flag for the main loop
        g.USER_INTERRUPT_REQUESTED = True

        # Attempt immediate provider cancellation if available
        try:
            from core.llm_router import LlmRouter
            router = LlmRouter()
            if hasattr(router, 'last_used_model') and router.last_used_model:
                # Get the current model's provider
                current_model = router.get_model([router.last_used_model])
                if current_model:
                    cancel_func = getattr(current_model.provider, 'cancel_current_generation', None)
                    if callable(cancel_func):
                        cancel_func()
                        print(colored("  → Provider cancellation initiated", "yellow"))
        except Exception:
            # Don't let provider cancellation errors break the signal handler
            pass

        print(colored("  → Breaking out of autonomous mode and returning to user input...", "yellow"))

    signal.signal(signal.SIGINT, signal_handler)

def check_interrupt():
    """Check if interrupt was requested and clear the flag."""
    if _interrupt_flag.is_set():
        _interrupt_flag.clear()
        return True
    return False

def offer_intelligent_recovery_options(context_chat):
    """
    Analyze interrupt metadata and offer context-appropriate recovery options.
    Returns user choice for how to proceed after interrupt.
    """
    if not hasattr(context_chat, 'metadata') or 'last_interrupt' not in context_chat.metadata:
        return 'continue'

    interrupt_info = context_chat.metadata['last_interrupt']
    print(colored("\n🔄 Recovery Options:", "cyan", attrs=['bold']))

    # Analyze the interrupt context and offer relevant options
    options = []

    if interrupt_info.get('had_partial_response', False):
        options.append(("partial", "Continue with partial response (if meaningful content was generated)"))


    if interrupt_info.get('models_attempted', []):
        failed_models = interrupt_info['models_attempted']
        options.append(("retry_diff", f"Retry with different model (avoid {', '.join(failed_models[:2])}...)"))

    options.extend([
        ("retry_same", "Retry with same settings"),
        ("continue", "Continue normally"),
        ("exit", "Exit to user input")
    ])

    # Display options
    for i, (key, description) in enumerate(options, 1):
        print(f"  {i}. {description}")

    # Get user choice with timeout
    print(colored("\nChoose option (1-{}) or press Enter for default [continue]: ".format(len(options)), "cyan"), end="", flush=True)

    try:
        import select

        # Non-blocking input with 5-second timeout
        ready, _, _ = select.select([sys.stdin], [], [], 5.0)
        if ready:
            choice = input().strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    selected_key = options[idx][0]
                    print(colored(f"  → Selected: {options[idx][1]}", "green"))
                    return selected_key

        # Default choice after timeout
        print(colored("  → Using default: Continue normally", "yellow"))
        return 'continue'

    except Exception:
        # Fallback for systems without select support
        return 'continue'

# --- Background Input Collector ---
class InputCollector:
    """
    Background thread that collects user input without interrupting AI tool execution.
    Queues user commands in args.message for processing at natural loop boundaries.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.running = False
        self.thread = None
        self.input_queue = queue.Queue()

    def start(self):
        """Start the background input collection thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._collect_input, daemon=True)
        self.thread.start()
        logging.debug("Background input collector started")

    def stop(self):
        """Stop the background input collection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logging.debug("Background input collector stopped")

    def _collect_input(self):
        """Main input collection loop running in background thread."""
        while self.running:
            try:
                # Check different input sources based on mode
                if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
                    self._check_gui_input()
                else:
                    self._check_cli_input()

                # Process any queued input
                self._process_queued_input()

                # Small sleep to prevent excessive CPU usage
                time.sleep(0.1)

            except Exception as e:
                logging.debug(f"Input collector error: {e}")
                time.sleep(0.5)  # Longer sleep on error

    def _check_gui_input(self):
        """Check for GUI input from visual_input_queue."""
        if hasattr(g, 'visual_input_queue') and not g.visual_input_queue.empty():
            try:
                user_input = g.visual_input_queue.get_nowait()
                if user_input:
                    # Handle both string and dict inputs
                    if isinstance(user_input, str):
                        if user_input.strip():
                            self.input_queue.put(user_input.strip())
                    elif isinstance(user_input, dict):
                        # Dict objects should be handled by permissions system, not regular input
                        # Put it back for proper handling
                        g.visual_input_queue.put(user_input)
                    else:
                        logging.debug(f"Unexpected input type in visual queue: {type(user_input)}")
            except queue.Empty:
                pass

    def _check_cli_input(self):
        """Check for CLI input using non-blocking method."""
        # Skip CLI input monitoring if we're in GUI mode - it can interfere
        # The GUI mode will handle input through the unified input system which now
        # checks for queued commands from this collector
        if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
            return

        try:
            # Use select on Unix systems for non-blocking input
            import select
            import sys

            # Check if stdin is available and not redirected
            if not sys.stdin.isatty():
                return

            # Check if input is available on stdin
            if select.select([sys.stdin], [], [], 0.0)[0]:
                try:
                    # Read a line of input
                    user_input = sys.stdin.readline().strip()
                    if user_input:
                        self.input_queue.put(user_input)
                except (EOFError, KeyboardInterrupt):
                    pass

        except ImportError:
            # Windows doesn't have select, use msvcrt if available
            try:
                import msvcrt
                if hasattr(msvcrt, 'kbhit') and msvcrt.kbhit():  # type: ignore
                    # Read characters until we get a complete line
                    line = ""
                    while hasattr(msvcrt, 'kbhit') and msvcrt.kbhit():  # type: ignore
                        char = msvcrt.getch().decode('utf-8', errors='ignore') if hasattr(msvcrt, 'getch') else ''  # type: ignore
                        if char == '\r':  # Enter key
                            if line.strip():
                                self.input_queue.put(line.strip())
                            line = ""
                            break
                        elif char == '\b':  # Backspace
                            if line:
                                line = line[:-1]
                        elif ord(char) >= 32:  # Printable characters
                            line += char
            except ImportError:
                # Neither select nor msvcrt available, skip CLI monitoring
                pass
        except Exception:
            # Ignore any other errors in input detection
            pass

    def _process_queued_input(self):
        """Process any input in the queue and add to args.message."""
        while not self.input_queue.empty():
            try:
                user_input = self.input_queue.get_nowait()

                # Add to message queue for processing at next loop iteration
                if user_input:
                    self.args.message.append(user_input)

                    queue_size = len(self.args.message)
                    if queue_size == 1:
                        print(colored(f"\n⚡ Command ready for instant execution: {user_input}", "cyan"))
                    else:
                        print(colored(f"\n⚡ Command ready for instant execution: {user_input} (⏳ {queue_size} total queued)", "cyan"))

                    # Set interrupt flag to break out of waiting states immediately
                    self._set_command_interrupt()

            except queue.Empty:
                break

    def _set_command_interrupt(self):
        """Set flag to interrupt waiting states and process commands immediately."""
        # Set a flag that can be checked by waiting functions
        if not hasattr(self, '_command_interrupt'):
            self._command_interrupt = threading.Event()
        self._command_interrupt.set()

    def has_pending_commands(self):
        """Check if there are pending commands to process."""
        return bool(self.args.message)

    def get_command_interrupt_event(self):
        """Get the command interrupt event for waiting functions."""
        if not hasattr(self, '_command_interrupt'):
            self._command_interrupt = threading.Event()
        return self._command_interrupt

# Global input collector instance
_input_collector = None

def get_input_collector(args: Optional[argparse.Namespace] = None) -> Optional[InputCollector]:
    """Get or create the global input collector instance."""
    global _input_collector
    if _input_collector is None and args:
        _input_collector = InputCollector(args)
    return _input_collector

def _get_next_priority_message(message_list: List[str]) -> Optional[str]:
    """
    Get the next message to process, prioritizing urgent commands.

    Priority order:
    1. /exit, /stop - Immediate termination commands
    2. /help, /save, /clear, /reset - User assistance commands
    3. Other commands (starting with /)
    4. Regular messages (FIFO order)

    Returns:
        The next message to process, or None if no messages available
    """
    if not message_list:
        return None

    # Define priority levels
    urgent_commands = ['/exit', '/stop']
    high_priority_commands = ['/help', '/save', '/clear', '/reset', '/screenshot']

    # Check for urgent commands first
    for msg in message_list:
        if any(msg.startswith(cmd) for cmd in urgent_commands):
            return msg

    # Check for high priority commands
    for msg in message_list:
        if any(msg.startswith(cmd) for cmd in high_priority_commands):
            return msg

    # Check for any other commands
    for msg in message_list:
        if msg.startswith('/'):
            return msg

    # Return first regular message (FIFO)
    return message_list[0] if message_list else None

# --- Custom Logging Setup ---
def parse_cli_args() -> argparse.Namespace:
    """Setup and parse CLI arguments, ensuring the script's functionality remains intact."""

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

# ConsoleFormatter moved to shared.utils.logging_utils

# SimpleFormatter moved to shared.utils.logging_utils

# FileFormatter moved to shared.utils.logging_utils

# setup_logging function moved to shared.utils.logging_utils

# swap_to_simple_logging function moved to shared.utils.logging_utils

# --- End Custom Logging Setup ---


# Display utility functions moved to shared.utils.display_utils




# print_startup_summary function moved to shared.utils.display_utils

# --- Eager setup of args ---
# This is done before heavy imports to allow --help to work quickly.
args = parse_cli_args()

# Store continue flag in global state for access by tools
g.CONTINUE_CONVERSATION = args.c

# Import figlet for later use
from pathlib import Path


# --- Main Application Imports (Lazy/Normal) ---
# Imports are moved below arg parsing and logging setup to allow for faster startup
# and to ensure logging is configured before any modules start logging.
from dotenv import load_dotenv
import warnings

# --- FIX: Import new command system components ---
from agent.commands.command_handler import CommandHandler
from agent.commands.base_command import CommandResult
from agent.core.app_context import AppContext
# --- END FIX ---

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")
warnings.filterwarnings("ignore", message="words count mismatch on*", module="phonemizer", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")
warnings.filterwarnings("ignore", message="Unrecognized FinishReason enum value*", module="proto.marshal.rules.enums", category=UserWarning)

# Removed ComputationalNotebook import - using direct tool calls instead
from core.permissions.path_detector import PathDetector
from agent.text_painter import TextStreamPainter



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Lazy load functions from shared audio utils to avoid heavy imports at startup
def get_listen_microphone():
    from shared.utils_audio import listen_microphone  # noqa: E402 - Use improved Faster-Whisper with small model
    return listen_microphone

try:
    from shared.utils_tts import TtsUtil  # type: ignore
except ImportError:
    class TtsUtil:
        @staticmethod
        def run(text, **kwargs):
            logging.warning(f"TtsUtil not found. Cannot speak: {text}")
            return json.dumps({"status": "error", "message": "TtsUtil not found"})

def get_extract_blocks():
    """
    Returns the extract_blocks function for code block extraction.
    This function acts as a getter to maintain compatibility with existing code.
    """
    return extract_blocks

def auto_correct_html_for_xml(text: str) -> str:
    """
    Automatically corrects common HTML void tags to be XML-compliant.
    Finds tags like <img>, <br>, <hr>, <input> that are not self-closed
    and adds the closing slash.

    Args:
        text (str): The text potentially containing non-compliant HTML.

    Returns:
        str: Text with void tags corrected for XML parsing.
    """
    # This regex finds unclosed void tags (img, br, hr, input, etc.)
    # It looks for the tag name, followed by attributes, and ensures it doesn't already end with '/>'.
    # The negative lookbehind `(?<!/)` ensures we don't match already self-closed tags.
    void_tag_pattern = re.compile(r"<((?:img|br|hr|input|link|meta)[^>]*?)(?<!/)>", re.IGNORECASE)

    # The substitution function adds the self-closing slash
    def self_close_tag(match):
        tag_content = match.group(1).strip()
        return f"<{tag_content} />"

    corrected_text = void_tag_pattern.sub(self_close_tag, text)

    return corrected_text

def detect_malformed_xml(text: str) -> List[Dict[str, str]]:
    """
    Detects malformed XML blocks using the standard library XML parser.

    This function uses xml.etree.ElementTree to properly validate XML syntax
    and provides detailed error messages with line and column information.

    Args:
        text (str): The text to analyze for malformed XML

    Returns:
        List[Dict[str, str]]: List of detected malformed XML issues with detailed error info
    """
    import xml.etree.ElementTree as ET

    errors = []

    # Find potential XML blocks using regex
    xml_pattern = r'<([a-zA-Z0-9_]+)[^>]*>.*?</\1>'

    for match in re.finditer(xml_pattern, text, re.DOTALL):
        xml_block = match.group(0)
        tag_name = match.group(1)

        try:
            # Try to parse the XML block with ElementTree
            ET.fromstring(xml_block)
            # If we get here, the XML is well-formed

        except ET.ParseError as e:
            # Get the position of the error in the original text
            match_start = match.start()
            text_before_match = text[:match_start]
            line_number = text_before_match.count('\n') + 1

            # Calculate relative position within the XML block
            xml_lines = xml_block.split('\n')
            error_line_in_block = getattr(e, 'lineno', 1) or 1
            error_col_in_block = getattr(e, 'offset', 1) or 1

            # Get context around the error
            context_start = max(0, error_line_in_block - 2)
            context_end = min(len(xml_lines), error_line_in_block + 2)
            context_lines = xml_lines[context_start:context_end]

            # Build detailed error information
            error_info = {
                'type': 'xml_parse_error',
                'tag_name': tag_name,
                'error_description': f"XML parsing failed: {e.msg}",
                'line_number': line_number + (error_line_in_block - 1),
                'column_number': error_col_in_block,
                'error_line_content': xml_lines[error_line_in_block - 1] if error_line_in_block <= len(xml_lines) else "",
                'context_lines': context_lines,
                'full_xml_block': xml_block,
                'suggested_fix': _get_xml_fix_suggestion(e.msg, xml_block)
            }

            errors.append(error_info)

    return errors


def _get_xml_fix_suggestion(error_msg: str, xml_block: str) -> str:
    """
    Generate helpful fix suggestions based on common XML parsing errors.

    Args:
        error_msg (str): The error message from ElementTree
        xml_block (str): The malformed XML block

    Returns:
        str: A suggested fix for the XML error
    """
    error_lower = error_msg.lower()

    if "mismatched tag" in error_lower or "expected" in error_lower:
        return "Check that all opening tags have matching closing tags with the same name"
    elif "not well-formed" in error_lower:
        return "Ensure all tags are properly closed and nested correctly"
    elif "unclosed token" in error_lower:
        return "Add missing closing tag or check for unclosed angle brackets"
    elif "invalid token" in error_lower:
        return "Check for invalid characters in tag names or attribute values"
    elif "unexpected end of data" in error_lower:
        return "The XML block appears to be incomplete - add missing closing tags"
    else:
        return f"Fix the XML syntax error: {error_msg}"


def generate_xml_error_context(errors: List[Dict[str, str]]) -> str:
    """
    Generate a context block with XML error information for the model.

    Args:
        errors: List of XML errors from detect_malformed_xml()

    Returns:
        str: Formatted context block with error details and suggestions
    """
    if not errors:
        return ""

    context_lines = ["XML Syntax Error Detected:"]
    context_lines.append("")

    for i, error in enumerate(errors, 1):
        if error['type'] == 'xml_parse_error':
            context_lines.append(f"Error {i}: {error['error_description']}")
            context_lines.append(f"  Tag: <{error['tag_name']}>")
            context_lines.append(f"  Location: Line {error['line_number']}, Column {error['column_number']}")
            context_lines.append(f"  Fix: {error['suggested_fix']}")
            context_lines.append("")

            # Show the error line with context
            if error.get('error_line_content'):
                context_lines.append("Error line:")
                context_lines.append(f"  {error['error_line_content']}")
                context_lines.append("")

            # Show context lines around the error
            if error.get('context_lines'):
                context_lines.append("Context:")
                for line in error['context_lines']:
                    context_lines.append(f"  {line}")
                context_lines.append("")

        elif error['type'] == 'tag_mismatch':
            # Keep backward compatibility for any remaining old-style errors
            context_lines.append(f"Error {i}: {error['error_description']}")
            context_lines.append(f"  Found: <{error['opening_tag']}>...</{error['closing_tag']}>")
            context_lines.append(f"  Fix: {error['suggested_fix']}")
            context_lines.append("")

        elif error['type'] == 'incomplete_block':
            context_lines.append(f"Error {i}: {error['error_description']}")
            context_lines.append(f"  Fix: {error['suggested_fix']}")
            context_lines.append("")

    context_lines.append("Due to these XML syntax errors, the affected tool blocks were ignored.")
    context_lines.append("Please correct the XML syntax and try again.")

    return f"<context>\n{chr(10).join(context_lines)}\n</context>"


def apply_default_llm_config():
    """
    Loads the cached LLM configuration from disk and applies it as the default
    for the current session. This makes LLM selections persistent.
    """
    llm_config = g.get_llm_config()
    if not llm_config:
        return # No saved config to apply

    # Preserve chronological selection order by using g.SELECTED_LLMS if available
    # Add error handling for corrupted config data
    try:
        if g.SELECTED_LLMS:
            selected_llms = [key for key in g.SELECTED_LLMS if isinstance(llm_config.get(key, {}), dict) and llm_config.get(key, {}).get('selected')]
        else:
            # Get selected models and sort them by their saved order
            selected_models_with_order = []
            for key, data in llm_config.items():
                if isinstance(data, dict) and data.get('selected'):
                    order = data.get('order', 999)  # Default high order for models without order
                    selected_models_with_order.append((order, key))

            # Sort by order and extract just the model keys
            selected_models_with_order.sort(key=lambda x: x[0])
            selected_llms = [key for order, key in selected_models_with_order]
    except Exception as e:
        logging.warning(colored(f"⚠️  LLM config appears corrupted, skipping auto-selection: {e}", "yellow"))
        logging.info(colored(f"🔍 Config content: {llm_config}", "cyan"))

        # Attempt to clean up corrupted config
        try:
            logging.info(colored("🔧 Attempting to reset corrupted LLM config...", "cyan"))
            g.save_llm_config({})  # Reset to empty config
            logging.info(colored("✅ LLM config reset successfully", "green"))
        except Exception as reset_error:
            logging.error(colored(f"❌ Failed to reset LLM config: {reset_error}", "red"))

        selected_llms = []

    # Only apply if there are actual selections, to not override command line flags unintentionally
    if selected_llms:
        g.SELECTED_LLMS = selected_llms
        # Count unique models by provider
        unique_models = list(set(selected_llms))  # Deduplicate
        provider_counts = {}
        for model in unique_models:
            emoji, provider_name = get_provider_emoji_and_name(model)
            provider_key = f"{emoji} {provider_name}"
            provider_counts[provider_key] = provider_counts.get(provider_key, 0) + 1

        provider_summary = ", ".join([f"{provider}: {count}" for provider, count in provider_counts.items()])
        logging.info(f"📋 Loaded latest LLM configuration. Total: {colored(str(len(unique_models)), 'green')} models ({provider_summary})")


async def llm_selection(args: argparse.Namespace, preselected_llms: Optional[List[str]] = None) -> List[str]:
    """
    Handles the LLM selection process using the enhanced LlmSelector UI.
    This function initializes the selector, awaits user input, and processes
    the results to configure the agent's LLM settings.
    """
    from agent.llm_selection import LlmSelector
    selector = LlmSelector()

    # The selector now handles its own UI, logic, and persistence.
    result = await selector.get_selection(
        preselected_llms=preselected_llms,
        force_local=args.local,
        save_selection=True  # Let the selector handle saving the config
    )

    # --- Process the structured result from the selector ---
    if result.get("status") != "Success":
        logging.warning(colored(f"# cli-agent: {result.get('message', 'LLM selection cancelled or failed.')}", "yellow") )
        return []

    # Handle the special 'any_local' case
    if result.get("selection_type") == "any_local":
        g.FORCE_LOCAL = True
        args.local = True
        args.llm = None # No specific LLM is set
        logging.info(colored("# cli-agent: 'Any but local' option selected. Local mode enabled.", "green") )
        return []

    # Handle specific model selections
    selected_configs = result.get("selected_configs", [])

    selected_llms = [config['model_key'] for config in selected_configs]

    # Update globals with the latest selection
    args.llm = selected_llms
    g.SELECTED_LLMS = selected_llms

    # Reload the config in `g` so the main loop gets the fresh settings
    g.load_llm_config()

    # Load previously selected models from saved configuration
    try:
        config = g.get_llm_config()

        # Load selected fallback models from config (using unified roles structure)
        previously_selected = g.SELECTED_LLMS

        if previously_selected:
            args.llm = previously_selected
            logging.info(colored(f"📋 Loaded {len(previously_selected)} selected models from config: {', '.join(previously_selected[:3])}{'...' if len(previously_selected) > 3 else ''}", "cyan"))

    except Exception as e:
        logging.warning(colored(f"⚠️  Failed to load saved LLM configuration: {e}", "yellow"))

    if not args.llm:
        logging.info(colored("# cli-agent: LLM set to auto, type (-h) for info", "green") )
    else:
        # Count unique models by provider and configuration
        # Use selected_llms directly to preserve the exact selection order from LLM selector
        # Deduplicate while preserving order (maintain user's selection order)
        provider_counts = {}

        for model in selected_llms:
            emoji, provider_name = get_provider_emoji_and_name(model)
            provider_key = f"{emoji} {provider_name}"
            provider_counts[provider_key] = provider_counts.get(provider_key, 0) + 1

        # Always show fallback models for visibility, even if none are configured
        count = len(selected_llms)
        color = "light_green" if count > 0 else "red"
        logging.info(colored(f"  🔄 {count} Fallback Models:".ljust(27), color) + " " + colorize_model_list(selected_llms))

    return selected_llms



def create_interruption_callback(
    response_buffer: List[str],
    painter: TextStreamPainter,
    lock: Optional[asyncio.Lock] = None,
):
    """
    Factory to create a stateful callback for handling LLM streams.

    This callback prints the stream, accumulates it into a buffer, and can
    interrupt by raising StreamInterruptedException when a complete code block
    (e.g., ```...```) or a complete tag for a known tool (e.g., <bash>...</bash>) is detected.
    This logic is consistent with the `extract_blocks` utility.

    Args:
        response_buffer: A list with a single string to act as a mutable buffer.
        painter: The TextStreamPainter for coloring output.
        lock: An optional asyncio.Lock for concurrent-safe printing.

    Returns:
        An async callback function.
    """
    # Pre-compile regex for efficiency. re.DOTALL allows '.' to match newlines.
    # CODE_BLOCK_REGEX = re.compile(r"```.*?(```)", re.DOTALL)

    # Dynamically build a regex for registered tool blocks. This is more robust
    # and prevents premature interruption on non-tool XML tags like <filepath>.
    # Define tags that should NOT interrupt the stream, allowing the agent to output them fully before pausing.
    non_interrupting_tags = {'think', 'todos'}

    # Get all tool delimiters and add the final <response> tag.
    all_delimiters = list(getattr(g, 'tool_classes', {}).keys()) + ['response']

    # In GUI mode, also exclude 'response' from interrupting since we want visualresponse only
    if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
        non_interrupting_tags = non_interrupting_tags.union({'response'})

    # Filter this list to get only the tags that SHOULD cause an interruption.
    interrupting_delimiters = [d for d in all_delimiters if d not in non_interrupting_tags]

    # Build the regex from the filtered list of interrupting tags.
    tool_tag_pattern = "|".join(re.escape(d) for d in interrupting_delimiters)
    XML_TOOL_BLOCK_REGEX = re.compile(f"<({tool_tag_pattern})[^>]*>.*?</\\1>", re.DOTALL) if tool_tag_pattern else None

    async def _callback(chunk: str):
        # Check for user interrupt immediately at start of callback for faster response
        if check_interrupt():
            raise StreamInterruptedException(response_buffer[0])

        # Batch processing for better performance while maintaining interrupt responsiveness
        char_batch = []
        batch_size = min(50, len(chunk))  # Process in small batches for responsiveness

        for i, char in enumerate(chunk):
            # Check for user interrupt every 10 characters or at batch boundaries for efficiency
            if i % 10 == 0 and check_interrupt():
                raise StreamInterruptedException(response_buffer[0] + ''.join(char_batch) + char)

            char_batch.append(char)

            # Process batch when full or at special characters
            should_process_batch = (
                len(char_batch) >= batch_size or
                char in '`><' or
                i == len(chunk) - 1  # Last character
            )

            if should_process_batch:
                batch_str = ''.join(char_batch)

                # Print the batch
                if  g.DEBUG_MODE:
                    if lock:
                        async with lock:
                            print(painter.apply_color(batch_str), end="", flush=True)
                    else:
                        print(painter.apply_color(batch_str), end="", flush=True)

                # Append batch to shared buffer
                response_buffer[0] += batch_str
                current_buffer = response_buffer[0]

                # Clear batch
                char_batch = []

                # Check for interruption conditions when we encounter special characters
                if char in '`><':
                    # # Check for a complete Markdown code block
                    # code_match = CODE_BLOCK_REGEX.search(current_buffer)
                    # if code_match:
                    #     full_block = code_match.group(0).strip()
                    #     if not g.SUMMARY_MODE:
                    #         print("\n")
                    #     raise StreamInterruptedException(full_block)

                    # Check for a complete block of a known tool
                    if XML_TOOL_BLOCK_REGEX:
                        tool_match = XML_TOOL_BLOCK_REGEX.search(current_buffer)
                        if tool_match:
                            full_block = tool_match.group(0).strip()
                            raise StreamInterruptedException(full_block)

    return _callback

def get_provider_emoji_and_name(model_key: str) -> tuple[str, str]:
    """
    Get the emoji and provider name for a model based on its provider class.
    Returns a tuple of (emoji, provider_name)
    """
    # Handle local Ollama models (contain colon)
    if ':' in model_key:
        return "🏠", "Ollama"

    # Try to get provider from LLM router if available
    llm_obj = LlmRouter.get_model([model_key])
    if llm_obj and llm_obj.provider:
        provider_class_name = llm_obj.provider.__class__.__name__
        # Determine emoji based on provider characteristics
        if 'Ollama' in provider_class_name:
            emoji = "🏠"
        elif 'Human' in provider_class_name:
            emoji = "👤"
        else:
            emoji = "☁️  "
        # Clean provider name from class name
        provider_name = provider_class_name.replace('API', '').replace('Client', '').replace('Interface', '')
        return emoji, provider_name
    else:
        return "⚠️", "Not found"


# Note: The type hint for 'UtilsManager' is a string to avoid circular dependency issues.
# The full definition of other functions like get_extract_blocks, g, LlmRouter, etc.
# are assumed to exist elsewhere in the project.


async def get_cli_input(prompt: str = " Enter your request: ") -> str:
    """
    Get input from CLI interface using the unified input system.

    This function provides a simple interface for CLI input, delegating
    to the unified input management system.
    """
    try:
        from agent.core.input_compatibility import get_unified_input_manager
        from agent.core.input_manager import InputSource

        manager = await get_unified_input_manager()

        # Force CLI input source
        user_input, images = await manager.get_user_input(
            prompt=prompt,
            force_source=InputSource.CLI
        )

        return user_input

    except Exception as e:
        # Fallback to simple input() if unified system fails
        import logging
        logging.warning(f"Unified CLI input failed, using fallback: {e}")
        return input(prompt)




async def construct_engineered_prompt(voice_agent: str, gui_mode: bool = False) -> Chat:
    """
    Constructs the initial Chat object with a dynamically generated system prompt.
    This prompt includes the agent's core principles and a list of available tools.
    """
    voice_mode_text = 'The user is interacting with you in speech mode, please be aware of potential transcription issues. Provide concise responses and be helpful to the user.\n' if voice_agent else ''
    base_instructions = f'''**SYSTEM: Agent Protocol & Capabilities**

You are a sophisticated AI agent capable of thinking and using tools to achieve your todos. Your primary directive is to understand user requests, formulate a plan, and perform actions to achieve the current needs.
{voice_mode_text}
### Guiding Principles (Your Core Logic)
The following loop is a suggestion of how to behave effectively, you may freely reorder your usage of the xml tools if it fits your needs, you aim for long term stability and progress in your responses.
1.  **DECOMPOSE**: Analyze the user's request and break it down into the required steps. Start with: `<think>...</think>`.
2.  **PLAN & TRACK**: Display/update your <todos>...</todos> to align your next step(s) with the stated long term goal(s).'''

    step_4 = "4.  **OBSERVE & ATTEND**: After executing a tool, ALWAYS use <think> and <todos> to remember and update your plan based on your observations. When you have finished your tasks or need intermittent information from the user, please prompt him with <response>."

    # Get disabled tools from global state (set by visual interface)
    user_disabled_tools = getattr(g, 'disabled_tools', [])

    # In GUI mode, modify the tool documentation and response behavior
    if gui_mode:
        # Replace 'response' instructions with 'visualresponse'
        base_instructions = base_instructions.replace("response", "visualresponse")
        step_4 = step_4.replace("response", "visualresponse")
        # Combine built-in disabled tools with user-disabled tools
        disabled_tools = ["response"] + user_disabled_tools
        tool_docs = _generate_tool_documentation(disabled_tools)
    else:
        tool_docs = _generate_tool_documentation(user_disabled_tools)




    # Combine all parts into the final instruction message
    default_inst = f"{base_instructions}\n{tool_docs}\n{step_4}"

    # Create and configure the new Chat object
    context_chat = Chat(debug_title="Main Context Chat")
    context_chat.set_instruction_message(default_inst)

    return context_chat


async def reset_agent_state(args: argparse.Namespace, context_chat: Chat) -> Chat:
    """
    Completely reset the agent state to a fresh start.
    Clears all conversation history, tool context, images, and reconstructs the engineered prompt.
    """
    from core.globals import g

    logging.info(colored("🔄 Performing complete agent reset...", "yellow"))

    # Clear chat context
    if context_chat:
        context_chat.messages.clear()
        if hasattr(context_chat, 'base64_images'):
            context_chat.base64_images.clear()
        if hasattr(context_chat, 'metadata'):
            context_chat.metadata.clear()

    # Reset tool context
    tool_context.reset()
    # Clear global tool execution state
    if hasattr(g, 'last_executed_tools_info'):
        g.last_executed_tools_info = None

    # Clear reset flag
    g.AGENT_RESET_REQUESTED = False

    # Reconstruct fresh engineered prompt
    fresh_context_chat = await construct_engineered_prompt(args.voice, args.gui)

    # Clear visual interface
    try:
        from tools.visualresponse import visualresponse
        session = visualresponse._get_session()
        if session and session.is_healthy():
            clear_payload = {
                "id": f"reset-clear-{int(time.time() * 1000000)}",
                "action": "clear",
                "type": "system"
            }
            session.send_update(clear_payload)
            logging.debug("Sent clear command to visual interface")
    except Exception as e:
        logging.debug(f"Failed to clear visual interface: {e}")

    logging.info(colored("✅ Agent reset completed - fresh start initialized", "green"))
    return fresh_context_chat


async def save_current_llm_config_as_default():
    """Saves the current session's LLM selection as the new default."""
    from core.globals import g # Lazy import
    logging.info(colored("💾 Saving current LLM selection as the new default...", "magenta"))

    # Get the complete, most recent configuration data for ALL models
    full_config = g.get_llm_config()

    # Get the set of models active in THIS session
    currently_selected_models = set(g.SELECTED_LLMS)

    # Update the 'selected' status in the full config based on the current session
    for model_key, data in full_config.items():
        if isinstance(data, dict):
            data['selected'] = model_key in currently_selected_models

    # Save the modified configuration back to the persistent file
    g.save_llm_config(full_config)

    logging.info(colored(f"✅ Default configuration updated with {len(currently_selected_models)} selected models.", "green"))


async def main(  # type: ignore[misc]
    input_event: Optional[asyncio.Event] = None,
    input_lock: Optional[threading.Lock] = None,
    shared_input: Optional[Dict[str, Any]] = None,
    tui_callback: Optional[Callable] = None
) -> None:
    try:
        # In TUI mode, create default args instead of parsing CLI
        if input_event:  # TUI mode
            # Create default args for TUI mode
            global args, g
            args = argparse.Namespace(
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
                min=False,
                image=False,
                full_output_mode=False,
                exit=False,
                debug_chats=False,
                online=False
            )

            # Setup logging properly for TUI mode (output goes through print/stdout)
            setup_logging(args.debug, args.log_file, tui_mode=True)

            # Setup signal handlers for graceful interruption
            setup_signal_handlers()

            # Display header in TUI mode (stdout is redirected before agent starts)
            header_dashes = colored("# # # # # # # # # # # # # # # # # # # # # # # # # #", "blue")
            header = figlet_format("CLI-Agent", font="slant")
            print(header_dashes)
            print(colored(header, "cyan", attrs=["bold"]))
            print(header_dashes)
            logging.info("Starting CLI-Agent...")

        # If not in TUI mode, display the header and perform standard logging setup.
        elif not tui_callback:
            setup_logging(args.debug, args.log_file)
            g.DEBUG_MODE = args.debug

            # Setup signal handlers for graceful interruption
            setup_signal_handlers()

            header_dashes = colored("# # # # # # # # # # # # # # # # # # # # # # # # # #", "blue")
            header = figlet_format("CLI-Agent", font="slant")
            print(header_dashes)
            print(colored(header, "cyan", attrs=["bold"]))
            print(header_dashes)
            logging.info("Starting CLI-Agent...")

        load_dotenv(g.CLIAGENT_ENV_FILE_PATH)

        if not args.llm and not args.local:
            apply_default_llm_config()

        # LLM roles are automatically loaded by g.load_llm_roles() during globals initialization

        # Configure TTS engine priority based on --remote-tts flag
        if getattr(args, 'online_voice', False):
            # User explicitly requested remote TTS - prioritize API engines
            remote_priority = ["openai", "elevenlabs", "piper", "coqui", "huggingface", "system"]
            g.set_tts_engine_priority(remote_priority)
            if not tui_callback: logging.info(colored("# cli-agent: Online voice enabled - using cloud TTS and transcription APIs", "green"))

        stdout_buffer, stderr_buffer = "", ""
        def stdout_callback(text: str):
            nonlocal stdout_buffer
            print(text, end="")
            stdout_buffer += text
        def stderr_callback(text: str):
            nonlocal stderr_buffer
            print(colored(text, "red"), end="")
            stderr_buffer += text

        web_server = None
        if args.gui:
            # Set GUI mode flag for behavior modifications
            g.GUI_MODE = True

            # Auto-start visualresponse server for GUI mode (this opens the main interface)
            try:
                from tools.visualresponse import visualresponse
                logging.info(colored("🌐 Starting GUI mode - opening visualresponse interface...", "blue"))
                # Get a session which will start the server and open the browser
                session = visualresponse._get_session()
                logging.info(colored(f"✅ GUI mode active - interface available at http://127.0.0.1:{session.port}", "green"))

                # Enable console capture for real-time log streaming to web interface
                try:
                    from tools.visualresponse import enable_console_capture
                    enable_console_capture()
                    logging.info(colored("📺 Console capture enabled - logs will stream to web interface", "cyan"))
                except Exception as e:
                    logging.warning(colored(f"⚠️  Could not enable console capture: {e}", "yellow"))
            except ImportError as e:
                logging.error(colored(f"❌ Could not start visualresponse server: {e}", "red"))
            except Exception as e:
                logging.error(colored(f"❌ Error starting visualresponse: {e}", "red"))

        context_chat: Optional[Chat] = None
        if args.c or args.regenerate:
            try:
                context_chat = Chat.load_from_json()
                if args.regenerate:
                    if not tui_callback: logging.info(colored("Loading previous chat for regeneration.", "green"))
                    if not context_chat or len(context_chat.messages) < 2:
                        if not tui_callback: logging.critical(colored("# cli-agent: No sufficient chat history found.", "red") )
                        exit(1)
                    if context_chat.messages[-1][0] == Role.ASSISTANT:
                        context_chat.messages.pop()
                    if not tui_callback: logging.info(colored("# cli-agent: Will regenerate response.", "green"))
                else:
                    if not tui_callback: logging.info(colored("Continuing previous chat.", "green"))
            except FileNotFoundError:
                if args.regenerate:
                    if not tui_callback: logging.critical(colored("No previous chat found to regenerate. Exiting.", "red") )
                    exit(1)
                else:
                    if not tui_callback: logging.warning(colored("No previous chat found. Starting a new chat.", "yellow") )
                    context_chat = None

        base64_images: List[str] = []
        if args.image:
            if not tui_callback: logging.info(colored("# cli-agent: Taking screenshot due to --img flag...", "green"))
            base64_images.extend(await handle_screenshot_capture())
            args.image = False

        if tui_callback:
            tui_callback("Initializing agent...", "system")

        if args.local and not g.SELECTED_LLMS:
            if not tui_callback: logging.info(colored("# cli-agent: Local mode (-l) detected. Selecting all available local models.", "green"))
            local_models = LlmRouter.get_models(force_local=True)
            if local_models:
                g.SELECTED_LLMS = [model.model_key for model in local_models]
            else:
                if not tui_callback: logging.warning(colored("# cli-agent: Local mode enabled, but no local models were found.", "yellow"))

        if args.llm == "__select__":
            await llm_selection(args, preselected_llms=g.SELECTED_LLMS)
            args.llm = None
        elif args.llm:
            g.SELECTED_LLMS = [args.llm]


        is_tui = bool(tui_callback)
        print_startup_summary(args, tui_mode=is_tui)

        # Model discovery is now manual-only via 'u' key in LLM selection menu
        # if not tui_callback:
        #     logging.info(colored("  - Starting background model discovery...", "blue"))
        # else:
        #     logging.info(colored("  - Starting background model discovery...", "blue"))

        # g.start_background_model_discovery()

        # Store the callbacks globally for lazy notebook initialization
        g.stdout_callback = stdout_callback
        g.stderr_callback = stderr_callback
        g.tui_callback = tui_callback  # Store tui_callback as well for consistent logging

        # Populate tool info for startup display
        if not hasattr(g, 'loaded_tool_info') or not g.loaded_tool_info:
            # Get disabled tools from global state (set by visual interface)
            user_disabled_tools = getattr(g, 'disabled_tools', [])
            _generate_tool_documentation(user_disabled_tools)

        if tui_callback: tui_callback("✔ Initialization complete.", "system_ok")
        else:
            # Display tool loading information
            if hasattr(g, 'loaded_tool_info') and g.loaded_tool_info:
                tool_info = g.loaded_tool_info

                # Simple count display
                total_tools = tool_info['tool_count']
                failed_count = len(tool_info.get('failed_tools', []))
                skipped_count = len(tool_info.get('skipped_tools', []))

                status_parts = [f"{total_tools} tools loaded"]
                if failed_count > 0:
                    status_parts.append(f"{failed_count} failed")
                if skipped_count > 0:
                    status_parts.append(f"{skipped_count} disabled")

                tools_status = ", ".join(status_parts)
                logging.info(colored(f"Tools ready: {tools_status}.", "blue"))
            else:
                # Fallback if detailed tool info is not available
                logging.info(colored("Tools ready.", "blue"))

        if context_chat is None:
            context_chat = await construct_engineered_prompt(args.voice, args.gui)

        # Install visual input bridge for browser integration
        try:
            from agent.visual_input_bridge import install_visual_input_bridge  # type: ignore
            install_visual_input_bridge()
        except (ImportError, Exception) as e:
            logging.debug(f"Visual input bridge not available - browser integration disabled: {e}")

        if '-m' in sys.argv or '--message' in sys.argv:
            if not args.message:
                multiline_input = handle_multiline_input()
                if multiline_input.strip():
                    args.message.append(multiline_input)

        if not tui_callback:
            logging.info(colored("Agent Ready.", "magenta"))
            show_unified_bindings_display(gui_mode=args.gui)
            swap_to_simple_logging()

        # Start background input collector for real-time command queuing
        input_collector = get_input_collector(args)
        if input_collector:
            input_collector.start()

        user_interrupt = False
        if not (args.c or args.regenerate):
            context_chat.add_message(Role.USER, "Please list the 5 most recently modified files of the cwd")
            if (args.gui):
                bootstrap_message_1 = f"""<think>
To list the 5 most recently modified files in the current working directory (excluding directories), a combination of ls, grep, and head commands can be used.
</think><todos>
- [ ] List 5 most recently modified files.
</todos><bash>
ls -tp | grep -v /$ | head -5
</bash><context>
{await bash.run_silent("ls -tp | grep -v /$ | head -5")}
</context><todos>
- [x] List 5 most recently modified files.
</todos><visualresponse>
  <action>create</action>
  <type>markdown</type>
  <content>
### Most Recently Modified Files:
{await bash.run_silent("ls -tp | grep -v /$ | head -5 | sed 's/^/- /'")}
  </content>
</visualresponse>"""
                bootstrap_message_2 = """<visualresponse>
  <action>create</action>
  <type>markdown</type>
  <content>
I have successfully viewed my permissions. Is there anything else I can assist you with?
  </content>
</visualresponse>"""
            else:
                bootstrap_message_1 = f"""<think>
To list the 5 most recently modified files in the current working directory (excluding directories), a combination of ls, grep, and head commands can be used.
</think><todos>
- [ ] List 5 most recently modified files.
</todos><bash>
ls -tp | grep -v /$ | head -5
</bash><context>
{await bash.run_silent("ls -tp | grep -v /$ | head -5")}
</context><todos>
- [x] List 5 most recently modified files.
</todos><response>
The 5 most recently modified files have been listed. Would you like me to inspect any of them?
</response>"""
                bootstrap_message_2 = """<response>
I have successfully viewed my permissions. Is there anything else I can assist you with?
</response>"""

            context_chat.add_message(Role.ASSISTANT, bootstrap_message_1)
            permissions_str = viewpermissions.run()
            if ("No automatic permissions set" not in permissions_str):
                context_chat.add_message(Role.USER, "view your permissions")
                context_chat.add_message(Role.ASSISTANT, f"""<todos>
- [ ] View permissions.
</todos><viewpermissions>
</viewpermissions><context>
{permissions_str}
</context><todos>
- [x] View permissions.
</todos>""" + bootstrap_message_2)

        assistant_response = ""

        # --- FIX: Instantiate CommandHandler with proper AppContext ---
        app_context = AppContext()
        app_context.settings.load_from_args(args)
        app_context.logger = logging.getLogger("command_handler_context")
        command_handler = CommandHandler(app_context)
        # --- END FIX ---

        initializing: bool = True

        # CommandHandler already initialized above - no need to reinitialize

        while True:
            from core.llm_router import LlmRouter
            router = LlmRouter()
            router.failed_models.clear()
            g.LLM_STRENGTHS = []
            assistant_response = ""

            g.FORCE_LOCAL = args.local
            g.DEBUG_CHATS = args.debug_chats
            g.LLM = args.llm

            # --- UNIFIED INPUT GATHERING ---
            # All input sources are now handled within get_user_input_with_bindings
            user_input, new_images = await get_user_input_with_bindings(
                args, context_chat, force_input=user_interrupt,
                input_event=input_event, input_lock=input_lock, shared_input=shared_input
            )

            # Reset user_interrupt flag after successful input to allow queue processing
            user_interrupt = False

            # --- UNIFIED COMMAND & IMAGE PROCESSING ---
            if new_images:
                base64_images.extend(new_images)

            if user_input and user_input.startswith('/'):
                response = await command_handler.handle_command(
                    user_input, context_chat=context_chat, command_handler=command_handler
                )
                if response.result == CommandResult.EXIT: exit(0)
                if response.result == CommandResult.ERROR: logging.error(colored(f"# cli-agent: {response.message}", "red"))
                continue # Always get next input after a command

            # Skip adding system commands to chat history
            if (user_input) and not (
                (isinstance(user_input, dict) and user_input.get('type') == 'llm_selection_update') or
                (isinstance(user_input, str) and 'llm_selection_update' in user_input and user_input.strip().startswith('{')) or
                (isinstance(user_input, str) and user_input.strip() == '__AGENT_RESET__')
            ):
                context_chat.add_message(Role.USER, user_input)
            else:
                continue

            initializing = False
            last_action_signature: Optional[str] = None
            stall_counter: int = 0
            MAX_STALLS: int = 2
            text_stream_painter = TextStreamPainter()

            # Agent loop
            while True:
                # Check for interrupt at the start of each loop iteration
                if check_interrupt():
                    print(colored("\n🛑 Agent loop interrupted by user.", "red"))
                    break

                try:
                    if assistant_response:
                        context_chat.add_message(Role.ASSISTANT, assistant_response)
                        assistant_response = ""

                    if context_chat:
                        context_chat.save_to_json()

                    UNCOMPRESSED_HISTORY = 4 # Compresses reasoning traces after x messages
                    if (len(context_chat.messages) > UNCOMPRESSED_HISTORY):
                        for i in range(UNCOMPRESSED_HISTORY,UNCOMPRESSED_HISTORY+1): # do prompt engineering to the 7th and 8th messages automatically
                            role = context_chat.messages[-i][0]
                            msg = context_chat.messages[-i][1]
                            if "<think>" in msg:
                                # Extract content within <think> tags
                                think_content_match = re.search(r'<think>(.*?)</think>', msg, re.DOTALL)
                                if think_content_match:
                                    think_content = think_content_match.group(1).strip()
                                    # Find the first line until a newline
                                    first_line_match = re.match(r'^([^\n]*)', think_content)
                                    if first_line_match:
                                        first_line = first_line_match.group(1).strip()
                                        # Replace the original <think> content with only the first line
                                        msg = re.sub(r'<think>(.*?)</think>', f'<think>\n{first_line}\n</think>', msg, flags=re.DOTALL)
                            context_chat.messages[-i] = (role, msg)

                    llm_config = g.get_llm_config()

                    response_buffer_list = [""]
                    interruption_callback = create_interruption_callback(response_buffer_list, text_stream_painter)
                    current_todo = f"<current_todo>\n{todos().get_next_unchecked_task()}\n</current_todo>"
                    # Simple single model execution
                    try:
                        if check_interrupt():
                            user_interrupt = True
                            break

                        # Use router for complexity assessment if available
                        if g.router_llm:
                            difficulty_chat = Chat("Act as a complexity assessment assistant. You view the users provided context and respond with a percentage value from 1-100 indicating the difficulty of formulating the next assistant response.")
                            difficulty_chat.add_message(Role.USER, "Hi, please think about the following conversation and assess the difficulty of responding:\n" + context_chat.print_chat(-3, None, True))
                            # In local mode, don't force preferred models since they might be cloud models
                            router_preferred = [] if g.FORCE_LOCAL else [g.router_llm] + g.FALLBACK_MODELS
                            response = await LlmRouter.generate_completion(difficulty_chat, preferred_models=router_preferred, force_preferred_model=False, force_free=False, force_local=g.FORCE_LOCAL)
                            # extract last number found in the response string
                            response_complexity: int = 50 # Default fallback
                            match = re.findall(r'\d+', response)
                            if match:
                                try:
                                    response_complexity = int(match[-1])
                                except ValueError:
                                    logging.warning(f"Could not convert '{match[-1]}' to int for complexity score. Using default.")

                            if response_complexity > 85:
                                selected_models = [g.planner_llm] + g.FALLBACK_MODELS if g.planner_llm else g.FALLBACK_MODELS
                            elif response_complexity > 20:
                                selected_models = g.FALLBACK_MODELS
                            else:
                                selected_models = [g.router_llm] + g.FALLBACK_MODELS
                        else:
                            selected_models = g.FALLBACK_MODELS

                        # Generate single response
                        # In local mode, don't force preferred models since they might be cloud models
                        main_preferred = [] if g.FORCE_LOCAL else selected_models
                        # Add the thinking prefix to the chat directly
                        context_chat.add_message(Role.ASSISTANT, "<think>")
                        assistant_response = await LlmRouter.generate_completion(
                            chat=context_chat,
                            preferred_models=main_preferred,
                            force_preferred_model=not g.FORCE_LOCAL,
                            force_local=g.FORCE_LOCAL,
                            base64_images=base64_images,
                            generation_stream_callback=interruption_callback
                        )

                        # Store current model for tool execution
                        g.current_llm_model = LlmRouter.get_last_used_model()

                    except UserInterruptedException:
                        # User interrupted model generation with Ctrl+C - handle gracefully with state preservation
                        print(colored("\n⚠️  Model generation interrupted by user (Ctrl+C)", "yellow", attrs=['bold']))

                        # Immediate chat state preservation
                        try:
                            context_chat.save_to_json()
                            print(colored("  → Chat context saved successfully", "green"))
                        except Exception as e:
                            print(colored(f"  ⚠️  Warning: Could not save chat context: {e}", "yellow"))

                        # Store interrupt metadata for intelligent recovery
                        if hasattr(context_chat, 'metadata'):
                            import time
                            # Get the model used for the attempt
                            model_attempted = selected_models[0] if 'selected_models' in locals() and selected_models else "unknown"
                            context_chat.metadata['last_interrupt'] = {
                                'timestamp': time.time(),
                                'had_partial_response': bool(assistant_response),
                                'models_attempted': [model_attempted]
                            }

                        user_interrupt = True
                        break
                    except StreamInterruptedException as e:
                        # Normal tool block detection - continue processing with the detected block
                        if g.DEBUG_MODE:
                            print(colored(f"\n🔧 Tool block detected: {e.response[:50]}{'...' if len(e.response) > 50 else ''}", "cyan"))
                        assistant_response = e.response
                        # Continue normal flow with the extracted tool block

                        # Add model information to previously stored tool execution info
                        if hasattr(g, 'last_executed_tools_info') and g.last_executed_tools_info:
                            used_model = LlmRouter.get_last_used_model()
                            for tool_info in g.last_executed_tools_info:
                                tool_info['model'] = used_model
                            logging.debug(f"Updated execution info with model: {used_model}")

                        base64_images = []

                    except Exception as e:
                        print(colored(f"An Exception was raised, e: {e}", "red"))
                        traceback.print_exc()
                        break

                    if not assistant_response:
                        break

                    if last_action_signature and assistant_response == last_action_signature:
                        stall_counter += 1
                    else:
                        stall_counter = 0
                    last_action_signature = assistant_response

                    if stall_counter >= MAX_STALLS:
                        print(colored("The agent is spamming its last tool use. Aborting...", "red"))
                        stall_counter, last_action_signature = 0, None
                        break

                    context_chat.add_message(Role.ASSISTANT, assistant_response)
                    # Note: AI response to visual interface will be sent after tool execution
                    # to properly detect if visualresponse was used

                    # Display the response in summary mode (since streaming doesn't show it)
                    if assistant_response.strip():
                        print(assistant_response)

                    # --- INTELLIGENT CORRECTION & VALIDATION ---
                    # 1. Proactively auto-correct common HTML tags to be XML-compliant
                    corrected_response = auto_correct_html_for_xml(assistant_response)
                    if corrected_response != assistant_response:
                        logging.debug(colored("🔧 Auto-corrected HTML tags for XML compliance.", "cyan"))
                        assistant_response = corrected_response

                    # 2. Check for any remaining malformed XML
                    xml_errors = detect_malformed_xml(assistant_response)
                    if xml_errors:
                        # Generate context error message and add to chat
                        error_context = generate_xml_error_context(xml_errors)
                        context_chat.add_message(Role.ASSISTANT, error_context)
                        assistant_response = ""
                        print(colored("⚠️  XML syntax errors detected and reported to model", "yellow"))
                        continue  # Skip tool execution and get corrected response

                    # --- NEW: Generic Tool Execution Loop ---
                    # Find all unique tool tags present in the assistant's response
                    tool_classes = getattr(g, 'tool_classes', {})
                    tool_tags_found = []
                    for tool_name, tool_class in tool_classes.items():
                        # Use a more robust regex to find any instance of the tool tag
                        if re.search(f'<{tool_class.get_delim()}[> ]', assistant_response, re.DOTALL):
                            tool_tags_found.append(tool_class)

                    # If any tool tags are found, process them
                    if tool_tags_found:

                        # --- Enhanced Path-Aware Permission Checking ---
                        # Check permissions using path detection for tools with file operations
                        should_execute = False
                        rejection_reason = []  # Initialize rejection tracking
                        if len(tool_tags_found) == 1 and any(k in (tool_tags_found[0].__module__ or "") for k in ["todos", "think", "searchweb", "grepsearch"]):
                            should_execute = True
                        elif len(tool_tags_found) == 1 and tool_tags_found[0].get_delim() == "bash":
                            # Check for sleep commands - auto-allow them like the bash tool already handles them specially
                            sleep_match = re.search(r'<bash>\s*sleep\s+(\d+)\s*</bash>', assistant_response, re.IGNORECASE | re.DOTALL)
                            if sleep_match:
                                should_execute = True

                            # Check for safe read-only commands - auto-allow them without permissions
                            if not should_execute:
                                safe_command_patterns = [
                                    r'<bash>\s*ls(\s+[^;&|`$()]*)*\s*</bash>',    # ls with safe flags
                                    r'<bash>\s*pwd\s*</bash>',                     # pwd
                                    r'<bash>\s*whoami\s*</bash>',                  # whoami
                                    r'<bash>\s*date(\s+[^;&|`$()]*)*\s*</bash>',   # date
                                    r'<bash>\s*hostname\s*</bash>',                # hostname
                                    r'<bash>\s*uname(\s+-[a-zA-Z]*)*\s*</bash>',   # uname with flags
                                ]

                                if any(re.search(pattern, assistant_response, re.IGNORECASE | re.DOTALL)
                                       for pattern in safe_command_patterns):
                                    should_execute = True
                                # If command doesn't match safe patterns, fall through to general permission system
                        if not should_execute and len(tool_tags_found) > 0:
                            # Analyze each tool for path-based permissions
                            permission_needed = False
                            tool_analyses = []

                            for tool in tool_tags_found:
                                delim = tool.get_delim()
                                blocks = get_extract_blocks()(assistant_response, [delim])

                                for block in blocks:
                                    analysis = PathDetector.analyze_tool_block(delim, block)
                                    tool_analyses.append(analysis)

                                    if analysis['needs_permission']:
                                        permission_needed = True

                            if not permission_needed:
                                # No permission needed for safe tools
                                should_execute = True
                            else:
                                # Check permissions for each tool that needs it
                                should_execute = True  # Start optimistic

                                for analysis in tool_analyses:
                                    if not analysis['needs_permission']:
                                        continue

                                    # Use enhanced permission system for path-aware tools
                                    permission_granted = await enhanced_permission_prompt(
                                        args,
                                        tool_name=analysis['tool_name'],
                                        action_description=analysis['action_description'],
                                        file_path=analysis['primary_path'],
                                        command_hash=analysis['command_hash'],
                                        command_pattern=analysis['command_pattern'],
                                        has_paths=analysis['has_paths'],
                                        xml_content=analysis.get('xml_content', ''),
                                        input_event=input_event,
                                        input_lock=input_lock,
                                        shared_input=shared_input,
                                        get_user_input_func=get_user_input_with_bindings
                                    )

                                    if not permission_granted:
                                        logging.info(colored("⛔ Code execution denied", "green"))
                                        # Store rejection details for informative user message
                                        if 'rejection_reason' not in locals():
                                            rejection_reason = []
                                        rejection_details = {
                                            'tool': analysis['tool_name'],
                                            'action': analysis['action_description'],
                                            'paths': analysis.get('detected_paths', []),
                                            'risk_level': analysis.get('risk_level', 'unknown'),
                                            'xml_content': analysis.get('xml_content', '')[:100] + '...' if len(analysis.get('xml_content', '')) > 100 else analysis.get('xml_content', '')
                                        }
                                        rejection_reason.append(rejection_details)

                                        should_execute = False
                                        break
                                    logging.info(colored("✅ Code execution permitted", "green"))

                        if should_execute:
                            try:
                                tool_results = []
                                executed_tools_info = []  # Track tool execution info for visual interface
                                for tool_class in tool_tags_found:
                                    tool_name = tool_class.get_delim()
                                    blocks = get_extract_blocks()(assistant_response, [tool_name])

                                    if not blocks:
                                        continue

                                    # --- FIX: Execute only the first block for each tool to prevent duplicate execution ---
                                    first_block = blocks[0]
                                    if len(blocks) > 1:
                                        logging.warning(colored(f"⚠️  Duplicate '{tool_name}' blocks detected. Only the first was executed.", "yellow"))

                                    # Track tool execution timing and status
                                    import time
                                    tool_start_time = time.time()
                                    tool_status = "success"
                                    tool_error = None

                                    try:
                                        # Send live tool start notification to loading area header (skip visualresponse)
                                        if hasattr(g, 'GUI_MODE') and g.GUI_MODE and tool_name != 'visualresponse':
                                            try:
                                                from tools.visualresponse import visualresponse
                                                session = visualresponse._get_session()
                                                if session and session.is_healthy():
                                                    live_payload = {
                                                        "id": f"loading-tools-{int(time.time() * 1000000)}",
                                                        "action": "loading_tool_start",
                                                        "tool_name": tool_name,
                                                        "tool_args": first_block.strip() if first_block else "",
                                                        "timestamp": time.time()
                                                    }

                                                    # Add model key metadata to tool start notifications
                                                    try:
                                                        from core.llm_router import LlmRouter
                                                        used_model = LlmRouter.get_last_used_model()
                                                        if used_model:
                                                            live_payload['model'] = used_model
                                                    except Exception as e:
                                                        logging.debug(f"Could not get model key for tool start: {e}")
                                                    session.send_update(live_payload)
                                                    logging.debug(f"Sent loading tool start notification for {tool_name}")
                                            except Exception as e:
                                                logging.debug(f"Failed to send loading tool start notification: {e}")

                                        # All tools now share the same simple run signature.
                                        # Suppress runtime warnings from tools (gRPC/SSL/TF) unless debug mode
                                        with suppress_warnings_runtime():
                                            tool_result = tool_class.run(first_block)

                                        # Handle both sync and async tools with proper error handling
                                        try:
                                            if asyncio.iscoroutine(tool_result):
                                                tool_result = await tool_result
                                        except Exception as e:
                                            tool_status = "error"
                                            tool_error = str(e)
                                            logging.error(colored(f"Error awaiting tool {tool_name}: {e}", "red"))
                                            tool_result = f"Error executing {tool_name}: {str(e)}"

                                        # Ensure tool_result is always a string, never a coroutine
                                        if asyncio.iscoroutine(tool_result):
                                            tool_status = "error"
                                            tool_error = "returned unresolved coroutine"
                                            logging.error(colored(f"Tool {tool_name} still returned coroutine after await - converting to error", "red"))
                                            tool_result = f"Error: Tool {tool_name} returned unresolved coroutine"

                                    except Exception as e:
                                        tool_status = "error"
                                        tool_error = str(e)
                                        logging.error(colored(f"Error executing tool {tool_name}: {e}", "red"))
                                        tool_result = f"Error executing {tool_name}: {str(e)}"

                                    # Check if tool result contains error (for tools that return error JSON instead of raising exceptions)
                                    if tool_status == "success" and tool_result:
                                        try:
                                            # Try to parse result as JSON to check for error key
                                            import json
                                            result_json = json.loads(tool_result)
                                            if isinstance(result_json, dict) and "error" in result_json:
                                                tool_status = "error"
                                                tool_error = result_json["error"]
                                                logging.debug(f"Tool {tool_name} returned error in result: {tool_error}")
                                        except (json.JSONDecodeError, TypeError):
                                            # If not valid JSON, keep original status
                                            pass

                                    # Calculate execution time
                                    tool_duration_ms = int((time.time() - tool_start_time) * 1000)

                                    # Send live tool completion notification to loading area header (skip visualresponse)
                                    if hasattr(g, 'GUI_MODE') and g.GUI_MODE and tool_name != 'visualresponse':
                                        try:
                                            from tools.visualresponse import visualresponse
                                            session = visualresponse._get_session()
                                            if session and session.is_healthy():
                                                live_payload = {
                                                    "id": f"loading-tools-{int(time.time() * 1000000)}",
                                                    "action": "loading_tool_complete",
                                                    "tool_name": tool_name,
                                                    "tool_status": tool_status,
                                                    "tool_duration_ms": tool_duration_ms,
                                                    "tool_error": tool_error,
                                                    "tool_result": tool_result[:500] if tool_result else "",  # Truncate for header display
                                                    "timestamp": time.time()
                                                }

                                                # Add model key metadata to tool completion notifications
                                                try:
                                                    from core.llm_router import LlmRouter
                                                    used_model = LlmRouter.get_last_used_model()
                                                    if used_model:
                                                        live_payload['model'] = used_model
                                                except Exception as e:
                                                    logging.debug(f"Could not get model key for tool completion: {e}")
                                                session.send_update(live_payload)
                                                logging.debug(f"Sent loading tool completion notification for {tool_name}")
                                        except Exception as e:
                                            logging.debug(f"Failed to send loading tool completion notification: {e}")

                                    # Store tool execution info for visual interface
                                    tool_info = {
                                        "name": tool_name,
                                        "status": tool_status,
                                        "duration_ms": tool_duration_ms,
                                        "error": tool_error,
                                        "args": first_block.strip() if first_block else "",
                                        "context_returned": tool_result.strip() if tool_result else "",
                                        "context_length": len(str(tool_result)) if tool_result else 0
                                    }

                                    # Add model key to tool info at creation time
                                    try:
                                        # Use stored current model from LLM response
                                        if hasattr(g, 'current_llm_model') and g.current_llm_model:
                                            tool_info["model"] = g.current_llm_model
                                        else:
                                            # Fallback to get_last_used_model
                                            used_model = LlmRouter.get_last_used_model()
                                            tool_info["model"] = used_model if used_model else "unknown"
                                    except Exception as e:
                                        logging.debug(f"Could not get model for tool {tool_name}: {e}")
                                        tool_info["model"] = "unknown"

                                    executed_tools_info.append(tool_info)

                                    # Wrap tools returned values as tool_ouput
                                    if tool_name and tool_name not in ["todos"] and tool_result:
                                        tool_results.append(tool_result)

                                returned_context = ""
                                for i, context in enumerate(tool_results):
                                    context = re.sub(r'\x1b\[[0-9;]*m', '', context) #  remove ANSI escape codes (also known as terminal escape sequences)
                                    if len(context) > readfile.CHUNK_LIMIT:
                                        # Try to create a resource instead of truncating
                                        if RESOURCE_MANAGER_AVAILABLE:
                                            try:
                                                # Get tool name from executed_tools_info
                                                tool_name = "unknown"
                                                if i < len(executed_tools_info):
                                                    tool_name = executed_tools_info[i].get('tool_name', 'unknown')

                                                # Create resource with timestamp for uniqueness
                                                import time
                                                timestamp = int(time.time())
                                                resource_id = f"tool_output_{tool_name}_{timestamp}_{i}"

                                                resource_manager = TextResourceManager._get_manager()
                                                resource_manager.create_resource(
                                                    content=context,
                                                    source=f"{tool_name}_output",
                                                    source_type="tool_output",
                                                    resource_id=resource_id
                                                )

                                                lines_count = len(context.splitlines())
                                                char_count = len(context)

                                                context = f"""Tool output from '{tool_name}' is large ({lines_count:,} lines, {char_count:,} characters) - stored as searchable resource.

Resource ID: {resource_id}

Use <searchtext><resource_id>{resource_id}</resource_id><pattern>YOUR_PATTERN</pattern></searchtext> to search this content.
Use <searchtext><resource_id>{resource_id}</resource_id><lines>1-50</lines></searchtext> to view specific line ranges.

Examples:
- Search for errors: <searchtext><resource_id>{resource_id}</resource_id><pattern>error|failed|exception</pattern><context_lines>2</context_lines></searchtext>
- View beginning: <searchtext><resource_id>{resource_id}</resource_id><lines>1-20</lines></searchtext>
- View end: <searchtext><resource_id>{resource_id}</resource_id><lines>{max(1, lines_count-19)}-{lines_count}</lines></searchtext>"""
                                            except Exception as e:
                                                # Fallback to truncation if resource creation fails
                                                chunk_size = readfile.CHUNK_LIMIT//4
                                                # Find closest newline before chunk_size (exclusive)
                                                start_pos = context.rfind('\n', 0, chunk_size)
                                                if start_pos == -1:  # No newline found, use beginning
                                                    start_pos = 0
                                                else:
                                                    start_pos += 1  # Exclude the newline itself

                                                # Find closest newline after -chunk_size (exclusive)
                                                end_search_pos = len(context) - chunk_size
                                                end_pos = context.find('\n', end_search_pos)
                                                if end_pos == -1:  # No newline found, use end
                                                    end_pos = len(context)

                                                context = context[:start_pos] + f"...output truncated (resource creation failed: {e})...\n" + context[end_pos:]
                                        else:
                                            # Fallback to original truncation when resource manager not available
                                            chunk_size = readfile.CHUNK_LIMIT//4
                                            # Find closest newline before chunk_size (exclusive)
                                            start_pos = context.rfind('\n', 0, chunk_size)
                                            if start_pos == -1:  # No newline found, use beginning
                                                start_pos = 0
                                            else:
                                                start_pos += 1  # Exclude the newline itself

                                            # Find closest newline after -chunk_size (exclusive)
                                            end_search_pos = len(context) - chunk_size
                                            end_pos = context.find('\n', end_search_pos)
                                            if end_pos == -1:  # No newline found, use end
                                                end_pos = len(context)

                                            context = context[:start_pos] + "...output truncated - TextResourceManager not available...\n" + context[end_pos:]
                                    returned_context += f"<context>\n{context}\n</context>"

                                # Store tool execution info in global context for visual response
                                if executed_tools_info:
                                    # Store tool execution info (model info will be added after LLM response)
                                    g.last_executed_tools_info = executed_tools_info
                                    logging.debug(f"Stored execution info for {len(executed_tools_info)} tools")

                                # Note: Automatic AI responses removed - only explicit visualresponse tool calls appear in interface
                                if (len(returned_context) > 0):
                                    context_chat.add_message(Role.ASSISTANT, returned_context)

                                    # Check for images from tool context and add them to the chat
                                    if tool_context.has_images():
                                        context_chat.base64_images.extend(tool_context.get_images())
                                        logging.info(colored(f"📸 Added {len(tool_context.get_images())} image(s) to chat context for vision analysis", "blue"))
                                        tool_context.clear_images()  # Clear images after adding them
                                    assistant_response = ""
                                    if ("response" in tool_name and "error" not in returned_context.lower()):
                                        # Break and wait for user input instead of continuing the loop
                                        break
                                continue # Continue the agent loop for the next action
                            except Exception:
                                error_output = f"<context>\n{traceback.format_exc()}\n</context>"
                                context_chat.add_message(Role.ASSISTANT, error_output)
                                assistant_response = ""
                                continue # Continue the agent loop for the next action
                        else:
                            # Build detailed rejection message
                            if 'rejection_reason' in locals() and rejection_reason:
                                rejection_details = []
                                for reason in rejection_reason:
                                    tool_detail = f"Tool: {reason['tool']}"
                                    action_detail = f"Action: {reason['action']}"
                                    risk_detail = f"Risk Level: {reason['risk_level']}"

                                    details = [tool_detail, action_detail, risk_detail]

                                    if reason['paths']:
                                        paths_detail = f"Paths: {', '.join(reason['paths'][:3])}" + ("..." if len(reason['paths']) > 3 else "")
                                        details.append(paths_detail)

                                    if reason['xml_content']:
                                        content_detail = f"Content: {reason['xml_content']}"
                                        details.append(content_detail)

                                    rejection_details.append(" | ".join(details))

                                detailed_reason = "\n".join(f"  - {detail}" for detail in rejection_details)
                                cancellation_notice = f"<context>\nCode execution rejected by user or guard.\n\nRejection Details:\n{detailed_reason}\n</context>"
                            else:
                                cancellation_notice = "<context>\nCode execution rejected by user or guard.\n</context>"

                            context_chat.add_message(Role.ASSISTANT, cancellation_notice)
                            assistant_response = ""
                            continue # Continue the agent loop for the next action

                    else:
                        # No tools or code found, this is a conversational response.
                        break # Ask for user input
                except (KeyboardInterrupt, UserInterruptedException):
                    user_interrupt = True
                    # This will now gracefully break the agent's inner loop
                    # and return to the user input prompt.
                    break
                except Exception as e:
                    if "ctrl+c" not in str(e).lower():
                        # Get detailed traceback information
                        tb = traceback.extract_tb(e.__traceback__)

                        # Enhanced error message with context
                        error_msg = f"An unexpected error occurred in the agent loop: {e}"

                        # Add specific information for NoneType iteration errors
                        if "argument of type 'NoneType' is not iterable" in str(e):
                            error_msg += "\n🔍 NoneType Iteration Error Details:"

                            # Find the exact line that caused the error
                            for frame in tb:
                                filename = frame.filename
                                line_number = frame.lineno
                                function_name = frame.name
                                line_text = frame.line or linecache.getline(filename, line_number).strip()

                                if any(op in line_text for op in [' in ', '.startswith(', '.endswith(', ' not in ']):
                                    error_msg += f"\n  📁 File: {filename}"
                                    error_msg += f"\n  📍 Line {line_number} in function '{function_name}'"
                                    error_msg += f"\n  💥 Code: {line_text}"

                                    # Try to identify which variable is None
                                    if ' in ' in line_text:
                                        parts = line_text.split(' in ')
                                        if len(parts) >= 2:
                                            suspect_var = parts[1].split()[0].strip('()[]{}')
                                            error_msg += f"\n  🎯 Suspected None variable: {suspect_var}"
                                    elif '.startswith(' in line_text or '.endswith(' in line_text:
                                        var_name = line_text.split('.')[0].strip()
                                        error_msg += f"\n  🎯 Suspected None variable: {var_name}"
                                    break

                        # Always show the full stack trace in debug mode or for critical errors
                        if args.debug or "NoneType" in str(e):
                            error_msg += "\n\n📋 Full Stack Trace:"
                            for i, frame in enumerate(tb):
                                error_msg += f"\n  {i+1}. {frame.filename}:{frame.lineno} in {frame.name}()"
                                if frame.line:
                                    error_msg += f"\n     → {frame.line.strip()}"

                        logging.critical(colored(error_msg, "red"))

                        # Also log with exc_info for complete traceback in debug mode
                        if args.debug:
                            logging.critical("Complete exception details:", exc_info=True)
                    user_interrupt = True
                    break # Ask for user input

            else:
                print("# cli-agent: No conversation to compact.")

            if args.exit and not args.message:
                # Check if there are active visual response servers before exiting
                try:
                    from tools.visualresponse import visualresponse
                    if (hasattr(visualresponse, '_session') and
                        visualresponse._session and
                        visualresponse._session.process and
                        visualresponse._session.process.is_alive()):

                        # Visual server is active - wait for user to close it
                        server_url = f"http://127.0.0.1:{visualresponse._session.port}"
                        print(colored(f"\n🌐 Visual response server is running at {server_url}", "cyan"))
                        print(colored("Press Ctrl+C to exit when you're done viewing the content.", "yellow"))

                        try:
                            # Keep the process alive until user interrupts
                            while (visualresponse._session.process and
                                   visualresponse._session.process.is_alive()):
                                import time
                                time.sleep(1)
                        except KeyboardInterrupt:
                            print(colored("\n🛑 Shutting down visual server...", "yellow"))
                            visualresponse.close_session()

                except ImportError:
                    pass  # Visual response tool not available

                exit(0)

    except (asyncio.CancelledError, KeyboardInterrupt):
        logging.warning(colored("CLI-Agent was interrupted. Shutting down gracefully...", "yellow") )
        _perform_graceful_shutdown()
    except Exception as e:
        if not isinstance(e, StreamInterruptedException):
            logging.critical(colored(f"CLI-Agent encountered a fatal error: {e}", "red"), exc_info=True)
        _perform_graceful_shutdown()


def _perform_graceful_shutdown():
    """Perform graceful shutdown including log cleanup."""
    # Stop input collector
    input_collector = get_input_collector()
    if input_collector:
        input_collector.stop()

    try:
        from shared.utils.log_manager import get_log_manager
        log_manager = get_log_manager()
        shutdown_stats = log_manager.shutdown()

        if shutdown_stats['log_cleanup']['files_removed'] > 0 or shutdown_stats['audio_cleanup']['files_removed'] > 0:
            total_files = shutdown_stats['log_cleanup']['files_removed'] + shutdown_stats['audio_cleanup']['files_removed']
            total_bytes = shutdown_stats['log_cleanup']['bytes_freed'] + shutdown_stats['audio_cleanup']['bytes_freed']
            logging.info(colored(f"🧹 Final cleanup: {total_files} files removed, {total_bytes / 1024:.1f} KB freed", "cyan"))

    except ImportError:
        pass  # Log manager not available
    except Exception as e:
        logging.debug(f"Error during graceful shutdown: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(colored("\nCLI-Agent was interrupted by user. Shutting down...", "yellow") )
        _perform_graceful_shutdown()
    except Exception as e:
        if not isinstance(e, StreamInterruptedException):
            print(colored(f"\nCLI-Agent encountered a fatal error during startup: {e}", "red"))
            traceback.print_exc()
        _perform_graceful_shutdown()
