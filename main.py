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

# --- EARLY DETECTION: Isolated Mode ---
# Check for --isolated flag BEFORE any heavy imports to enable fast Docker handoff
if '--isolated' in sys.argv:
    # Add project root to path for imports
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import and run Docker orchestrator (this will exec and never return)
    from shared.utils.docker_runner import run_in_isolated_container
    run_in_isolated_container(sys.argv)
    # Never reaches here - process is replaced with docker run

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
from typing import Dict, List, Optional, Callable, Any, Tuple
from pyfiglet import figlet_format

# Import with suppression in non-debug mode
if '--debug' not in sys.argv:
    with suppress_stderr_completely():
        pass
else:
    pass

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

# Import TextResourceManager for general resource creation
try:
    from shared.utils.text_resource_manager import TextResourceManager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False
from core.globals import g
from core.constants import NON_INTERRUPTING_TAGS, NON_VALIDATING_TAGS
from core.llm_router import LlmRouter, StreamInterruptedException, UserInterruptedException, ModelGenerationFailedException
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

# Import project storage utilities from extracted module
from shared.utils.project_storage import get_project_storage_path

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
    interrupt_count = [0]  # Use list to allow modification in nested function
    reset_timer = [None]  # Timer to reset interrupt count

    def reset_interrupt_count():
        """Reset interrupt count after timeout."""
        interrupt_count[0] = 0

    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) - first press cancels AI, second press exits immediately."""
        interrupt_count[0] += 1

        # Cancel existing reset timer
        if reset_timer[0] is not None:
            reset_timer[0].cancel()

        if interrupt_count[0] == 1:
            # First interrupt: Cancel AI generation and return to input
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
            print(colored("  → Press Ctrl+C again within 2 seconds to exit immediately", "yellow"))

            # Reset counter after 2 seconds
            reset_timer[0] = threading.Timer(2.0, reset_interrupt_count)
            reset_timer[0].daemon = True
            reset_timer[0].start()
        else:
            # Second interrupt: Exit immediately
            print(colored("\n🛑 SECOND INTERRUPT - Exiting immediately...", "red", attrs=['bold']))
            sys.exit(0)

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
        """Check for GUI input from visual_input_queue with session context support."""
        if hasattr(g, 'visual_input_queue') and not g.visual_input_queue.empty():
            try:
                user_input = g.visual_input_queue.get_nowait()
                if user_input:
                    # Handle both string and dict inputs
                    if isinstance(user_input, str):
                        if user_input.strip():
                            self.input_queue.put(user_input.strip())
                    elif isinstance(user_input, dict):
                        # Extract session_id if present
                        session_id = user_input.get('session_id')

                        # Set active session for tool routing
                        if session_id:
                            g.set_active_session(session_id)
                            logging.debug(f"Set active session to {session_id}")

                        # Check if it's a regular message or system message
                        if 'message' in user_input:
                            message = user_input['message']
                            if isinstance(message, str) and message.strip():
                                self.input_queue.put(message.strip())
                        elif user_input.get('type') in ['permission_response', 'interrupt_command']:
                            # System messages - put back for proper handling
                            g.visual_input_queue.put_nowait(user_input)
                        else:
                            logging.debug(f"Unexpected dict structure in visual queue: {user_input.keys()}")
                    else:
                        logging.debug(f"Unexpected input type in visual queue: {type(user_input)}")
            except asyncio.QueueEmpty:
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

            except asyncio.QueueEmpty:
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

# --- Tool Listing Functionality ---
def list_tools_and_exit() -> None:
    """
    List all available tools with their status and exit.
    Used by --list-tools argument.
    """
    from agent.core.tool_discovery import _generate_tool_documentation
    from core.tool_config import get_tool_config_manager

    # Get disabled tools from config
    tool_config = get_tool_config_manager()
    disabled_tools = tool_config.get_disabled_tools()

    # Generate tool documentation (also populates g.tool_classes)
    _generate_tool_documentation(disabled_tools)

    # Get tool info
    tool_info = g.loaded_tool_info
    all_tool_names = set(tool_info['loaded_tools'])
    disabled_tool_names = set(disabled_tools)
    skipped_tool_names = set(tool_info.get('skipped_tools', []))

    # Collect all tools (enabled, disabled, and skipped)
    all_tools_dict = {}

    # Add loaded (enabled) tools
    for tool_name in tool_info['loaded_tools']:
        tool_class = g.tool_classes.get(tool_name)
        if tool_class:
            try:
                info = tool_class.get_tool_info()
                all_tools_dict[tool_name] = {
                    'status': 'enabled',
                    'description': info.get('description', 'No description')
                }
            except Exception as e:
                all_tools_dict[tool_name] = {
                    'status': 'enabled',
                    'description': f'Error getting info: {e}'
                }

    # Add disabled tools (need to discover them)
    import importlib
    from pathlib import Path
    tools_dir = Path(__file__).parent / "tools"

    for disabled_name in disabled_tools:
        if disabled_name not in all_tools_dict:
            # Try to load the tool to get its description
            try:
                module = importlib.import_module(f"tools.{disabled_name}")
                if hasattr(module, disabled_name):
                    tool_class = getattr(module, disabled_name)
                    if hasattr(tool_class, 'get_tool_info'):
                        info = tool_class.get_tool_info()
                        all_tools_dict[disabled_name] = {
                            'status': 'disabled',
                            'description': info.get('description', 'No description')
                        }
                    else:
                        all_tools_dict[disabled_name] = {
                            'status': 'disabled',
                            'description': 'Tool exists but missing get_tool_info()'
                        }
                else:
                    all_tools_dict[disabled_name] = {
                        'status': 'disabled',
                        'description': 'Tool class not found'
                    }
            except Exception:
                all_tools_dict[disabled_name] = {
                    'status': 'disabled',
                    'description': 'Unable to load tool info'
                }

    # Handle skipped tools
    for skipped in skipped_tool_names:
        tool_name = skipped.split(' ')[0]  # Extract tool name from "toolname (reason)"
        if tool_name not in all_tools_dict:
            all_tools_dict[tool_name] = {
                'status': 'skipped',
                'description': skipped
            }

    # Print formatted output
    print("\n" + "="*80)
    print("AVAILABLE TOOLS")
    print("="*80 + "\n")

    # Sort by name
    for tool_name in sorted(all_tools_dict.keys()):
        tool_data = all_tools_dict[tool_name]
        status = tool_data['status']
        description = tool_data['description']

        # Color-code status
        if status == 'enabled':
            status_str = colored("✓ ENABLED ", "green")
        elif status == 'disabled':
            status_str = colored("✗ DISABLED", "red")
        else:
            status_str = colored("○ SKIPPED ", "yellow")

        # Format tool name
        tool_name_str = colored(f"{tool_name:20s}", "cyan", attrs=["bold"])

        # Print line
        print(f"{status_str}  {tool_name_str}  {description}")

    # Print summary
    print("\n" + "-"*80)
    enabled_count = sum(1 for t in all_tools_dict.values() if t['status'] == 'enabled')
    disabled_count = sum(1 for t in all_tools_dict.values() if t['status'] == 'disabled')
    skipped_count = sum(1 for t in all_tools_dict.values() if t['status'] == 'skipped')

    print(f"Summary: {enabled_count} enabled, {disabled_count} disabled, {skipped_count} skipped")
    print("-"*80 + "\n")

    sys.exit(0)

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
    parser.add_argument("-s", "--speak", action="store_true", default=False,
                        help="Enable text-to-speech output only (no microphone input).")
    parser.add_argument("--online_voice", action="store_true", default=False,
                        help="Enable cloud-based speech transcription and TTS (Groq/OpenAI/ElevenLabs) with local fallback. By default, uses local engines only.")

    parser.add_argument("--full_output", dest='full_output_mode', action="store_true", default=True,
                        help="Full output mode. Shows live model output instead of hiding it.")
    parser.add_argument("-img", "--image", action="store_true", default=False,
                        help="Take a screenshot using Spectacle for region selection (with automatic fallbacks if not available).")
    parser.add_argument("-o", "--online", action="store_true", default=False,
                        help="Force use of cloud AI.")

    # Role-specific LLM selection arguments (support multiple models for fallback chains)
    parser.add_argument("--override", type=str, nargs='+', default=None,
                        help="Override LLM model(s) - bypasses router/planner selection. Supports multiple models for fallback chain (e.g., --override gemini-2.0-flash claude-haiku-4-5)")
    parser.add_argument("--router", type=str, nargs='+', default=None,
                        help="Router LLM model(s) for complexity assessment. Supports multiple models for fallback chain.")
    parser.add_argument("--planner", type=str, nargs='+', default=None,
                        help="Planner LLM model(s) for high-complexity tasks. Supports multiple models for fallback chain.")
    parser.add_argument("--defaults", type=str, nargs='+', default=None,
                        help="Default/fallback LLM model(s) for general tasks. Supports multiple models.")

    # Backward compatibility: --llm is an alias for --override
    parser.add_argument("-llm", "--llm", type=str, nargs='?', const="__select__", default=None, dest='override',
                        help="[DEPRECATED: use --override] Specify override LLM model. Use without value to open selection menu.")

    parser.add_argument("--gui", action="store_true", default=False,
                        help="Open a web interface for the chat (opens browser automatically)")
    parser.add_argument("--api", action="store_true", default=False,
                        help="Start API server without opening browser (for headless/remote use)")
    parser.add_argument("--debug-chats", action="store_true", default=False,
                        help="Enable debug windows for chat contexts without full debug logging. Sets g.DEBUG_CHATS=True.")
    parser.add_argument("--private_remote_wake_detection", action="store_true", default=False,
                        help="Use private remote wake detection")

    # --- Logging Arguments ---
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable detailed debug logging to the console.")
    # NOTE: File logging is ALWAYS enabled at DEBUG level to ~/.cli-agent/logs/session.log
    # This ensures production issues can always be diagnosed from logs
    # --- End Logging Arguments ---

    parser.add_argument("--clear-cache", type=int, nargs='?', const=2, default=None, metavar='DAYS',
                        help="Clear LLM cache entries older than N days (default: 2 days). Use --clear-cache 365 to clear the recent year. Exits after clearing.")

    parser.add_argument("--isolated", action="store_true", default=False,
                        help="Run in isolated Docker container (builds cached image on first use, ~2-3 min initial build, instant subsequent runs)")

    parser.add_argument("-e", "--exit", action="store_true", default=False,
                        help="Exit after all automatic messages have been parsed successfully")

    # --- Tool Management Arguments ---
    parser.add_argument("--list-tools", action="store_true", default=False,
                        help="List all available tools with their status and exit")
    parser.add_argument("--enable", type=str, nargs='+', default=None, metavar='TOOL',
                        help="Enable specific tools (removes from disabled list). Example: --enable bash readfile")
    parser.add_argument("--disable", type=str, nargs='+', default=None, metavar='TOOL',
                        help="Disable specific tools. Example: --disable bash writefile editfile")
    parser.add_argument("--read-only", action="store_true", default=False,
                        help="Run in read-only mode (only enables safe, non-modifying tools: grepsearch, paralleltools, readdirectory, readfile, searchconversations, searchresource, think, todos, viewpermissions, visualresponse)")
    parser.add_argument("--allow-write-path", type=str, nargs='+', default=None, metavar='PATH',
                        help="Allow write operations only to specific paths (use with --read-only for fine-grained control). Example: --allow-write-path ~/.cli-agent/suggested_prompts.json")
    # --- End Tool Management Arguments ---

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

# --- Eager setup of args ---
# This is done before heavy imports to allow --help to work quickly.
args = parse_cli_args()

# --- Handle Cache Clearing ---
# Clear cache and exit if --clear-cache was specified
if args.clear_cache is not None:
    from core.cache.llm_cache_manager import LLMCacheManager
    import os

    cache_db_path = os.path.join(g.CLIAGENT_CACHE_DIR, "llm_cache.db")
    cache_manager = LLMCacheManager(cache_db_path)

    # Get stats before clearing
    stats_before = cache_manager.get_cache_statistics()

    # Clear entries older than specified days
    removed = cache_manager.cleanup_expired_entries(max_age_days=args.clear_cache)

    # Vacuum database to reclaim space
    cache_manager.vacuum_database()

    # Get stats after clearing
    stats_after = cache_manager.get_cache_statistics()

    # Print summary
    print("\n✅ Cache Cleanup Summary:")
    print(f"   Removed: {removed} entries older than {args.clear_cache} days")
    print(f"   Before: {stats_before.get('total_entries', 0)} entries, {stats_before.get('db_file_size_bytes', 0) / (1024*1024):.2f} MB")
    print(f"   After:  {stats_after.get('total_entries', 0)} entries, {stats_after.get('db_file_size_bytes', 0) / (1024*1024):.2f} MB")
    print(f"   Space reclaimed: {(stats_before.get('db_file_size_bytes', 0) - stats_after.get('db_file_size_bytes', 0)) / (1024*1024):.2f} MB\n")

    sys.exit(0)

# --- Handle Tool Management Arguments ---
# Process --enable and --disable arguments before listing/starting
from core.tool_config import get_tool_config_manager
tool_config_manager = get_tool_config_manager()

if hasattr(args, 'enable') and args.enable:
    tool_config_manager.enable_tools(args.enable)
    print(colored(f"✓ Enabled tools: {', '.join(args.enable)}", "green"))

if hasattr(args, 'disable') and args.disable:
    tool_config_manager.disable_tools(args.disable)
    print(colored(f"✓ Disabled tools: {', '.join(args.disable)}", "red"))

# --- Handle Allow Write Path (process before read-only mode) ---
# Process --allow-write-path to expand and store allowed write paths
if hasattr(args, 'allow_write_path') and args.allow_write_path:
    import os
    allowed_write_paths = []
    for path in args.allow_write_path:
        # Expand ~ and relative paths to absolute paths
        expanded_path = os.path.abspath(os.path.expanduser(path))
        allowed_write_paths.append(expanded_path)

    # Store in args and g for later access by tools
    args.allowed_write_paths = allowed_write_paths
    g.ALLOWED_WRITE_PATHS = allowed_write_paths
    print(colored(f"✓ Allowing writes to {len(allowed_write_paths)} specific path(s):", "green"))
    for path in allowed_write_paths:
        print(colored(f"  • {path}", "cyan"))

    # Auto-approve permissions for these paths
    try:
        from core.permissions.permissions_manager import PermissionsManager
        perms_manager = PermissionsManager()

        # Add always-allow permission for each write tool + path combination
        write_tools = ['writefile', 'editfile', 'insertline', 'deletelines']
        for tool_name in write_tools:
            for path in allowed_write_paths:
                perms_manager.add_always_allow(tool_name, file_path=path)

        print(colored(f"  Auto-approved permissions for {len(write_tools)} write tools", "green"))
    except Exception as e:
        print(colored(f"  Warning: Could not auto-approve permissions: {e}", "yellow"))
else:
    args.allowed_write_paths = []
    g.ALLOWED_WRITE_PATHS = []

# --- Handle Read-Only Mode ---
# In read-only mode, disable all tools except safe read-only ones
if hasattr(args, 'read_only') and args.read_only:
    # Define read-only tools (safe, non-modifying tools)
    READ_ONLY_TOOLS = [
        'grepsearch', 'paralleltools', 'readdirectory', 'readfile',
        'searchconversations', 'searchresource', 'think', 'todos',
        'viewpermissions', 'visualresponse', 'response'
    ]

    # If --allow-write-path is specified, also allow restricted write tools
    WRITE_TOOLS = ['writefile', 'editfile', 'insertline', 'deletelines']
    if hasattr(args, 'allow_write_path') and args.allow_write_path:
        READ_ONLY_TOOLS.extend(WRITE_TOOLS)
        print(colored(f"  Enabled write tools (restricted to {len(args.allowed_write_paths)} path(s))", "cyan"))

    # Get all available tools by doing a quick discovery
    from pathlib import Path
    tools_dir = Path(__file__).parent / "tools"
    all_tool_names = []

    # Scan tools directory
    for file in tools_dir.glob("*.py"):
        if file.name.startswith("_") or file.stem == "__init__":
            continue
        all_tool_names.append(file.stem)

    # Scan optional tools
    optional_dir = tools_dir / "optional"
    if optional_dir.exists():
        for file in optional_dir.glob("*.py"):
            if file.name.startswith("_") or file.stem == "__init__":
                continue
            all_tool_names.append(file.stem)

    # Disable all tools that aren't in the read-only list
    tools_to_disable = [t for t in all_tool_names if t not in READ_ONLY_TOOLS]

    if tools_to_disable:
        tool_config_manager.set_disabled_tools(tools_to_disable)
        print(colored(f"✓ Read-only mode enabled: {len(READ_ONLY_TOOLS)} safe tools available", "green"))
        print(colored(f"  Disabled {len(tools_to_disable)} modifying tools", "yellow"))

# --- Handle Tool Listing ---
# List tools and exit if --list-tools was specified
if args.list_tools:
    list_tools_and_exit()

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
    Also detects tag mismatches like <type>...</content>.

    Args:
        text (str): The text to analyze for malformed XML

    Returns:
        List[Dict[str, str]]: List of detected malformed XML issues with detailed error info
    """
    import xml.etree.ElementTree as ET
    from shared.common_utils import _mask_quoted_xml_tags

    errors = []

    # NOTE: Pre-check for mismatched tags removed due to false positives with nested XML
    # The regex pattern r'<tag>(.*?)</different_tag>' will incorrectly match nested structures
    # Example: <outer><inner>...</inner></outer> matches as <outer>...</inner> (false positive)
    # Instead, we rely solely on the XML parser below to detect malformed XML

    # Mask quoted XML tags before validation to prevent false positives
    # e.g., "<bash>" in think tags should not be validated as XML
    masked_text, quote_replacements = _mask_quoted_xml_tags(text)

    # Find potential XML blocks using regex (now on masked text)
    xml_pattern = r'<([a-zA-Z0-9_]+)[^>]*>.*?</\1>'

    for match in re.finditer(xml_pattern, masked_text, re.DOTALL):
        xml_block = match.group(0)
        tag_name = match.group(1)

        # Skip validation for non-validating tags (think, todos) which contain free-form text
        if tag_name in NON_VALIDATING_TAGS:
            logging.debug(f"Skipping XML validation for non-validating tag '<{tag_name}>' (contains free-form text)")
            continue

        try:
            # Try to parse the XML block with ElementTree
            ET.fromstring(xml_block)
            # If we get here, the XML is well-formed

        except ET.ParseError:
            # Try to fix the XML with automatic escaping (same logic as extract_blocks)
            try:
                import html

                # Strategy: Use regex to identify XML tags vs content, then escape only content
                parts = []
                last_end = 0

                # Pattern to match XML tags (opening, closing, or self-closing)
                tag_pattern = r'</?[a-zA-Z0-9_]+[^>]*?>'

                for tag_match in re.finditer(tag_pattern, xml_block):
                    # Content before this tag needs escaping
                    content_before = xml_block[last_end:tag_match.start()]
                    if content_before:
                        parts.append(html.escape(content_before, quote=False))

                    # The tag itself should not be escaped
                    parts.append(tag_match.group(0))

                    last_end = tag_match.end()

                # Handle any remaining content after the last tag
                remaining_content = xml_block[last_end:]
                if remaining_content:
                    parts.append(html.escape(remaining_content, quote=False))

                fixed_xml = ''.join(parts)

                # Try parsing again with escaped content
                ET.fromstring(fixed_xml)

                # If we get here, the XML is valid after escaping - don't report as error
                logging.debug(f"XML in tag '<{tag_name}>' auto-fixed with content escaping")
                continue  # Skip adding to errors list

            except ET.ParseError as e:
                # Even after escaping, it still fails - report the error
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
        args.defaults = None # No specific defaults are set
        logging.info(colored("# cli-agent: 'Any but local' option selected. Local mode enabled.", "green") )
        return []

    # Handle specific model selections
    selected_configs = result.get("selected_configs", [])

    selected_llms = [config['model_key'] for config in selected_configs]

    # Update globals with the latest selection (LLM selector selects default/fallback models)
    args.defaults = selected_llms
    g.SELECTED_LLMS = selected_llms

    # Reload the config in `g` so the main loop gets the fresh settings
    g.load_llm_config()

    # Load previously selected models from saved configuration
    try:
        config = g.get_llm_config()

        # Load selected fallback models from config (using unified roles structure)
        previously_selected = g.SELECTED_LLMS

        if previously_selected:
            args.defaults = previously_selected
            logging.info(colored(f"📋 Loaded {len(previously_selected)} selected models from config: {', '.join(previously_selected[:3])}{'...' if len(previously_selected) > 3 else ''}", "cyan"))

    except Exception as e:
        logging.warning(colored(f"⚠️  Failed to load saved LLM configuration: {e}", "yellow"))

    if not args.defaults and not args.override:
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
    token_broadcast_callback = None,
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
        token_broadcast_callback: Optional async callback to emit token_stream events.
            Called with (token, tokens_generated, elapsed_ms).

    Returns:
        An async callback function.
    """
    # Pre-compile regex for efficiency. re.DOTALL allows '.' to match newlines.
    # CODE_BLOCK_REGEX = re.compile(r"```.*?(```)", re.DOTALL)

    # Dynamically build a regex for registered tool blocks. This is more robust
    # and prevents premature interruption on non-tool XML tags like <filepath>.
    # Define tags that should NOT interrupt the stream (from constants for consistency)
    non_interrupting_tags = NON_INTERRUPTING_TAGS

    # Get all tool delimiters and add the final <response> tag.
    all_delimiters = list(getattr(g, 'tool_classes', {}).keys())

    # In GUI mode, also exclude 'response' from interrupting since we want visualresponse only
    # if hasattr(g, 'GUI_MODE') and g.GUI_MODE:
    #     non_interrupting_tags = non_interrupting_tags.union({'response'})

    # Filter this list to get only the tags that SHOULD cause an interruption.
    interrupting_delimiters = [d for d in all_delimiters if d not in non_interrupting_tags]

    # Build the regex from the filtered list of interrupting tags.
    tool_tag_pattern = "|".join(re.escape(d) for d in interrupting_delimiters)
    XML_TOOL_BLOCK_REGEX = re.compile(f"<({tool_tag_pattern})[^>]*>.*?</\\1>", re.DOTALL) if tool_tag_pattern else None

    # Repetition detection state (detects when model loops on same 1-5 chars)
    repetition_state = {'pattern': '', 'count': 0, 'check_pos': 0}

    # Token streaming state (for broadcasting real-time token events to frontend)
    import time
    token_state = {
        'start_time': time.time(),
        'tokens_generated': 0,
        'last_broadcast_time': time.time()
    }

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

                # Print the batch (always show streaming output)
                if lock:
                    async with lock:
                        print(painter.apply_color(batch_str), end="", flush=True)
                else:
                    print(painter.apply_color(batch_str), end="", flush=True)

                # Append batch to shared buffer
                response_buffer[0] += batch_str
                current_buffer = response_buffer[0]

                # Emit token stream event if broadcast callback is provided
                if token_broadcast_callback:
                    try:
                        token_state['tokens_generated'] += len(batch_str)
                        elapsed_ms = int((time.time() - token_state['start_time']) * 1000)

                        # Emit token event (throttle to avoid spamming - once per 100ms or when batch ends)
                        current_time = time.time()
                        if current_time - token_state['last_broadcast_time'] >= 0.1 or i == len(chunk) - 1:
                            if asyncio.iscoroutinefunction(token_broadcast_callback):
                                await token_broadcast_callback(batch_str, token_state['tokens_generated'], elapsed_ms)
                            else:
                                token_broadcast_callback(batch_str, token_state['tokens_generated'], elapsed_ms)
                            token_state['last_broadcast_time'] = current_time
                    except Exception as e:
                        # Don't let broadcasting errors interrupt the stream
                        logging.debug(f"Token broadcast error: {e}")

                # Clear batch
                char_batch = []

                # Check for repetition loops (model stuck repeating same 1-5 chars)
                buffer_len = len(current_buffer)
                if buffer_len > repetition_state['check_pos'] + 100:  # Check every 100 chars
                    # Look at last 100 characters
                    recent = current_buffer[-100:]

                    # Look at a wider context (last 500 chars) to check for mixed content
                    # This helps distinguish legitimate code indentation from pure spam
                    wider_context = current_buffer[-500:] if len(current_buffer) >= 500 else current_buffer
                    context_non_whitespace = len(wider_context.strip())
                    context_total = len(wider_context)

                    # If the wider context has >10% non-whitespace chars, likely legitimate code
                    has_mixed_content = context_total > 0 and (context_non_whitespace / context_total) > 0.1

                    # Try patterns of length 1-5
                    for pattern_len in range(1, 6):
                        if len(recent) < pattern_len * 200:  # Need at least 200 repetitions
                            continue
                        pattern = recent[-pattern_len:]

                        # Skip whitespace-only patterns if there's mixed content in wider context
                        # This allows legitimate indentation while catching pure whitespace spam
                        if pattern.strip() == '' and has_mixed_content:
                            continue

                        # Count how many times this pattern repeats at the end
                        count = 0
                        pos = len(recent) - pattern_len
                        while pos >= 0 and recent[pos:pos+pattern_len] == pattern:
                            count += 1
                            pos -= pattern_len

                        if count >= 200:  # Pattern repeats 200+ times
                            logging.error(colored(f"\n⚠️  Model repetition loop detected: '{pattern}' repeated {count} times", "red"))
                            raise StreamInterruptedException(f"[REPETITION_LOOP_DETECTED: '{pattern}' x{count}]")

                    repetition_state['check_pos'] = buffer_len

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

                            # Special case: if this is inside an unclosed <paralleltools> block,
                            # don't interrupt yet - wait for the outer </paralleltools> tag
                            paralleltools_start = current_buffer.find('<paralleltools')
                            if paralleltools_start != -1:
                                # Check if we have a closing </paralleltools> tag
                                paralleltools_end = current_buffer.find('</paralleltools>')
                                if paralleltools_end == -1:
                                    # paralleltools is still open, don't interrupt on inner tools
                                    continue

                            raise StreamInterruptedException(full_block)

    return _callback

def get_provider_emoji_and_name(model_key: str) -> Tuple[str, str]:
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



def build_system_prompt(voice_mode: bool = False, gui_mode: bool = False) -> str:
    """
    Build system prompt with current g.disabled_tools state.

    :param voice_mode: Whether voice mode is enabled
    :param gui_mode: Whether GUI mode is enabled
    :return: Complete system prompt string
    """
    voice_mode_text = 'The user is interacting with you in speech mode, please be aware of potential transcription issues. Provide concise responses and be helpful to the user.\n' if voice_mode else ''
    base_instructions = f'''**SYSTEM: Agent Protocol & Capabilities**

You are a sophisticated AI agent capable of thinking and using tools to achieve your todos. Your primary directive is to understand user requests, formulate a plan, and perform actions to achieve the current needs.
{voice_mode_text}
### Guiding Principles (Your Core Logic)
The following loop is a suggestion of how to behave effectively, you may freely reorder your usage of the xml tools if it fits your needs, you aim for long term stability and progress in your responses.
1.  **DECOMPOSE**: Analyze the user's request and break it down into the required steps. Start with: `<think>...</think>`.
2.  **PLAN & TRACK**: Display/update your <todos>...</todos> to align your next step(s) with the stated long term goal(s).'''

    step_4 = "4.  **OBSERVE & ATTEND**: Regularly use the <think> and <todos> tools, especially after you've received a new <context>, to make your observations explicit and guide your current actions. When you need to respond to the user, please prompt him with <response>."

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
    return f"{base_instructions}\n{tool_docs}\n{step_4}"


def update_system_prompt() -> None:
    """
    Update g.chat's system prompt (messages[0]) with current disabled_tools state.
    This allows runtime modification of available tools without restarting the agent.
    """
    if not hasattr(g, 'chat') or not g.chat:
        logging.debug("Cannot update system prompt: g.chat not initialized")
        return

    if not g.chat.messages or len(g.chat.messages) == 0:
        logging.warning("Cannot update system prompt: chat has no messages")
        return

    if g.chat.messages[0].role != Role.SYSTEM:
        logging.warning("Cannot update system prompt: first message is not SYSTEM role")
        return

    # Regenerate prompt with current disabled_tools
    prompt = build_system_prompt(voice_mode=g.voice_mode, gui_mode=g.GUI_MODE)
    g.chat.messages[0].content = prompt

    disabled_count = len(getattr(g, 'disabled_tools', []))
    logging.info(f"✓ Updated system prompt ({disabled_count} tools disabled)")


async def construct_engineered_prompt(voice_agent: str, gui_mode: bool = False, add_bootstrap: bool = True) -> Chat:
    """
    Constructs the initial Chat object with a dynamically generated system prompt.
    This prompt includes the agent's core principles and a list of available tools.

    :param voice_agent: Whether voice mode is enabled
    :param gui_mode: Whether GUI mode is enabled
    :param add_bootstrap: Whether to add bootstrap messages (file list + permissions)
    """
    # Generate system prompt using current disabled_tools state
    default_inst = build_system_prompt(voice_mode=voice_agent, gui_mode=gui_mode)

    # Create and configure the new Chat object
    context_chat = Chat(debug_title="Main Context Chat")
    context_chat.is_main_context = True  # Mark as main context for WebSocket broadcasting
    context_chat.set_instruction_message(default_inst)

    # Add bootstrap messages if requested
    if add_bootstrap:
        from tools.bash import bash
        from tools.viewpermissions import viewpermissions

        context_chat.add_message(Role.USER, "Please list the 5 most recently modified files of the cwd")

        if gui_mode:
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
        if "No automatic permissions set" not in permissions_str:
            context_chat.add_message(Role.USER, "view your permissions")
            context_chat.add_message(Role.ASSISTANT, f"""<think>
This task can be completed using the viewpermssions tool.</think><todos>
- [ ] View permissions.
</todos><viewpermissions>
</viewpermissions><context>
{permissions_str}
</context><todos>
- [x] View permissions.
</todos>""" + bootstrap_message_2)

        # Store count of bootstrap messages in metadata (includes system message)
        message_count_after = len(context_chat.messages)
        context_chat.metadata['bootstrap_message_count'] = message_count_after

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

    # Reconstruct fresh engineered prompt (without bootstrap on reset)
    fresh_context_chat = await construct_engineered_prompt(args.voice, args.gui or args.api, add_bootstrap=False)

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
            setup_logging(args.debug, tui_mode=True)

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
            setup_logging(args.debug)
            g.DEBUG_MODE = args.debug

            # Sync command line args to global state
            g.auto_execute = args.yolo
            # Voice mode: full voice (mic input + TTS) or TTS-only mode
            g.voice_mode = args.voice or args.speak
            g.tts_only_mode = args.speak and not args.voice  # TTS without mic input
            g.local_mode = args.local
            g.online_mode = args.online

            # Setup signal handlers for graceful interruption
            setup_signal_handlers()

            header_dashes = colored("# # # # # # # # # # # # # # # # # # # # # # # # # #", "blue")
            header = figlet_format("CLI-Agent", font="slant")
            print(header_dashes)
            print(colored(header, "cyan", attrs=["bold"]))
            print(header_dashes)
            logging.info("Starting CLI-Agent...")

        load_dotenv(g.CLIAGENT_ENV_FILE_PATH)

        # Apply default LLM config if no role-specific models were specified
        if not args.override and not args.router and not args.planner and not args.defaults and not args.local:
            apply_default_llm_config()

        # LLM roles are automatically loaded by g.load_llm_roles() during globals initialization

        # Auto-load profile 1 at startup (unless CLI args override it)
        if not args.override and not args.router and not args.planner and not args.defaults:
            try:
                from core.llm_profiles import get_profile_manager
                manager = get_profile_manager()

                # FATAL ERROR: Delete any corrupted profiles on startup
                deleted_profiles = manager.delete_corrupted_profiles_on_startup()
                if deleted_profiles:
                    # Reset all LLM roles to None (blank slate)
                    logging.info("⚠️  Initializing with blank slate - no models selected")
                    g.update_llm_role('override', None)
                    g.update_llm_role('router', None)
                    g.update_llm_role('planner', None)
                    g.update_llm_role('fallbacks', model_keys=None)

                # Always load profile "1" if no corrupted profiles were deleted
                if not deleted_profiles:
                    profile = manager.get_profile("1")
                    if profile:
                        profile.apply()
                        logging.info(colored("📌 Auto-loaded profile: '1'", "green"))
                    else:
                        logging.warning("Profile '1' not found, starting with blank slate")
            except Exception as e:
                logging.warning(colored(f"Failed to auto-load profile: {e}", "yellow"))

        # Apply command-line role-specific model selections (overrides config file)
        if args.override:
            models = args.override if isinstance(args.override, list) else [args.override]
            g.update_llm_role('override', model_keys=models)
            logging.info(colored(f"🎯 Override LLM set via CLI: {', '.join(models)}", "cyan"))

        if args.router:
            models = args.router if isinstance(args.router, list) else [args.router]
            g.update_llm_role('router', model_keys=models)
            logging.info(colored(f"🧭 Router LLM set via CLI: {', '.join(models)}", "cyan"))

        if args.planner:
            models = args.planner if isinstance(args.planner, list) else [args.planner]
            g.update_llm_role('planner', model_keys=models)
            logging.info(colored(f"📋 Planner LLM set via CLI: {', '.join(models)}", "cyan"))

        if args.defaults:
            models = args.defaults if isinstance(args.defaults, list) else [args.defaults]
            g.update_llm_role('fallbacks', model_keys=models)
            logging.info(colored(f"⚙️  Default LLMs set via CLI: {', '.join(models)}", "cyan"))

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
        if args.gui or args.api:
            # Set GUI mode flag for behavior modifications
            g.GUI_MODE = True

            # Set API_MODE flag when using --api (headless)
            if args.api:
                g.API_MODE = True

            # Auto-start visualresponse server for GUI mode
            try:
                from tools.visualresponse import visualresponse
                open_browser = args.gui and not args.api  # Only open browser with --gui, not --api
                mode_msg = "🌐 Starting GUI mode - opening browser..." if open_browser else "🌐 Starting API server (headless mode)..."
                logging.info(colored(mode_msg, "blue"))
                logging.debug(f"About to call visualresponse._get_session_async(open_browser={open_browser})")

                # Get a session which will start the server (async version for use in async context)
                session = await visualresponse._get_session_async(open_browser=open_browser)
                logging.info(colored(f"✅ Server active - interface available at http://127.0.0.1:{session.port}", "green"))
                logging.debug(f"Session created: {session}, server instance: {session.server if hasattr(session, 'server') else 'N/A'}")

                # Enable console capture for real-time log streaming to web interface
                try:
                    from tools.visualresponse import enable_console_capture
                    enable_console_capture()
                    logging.info(colored("📺 Console capture enabled - logs will stream to web interface", "cyan"))
                except Exception as e:
                    logging.warning(colored(f"⚠️  Could not enable console capture: {e}", "yellow"))
            except ImportError as e:
                logging.error(colored(f"❌ Could not start visualresponse server: {e}", "red"))
                logging.error(traceback.format_exc())
            except Exception as e:
                logging.error(colored(f"❌ Error starting visualresponse: {e}", "red"))
                logging.error(traceback.format_exc())

        context_chat: Optional[Chat] = None
        if args.c or args.regenerate:
            try:
                context_chat = Chat.load_from_json()

                # Reset bootstrap_message_count for loaded chats since they have grown beyond bootstrap
                if context_chat and 'bootstrap_message_count' in context_chat.metadata:
                    context_chat.metadata['bootstrap_message_count'] = 0

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

        images_for_message: List[str] = []

        # Add any images from session context (web UI uploads)
        session = g.get_active_session()
        if session and session.current_images:
            images_for_message.extend(session.current_images)
            session.clear_images()
            logging.debug(f"Added {len(images_for_message)} images from session {session.session_id}")
        # Backward compat: also check legacy global
        elif hasattr(g, 'current_images') and g.current_images:
            images_for_message.extend(g.current_images)
            g.current_images = []
            logging.debug(f"Added {len(images_for_message)} images from legacy global")

        if args.image:
            if not tui_callback: logging.info(colored("# cli-agent: Taking screenshot due to --img flag...", "green"))
            images_for_message.extend(await handle_screenshot_capture())
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

        # Handle interactive LLM selection (triggered by --llm without value or --override without value)
        if args.override == "__select__":
            await llm_selection(args, preselected_llms=g.SELECTED_LLMS)
            args.override = None
        elif args.override and not isinstance(args.override, list):
            # Single override model passed as string (convert to list for consistency)
            args.override = [args.override]


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
            # Load tool configuration manager and handle CLI arguments
            from core.tool_config import get_tool_config_manager
            tool_config = get_tool_config_manager()

            # Handle --enable and --disable CLI arguments
            if hasattr(args, 'enable') and args.enable:
                tool_config.enable_tools(args.enable)
                logging.info(f"Enabled tools from CLI: {args.enable}")

            if hasattr(args, 'disable') and args.disable:
                tool_config.disable_tools(args.disable)
                logging.info(f"Disabled tools from CLI: {args.disable}")

            # Get disabled tools from tool config (merged with CLI args)
            user_disabled_tools = tool_config.get_disabled_tools()

            # Also check global state for web UI disabled tools
            if hasattr(g, 'disabled_tools') and g.disabled_tools:
                # Merge with global state (web UI may have set this)
                user_disabled_tools = list(set(user_disabled_tools + g.disabled_tools))

            # Update global state with final disabled tools list
            g.disabled_tools = user_disabled_tools

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

        if args.c or args.regenerate:
            # Try to restore previous chat state for persistent conversation across restarts
            storage_path = str(get_project_storage_path()) + "/recent_chat.json"

            if os.path.exists(storage_path):
                try:
                    context_chat = Chat.load_from_json()
                    logging.info(colored(f"📚 Restored previous chat with {len(context_chat.messages)} messages", "green"))
                except Exception as e:
                    logging.warning(colored(f"⚠️  Failed to load saved chat: {e}, creating new one", "yellow"))
                    context_chat = None
        else:
            # Only add bootstrap if not using -c or --regenerate flags
            context_chat = await construct_engineered_prompt(args.voice, args.gui or args.api, True)
            logging.info(colored("✨ Created new chat session", "green"))

        # Set global chat reference for API/web interface
        g.chat = context_chat

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

        assistant_response = ""

        # --- FIX: Instantiate CommandHandler with proper AppContext ---
        app_context = AppContext()
        app_context.settings.load_from_args(args)
        app_context.logger = logging.getLogger("command_handler_context")
        command_handler = CommandHandler(app_context)
        # --- END FIX ---

        initializing: bool = True

        # CommandHandler already initialized above - no need to reinitialize

        # Give WebSocket server time to accept connections before processing queued messages
        # This ensures frontend can load chat/config before LLM call starts
        if args.gui and args.message:
            await asyncio.sleep(0.5)
            logging.debug("Allowed WebSocket connection time before processing queued messages")

        while True:
            from core.llm_router import LlmRouter
            router = LlmRouter()
            router.failed_models.clear()
            assistant_response = ""

            # Sync context_chat with g.chat (in case API updated it)
            # Only sync if g.chat has more messages (legitimate update), not if it's a fresh/empty chat
            if hasattr(g, 'chat') and g.chat is not None and g.chat is not context_chat:
                if len(g.chat.messages) >= len(context_chat.messages):
                    context_chat = g.chat
                    logging.info("Synced context_chat with g.chat (updated via API)")
                else:
                    # g.chat has fewer messages - API created a new chat but we should keep our context
                    g.chat = context_chat
                    logging.debug("Ignored g.chat sync - has fewer messages than current context")

            g.FORCE_LOCAL = args.local
            g.DEBUG_CHATS = args.debug_chats
            g.LLM = args.override

            # --- UNIFIED INPUT GATHERING ---
            # All input sources are now handled within get_user_input_with_bindings
            try:
                user_input, new_images = await get_user_input_with_bindings(
                    args, context_chat, force_input=user_interrupt,
                    input_event=input_event, input_lock=input_lock, shared_input=shared_input
                )
            except (KeyboardInterrupt, EOFError):
                # Handle Ctrl+C during input waiting
                print(colored("\n🛑 INTERRUPT SIGNAL RECEIVED - Cancelling current operation...", "red", attrs=['bold']))
                _interrupt_flag.set()
                g.USER_INTERRUPT_REQUESTED = True
                user_interrupt = True
                continue

            # Reset user_interrupt flag after successful input to allow queue processing
            user_interrupt = False

            # Reset tools executed flag for new user input
            tools_executed_this_turn = False

            # --- UNIFIED COMMAND & IMAGE PROCESSING ---
            if new_images:
                images_for_message.extend(new_images)

            # Add any images from session context (web UI uploads)
            session = g.get_active_session()
            if session and session.current_images:
                images_for_message.extend(session.current_images)
                session.clear_images()
                logging.debug(f"Added {len(images_for_message)} images from session {session.session_id}")
            # Backward compat: also check legacy global
            elif hasattr(g, 'current_images') and g.current_images:
                images_for_message.extend(g.current_images)
                g.current_images = []
                logging.debug(f"Added {len(images_for_message)} images from legacy global")

            if user_input and user_input.startswith('/'):
                response = await command_handler.handle_command(
                    user_input, context_chat=context_chat, command_handler=command_handler
                )
                if response.result == CommandResult.EXIT: exit(0)
                if response.result == CommandResult.ERROR: logging.error(colored(f"# cli-agent: {response.message}", "red"))
                continue

            # Skip adding system commands to chat history
            if (user_input) and not (
                (isinstance(user_input, dict) and user_input.get('type') == 'llm_selection_update') or
                (isinstance(user_input, str) and 'llm_selection_update' in user_input and user_input.strip().startswith('{')) or
                (isinstance(user_input, str) and user_input.strip() == '__AGENT_RESET__')
            ):
                context_chat.add_message(
                    Role.USER,
                    user_input,
                    images=images_for_message if images_for_message else None
                )

                # Auto-save after user message
                try:
                    context_chat.save_to_json(also_archive=False)
                except Exception as e:
                    logging.debug(f"Failed to auto-save chat: {e}")

                if args.gui or args.api:
                    try:
                        from api.server import broadcast_message_added
                        await broadcast_message_added(context_chat.messages[-1])
                    except Exception:
                        pass

                images_for_message = []
            else:
                continue

            initializing = False
            last_action_signature: Optional[str] = None
            stall_counter: int = 0
            MAX_STALLS: int = 2
            text_stream_painter = TextStreamPainter()

            # Agent loop
            tools_executed_this_turn = False  # Track if tools were executed in this conversation turn
            while True:
                # Check for interrupt at the start of each loop iteration
                if check_interrupt():
                    print(colored("\n🛑 Agent loop interrupted by user.", "red"))
                    break

                try:
                    if assistant_response:
                        context_chat.add_message(Role.ASSISTANT, assistant_response)

                        # Broadcast initial assistant response
                        if args.gui or args.api:
                            try:
                                from api.server import broadcast_message_added
                                await broadcast_message_added(context_chat.messages[-1])
                            except Exception:
                                pass

                        assistant_response = ""

                    if context_chat:
                        context_chat.save_to_json()

                    UNCOMPRESSED_HISTORY = 1
                    if (len(context_chat.messages) > UNCOMPRESSED_HISTORY):
                        for i in range(UNCOMPRESSED_HISTORY,UNCOMPRESSED_HISTORY+1):
                            message = context_chat.messages[-i]
                            if "<think>" in message.content:
                                think_content_match = re.search(r'<think>(.*?)</think>', message.content, re.DOTALL)
                                if think_content_match:
                                    think_content = think_content_match.group(1).strip()
                                    first_line_match = re.match(r'^([^\n]*)', think_content)
                                    if first_line_match:
                                        first_line = first_line_match.group(1).strip()
                                        message.content = re.sub(r'<think>(.*?)</think>', f'<think>\n{first_line}\n</think>', message.content, flags=re.DOTALL)

                    llm_config = g.get_llm_config()

                    response_buffer_list = [""]

                    # Create token broadcast callback for real-time frontend updates
                    async def token_broadcast_callback(token: str, tokens_generated: int, elapsed_ms: int):
                        """Emit token_stream events to frontend via WebSocket"""
                        try:
                            if args.gui or args.api:
                                from api.server import broadcast_token_stream
                                await broadcast_token_stream(token, tokens_generated, elapsed_ms)
                        except Exception as e:
                            logging.debug(f"Failed to broadcast token: {e}")

                    interruption_callback = create_interruption_callback(
                        response_buffer_list,
                        text_stream_painter,
                        token_broadcast_callback=token_broadcast_callback
                    )
                    current_todo = f"<current_todo>\n{todos().get_next_unchecked_task()}\n</current_todo>"
                    # Simple single model execution
                    try:
                        if check_interrupt():
                            user_interrupt = True
                            break

                        # Check for override model first - if set, use it exclusively and skip routing
                        # (Override can be set via --override CLI arg or /override_llm command)
                        if g.override_llm:
                            # Use override model with its fallback chain + global fallbacks
                            selected_models = g.override_llms_list + g.FALLBACK_MODELS
                        # Use router for complexity assessment if available
                        elif g.router_llm:
                            difficulty_chat = Chat("You are a complexity assessment assistant. Analyze the conversations needs and respond with a chain of thought followed by a score.")
                            difficulty_chat.add_message(Role.USER, "Analyze the complexity of formulating the next response in this conversation. Here are some few shot examples for you to reenact: The user provided a simple greeting (Complexity: 1%), A fileread of a known filepath is needed for the next step (15%), searching the web or filesystem (35%), applying edits to a file (50%), managing running processes (60%), authoring a new file (80%), implementing a script (90%), implementing a visualization (100%). Based on your level of uncertainty about the next action regress your score to 50%.\nProvide *brief* reasoning, then end with: COMPLEXITY_SCORE: [number 1-100]\n\nConversation:\n" + context_chat.print_chat(-3, None, True))
                            # In local mode, don't force preferred models since they might be cloud models
                            # Use router model with its fallback chain + global fallbacks
                            response = await LlmRouter.generate_completion(difficulty_chat, preferred_models=g.router_llms_list + g.FALLBACK_MODELS, hidden_reason="Complexity Assessment", force_free=True)
                            # Extract complexity score from formatted response
                            response_complexity: int = 50 # Default fallback
                            score_match = re.search(r'COMPLEXITY_SCORE:\s*(\d+)', response)
                            if score_match:
                                try:
                                    response_complexity = int(score_match.group(1))
                                except ValueError:
                                    logging.warning(f"Could not convert '{score_match.group(1)}' to int for complexity score. Using default.")
                            else:
                                # Fallback: extract last number if formatted score not found
                                match = re.findall(r'\d+', response)
                                if match:
                                    try:
                                        response_complexity = int(match[-1])
                                    except ValueError:
                                        pass
                            print(f"Score match: {score_match}")
                            if response_complexity >= 85:
                                # Use planner model with its fallback chain + global fallbacks
                                selected_models = g.planner_llms_list + g.FALLBACK_MODELS if g.planner_llms_list else g.FALLBACK_MODELS
                            elif response_complexity >= 20:
                                selected_models = g.FALLBACK_MODELS
                            else:
                                # Use router model with its fallback chain + global fallbacks
                                selected_models = g.router_llms_list + g.FALLBACK_MODELS
                        else:
                            selected_models = g.FALLBACK_MODELS

                        # Validate that at least one LLM is configured
                        if not selected_models or all(m is None or m == '' for m in selected_models):
                            error_message = "❌ No LLM configured. Please select at least one model (router, planner, or fallback) before sending messages."

                            # Debug logging to understand why validation failed
                            logging.error("LLM validation failed:")
                            logging.error(f"  selected_models: {selected_models}")
                            logging.error(f"  g.planner_llms_list: {g.planner_llms_list}")
                            logging.error(f"  g.router_llms_list: {g.router_llms_list}")
                            logging.error(f"  g.FALLBACK_MODELS: {g.FALLBACK_MODELS}")
                            logging.error(f"  g.llm_roles: {g.llm_roles}")

                            logging.error(error_message)
                            context_chat.add_message(
                                Role.ASSISTANT,
                                error_message
                            )
                            # Skip the LLM call and continue to next input
                            continue

                        # Generate single response
                        # Add the thinking prefix to the chat directly
                        # context_chat.add_message(Role.ASSISTANT, "<think>\n")
                        # context_chat.add_message(Role.ASSISTANT, "<think>")
                        # context_chat.add_message(Role.ASSISTANT, "<think>")
                        # context_chat.add_message(Role.ASSISTANT, "<think>")
                        assistant_response = await LlmRouter.generate_completion(
                            chat=context_chat,
                            preferred_models=selected_models,
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

                        # Store current model for tool execution (same as line 1812)
                        g.current_llm_model = LlmRouter.get_last_used_model()

                        # Store current model for tool execution
                        # (tool_info will be created and broadcast after tools actually execute)

                        base64_images = []

                    except ModelGenerationFailedException as e:
                        print(colored(f"Model generation failed: {e}", "red"))
                        break
                    except Exception as e:
                        print(colored(f"An Exception was raised, e: {e}", "red"))
                        traceback.print_exc()
                        break

                    # Don't check "if not assistant_response" here - it gets cleared after tool execution
                    # The actual break logic is at line ~2402 when response/visualresponse tools execute

                    if last_action_signature and assistant_response == last_action_signature:
                        stall_counter += 1
                    else:
                        stall_counter = 0
                    last_action_signature = assistant_response

                    if stall_counter >= MAX_STALLS:
                        print(colored("The agent is spamming its last tool use. Aborting...", "red"))
                        stall_counter, last_action_signature = 0, None
                        break

                    # If assistant_response is empty (after tool execution cleared it), skip processing and continue loop
                    # This allows the agent to generate the next response without treating empty string as "no tools found"
                    if not assistant_response.strip():
                        continue  # Go back to top of agent loop to generate next response

                    # Track model used for this LLM response
                    metadata = {}
                    try:
                        response_model = g.current_llm_model if hasattr(g, 'current_llm_model') and g.current_llm_model else LlmRouter.get_last_used_model()
                        if response_model:
                            metadata['response_model'] = response_model
                    except Exception as e:
                        logging.debug(f"Could not get response model: {e}")

                    context_chat.add_message(Role.ASSISTANT, assistant_response, metadata=metadata)
                    # Note: AI response to visual interface will be sent after tool execution
                    # to properly detect if visualresponse was used

                    # # Display the response in summary mode (since streaming doesn't show it)
                    # print(assistant_response)

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
                        delim = tool_class.get_delim()
                        # Check for XML tags: <bash>...</bash>
                        has_xml_tag = re.search(f'<{delim}[> ]', assistant_response, re.DOTALL)
                        # Check for markdown code blocks: ```bash...```
                        has_markdown = re.search(rf'```{delim}\s*\n', assistant_response, re.DOTALL)
                        if has_xml_tag or has_markdown:
                            tool_tags_found.append(tool_class)

                    # Store original response for TTS (ALWAYS, not just when tools found)
                    # This ensures TTS works for all responses including visualresponse-only
                    original_assistant_response = assistant_response

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

                                # --- LAYER 1: ALLOWED_WRITE_PATHS Enforcement ---
                                # Check write path restrictions BEFORE showing permission prompts
                                # This provides instant denial for unauthorized write attempts
                                allowed_write_paths = getattr(g, 'ALLOWED_WRITE_PATHS', [])
                                if allowed_write_paths:
                                    write_tools = ['writefile', 'editfile', 'insertline', 'deletelines']
                                    for analysis in tool_analyses:
                                        tool_name = analysis['tool_name']
                                        primary_path = analysis.get('primary_path')

                                        # If this is a write tool with a target path
                                        if tool_name in write_tools and primary_path:
                                            # Resolve to absolute path for consistent checking
                                            import os
                                            abs_path = os.path.abspath(primary_path)

                                            # Check if path is in allowed list
                                            if abs_path not in allowed_write_paths:
                                                # INSTANT DENIAL - No permission prompt shown
                                                should_execute = False
                                                rejection_reason = [{
                                                    'tool': tool_name,
                                                    'action': analysis['action_description'],
                                                    'paths': [abs_path],
                                                    'risk_level': 'restricted',
                                                    'xml_content': f"Write access denied: '{abs_path}' is not in the allowed write paths list.\nAllowed paths: {', '.join(allowed_write_paths)}",
                                                    'restriction_type': 'ALLOWED_WRITE_PATHS'
                                                }]
                                                logging.info(colored(f"⛔ Write denied by ALLOWED_WRITE_PATHS enforcement: {abs_path}", "red"))
                                                logging.info(colored(f"   Allowed paths: {allowed_write_paths}", "yellow"))
                                                break

                                # Only proceed to permission prompt if not already denied by path restriction
                                if should_execute:
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
                                tools_executed_this_turn = True  # Mark that tools were executed
                                should_break_agent_loop = False  # Flag to break outer agent loop after tool execution
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

                                        # Broadcast tool_update event: running status (frontend live feedback)
                                        if args.gui or args.api:
                                            try:
                                                from api.server import broadcast_tool_update
                                                description = f"Execute: {first_block.strip()[:100]}" if first_block else f"Running {tool_name}"
                                                await broadcast_tool_update(
                                                    tool_name=tool_name,
                                                    status="running",
                                                    description=description
                                                )
                                            except Exception as e:
                                                logging.debug(f"Failed to broadcast tool_update: {e}")

                                        # Check if tool is disabled before execution
                                        user_disabled_tools = getattr(g, 'disabled_tools', [])
                                        if tool_name in user_disabled_tools:
                                            tool_status = "error"
                                            tool_error = "tool_disabled"
                                            tool_result = f"ERROR: The '{tool_name}' tool has been disabled by the user and cannot be executed. You must find an alternative approach that does not use this tool."
                                            logging.warning(colored(f"⚠️  Blocked execution of disabled tool: {tool_name}", "yellow"))
                                        else:
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

                                    # Broadcast tool_update event: completion status (frontend live feedback)
                                    if args.gui or args.api:
                                        try:
                                            from api.server import broadcast_tool_update
                                            result_preview = tool_result[:200] if tool_result else ""
                                            await broadcast_tool_update(
                                                tool_name=tool_name,
                                                status=tool_status,
                                                result=result_preview,
                                                duration_ms=tool_duration_ms,
                                                error=tool_error
                                            )
                                        except Exception as e:
                                            logging.debug(f"Failed to broadcast tool_update: {e}")

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

                                    # Add model key to tool info - use the model that generated this response
                                    try:
                                        # Use g.current_llm_model which was set when the LLM response was generated
                                        # (not get_last_used_model which might be from a later LLM call)
                                        used_model = g.current_llm_model if hasattr(g, 'current_llm_model') else LlmRouter.get_last_used_model()
                                        tool_info["model"] = used_model
                                        if used_model and used_model != 'unknown':
                                            logging.debug(f"✓ Added model '{used_model}' to tool_info for {tool_name}")
                                        else:
                                            logging.warning(f"⚠ No model available for tool {tool_name}, using 'unknown'")
                                    except Exception as e:
                                        logging.error(f"❌ Could not get model for tool {tool_name}: {e}")
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

Use <searchresource><resource_id>{resource_id}</resource_id><search>YOUR_PATTERN</search></searchresource> to search this content.
Use <searchresource><resource_id>{resource_id}</resource_id><lines>1-50</lines></searchresource> to view specific line ranges.

Examples:
- Search for errors: <searchresource><resource_id>{resource_id}</resource_id><search>error|failed|exception</search><context>2</context></searchresource>
- View beginning: <searchresource><resource_id>{resource_id}</resource_id><lines>1-20</lines></searchresource>
- View end: <searchresource><resource_id>{resource_id}</resource_id><lines>{max(1, lines_count-19)}-{lines_count}</lines></searchresource>"""
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
                                    g.last_executed_tools_info = executed_tools_info
                                    logging.debug(f"Stored execution info for {len(executed_tools_info)} tools")

                                # Add tool results to chat - this appends to existing assistant message
                                if (len(returned_context) > 0):
                                    metadata = {}
                                    if executed_tools_info:
                                        metadata['tool_info'] = executed_tools_info
                                    context_chat.add_message(Role.ASSISTANT, returned_context, metadata=metadata)

                                    # Broadcast the updated assistant message (content has grown)
                                    if args.gui or args.api:
                                        try:
                                            from api.server import broadcast_message_added
                                            await broadcast_message_added(context_chat.messages[-1])
                                            logging.debug("Broadcasted updated assistant message")
                                        except Exception:
                                            pass

                                    # Check for images from tool context and add them to the chat
                                    if tool_context.has_images():
                                        context_chat.base64_images.extend(tool_context.get_images())
                                        logging.info(colored(f"📸 Added {len(tool_context.get_images())} image(s) to chat context for vision analysis", "blue"))
                                        tool_context.clear_images()  # Clear images after adding them

                                # Clear assistant_response before checking if we should break/continue
                                assistant_response = ""

                                # --- TTS: Speak the response after tools have executed ---
                                # This ensures visualresponse content is processed before TTS
                                # Use original_assistant_response since assistant_response was cleared
                                # IMPORTANT: This must run BEFORE the break statement to ensure TTS for visualresponse!
                                # Skip backend TTS in GUI mode - frontend handles it via /api/audio/tts endpoint
                                if g.voice_mode and not g.GUI_MODE and 'original_assistant_response' in locals() and original_assistant_response.strip():
                                    try:
                                        from shared.utils_audio import filter_latex_for_tts
                                        from shared.audio_router import route_tts_audio, generate_tts_audio
                                        # Note: re is already imported at module level (line 90)

                                        logging.debug(f"TTS: Starting TTS processing (voice_mode={g.voice_mode})")
                                        logging.debug(f"TTS: Response length: {len(original_assistant_response)} chars")

                                        # --- TIER 1: Tool-Aware Content Filtering ---
                                        # Check if visualresponse tool is being used
                                        skip_tts = False
                                        response_text = None

                                        # Detect visualresponse tool call using extract_blocks for proper parsing
                                        visualresponse_blocks = get_extract_blocks()(original_assistant_response, ['visualresponse'])
                                        if visualresponse_blocks:
                                            logging.debug(f"TTS: Found {len(visualresponse_blocks)} visualresponse blocks")
                                            # Parse the visualresponse content (which is now the inner XML)
                                            visualresponse_content = visualresponse_blocks[0]
                                            # Check the type tag inside visualresponse
                                            type_match = re.search(r'<type>(.*?)</type>', visualresponse_content, re.DOTALL | re.IGNORECASE)
                                            if type_match:
                                                viz_type = type_match.group(1).strip().lower()
                                                logging.debug(f"TTS: Visualresponse type: {viz_type}")
                                                if viz_type in ['html', 'code', 'javascript', 'css']:
                                                    # Skip TTS for visual-only content types
                                                    skip_tts = True
                                                    logging.debug(f"TTS: Skipping TTS for visual-only type: {viz_type}")
                                                elif viz_type == 'markdown':
                                                    # For markdown, extract the content and speak it
                                                    # Use extract_blocks on the inner content to handle escaped chars
                                                    content_blocks = get_extract_blocks()(visualresponse_content, ['content'])
                                                    if content_blocks:
                                                        markdown_content = content_blocks[0].strip()
                                                        logging.debug(f"TTS: Extracted markdown content ({len(markdown_content)} chars)")
                                                        # Remove CDATA wrapper if present
                                                        if markdown_content.startswith('<![CDATA['):
                                                            markdown_content = markdown_content[9:-3].strip()
                                                        # Note: markdown formatting will be removed by filter_latex_for_tts()
                                                        response_text = markdown_content
                                                    else:
                                                        logging.debug("TTS: No content blocks found in visualresponse")
                                        else:
                                            logging.debug("TTS: No visualresponse blocks found")

                                        if not skip_tts and response_text is None:
                                            # Extract text from <response> tags if present, otherwise use full response
                                            response_match = re.search(r'<response>(.*?)</response>', original_assistant_response, re.DOTALL)
                                            if response_match:
                                                response_text = response_match.group(1).strip()
                                                logging.debug(f"TTS: Extracted from <response> tag ({len(response_text)} chars)")
                                            else:
                                                # Remove all XML tags for TTS
                                                response_text = re.sub(r'<[^>]+>', '', original_assistant_response).strip()
                                                logging.debug(f"TTS: Extracted by stripping XML tags ({len(response_text)} chars)")

                                        # --- TIER 2: LaTeX/Math Formula Filtering ---
                                        if not skip_tts and response_text:
                                            original_length = len(response_text)
                                            response_text = filter_latex_for_tts(response_text)
                                            logging.debug(f"TTS: After LaTeX filtering: {len(response_text)} chars (was {original_length})")

                                        # Speak the filtered text using audio router
                                        if not skip_tts and response_text:
                                            logging.debug(f"TTS: Speaking {len(response_text)} chars: '{response_text[:100]}...'")
                                            # Generate audio first
                                            audio_data = await generate_tts_audio(
                                                text=response_text,
                                                speed=0.9,
                                                volume=0.8,
                                                use_piper=True
                                            )
                                            # Route audio to appropriate playback mechanism
                                            if audio_data:
                                                await route_tts_audio(
                                                    text=response_text,
                                                    audio_data=audio_data,
                                                    speed=0.9,
                                                    volume=0.8
                                                )
                                        else:
                                            logging.debug(f"TTS: Not speaking (skip_tts={skip_tts}, response_text={'None' if response_text is None else f'{len(response_text)} chars'})")
                                    except Exception as e:
                                        logging.warning(f"Failed to speak response via TTS: {e}")
                                        logging.debug(f"TTS error traceback: {traceback.format_exc()}")

                                # Now check if we should break (after TTS has been processed)
                                if (tool_name in ["response", "visualresponse"] and "error" not in returned_context.lower()):
                                    # Set flag to break from outer agent loop after tool execution completes
                                    should_break_agent_loop = True
                                    break  # Break from tool execution for loop

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

                        # Check if we should break from agent loop after tool execution
                        if should_break_agent_loop:
                            logging.debug("Breaking from agent loop after visualresponse/response tool execution")
                            break  # Break from outer agent while loop

                    else:
                        # No tools or code found, this is a conversational response.
                        # In GUI mode, notify frontend if we expected tool calls (but only if no tools were executed this turn)
                        if hasattr(g, 'GUI_MODE') and g.GUI_MODE and assistant_response.strip() and not tools_executed_this_turn:
                            try:
                                from tools.visualresponse import visualresponse
                                session = visualresponse._get_session()
                                if session:
                                    import time
                                    await session.send_update_async({
                                        'type': 'assistant_message',
                                        'action': 'display',
                                        'message': '<context>\n⚠️ Warning: Expected tool call but received plain text response.\nThe model may have failed to use required tools.\n\nResponse received:\n' + assistant_response[:200] + ('...' if len(assistant_response) > 200 else '') + '\n</context>',
                                        'id': int(time.time()),
                                        'timestamp': time.time()
                                    })
                                    logging.warning("Sent 'no tool call' warning to frontend")
                            except Exception as e:
                                logging.debug(f"Failed to send warning to frontend: {e}")
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
