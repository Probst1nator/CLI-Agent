# main.py
import time
# Global timer for startup and operations
start_time = time.time()
import logging
import re
from termcolor import colored
from typing import List, Optional, Tuple
import sys
import argparse
import os
import math
import asyncio
from collections import Counter

# --- Custom Logging Setup ---

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
    """A simple formatter that just returns the message, used after startup."""
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

def setup_logging(debug_mode: bool, log_file: Optional[str] = None):
    """Configure the root logger for console and optional file output."""
    logger = logging.getLogger() # Get root logger
    logger.setLevel(logging.DEBUG) # Capture all messages at the root

    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_log_level = logging.DEBUG if debug_mode else logging.INFO
    console_handler.setLevel(console_log_level)
    console_formatter = ConsoleFormatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # --- File Handler (optional) ---
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG) # Always log everything to the file
        file_formatter = FileFormatter("%(elapsed)s [%(levelname)-8s] %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logging.info(colored(f"Logging to file: {log_file}", "magenta")) # Use standard logging so it gets timed

    # --- FIX: Silence Noisy Loggers in Production Mode ---
    if not debug_mode:
        # These libraries produce a lot of INFO-level logs we don't need in a normal run.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("py_classes.ai_providers.cls_ollama_interface").setLevel(logging.WARNING)
        logging.getLogger("numexpr").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)

def swap_to_simple_logging():
    """Swaps the console handler's formatter to a simpler one without the timer."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            # We found the console handler, now swap its formatter
            simple_formatter = SimpleFormatter("%(message)s")
            handler.setFormatter(simple_formatter)
            break

# --- End Custom Logging Setup ---


def print_startup_summary(args: argparse.Namespace):
    """Prints a summary of the current agent configuration using the logger."""
    
    # Helper for status strings
    def get_status_str(state, on_text='ON', off_text='OFF'):
        return colored(on_text, 'green', attrs=['bold']) if state else colored(off_text, 'red')

    # Column widths for alignment
    CMD_WIDTH = 24
    DESC_WIDTH = 28
    
    # Use logger.info to ensure timed prefix is applied
    from py_classes.globals import g # Lazy import to avoid circular dependency at top level
    
    logging.info(colored("--- Agent Configuration ---", "yellow"))
    
    # Toggles - show filtered models when local mode is on
    if g.SELECTED_LLMS:
        if args.local:
            # Filter to only local models (those with ':' in the name)
            local_models = [model for model in g.SELECTED_LLMS if ':' in model]
            if local_models:
                llm_status = f"{colored(str(len(local_models)), 'green')} local models ({colored(str(len(g.SELECTED_LLMS)), 'yellow')} total selected)"
            else:
                llm_status = colored('No local models available', 'red')
        else:
            unique_models = list(set(g.SELECTED_LLMS))
            llm_status = f"{colored(str(len(unique_models)), 'green')} models selected"
    else:
        llm_status = colored('Default', 'yellow')
    
    # Calculate guard count for Auto Execution status and eval count for MCT status
    llm_config = g.get_llm_config()
    guard_count = sum(
        data.get('guard', 0) 
        for data in llm_config.values() 
        if data.get('selected')
    )
    eval_count = sum(
        data.get('eval', 0) 
        for data in llm_config.values() 
        if data.get('selected')
    )
    
    # Auto Execution status with guard info
    if args.auto:
        if guard_count > 0:
            auto_status_text_on = f'ON ({guard_count} guards)'
        else:
            auto_status_text_on = 'ON (manual confirmation)'
        auto_status_text_off = 'OFF'
    else:
        auto_status_text_on = f'ON ({guard_count} guards)' if guard_count > 0 else 'ON (manual confirmation)'
        auto_status_text_off = 'OFF'
    
    # --- MCT status string for branch count display ---
    eval_suffix = f', {eval_count} evals' if eval_count > 0 else ''
    if args.mct == -1:
        # -mct used without integer - show auto mode
        if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 0:
            mct_status_text_on = f'ON ({len(g.SELECTED_LLMS)} branching LLMs{eval_suffix})'
        else:
            mct_status_text_on = f'ON (3 branches default{eval_suffix})'
        mct_status_text_off = 'OFF'
    else:
        mct_status_text_on = f'ON ({args.mct} branches{eval_suffix})'
        mct_status_text_off = f'OFF ({args.mct} branch)'
    
    config_lines = [
        f"  {colored('Local Mode (-l)'.ljust(CMD_WIDTH), 'white')} {colored('Use only local LLMs'.ljust(DESC_WIDTH), 'light_grey')} (Status: {get_status_str(args.local)})",
        f"  {colored('Auto Execution (-a)'.ljust(CMD_WIDTH), 'white')} {colored('Automatic code execution'.ljust(DESC_WIDTH), 'light_grey')} (Status: {get_status_str(args.auto, on_text=auto_status_text_on, off_text=auto_status_text_off)})",
        f"  {colored('MCT Branches (--mct)'.ljust(CMD_WIDTH), 'white')} {colored('Monte Carlo Tree Search'.ljust(DESC_WIDTH), 'light_grey')} (Status: {get_status_str(args.mct > 1 or args.mct == -1, on_text=mct_status_text_on, off_text=mct_status_text_off)})",
        f"  {colored('LLM Selection (--llm)'.ljust(CMD_WIDTH), 'white')} {colored('Specific LLM(s) to use'.ljust(DESC_WIDTH), 'light_grey')} (Status: {llm_status})",
        f"  {colored('Voice Mode (-v)'.ljust(CMD_WIDTH), 'white')} {colored('Voice input/output'.ljust(DESC_WIDTH), 'light_grey')} (Status: {get_status_str(args.voice)})"
    ]
    for line in config_lines:
        logging.info(line)
    logging.info(colored("Type '-h' for a full list of commands.", "dark_grey"))


def parse_cli_args() -> argparse.Namespace:
    """Setup and parse CLI arguments, ensuring the script's functionality remains intact."""

    # --- FIX: Pre-process sys.argv to handle shell escaping issues. ---
    # This specifically targets cases like '-mct\ 2' which a shell or debugger might
    # pass as a single argument string '-mct 2'. Argparse expects two separate tokens.
    raw_args = sys.argv[1:]
    processed_args = []
    for arg in raw_args:
        # Specifically look for '-mct' or '--mct' followed by a space and digits, all in one token.
        match = re.match(r'^(-mct|--mct)\s+(\d+)$', arg)
        if match:
            # Split the single token into two: the flag and its value.
            processed_args.append(match.group(1)) # e.g., '-mct'
            processed_args.append(match.group(2)) # e.g., '2'
        else:
            processed_args.append(arg)
    
    parser = argparse.ArgumentParser(
        description="AI CLI-Agent with backend options and more.",
        add_help=False  # Disable automatic help to manually handle unrecognized args
    )

    parser.add_argument("-h", "--help", action="store_true", default=False,
                        help="Display this help")
    parser.add_argument("-a", "--auto", action="store_true", default=False,
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
    parser.add_argument("--speak", action="store_true", default=False,
                        help="Text-to-speech output.")
    parser.add_argument("--full_output", dest='full_output_mode', action="store_true", default=False,
                        help="Full output mode. Shows live model output instead of hiding it.")
    parser.add_argument("-f", "--fast", action="store_true", default=False,
                        help="Use only fast LLMs. Sets g.FORCE_FAST=True.")
    parser.add_argument("-img", "--image", action="store_true", default=False,
                        help="Take a screenshot using Spectacle for region selection (with automatic fallbacks if not available).")
    parser.add_argument("-mct", "--mct", type=int, nargs='?', const=-1, default=1,
                        help="Enable Monte Carlo Tree Search for acting. Use -mct alone to auto-select branches (3 default, or all selected LLMs if using --llm), or -mct N for specific count (default: 1).")
    parser.add_argument("-sbx", "--sandbox", action="store_true", default=False,
                        help="Use weakly sandboxed python execution. Sets g.USE_SANDBOX=True.")
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
    parser.add_argument("--local-exec-confirm", action="store_true", default=False,
                        help="Use local LLM for auto-execution confirmation instead of cloud models.")
    
    # --- Logging Arguments ---
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable detailed debug logging to the console.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to a file to write detailed logs.")
    # --- End Logging Arguments ---

    parser.add_argument("-e", "--exit", action="store_true", default=False,
                        help="Exit after all automatic messages have been parsed successfully")
    
    # Parse the processed arguments instead of the raw ones from sys.argv
    args, unknown_args = parser.parse_known_args(processed_args)

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


# Display cool ASCII header
from pyfiglet import figlet_format

# Now print the header and startup summary
header_dashes = colored("# # # # # # # # # # # # # # # # # # # # # # # # # #", "blue")
header = figlet_format("CLI-Agent", font="slant")
print(header_dashes)
print(colored(header, "cyan", attrs=["bold"]))
print(header_dashes)


# --- Main Application Imports (Lazy/Normal) ---
# Imports are moved below arg parsing and logging setup to allow for faster startup
# and to ensure logging is configured before any modules start logging.
import datetime
import json
import traceback
from dotenv import load_dotenv
import pyperclip
import socket
import warnings
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.widgets import CheckboxList, Frame, Label
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
import base64
import tempfile
import subprocess
import importlib.util

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")
warnings.filterwarnings("ignore", message="words count mismatch on*", module="phonemizer", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")
warnings.filterwarnings("ignore", message="Unrecognized FinishReason enum value*", module="proto.marshal.rules.enums", category=UserWarning)

from py_classes.cls_computational_notebook import ComputationalNotebook
from py_classes.cls_util_manager import UtilsManager
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_llm_router import LlmRouter, StreamInterruptedException
from py_classes.cls_chat import Chat, Role
from py_classes.utils.cls_utils_web_server import WebServer
from py_classes.globals import g
from py_classes.cls_text_stream_painter import TextStreamPainter

# --- Final Logging Setup ---
# This is done *after* all imports to ensure our configuration is not overwritten by a rogue library.
setup_logging(args.debug, args.log_file)
logging.info("Starting CLI-Agent...")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.info("Default imports success...")

# Lazy load functions from py_methods.utils to avoid heavy imports at startup
def get_extract_blocks():
    from py_methods.utils import extract_blocks  # noqa: E402
    return extract_blocks


def get_listen_microphone():
    from py_methods.utils import listen_microphone  # noqa: E402
    return listen_microphone

def get_take_screenshot():
    from py_methods.utils import take_screenshot  # noqa: E402
    return take_screenshot

def get_update_cmd_collection():
    from py_methods.utils import update_cmd_collection  # noqa: E402
    return update_cmd_collection

# Try importing with a direct import
try:
    from utils.viewimage import ViewImage
except ImportError:
    spec = importlib.util.spec_from_file_location("ViewImage", 
                                                 os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                             "utils", "viewimage.py"))
    viewimage_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewimage_module)
    ViewImage = viewimage_module.ViewImage

try:
    from utils.tts import TtsUtil
except ImportError:
    class TtsUtil:
        @staticmethod
        def run(text, **kwargs):
            logging.warning(f"TtsUtil not found. Cannot speak: {text}")
            return json.dumps({"status": "error", "message": "TtsUtil not found"})
from utils.todos import TodosUtil

logging.info("Util imports success...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        logging.warning(f"Could not determine local IP: {e}")
        return None

def apply_default_llm_config():
    """
    Loads the cached LLM configuration from disk and applies it as the default
    for the current session. This makes LLM selections persistent.
    """
    llm_config = g.get_llm_config()
    if not llm_config:
        return # No saved config to apply

    selected_llms = [key for key, data in llm_config.items() if data.get('selected')]
    evaluator_llms = [key for key, data in llm_config.items() if data.get('eval', 0) > 0]
    
    # Only apply if there are actual selections, to not override command line flags unintentionally
    if selected_llms:
        g.SELECTED_LLMS = selected_llms
        g.EVALUATOR_LLMS = evaluator_llms
        # Count unique models by provider
        unique_models = list(set(selected_llms))  # Deduplicate
        provider_counts = {}
        for model in unique_models:
            if ':' in model:
                provider = "🏠 Local" 
            else:
                provider_name = model.split('-')[0].title()
                provider = f"☁️  {provider_name}"
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        provider_summary = ", ".join([f"{provider}: {count}" for provider, count in provider_counts.items()])
        logging.info(f"📋 Loaded default LLM configuration. Total: {colored(str(len(unique_models)), 'green')} models ({provider_summary})")
        
        if evaluator_llms:
            logging.info(f"⚖️  Default evaluators: {colored(str(len(evaluator_llms)), 'green')} models ({', '.join(evaluator_llms)})")

async def llm_selection(args: argparse.Namespace, preselected_llms: Optional[List[str]] = None) -> List[str]:
    """
    Handles the LLM selection process using the enhanced LlmSelector UI.
    This function initializes the selector, awaits user input, and processes
    the results to configure the agent's LLM settings.
    """
    from py_classes.cls_llm_selection import LlmSelector
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
    
    # Identify evaluator models and set them in globals
    evaluator_models = [config['model_key'] for config in selected_configs if config.get('eval', 0) > 0]
    g.EVALUATOR_LLMS = evaluator_models
    
    selected_llms = [config['model_key'] for config in selected_configs]
    
    # Update globals with the latest selection
    args.llm = selected_llms
    g.SELECTED_LLMS = selected_llms

    # Reload the config in `g` so the main loop gets the fresh settings
    g.load_llm_config()

    if not args.llm:
        logging.info(colored("# cli-agent: LLM set to auto, type (-h) for info", "green") )
    else:
        # Count unique models by provider and configuration
        unique_models = list(set(args.llm))  # Deduplicate
        provider_counts = {}
        branches_count = eval_count = guard_count = 0
        
        for model in unique_models:
            if ':' in model:
                provider = "🏠 Local"
            else:
                provider_name = model.split('-')[0].title()
                provider = f"☁️  {provider_name}"
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        for config in selected_configs:
            if config.get('beams', 0) > 0 or config.get('branches', 0) > 0:  # Support both old and new terminology
                branches_count += 1
            if config.get('eval', 0) > 0:
                eval_count += 1
            if config.get('guard', 0) > 0:
                guard_count += 1
        
        provider_summary = ", ".join([f"{provider}: {count}" for provider, count in provider_counts.items()])
        
        config_summary = []
        if branches_count > 0:
            config_summary.append(f"🌳 Branches: {colored(str(branches_count), 'green')}")
        if eval_count > 0:
            config_summary.append(f"⚖️  Eval: {colored(str(eval_count), 'green')}")
        if guard_count > 0:
            config_summary.append(f"🛡️  Guard: {colored(str(guard_count), 'green')}")
        
        logging.info(f"📋 LLM(s) selection updated. Total: {colored(str(len(unique_models)), 'green')} models ({provider_summary})")
        
        if config_summary:
            logging.info(f"⚙️  Configurations: {', '.join(config_summary)}")
        
        if g.EVALUATOR_LLMS:
            logging.info(f"⚖️  Evaluators: {colored(str(len(g.EVALUATOR_LLMS)), 'green')} models")
        
        # Auto-sync args.auto with guard LLMs when selection changes
        if guard_count > 0 and not args.auto:
            args.auto = True
            logging.info(colored(f"🔄 Auto-execution enabled: {guard_count} guard LLMs detected", "green"))
        elif guard_count == 0 and args.auto:
            # Inform user that auto is on but no guards are available
            logging.info(colored("⚠️  Auto-execution is enabled but no guard LLMs configured (manual confirmation required)", "yellow"))

    return selected_llms


async def utils_selection(args: argparse.Namespace) -> List[str]:
    """
    Handles the Utils selection process, supporting multi-selection mode.
    """
    utils_manager = UtilsManager()
    available_utils = utils_manager.get_util_names()
    util_choices = []
    add_new_tool_id = "__add_new_tool__"
    util_choices.append((add_new_tool_id, HTML('<special>+ Add New Tool</special>')))
    for util_name in available_utils:
        styled_text = HTML(f'<util>{util_name}</util>')
        util_choices.append((util_name, styled_text))
    style = Style.from_dict({'util': 'ansigreen', 'special': 'ansiyellow bold'})
    checkbox_list = CheckboxList(values=util_choices)
    bindings = KeyBindings()
    @bindings.add("e")
    def _execute(event) -> None:
        selected = [value for value, _ in util_choices if value in checkbox_list.current_values]
        app.exit(result=selected)
    @bindings.add("a")
    def _abort(event) -> None:
        app.exit(result=None)
    instructions = Label(text="Use arrow keys to navigate, Space to select/deselect utils, 'e' to confirm or 'a' to abort")
    root_container = HSplit([Frame(title="Select Utils to Use", body=checkbox_list), instructions])
    layout = Layout(root_container)
    app = Application(layout=layout, key_bindings=bindings, full_screen=False, style=style)
    try:
        selected_utils = await app.run_async()
        if selected_utils is not None and len(selected_utils) > 0:
            if add_new_tool_id in selected_utils:
                return [add_new_tool_id]
            utils_list = ", ".join(selected_utils)
            logging.info(colored(f"# cli-agent: Selected utils: {utils_list}", "green"))
            return selected_utils
        else:
            logging.warning(colored("# cli-agent: No utils selected or selection cancelled", "yellow"))
            return []
    except asyncio.CancelledError:
        logging.warning(colored("# cli-agent: Utils selection was interrupted", "yellow"))
        return []
    except Exception as e:
        logging.error(colored(f"# cli-agent: Error during utils selection: {str(e)}", "red"))
        if args.debug:
            logging.exception("Utils selection failed with traceback:")
        return []
def create_interruption_callback(
    response_buffer: List[str],
    painter: TextStreamPainter,
    lock: Optional[asyncio.Lock] = None,
):
    """
    Factory to create a stateful callback for handling LLM streams.

    This callback prints the stream, accumulates it into a buffer, and can
    interrupt by raising StreamInterruptedException when a complete code block
    (e.g., ```...```) or a complete tag (e.g., <tool>...</tool>) is detected.
    This logic is consistent with the `extract_blocks` utility.

    Args:
        response_buffer: A list with a single string to act as a mutable buffer.
        painter: The TextStreamPainter for coloring output.
        lock: An optional asyncio.Lock for concurrent-safe printing.

    Returns:
        An async callback function.
    """
    # Pre-compile regex for efficiency. re.DOTALL allows '.' to match newlines.
    CODE_BLOCK_REGEX = re.compile(r"```.*?(```)", re.DOTALL)
    TAG_BLOCK_REGEX = re.compile(r"<([a-zA-Z0-9_]+)[^>]*>.*?</\1>", re.DOTALL)
    
    async def _callback(chunk: str):
        # Process the chunk character by character to ensure proper interruption
        for char in chunk:
            # 1. Print the incoming character with the live "typing" effect (only if not in summary mode)
            if not g.SUMMARY_MODE:
                if lock:
                    async with lock:
                        print(painter.apply_color(char), end="", flush=True)
                else:
                    print(painter.apply_color(char), end="", flush=True)

            # 2. Append the character to the shared buffer
            response_buffer[0] += char
            current_buffer = response_buffer[0]

            # 3. Check for interruption conditions after each character, but only when necessary
            # Only check for patterns when we encounter potentially significant characters
            should_check = char in '`><'
            
            if should_check:
                # Check for a complete Markdown code block
                code_match = CODE_BLOCK_REGEX.search(current_buffer)
                if code_match:
                    full_block = code_match.group(0).strip()
                    if not g.SUMMARY_MODE:
                        print("\n") # Add a newline for clean UI separation only in normal mode
                    raise StreamInterruptedException(full_block)

                # Check for a complete XML/HTML-like tag block
                tag_match = TAG_BLOCK_REGEX.search(current_buffer)
                if tag_match:
                    full_block = tag_match.group(0).strip()
                    if not g.SUMMARY_MODE:
                        print("\n")
                    raise StreamInterruptedException(full_block)
                    
                # CRITICAL: Also interrupt if we detect hallucinated execution output
                if "<execution_output>" in current_buffer:
                    logging.warning(colored("🚨 INTERRUPTING: Detected hallucinated execution_output tag!", "red"))
                    # Find the content up to the execution_output tag
                    output_pos = current_buffer.find("<execution_output>")
                    content_before_output = current_buffer[:output_pos].strip()
                    if not g.SUMMARY_MODE:
                        print("\n")
                    raise StreamInterruptedException(content_before_output)

    return _callback

async def confirm_code_execution(args: argparse.Namespace, code_to_execute: str, utils_manager: 'UtilsManager') -> bool:
    """
    Handles code execution confirmation based on the 'Guard' configuration in the LLM selector.
    If Guard sum > 0, it runs a majority vote. If Guard sum == 0, it requires manual user confirmation.
    """
    # Fast-path for simple, non-destructive shell commands
    if not get_extract_blocks()(code_to_execute, "python"):
        always_permitted_bash = ["ls ", "pwd ", "cd ", "echo ", "print ", "cat ", "head ", "tail ", "grep ", "sed ", "awk ", "sort "]
        bash_code = "\n".join(get_extract_blocks()(code_to_execute, ["bash", "shell"]))
        bash_code_lines = [line for line in bash_code.split("\n") if line.strip() and not line.strip().startswith("#")]
        if bash_code_lines and all(any(line.strip().startswith(cmd) for cmd in always_permitted_bash) for line in bash_code_lines):
            logging.info(colored("✅ Code execution permitted automatically (safe command list)", "green"))
            return True

    # --- New Guard-based Logic ---
    llm_config = g.get_llm_config()
    total_guards = sum(
        data.get('guard', 0) 
        for data in llm_config.values() 
        if data.get('selected')
    )

    # 1. If no guards are configured, ALWAYS require manual confirmation.
    if total_guards == 0:
        logging.warning(colored("🛡️", "red", attrs=["bold"]) + colored(" No Guard models configured. Manual confirmation required for execution.", "yellow"))
        user_input = await get_user_input_with_bindings(args, None, colored("(Press Enter to confirm or 'n' to abort)", "cyan"))
        if user_input.lower() == 'n':
            logging.error(colored("❌ Code execution aborted by user", "red"))
            return False
        else:
            logging.info(colored("✅ Code execution permitted by user", "green"))
            return True

    # 2. If guards ARE configured, proceed with the Guard Council vote.
    guard_council = []
    for model_key, data in llm_config.items():
        if data.get('selected') and data.get('guard', 0) > 0:
            guard_council.extend([model_key] * data['guard'])

    logging.info(colored("🛡️", "red", attrs=["bold"]) + colored(f"  {len(guard_council)} Guards verifying code request: {', '.join(guard_council)}", "cyan"))
    
    async def get_verdict(model_key: str):
        # This is a self-contained chat for the guard to prevent context pollution
        execution_guard_chat = Chat(
            instruction_message=f"""You are a Code Execution Guardian. Your primary goal is to prevent unsafe or incomplete code execution. Analyze the code for safety and completeness.
- SAFE operations: Reading files (ls, cat), simple data retrieval (curl to public APIs), basic system info (ps, uname), file modifications with clear comments.
- UNSAFE operations: File deletions (rm -rf), system modifications (sudo), unrestricted network access, running unknown executables.
- INCOMPLETE code: Placeholders like `YOUR_API_KEY`, `TODO`, or unimplemented functions.

**Process:**
1. **Brief Analysis:** Explain your reasoning in 1-2 sentences.
2. **Single Word Verdict:** End your response with exactly one word: `yes`, `no`, or `unfinished`.

**The following classes are custom and considered trusted, they must still be verified for reliable and intentional usage in code**
{utils_manager.get_util_names()}
""",
            debug_title=f"🛡️  Guard Vote ({model_key})"
        )
        analysis_prompt = f"Analyze this code for safe execution and completeness:\n{code_to_execute}"
        execution_guard_chat.add_message(Role.USER, analysis_prompt)
        response = await LlmRouter.generate_completion(
            execution_guard_chat, 
            [model_key], 
            force_preferred_model=True,
            decision_patterns={"guard": r'\b(yes|no|unfinished)\b'}
        )
        
        # Find all verdict matches and take the LAST one (as instructed to end with verdict)
        verdict_matches = list(re.finditer(r'\b(yes|no|unfinished)\b', response.lower(), re.DOTALL | re.MULTILINE))
        if verdict_matches:
            verdict_match = verdict_matches[-1]  # Take the last match
            verdict = verdict_match.group(1)
        else:
            verdict = "no"  # Default to no if no verdict found
        analysis = response.strip()
        
        return verdict, analysis, model_key

    tasks = [get_verdict(model) for model in guard_council]
    results = await asyncio.gather(*tasks)
    
    vote_counts = Counter(v[0] for v in results)
    
    # Create detailed verdict summary like branch selection voting
    verdict_details = []
    for verdict in ['yes', 'no', 'unfinished']:
        count = vote_counts.get(verdict, 0)
        if count > 0:
            models_with_verdict = [model for v, _, model in results if v == verdict]
            verdict_details.append(f"{verdict} ({count} votes: {', '.join(models_with_verdict)})")
    
    logging.info(colored("🛡️", "red", attrs=["bold"]) + colored(f"  Guard verdict summary: {', '.join(verdict_details)}", "light_blue"))

    # Determine final verdict with safety-first tie-breaking (no > unfinished > yes)
    if vote_counts.get('no', 0) >= vote_counts.get('unfinished', 0) and vote_counts.get('no', 0) >= vote_counts.get('yes', 0):
        final_verdict = 'no'
    elif vote_counts.get('unfinished', 0) >= vote_counts.get('yes', 0):
        final_verdict = 'unfinished'
    else:
        final_verdict = 'yes'

    aggregated_analysis = "\n".join([f"- ({model}): {analysis}" for v, analysis, model in results if v == final_verdict])
    
    if 'yes' == final_verdict:
        logging.info(colored("✅ Code guard permitted execution by majority vote. Reasoning:\n", "green") + colored(aggregated_analysis, "light_green"))
        return True
    elif 'unfinished' == final_verdict:
        logging.warning(colored("⚠️ Code guard prompted code revision by majority vote. Reasoning:\n", "yellow") + colored(aggregated_analysis, "light_magenta"))
        args.message.insert(0, "The code was deemed safe but unfinished. Explaining the code and its use via comments can convince the reviewers. Code review:\n" + aggregated_analysis)
        logging.info(colored("💬 Auto prompting with reasoning...", "blue"))
        return False
    else: # 'no'
        logging.error(colored("❌ Code execution aborted by majority vote. Reasoning:\n", "red") + colored(aggregated_analysis, "magenta"))
        user_response = await get_user_input_with_bindings(args, None, colored("Do you want to manually execute this code? (y/n): ", "cyan"))
        if user_response.lower() == 'y':
            logging.info(colored("✅ User overrode guard and permitted execution.", "green"))
            return True
        logging.error(colored("❌ Execution remains aborted.", "red"))
        return False

async def select_best_branch(
    context_chat: Chat,
    assistant_responses: List[str],
) -> Tuple[int, str]:
    """
    Uses a council of LLM judges to select the best response from multiple MCT branches.
    
    Returns:
        Tuple[int, str]: (selected_index, vote_summary_string)
    """
    # --- 1. Identify Evaluator Models ---
    llm_config = g.get_llm_config()
    evaluator_council = []
    for model_key, data in llm_config.items():
        if data.get('selected') and data.get('eval', 0) > 0:
            evaluator_council.extend([model_key] * data['eval'])

    # Fallback: If no evaluators are configured, use the first selected LLM as the judge.
    if not evaluator_council:
        if g.SELECTED_LLMS:
            evaluator_council.append(g.SELECTED_LLMS[0])
            logging.info(colored(f"No specific evaluator set. Using primary LLM to judge responses: {evaluator_council[0]}", "cyan"))
        else:
            logging.error(colored("⚠️ No evaluators or selected LLMs to judge branches. Defaulting to first branch.", "red"))
            return 0, "No evaluator available."

    num_alternatives = len(assistant_responses)
    # The prompt refers to indices 0..N-1
    selection_prompt = f"""You are a branch selection judge. Your task is to review {num_alternatives} response options and select the best one based on helpfulness, accuracy, and completeness.

**Available options:**
"""
    for i, response in enumerate(assistant_responses):
        selection_prompt += f"\n\n--- INDEX {i} ---\n{response}"

    selection_prompt += """

**Your task:**
1.  Carefully review all the indexed responses.
2.  Provide a brief comparative analysis explaining your choice.
3.  **Crucially**, end your entire response with a single line: `Selected index: [number]` where `[number]` is the integer index of your choice.
"""

    async def get_vote(model_key: str):
        # This is a self-contained chat for the judge to prevent context pollution
        judge_chat = Chat(
            instruction_message="You are an impartial judge. Your only goal is to select the best response from the provided list according to the user's instructions.",
            debug_title=f"⚖️  Branch Judge ({model_key})"
        )
        judge_chat.add_message(Role.USER, selection_prompt)
        
        response = await LlmRouter.generate_completion(
            judge_chat, 
            [model_key], 
            force_preferred_model=True,
            decision_patterns={"eval": r'Selected index:\s*(\d+)'}
        )
        
        match = re.search(r'Selected index:\s*(\d+)', response)
        if match:
            selected_index = int(match.group(1))
            if 0 <= selected_index < num_alternatives:
                return selected_index, model_key
        return None, model_key

    # --- 2. Gather votes from the council ---
    logging.info(colored("⚖️", "blue", attrs=["bold"]) + colored(f"  Branch selection by {len(evaluator_council)} voters: {', '.join(evaluator_council)}", "cyan"))
    tasks = [get_vote(model) for model in evaluator_council]
    results = await asyncio.gather(*tasks)

    # Filter out failed votes (where index is None)
    valid_votes = [res for res in results if res[0] is not None]

    if not valid_votes:
        logging.warning(colored("⚠️ All evaluators failed to pick a valid branch. Defaulting to first branch.", "yellow"))
        return 0, "All evaluators failed."
    
    # Show individual vote details
    vote_details = []
    for vote_result in valid_votes:
        index, model = vote_result
        vote_details.append(f"{model} voted for index {index}")
    if vote_details:
        logging.info(colored("📊", "blue") + colored(f" Individual votes: {'; '.join(vote_details)}", "light_blue"))

    # --- 3. Tally votes and create summary ---
    vote_counts = Counter(vote[0] for vote in valid_votes)
    
    # Group models by their vote
    votes_by_index = {}
    for index, model in valid_votes:
        if index not in votes_by_index:
            votes_by_index[index] = []
        votes_by_index[index].append(model)
        
    # Create the summary string
    summary_parts = []
    for index, count in vote_counts.most_common():
        models = ", ".join(votes_by_index[index])
        s = 's' if count > 1 else ''
        summary_parts.append(f"Index {index} ({count} vote{s}: {models})")
    
    vote_summary = f"Vote Summary: {'; '.join(summary_parts)}"

    winning_index = vote_counts.most_common(1)[0][0]

    return winning_index, vote_summary

async def save_current_llm_config_as_default():
    """Saves the current session's LLM selection as the new default."""
    from py_classes.globals import g # Lazy import
    logging.info(colored("💾 Saving current LLM selection as the new default...", "magenta"))
    
    # Get the complete, most recent configuration data for ALL models
    # force_reread ensures we have the latest base config before modifying.
    full_config = g.get_llm_config(force_reread=True) 
    
    # Get the set of models active in THIS session
    currently_selected_models = set(g.SELECTED_LLMS)
    
    # Update the 'selected' status in the full config based on the current session
    for model_key, data in full_config.items():
        data['selected'] = model_key in currently_selected_models
    
    # Save the modified configuration back to the persistent file
    g.save_llm_config(full_config)
    
    logging.info(colored(f"✅ Default configuration updated with {len(currently_selected_models)} selected models.", "green"))


def handle_multiline_input() -> str:
    """Handles multiline input from the user."""
    print(colored("Enter your multiline input. Type '--f' on a new line when finished.", "blue"))
    lines = []
    while True:
        try:
            line = input()
            if line == "--f":
                break
            lines.append(line)
        except EOFError:
            break
    return "\n".join(lines)

async def get_user_input_with_bindings(
    args: argparse.Namespace,
    context_chat: Chat,
    prompt: str = colored("💬 Enter your request: ", 'blue', attrs=["bold"]),
    input_override: str = None,
    force_input: bool = False
) -> str:
    """
    Gets user input, handling special keybindings.
    """
    while True:
        if prompt == "":
            user_input = ""
        elif input_override:
            user_input = input_override
        elif force_input:
            user_input = input(prompt)
        else:
            try:
                if context_chat:
                    user_input = input(f"\n[Tokens: {math.ceil(len(context_chat.__str__())*3/4)} | Messages: {len(context_chat.messages)}] " + prompt)
                else:
                    user_input = input(prompt)
            except KeyboardInterrupt:
                logging.warning(colored("\n# cli-agent: Exiting due to Ctrl+C.", "yellow"))
                exit()
        
        if user_input == "-r" or user_input == "--r":
            if not context_chat or len(context_chat.messages) < 2:
                logging.error(colored("# cli-agent: No chat history found, cannot regenerate.", "red"))
                continue
            logging.info(colored("# cli-agent: Regenerating last response.", "green"))
            context_chat.messages.pop()
            user_input = ""
        elif user_input == "-l" or user_input == "--local":
            args.local = not args.local
            g.FORCE_LOCAL = args.local
            logging.info(colored(f"# cli-agent: Local mode toggled {'on' if args.local else 'off'}.", "green"))
            continue
        elif user_input == "-llm" or user_input == "--llm":
            logging.info(colored("# cli-agent: Opening LLM selection.", "green"))
            await llm_selection(args, preselected_llms=g.SELECTED_LLMS)
            continue
        elif user_input == "-a" or user_input == "--auto":
            args.auto = not args.auto
            
            # Check guard count for context
            llm_config = g.get_llm_config()
            guard_count = sum(
                data.get('guard', 0) 
                for data in llm_config.values() 
                if data.get('selected')
            )
            
            status_msg = f"# cli-agent: Automatic execution toggled {'on' if args.auto else 'off'}"
            if args.auto and guard_count > 0:
                status_msg += f" ({guard_count} guards active)"
            elif args.auto and guard_count == 0:
                status_msg += " (manual confirmation required)"
            elif not args.auto and guard_count > 0:
                status_msg += f" (ignoring {guard_count} available guards)"
            
            logging.info(colored(f"{status_msg}.", "green"))
            continue
        elif user_input == "-mct" or user_input == "--mct":
            args.mct = int(input(colored(f"# cli-agent: Enter a branch count (current: {args.mct}): ", "green")))
            logging.info(colored(f"# cli-agent: Monte Carlo Tree Search count set to {args.mct}", "green"))
            if context_chat:
                context_chat.debug_title = "MCTs Branching - Main Context Chat" if args.mct > 1 else "Main Context Chat"
            continue
        elif user_input == "-f" or user_input == "--fast":
            args.fast = not args.fast
            g.FORCE_STRONG = False
            logging.info(colored(f"# cli-agent: Fast LLM mode toggled {'on' if args.fast else 'off'}.", "green"))
            continue
        elif user_input == "-v" or user_input == "--v":
            args.voice = not args.voice
            logging.info(colored(f"# cli-agent: Voice mode toggled {'on' if args.voice else 'off'}.", "green"))
            continue
        elif user_input == "--speak":
            args.speak = not args.speak
            logging.info(colored(f"# cli-agent: Text-to-speech mode toggled {'on' if args.speak else 'off'}.", "green"))
            continue
        elif user_input == "--full_output":
            args.full_output_mode = not args.full_output_mode
            # Invert the logic: full_output_mode = True means SUMMARY_MODE = False
            g.SUMMARY_MODE = not args.full_output_mode
            logging.info(colored(f"# cli-agent: Full output mode toggled {'on' if args.full_output_mode else 'off'}.", "green"))
            continue
        elif user_input in ["-img", "--img", "-screenshot", "--screenshot"] or args.image:
            logging.info(colored("# cli-agent: Taking screenshot.", "green"))
            args.image = False
            await handle_screenshot_capture(context_chat)
            continue
        elif user_input == "-p" or user_input == "--p":
            logging.info(colored("# cli-agent: Printing chat history.", "green"))
            os.system('clear')
            print(colored("Chat history:", "green"))
            if context_chat:
                context_chat.print_chat()
            else:
                print(colored("No chat history available.", "yellow"))
            continue
        elif user_input == "-m" or user_input == "--m":
            return handle_multiline_input()
        elif user_input in ["-o", "--o", "-online", "--online"]:
            args.online = not args.online
            logging.info(colored(f"# cli-agent: Online mode toggled {'on' if args.online else 'off'}.", "green"))
            continue
        elif user_input in ["-e", "--e", "--exit"]:
            logging.info(colored("# cli-agent: Exiting...", "green"))
            exit(0)
        elif user_input in ["-h", "--h", "--help"]:
            print_startup_summary(args) # Reuse the summary function
            # Print the detailed command list
            CMD_WIDTH = 20
            print(colored("\n--- Other Commands ---", "yellow"))
            print(f"  {colored('-r, --regenerate'.ljust(CMD_WIDTH), 'white')}Regenerate last response")
            print(f"  {colored('-img, --image'.ljust(CMD_WIDTH), 'white')}Take screenshot")
            print(f"  {colored('-p, --p'.ljust(CMD_WIDTH), 'white')}Print chat history")
            print(f"  {colored('-m, --m'.ljust(CMD_WIDTH), 'white')}Enter multiline input mode")
            print(f"  {colored('-e, --exit'.ljust(CMD_WIDTH), 'white')}Exit CLI-Agent")
            print(f"  {colored('-h, --help'.ljust(CMD_WIDTH), 'white')}Show this help")
            print(f"  {colored('-s, --save'.ljust(CMD_WIDTH), 'white')}Save current LLM selection as default")
            print(f"  {colored('--full_output'.ljust(CMD_WIDTH), 'white')}Show full model output instead of summary")
            continue
        elif user_input in ["-s", "--save"]:
            await save_current_llm_config_as_default()
            continue
        return user_input
    if args.image:
        args.image = False

async def handle_screenshot_capture(context_chat: Optional[Chat]) -> str:
    """
    Handles the screenshot capture process.
    """
    base64_images: List[str] = []
    screenshots_paths: List[str] = []
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        base64_images = []
        try:
            logging.info(colored("Attempting screenshot with Spectacle (region selection)...", "green"))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_filename = temp_file.name
            subprocess.run(['spectacle', '-rbno', temp_filename], check=True)
            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                with open(temp_filename, 'rb') as image_file:
                    png_data = image_file.read()
                base64_img = base64.b64encode(png_data).decode('utf-8')
                base64_images = [base64_img]
                pyperclip.copy(base64_img)
                logging.info(colored("Screenshot captured successfully. (+Copied to clipboard)", "green"))
            else:
                logging.warning(colored(f"No screenshot was captured (attempt {attempt}).", "yellow"))
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
        except subprocess.CalledProcessError:
            logging.error(colored(f"Spectacle command failed on attempt {attempt}. Is it installed?", "red"))
        except Exception as e:
            logging.exception(f"Unexpected error during screenshot capture attempt {attempt}: {e}")
        if base64_images:
            break
        if attempt < max_attempts and not base64_images:
            logging.warning(colored("Retrying screenshot capture...", "yellow"))
            await asyncio.sleep(2)
    if base64_images:
        images_dir = os.path.join(g.AGENTS_SANDBOX_DIR, "screenshots")
        os.makedirs(images_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, base64_img in enumerate(base64_images):
            try:
                img_data = base64.b64decode(base64_img)
                img_path = os.path.join(images_dir, f"screenshot_{timestamp}_{i+1}.png")
                screenshots_paths.append(img_path)
                with open(img_path, "wb") as img_file:
                    img_file.write(img_data)
                logging.info(colored(f"Screenshot saved to {img_path}", "green"))
            except Exception as e:
                logging.error(colored(f"Error saving screenshot {i+1}: {e}", "red"))
    else:
        logging.error(colored("No images were captured.", "red"))
    if not base64_images:
        logging.warning(colored("# cli-agent: No screenshot was captured after multiple attempts.", "yellow"))
        return ""
    logging.info(colored("Screenshot preprocesssing...", "green"))
    context_chat.add_message(Role.USER, f"""I am inquiring about a screenshot let's have a look at it.
```python
image_path = '{screenshots_paths[0]}'
description = ViewImage.run(image_path, "Describe the screenshot in detail, focusing on any text, images, or notable features.")
print(f"Screenshot description: {{description}}")
```
<execution_output>
Screenshot description: {ViewImage.run(screenshots_paths[0], 'Describe the screenshot in detail, focusing on any text, images, or notable features.')}
</execution_output>
Perfect, use this description as needed for the next steps.\n""")
    return base64_images

def preprocess_consecutive_sudo_commands(code: str) -> str:
    """Combine consecutive shell commands with sudo to reduce password prompts."""
    lines = code.strip().split('\n')
    processed_lines = []
    shell_commands = []
    for line in lines:
        line = line.strip()
        if line.startswith('!'):
            shell_commands.append(line[1:])
        else:
            if shell_commands:
                combined = ' && '.join(shell_commands)
                processed_lines.append(f'!{combined}')
                shell_commands = []
            processed_lines.append(line)
    if shell_commands:
        combined = ' && '.join(shell_commands)
        processed_lines.append(f'!{combined}')
    return '\n'.join(processed_lines)

def extract_paths(user_input: str) -> Tuple[List[str], List[str]]:
    """
    Extracts local file paths, folder paths, and online URLs from a string.
    """
    token_pattern = re.compile(r'(["\'])(.+?)\1|(\S+)')
    trailing_chars_to_strip = '.,;:"\'!?)>]}'
    local_paths = set()
    online_paths = set()
    for match in token_pattern.finditer(user_input):
        candidate = match.group(2) or match.group(3)
        if not candidate:
            continue
        cleaned_candidate = candidate.rstrip(trailing_chars_to_strip)
        if cleaned_candidate.startswith(('http://', 'https://')):
            online_paths.add(cleaned_candidate)
        elif cleaned_candidate.startswith(('/', './', '~/', 'file://')) or re.match(r'^[a-zA-Z]:[\\/]', cleaned_candidate):
            local_paths.add(cleaned_candidate)
    return list(local_paths), list(online_paths)

async def main() -> None:
    try:
        # The `args` and logger are already set up globally at the start of the script.
        
        load_dotenv(g.CLIAGENT_ENV_FILE_PATH)
        
        # Only apply the cached LLM config if no specific LLM flags are used. This ensures CLI args have priority.
        if not args.llm and not args.local and not args.fast:
            apply_default_llm_config()

        default_inst = f'''**SYSTEM: Agent Protocol & Capabilities**

You are a sophisticated AI agent operating within a command-line interface (CLI) and a python notebook. Your primary directive is to understand user requests, formulate a plan, and execute code to achieve the goal.

### 1. Guiding Principles (Your Core Logic)
You must follow this four-step loop for every task:
1.  **THINK & DECOMPOSE**: Analyze the user's request. Break it down into small, logical steps.
2.  **PLAN & TRACK (Todos)**: Add your plan to the `todos` list. Keep it updated, marking steps complete as you go.
3.  **ACT (Execute Code)**: Use your `bash` and `python` tools to execute the current step.
4.  **VERIFY & REFINE**: Check the results of your actions. Refine your plan based on what you learned.

### 2. Execution Environments
You can switch between two environments by using the appropriate code block language:
- **`bash`**: For direct system interaction, file system navigation, and simple commands.
- **`python`**: For complex logic, data manipulation, and accessing your specialized `utils` library.

### 3. Toolbox & Dynamic Hints
Your specialized tools (utilities) are available within any `python` code block.
**IMPORTANT**: For each task, a dynamic list of relevant tools and their exact usage syntax will be provided in the `# HINTS` section of the user's message. You MUST use this section as your primary reference for available tools.

### 4. Rules of Engagement
- **Workspace**: Your primary working directory is `{g.AGENTS_SANDBOX_DIR}`.
- **Safety**: Be cautious. Prefer creating new files over overwriting existing ones.
- **Clarity**: Announce your plan before executing code.
'''

        g.INITIAL_MCT_VALUE = args.mct
        if os.getenv("DEFAULT_FORCE_LOCAL") == get_local_ip():
            args.local = True
        if args.voice:
            args.auto = True
            logging.info(colored("# cli-agent: Voice mode enabled, automatically enabling auto execution mode", "green"))
        if args.sandbox:
            g.USE_SANDBOX = True
        
        stdout_buffer, stderr_buffer = "", ""
        def stdout_callback(text: str):
            nonlocal stdout_buffer
            print(text, end="")
            stdout_buffer += text
        def stderr_callback(text: str):
            nonlocal stderr_buffer
            print(colored(text, "red"), end="")
            stderr_buffer += text
        def input_callback(previous_output: str) -> bool | str:
            logging.warning(colored("⏳ Process seems idle, continuing to wait...", "yellow"))
            return True
            
        web_server = None
        if args.gui:
            web_server = WebServer()
            g.web_server = web_server
            web_server.start()
        
        context_chat: Optional[Chat] = None
        if args.c or args.regenerate:
            try:
                context_chat = Chat.load_from_json()
                if args.regenerate:
                    logging.info(colored("Loading previous chat for regeneration.", "green"))
                    if not context_chat or len(context_chat.messages) < 2:
                        logging.critical(colored("# cli-agent: No sufficient chat history found.", "red"))
                        exit(1)
                    if context_chat.messages[-1][0] == Role.ASSISTANT:
                        context_chat.messages.pop()
                    logging.info(colored("# cli-agent: Will regenerate response.", "green"))
                else:
                    logging.info(colored("Continuing previous chat.", "green"))
            except FileNotFoundError:
                if args.regenerate:
                    logging.critical(colored("No previous chat found to regenerate. Exiting.", "red"))
                    exit(1)
                else:
                    logging.warning(colored("No previous chat found. Starting a new chat.", "yellow"))
                    context_chat = None
        
        if args.mct and context_chat:
            context_chat.debug_title = "MCTs Branching - Main Context Chat"
        if args.image:
            logging.info(colored("# cli-agent: Taking screenshot due to --img flag...", "green"))
            base64_images = await handle_screenshot_capture(context_chat)
            args.image = False
        
        # Handle -l to auto-select all local models
        if args.local and not g.SELECTED_LLMS:
            logging.info(colored("# cli-agent: Local mode (-l) detected. Selecting all available local models.", "green"))
            # Get all local models that are not guard models
            local_models = [m for m in LlmRouter.get_models(force_local=True) if not any(s == AIStrengths.GUARD for s in m.strengths)]
            if local_models:
                g.SELECTED_LLMS = [model.model_key for model in local_models]
            else:
                logging.warning(colored("# cli-agent: Local mode enabled, but no local models were found.", "yellow"))

        if args.llm == "__select__":
            logging.info(colored("# cli-agent: Opening LLM selection...", "green"))
            await llm_selection(args, preselected_llms=g.SELECTED_LLMS)
            args.llm = None
        elif args.llm:
            g.SELECTED_LLMS = [args.llm]
        
        # Sync args.auto with guard LLMs before showing startup summary
        llm_config = g.get_llm_config()
        guard_count = sum(
            data.get('guard', 0) 
            for data in llm_config.values() 
            if data.get('selected')
        )
        if guard_count > 0 and not args.auto:
            args.auto = True
            logging.info(colored(f"🔄 Auto-execution enabled: {guard_count} guard LLMs detected", "green"))
        
        # Now that LLM selections are made, print the startup summary
        print_startup_summary(args)
        
        logging.info(colored("Initializing sandbox environment...", "blue"))
        notebook = ComputationalNotebook(stdout_callback=stdout_callback, stderr_callback=stderr_callback, input_prompt_handler=input_callback)
        utils_manager = UtilsManager()
        
        logging.info(colored("Sandbox environment ready.", "blue"))

        if context_chat is None:
            context_chat = Chat(debug_title="Main Context Chat")
            if args.sandbox:
                default_inst += f"\nPlease try to stay within your sandbox directory at {g.AGENTS_SANDBOX_DIR}\n"
            context_chat.set_instruction_message(default_inst)
            
            # Helper logic for the new bootstrap message
            def get_recent_files_output():
                directory = os.getcwd()
                output_lines = [f"Working Directory: {directory}\n"]
                try:
                    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
                    all_files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)
                    output_lines.append("5 last modified files:")
                    if not all_files:
                        output_lines.append("-> No files found in this directory.")
                    else:
                        for filename in all_files[:5]:
                            output_lines.append(f"- {filename}")
                except FileNotFoundError:
                    output_lines.append(f"Error: The directory '{directory}' does not exist.")
                return "\n".join(output_lines)

            # This new bootstrap message is a concise, pre-filled sequence to establish context.
            initial_bootstrap_message = f"""Let's get a quick overview of our current state by listing the 5 most recently modified files.

```python
import os

# Get the current working directory
directory = os.getcwd()
print(f"Working Directory: {{directory}}")

try:
    # 1. Get all entries in the directory that are files
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # 2. Sort the files by their modification time (newest first)
    all_files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)

    # 3. Print the 5 most recent files
    print("\\n5 last modified files:")
    if not all_files:
        print("-> No files found in this directory.")
    else:
        for filename in all_files[:5]:
            print(f"- {{filename}}")

except FileNotFoundError:
    print(f"Error: The directory '{{directory}}' does not exist.")
```
<execution_output>
{get_recent_files_output()}
</execution_output>
"""

        if '-m' in sys.argv or '--message' in sys.argv:
            if not args.message:
                logging.info(colored("# cli-agent: Entering multiline input mode.", "green"))
                multiline_input = handle_multiline_input()
                if multiline_input.strip():
                    args.message.append(multiline_input)

        logging.info(colored("Ready.", "magenta"))
        
        # Swap to the simple logger now that startup is complete
        swap_to_simple_logging()

        user_interrupt = False
        context_chat.add_message(Role.USER, initial_bootstrap_message)
        while True:
            LlmRouter().failed_models.clear()
            user_input: Optional[str] = None
            
            # Set LLM strengths based on --fast or --strong flags
            g.LLM_STRENGTHS = []
            if args.fast:
                g.LLM_STRENGTHS = [AIStrengths.SMALL]
            
            # Sync args.auto with guard LLMs - auto-enable when guards are present
            llm_config = g.get_llm_config()
            guard_count = sum(
                data.get('guard', 0) 
                for data in llm_config.values() 
                if data.get('selected')
            )
            if guard_count > 0 and not args.auto:
                args.auto = True
                logging.debug(colored(f"Auto-execution enabled: {guard_count} guard LLMs detected", "green"))
            
            # Explicitly set global flags from args
            g.FORCE_LOCAL = args.local
            g.DEBUG_CHATS = args.debug_chats
            g.FORCE_FAST = args.fast
            g.LLM = args.llm
            g.FORCE_ONLINE = args.online
            g.SUMMARY_MODE = not args.full_output_mode
            
            temperature = 0.85 if g.MCT > 1 else 0
            
            logging.debug(f"MCT active: args.mct={args.mct}, g.MCT={g.MCT}, g.SELECTED_LLMS={g.SELECTED_LLMS}")

            if context_chat:
                context_chat.save_to_json()

            if args.regenerate:
                user_input = ""
                logging.info(colored("# cli-agent: Proceeding with regeneration...", "green"))
                args.regenerate = False
            elif args.voice:
                user_input, _, wake_word_used = get_listen_microphone()(private_remote_wake_detection=args.private_remote_wake_detection)
            elif args.message:
                user_input = args.message.pop(0)
                msg_log = f"💬 Processing message: {user_input}"
                if args.message:
                    msg_log += colored(f" (⏳ {len(args.message)} more queued)", 'blue')
                print(colored(msg_log, 'blue', attrs=['bold']), flush=True)
            else:
                user_input = await get_user_input_with_bindings(args, context_chat, force_input=user_interrupt)
                try:
                    from utils.viewfiles import ViewFiles
                    local_paths, _ = extract_paths(user_input)
                    for path in local_paths:
                        expanded_path = os.path.expanduser(path)
                        if not os.path.exists(expanded_path):
                            continue
                        
                        # --- FIX: Use ViewFiles for both files and directories, with correct arguments ---
                        if os.path.isfile(expanded_path) or os.path.isdir(expanded_path):
                            entity_type = "file" if os.path.isfile(expanded_path) else "directory"
                            logging.info(colored(f"# cli-agent: Auto-viewing {entity_type}: {path}", "green"))
                            # Correctly call ViewFiles.run with the 'paths' keyword and a list.
                            view_result_str = ViewFiles.run(paths=[expanded_path])
                            view_result = json.loads(view_result_str)
                            # The 'result' key contains a pre-formatted markdown string.
                            if view_result.get("result"):
                                user_input += f"\n\n# Content of: {path}\n{view_result['result']}"
                except (ImportError, FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
                    logging.debug("Path augmentation feature failed silently.")
                user_interrupt = False

            if LlmRouter.has_unconfirmed_data():
                LlmRouter.confirm_finetuning_data()

            action_counter, assistant_response = 0, ""
            text_stream_painter = TextStreamPainter()
            base64_images: List[str] = []

            if (user_input):
                # Capture the output of adding the task and listing the todos for the prompt context.
                add_output_str = TodosUtil._run_logic("add", task="user_input: " + user_input)
                list_output_str = TodosUtil._run_logic("list")

                # Construct the prompt for the agent, showing the final state of the list.
                todos_prompt = f"""**TodosUtil (`python`)**
```python
# Let's add my current request to the todos, please first review your todos and prioritize my task
TodosUtil.run('add', task='{user_input}')
TodosUtil.run('list')
```
<execution_output>
{add_output_str}
{list_output_str}
</execution_output>
"""
                context_chat.add_message(Role.USER, todos_prompt)

                # Use the captured string to generate hints for the next step.
                try:
                    prompt_subfix = ""
                    # Check if the list is not empty before adding the header.
                    if "Your to-do list is empty." not in list_output_str:
                        prompt_subfix += f"\n\n# UPCOMING TODOS\n{list_output_str}"
                    
                    # Use the same captured string for hint generation.
                    guidance_prompt = utils_manager.get_relevant_tools_prompt(list_output_str, top_k=5)
                    if guidance_prompt:
                        # Format the suggestions nicely with colors and proper spacing
                        logging.info(colored("🔧 Suggested utilities based on your todo list:", "cyan", attrs=["bold"]))
                        for line in guidance_prompt.split('\n'):
                            if line.strip():
                                logging.info(colored(f"   {line.strip()}", "light_blue"))
                    if guidance_prompt:
                        prompt_subfix += f"\n\n# HINTS\n{guidance_prompt}"
                    
                    if prompt_subfix:
                        context_chat.add_message(Role.USER, prompt_subfix)
                except Exception as e:
                    # Handle potential errors in hint generation gracefully.
                    logging.warning(f"Could not generate hints: {e}")

            last_action_signature: Optional[str] = None
            stall_counter: int = 0
            MAX_STALLS: int = 2
            
            while True:
                try:
                    # --- REVISED BRANCHING AND MODEL SELECTION LOGIC ---
                    response_branches: List[str] = []
                    try:
                        if assistant_response:
                            logging.warning(colored("WARNING: Unhandled assistant response detected.", "yellow"))
                            context_chat.add_message(Role.ASSISTANT, assistant_response)
                            assistant_response = ""

                        # 1. Get the full LLM configuration from globals.
                        llm_config = g.get_llm_config()
                        
                        # 2. Identify models for branching, respecting their beam counts.
                        # A model with N beams contributes N branches to the MCT.
                        models_for_tasks = []
                        for model_key, data in llm_config.items():
                            if data.get('selected') and model_key not in LlmRouter().failed_models:
                                beam_count = data.get('beams', 0)
                                if beam_count > 0:
                                    # Add the model to the task list 'beam_count' times
                                    models_for_tasks.extend([model_key] * beam_count)

                        # Handle MCT parameter logic
                        user_mct_value = args.mct  # Preserve original user setting
                        available_branches = len(models_for_tasks)
                        
                        # Determine effective MCT count based on user input
                        if user_mct_value == -1:  # -mct used without integer
                            if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 0:
                                # Use all selected LLMs
                                effective_mct_count = len(g.SELECTED_LLMS)
                            else:
                                # No LLMs selected, default to 3
                                effective_mct_count = 3
                        else:
                            # Specific integer provided
                            effective_mct_count = user_mct_value
                        
                        # Limit models to effective MCT count
                        if available_branches > effective_mct_count:
                            models_for_tasks = models_for_tasks[:effective_mct_count]
                        
                        num_branches = len(models_for_tasks)
                        g.MCT = num_branches  # Don't override args.mct, keep user's original setting
                        temperature = 0.85 if g.MCT > 1 else 0

                        if not models_for_tasks:
                            # Fallback: if no beams are set, use available selected models up to effective MCT count.
                            selected_models = [
                                model_key for model_key, data in llm_config.items() 
                                if data.get('selected') and model_key not in LlmRouter().failed_models
                            ]
                            if selected_models:
                                # Use up to effective MCT count of available models
                                models_for_tasks = selected_models[:effective_mct_count]
                                num_branches = len(models_for_tasks)
                                g.MCT = num_branches  # Don't override args.mct
                                if num_branches == 1:
                                    logging.warning(colored("No models with beams > 0 configured. Using single model for execution.", "yellow"))
                                else:
                                    logging.warning(colored(f"No models with beams > 0 configured. Using {num_branches} models for MCT execution.", "yellow"))
                            else:
                                logging.error(colored("❌ No available models to process the request. Check LLM selection.", "red"))
                                break

                        # 3. Execute generation tasks.
                        if num_branches > 1:
                            # --- MCT Branching Execution ---
                            logging.info(colored("🌿", "green", attrs=["bold"]) + colored(f" Generating {num_branches} MCT branches with models: {', '.join(models_for_tasks)}", "cyan"))
                            print_lock = asyncio.Lock()
                            branch_model_map = {}  # Track which model generated which branch
                            async def generate_branch(model_key: str, branch_index: int, lock: asyncio.Lock):
                                branch_model_map[branch_index] = model_key  # Store the model name
                                response_buffer_list = [""]
                                branch_update_callback = create_interruption_callback(
                                    response_buffer=response_buffer_list,
                                    painter=text_stream_painter,
                                    lock=lock
                                )
                                try:
                                    context_chat.debug_title = f"MCT Branch {branch_index+1}/{num_branches}"
                                    await LlmRouter.generate_completion(
                                        chat=context_chat,
                                        preferred_models=[model_key],
                                        force_preferred_model=True,
                                        temperature=temperature,
                                        base64_images=base64_images,
                                        generation_stream_callback=branch_update_callback,
                                        strengths=g.LLM_STRENGTHS,
                                        thinking_budget=None,
                                        exclude_reasoning_tokens=True
                                    )
                                    return response_buffer_list[0]
                                except StreamInterruptedException as e:
                                    return e.response
                                except Exception as e:
                                    if not isinstance(e, StreamInterruptedException):
                                        async with lock:
                                            logging.error(colored(f"❌ Branch {branch_index+1} ({model_key}) failed: {e}", "red"))
                                        if model_key: LlmRouter().failed_models.add(model_key)
                                    return None

                            tasks = [generate_branch(model_key, i, print_lock) for i, model_key in enumerate(models_for_tasks)]
                            branch_results = await asyncio.gather(*tasks)
                            response_branches = [res for res in branch_results if res and res.strip()]
                            if not g.SUMMARY_MODE:
                                print()
                        else:
                            # --- Single Branch Execution ---
                            model_for_branch = models_for_tasks[0]
                            context_chat.debug_title = "Main Context"
                            response_buffer_list = [""]
                            interruption_callback = create_interruption_callback(
                                response_buffer=response_buffer_list,
                                painter=text_stream_painter
                            )
                            try:
                                await LlmRouter.generate_completion(
                                    chat=context_chat,
                                    preferred_models=[model_for_branch],
                                    force_preferred_model=True,
                                    temperature=temperature,
                                    base64_images=base64_images,
                                    generation_stream_callback=interruption_callback,
                                    strengths=g.LLM_STRENGTHS,
                                    thinking_budget=None,
                                    exclude_reasoning_tokens=True
                                )
                                if response_buffer_list[0].strip(): 
                                    response_branches.append(response_buffer_list[0])
                            except StreamInterruptedException as e:
                                if e.response and e.response.strip(): 
                                    response_branches.append(e.response)

                        base64_images = [] # Clear images after use
                    except KeyboardInterrupt:
                        logging.warning(colored("\n-=- User interrupted model generation -=-", "yellow"))
                        if args.message: args.message = []
                        context_chat.add_message(Role.ASSISTANT, response_buffer_list[0] if 'response_buffer_list' in locals() else "")
                        break
                    except Exception as e:
                        if not isinstance(e, StreamInterruptedException):
                            LlmRouter.clear_unconfirmed_finetuning_data()
                            logging.error(colored(f"Error generating response: {e}", "red"), exc_info=args.debug)
                            break
                    
                    if response_branches:
                        if num_branches > 1 and len(response_branches) > 1:
                            try:
                                selected_branch_index, vote_summary = await select_best_branch(context_chat, response_branches)
                                assistant_response = response_branches[selected_branch_index]
                                winning_model_name = branch_model_map.get(selected_branch_index, "Unknown")
                                
                                logging.info(colored(f"✅ Picked branch {selected_branch_index}. {vote_summary}", "green"))

                                # If in summary mode (not full output mode), print the picked branch's full response with model info
                                if not args.full_output_mode:
                                    print(colored(f"--- [Picked Branch: {winning_model_name}] ---", "blue", attrs=['bold']))
                                    print(text_stream_painter.apply_color(assistant_response), flush=True)
                                    print(colored("---------------------------------------------------------", "blue", attrs=['bold']))
                                else:
                                    # In normal mode, just print the response as before
                                    print(text_stream_painter.apply_color(assistant_response), flush=True) 
                            except Exception as e:
                                logging.error(colored(f"Error during MCT branch selection: {e}", "red"), exc_info=args.debug)
                                logging.warning(colored("⚠️ Defaulting to first branch.", "yellow"))
                                assistant_response = response_branches[0]
                        else:
                            # Single branch or only one successful branch
                            assistant_response = response_branches[0]
                            if num_branches > 1:
                                logging.info(colored("✅ Using single successful branch.", "green"))
                            
                            # If in summary mode (not full output mode), the single response wasn't streamed, so print it now.
                            if not args.full_output_mode:
                                print(text_stream_painter.apply_color(assistant_response), flush=True)
                    else:
                        logging.error(colored("All generation branches failed.", "red"))
                        break
                    
                    # assistant response deduplication
                    if last_action_signature and assistant_response == last_action_signature:
                        stall_counter += 1
                        logging.warning(colored(f"Stall counter: {stall_counter}/{MAX_STALLS}", "yellow"))
                    else:
                        stall_counter = 0
                    last_action_signature = assistant_response

                    if stall_counter >= MAX_STALLS:
                        logging.error(colored("! Agent appears to be stalled. Intervening.", "red"))
                        context_chat.add_message(Role.USER, "My last two attempts have failed. I need to stop, re-evaluate my strategy, and devise a new plan.")
                        stall_counter, last_action_signature = 0, None
                        break

                    # assistant response generated and consumable
                    context_chat.add_message(Role.ASSISTANT, assistant_response)
                    
                    # extract actions
                    shell_blocks = get_extract_blocks()(assistant_response, ["shell", "bash"])
                    python_blocks = get_extract_blocks()(assistant_response, ["python", "tool_code"])


                    # Agent to User data output
                    if args.voice or args.speak:
                        verbal_text = re.sub(r'```[^`]*```', '', assistant_response)
                        if python_blocks and shell_blocks: verbal_text += "I've implemented shell and python code."
                        elif python_blocks: verbal_text += "I've implemented python code."
                        elif shell_blocks: verbal_text += "I've implemented shell code."
                        TtsUtil.run(text=verbal_text)
                    
                    # No actions taken and todos are left
                    if not python_blocks and not shell_blocks:
                        try:
                            TodosUtil.run("add", task="HIGH PRIORITY: update the todo list entries")
                            all_todos = TodosUtil._load_todos()
                            if all_todos and any(not todo.get('completed', False) for todo in all_todos):
                                todos_str = TodosUtil.run("list")
                                auto_prompt = f"The agent has not returned any code, please ensure the todos are completed before finishing your turn.\n```python\nTodosUtil.run('list')```\n<execution_output>\n{TodosUtil._run_logic('list')}\n</execution_output>\n"
                                assistant_response = ""
                                context_chat.add_message(Role.USER, auto_prompt)
                                continue

                        except Exception as e:
                            logging.warning(colored(f"Could not check for pending to-dos. Error: {e}", "red"))
                        break
                    

                    formatted_code = ""
                    if shell_blocks: formatted_code += "```bash\n" + "\n".join(shell_blocks) + "\n```\n"
                    if python_blocks: formatted_code += "```python\n" + python_blocks[0] + "\n```"

                    context_chat.save_to_json()
                    if await confirm_code_execution(args, formatted_code, utils_manager):
                        logging.info(colored("🔄 Executing code...", "cyan"))
                        try:
                            if shell_blocks:
                                for shell_line in shell_blocks:
                                    l_shell_line = shell_line.strip()
                                    if 'sudo ' in l_shell_line:
                                        l_shell_line = preprocess_consecutive_sudo_commands(l_shell_line)
                                        if 'sudo ' in l_shell_line and 'sudo -A ' not in l_shell_line:
                                            l_shell_line = l_shell_line.replace("sudo ", "sudo -A ")
                                    notebook.execute(l_shell_line)
                            if python_blocks:
                                notebook.execute(python_blocks[0], is_python_code=True)
                            logging.info(colored("✅ Code execution completed.", "cyan"))

                            tool_output = ""
                            # Perform the string replacements outside the f-string expression
                            processed_stdout = stdout_buffer.replace('\n⚙️  ', '\n').replace('\n🐍  ', '\n').strip()
                            
                            if stdout_buffer.strip():
                                tool_output += f"```stdout\n{processed_stdout}\n```\n"  # noqa: F541
                            if stderr_buffer.strip():
                                tool_output += f"```stderr\n{stderr_buffer.strip()}\n```\n"
                            tool_output = re.sub(r'\x1b\[[0-9;]*m', '', tool_output) # noqa: F841, E501


                            if len(tool_output) > 4000:
                                tool_output = tool_output[:g.OUTPUT_TRUNCATE_HEAD_SIZE] + "\n[...output truncated...]\n" + tool_output[-g.OUTPUT_TRUNCATE_TAIL_SIZE:]

                            if not tool_output.strip():
                                tool_output = "<execution_output>\nThe execution completed without output.\n</execution_output>"
                            else:
                                tool_output = f"<execution_output>\n{tool_output.strip()}\n</execution_output>"
                            
                            context_chat.add_message(Role.ASSISTANT, tool_output)
                            assistant_response = ""
                            stdout_buffer, stderr_buffer = "", ""
                            action_counter += 1
                            continue
                        except Exception as e:
                            logging.error(colored(f"❌ Error executing code: {e}", "red"), exc_info=args.debug)
                            error_output = f"<execution_output>\n```error\n{traceback.format_exc()}\n```\n</execution_output>"
                            context_chat.add_message(Role.ASSISTANT, f"{assistant_response}{error_output}")
                            assistant_response, stdout_buffer, stderr_buffer = "", "", ""
                            break
                    else:
                        logging.warning(colored("✖️  Execution cancelled.", "yellow"))
                        cancellation_notice = "<execution_output>\nCode execution cancelled\n</execution_output>"
                        context_chat.add_message(Role.ASSISTANT, f"{assistant_response}{cancellation_notice}")
                        assistant_response, stdout_buffer, stderr_buffer = "", "", ""
                        break
                except KeyboardInterrupt:
                    logging.warning(colored("=== User interrupted execution (Ctrl+C) ===", "yellow"))
                    user_interrupt = True
                    break
                except Exception as e:
                    LlmRouter.clear_unconfirmed_finetuning_data()
                    if "ctrl+c" in str(e).lower():
                        logging.warning(colored("=== User interrupted execution (Ctrl+C) ===", "yellow"))
                        user_interrupt = True
                        break
                    logging.critical(colored(f"An unexpected error occurred in the agent loop: {e}", "red"), exc_info=args.debug)
                    try:
                        error_output = f"<execution_output>\n```error\n{traceback.format_exc()}\n```\n</execution_output>"
                        context_chat.add_message(Role.ASSISTANT, f"{assistant_response}{error_output}")
                    except Exception as context_e:
                        logging.error(colored(f"Failed to add error to context: {context_e}", "red"))
                    assistant_response, stdout_buffer, stderr_buffer = "", "", ""
                    break
            
            if context_chat:
                context_chat.save_to_json()
            if args.exit and not args.message:
                logging.info(colored("All automatic messages processed successfully. Exiting...", "green"))
                exit(0)
        logging.info(colored("CLI-Agent is shutting down.", "cyan"))

    except (asyncio.CancelledError, KeyboardInterrupt):
        logging.warning(colored("CLI-Agent was interrupted. Shutting down gracefully...", "yellow"))
    except Exception as e:
        if not isinstance(e, StreamInterruptedException):
            logging.critical(colored(f"CLI-Agent encountered a fatal error: {e}", "red"), exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(colored("\nCLI-Agent was interrupted by user. Shutting down...", "yellow"))
    except Exception as e:
        if not isinstance(e, StreamInterruptedException):
            print(colored(f"\nCLI-Agent encountered a fatal error during startup: {e}", "red"))
            traceback.print_exc()