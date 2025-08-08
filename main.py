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
    
    # Toggles
    llm_status = colored(', '.join(g.SELECTED_LLMS), 'cyan') if g.SELECTED_LLMS else colored('Default', 'yellow')
    
    # --- FIX: Use a more descriptive status string for MCT that shows branch count in both ON and OFF states ---
    mct_status_text_on = f'ON ({args.mct} branches)'
    # Use singular 'branch' for the off case.
    mct_status_text_off = f'OFF ({args.mct} branch)'
    
    config_lines = [
        f"  {colored('Local Mode (-l)'.ljust(CMD_WIDTH), 'white')} {colored('Use only local LLMs'.ljust(DESC_WIDTH), 'light_grey')} (Status: {get_status_str(args.local)})",
        f"  {colored('Auto Execution (-a)'.ljust(CMD_WIDTH), 'white')} {colored('Automatic code execution'.ljust(DESC_WIDTH), 'light_grey')} (Status: {get_status_str(args.auto)})",
        f"  {colored('MCT Branches (--mct)'.ljust(CMD_WIDTH), 'white')} {colored('Monte Carlo Tree Search'.ljust(DESC_WIDTH), 'light_grey')} (Status: {get_status_str(args.mct > 1, on_text=mct_status_text_on, off_text=mct_status_text_off)})",
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
    parser.add_argument("-s", "--speak", action="store_true", default=False,
                        help="Text-to-speech output.")
    parser.add_argument("-f", "--fast", action="store_true", default=False,
                        help="Use only fast LLMs. Sets g.FORCE_FAST=True.")
    parser.add_argument("-strong", "--strong", action="store_true", default=False,
                        help="Use strong LLMs (slow!)")
    parser.add_argument("-img", "--image", action="store_true", default=False,
                        help="Take a screenshot using Spectacle for region selection (with automatic fallbacks if not available).")
    parser.add_argument("-mct", "--mct", type=int, nargs='?', const=3, default=1,
                        help="Enable Monte Carlo Tree Search for acting, with an optional number of branches (default: 3).")
    parser.add_argument("-sbx", "--sandbox", action="store_true", default=False,
                        help="Use weakly sandboxed python execution. Sets g.USE_SANDBOX=True.")
    parser.add_argument("-o", "--online", action="store_true", default=False,
                        help="Force use of cloud AI.")
    
    parser.add_argument("-llm", "--llm", type=str, nargs='?', const="__select__", default=None,
                        help="Specify the LLM model key to use (e.g., 'gpt-4', 'gemini-pro'). Use without value to open selection menu.")
    
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
import asyncio
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

from py_classes.cls_computational_notebook import ComputationalNotebook
from py_classes.cls_util_manager import UtilsManager
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_llm_router import Llm, LlmRouter, StreamInterruptedException
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
try:
    from utils.todos import TodosUtil
except ImportError:
    class TodosUtil:
        @staticmethod
        def _load_todos(**kwargs):
            logging.warning("TodosUtil not found.")
            return []
        @staticmethod
        def _format_todos(**kwargs):
            logging.warning("TodosUtil not found.")
            return ""

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

def save_selected_llms(selected_llms: List[str]) -> None:
    """Save the selected LLMs to persistent storage."""
    try:
        llm_selection_file = os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, "selected_llms.json")
        os.makedirs(g.CLIAGENT_PERSISTENT_STORAGE_PATH, exist_ok=True)
        with open(llm_selection_file, 'w') as f:
            json.dump(selected_llms, f)
    except Exception as e:
        logging.warning(f"Could not save selected LLMs: {e}")

def load_selected_llms() -> List[str]:
    """Load the previously selected LLMs from persistent storage."""
    try:
        llm_selection_file = os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, "selected_llms.json")
        if os.path.exists(llm_selection_file):
            with open(llm_selection_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Could not load selected LLMs: {e}")
    return []


async def llm_selection(args: argparse.Namespace, preselected_llms: List[str] = None) -> List[str]:
    """
    Handles the LLM selection process, supporting multi-selection mode.
    """
    available_llms = Llm.get_available_llms(exclude_guards=True)
    llm_choices = []
    llm_choices.append(("any_local", HTML('<provider>Any</provider> - <model>Any but local</model> - <pricing>Automatic selection</pricing>')))
    from py_classes.ai_providers.cls_ollama_interface import OllamaClient
    ollama_status = {}
    try:
        # This will now be timed because logging is set up
        ollama_status = OllamaClient.get_comprehensive_model_status(["localhost", "192.168.178.37"])
    except Exception:
        pass
    for llm in available_llms:
        status_indicator = ""
        if llm.provider.__class__.__name__ == "OllamaClient":
            model_base_name = llm.model_key.split(':')[0]
            if model_base_name in ollama_status:
                if ollama_status[model_base_name]['downloaded']:
                    status_indicator = ' <downloaded>✓ Downloaded</downloaded>'
                else:
                    status_indicator = ' <notdownloaded>⬇ Available</notdownloaded>'
            else:
                status_indicator = ' <notdownloaded>⬇ Available</notdownloaded>'
        styled_text = HTML(
            f'<provider>{llm.provider.__class__.__name__}</provider> - '
            f'<model>{llm.model_key}</model> - '
            f'<pricing>{f"${llm.pricing_in_dollar_per_1M_tokens}/1M tokens" if llm.pricing_in_dollar_per_1M_tokens else "Free"}</pricing> - '
            f'<context>Context: {llm.context_window}</context>'
            f'{status_indicator}'
        )
        llm_choices.append((llm.model_key, styled_text))
    style = Style.from_dict({
        'model': 'ansicyan',
        'provider': 'ansigreen',
        'pricing': 'ansimagenta',
        'context': 'ansiblue',
        'downloaded': 'ansibrightgreen',
        'notdownloaded': 'ansibrightred',
    })
    default_selected = []
    if preselected_llms is not None:
        default_selected = [llm for llm in preselected_llms if llm in [choice[0] for choice in llm_choices]]
    else:
        saved_llms = load_selected_llms()
        default_selected = [llm for llm in saved_llms if llm in [choice[0] for choice in llm_choices]]
    checkbox_list = CheckboxList(values=llm_choices, default_values=default_selected)
    bindings = KeyBindings()
    @bindings.add("e")
    def _execute(event) -> None:
        selected = [value for value, _ in llm_choices if value in checkbox_list.current_values]
        app.exit(result=selected)
    @bindings.add("a")
    def _abort(event) -> None:
        app.exit(result=None)
    instructions = Label(text="Use arrow keys to navigate, Space to select/deselect LLMs, 'e' to confirm or 'a' to abort")
    root_container = HSplit([Frame(title="Select LLMs to use", body=checkbox_list), instructions])
    layout = Layout(root_container)
    app = Application(layout=layout, key_bindings=bindings, full_screen=False, style=style)
    try:
        selected_llms = await app.run_async()
        if selected_llms is not None and len(selected_llms) > 0:
            save_selected_llms(selected_llms)
            if "any_local" in selected_llms:
                g.FORCE_LOCAL = True
                args.local = True
                args.llm = None
                logging.info(colored("# cli-agent: 'Any but local' option selected. Local mode enabled.", "green"))
                return []
            args.llm = selected_llms if selected_llms else []
            if not args.llm:
                logging.info(colored("# cli-agent: LLM set to auto, type (--h) for info", "green"))
            else:
                logging.info(colored(f"# cli-agent: LLM(s) set to {args.llm}, type (--h) for info", "green"))
            return selected_llms
        else:
            logging.warning(colored("# cli-agent: No LLMs selected or selection cancelled", "yellow"))
            return []
    except asyncio.CancelledError:
        logging.warning(colored("# cli-agent: LLM selection was interrupted", "yellow"))
        return []
    except Exception as e:
        logging.error(colored(f"# cli-agent: Error during LLM selection: {str(e)}", "red"))
        if args.debug:
            logging.exception("LLM selection failed with traceback:")
        return []

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
    newline_on_interrupt: bool = False
):
    """
    Factory to create a stateful callback for handling LLM streams.

    This callback prints the stream, accumulates it into a buffer, and can
    interrupt by raising StreamInterruptedException when a complete code block is detected.

    Args:
        response_buffer: A list with a single string to act as a mutable buffer.
        painter: The TextStreamPainter for coloring output.
        lock: An optional asyncio.Lock for concurrent-safe printing.
        newline_on_interrupt: If True, prints a newline before raising interruption.

    Returns:
        An async callback function.
    """
    async def _callback(chunk: str):

        for char in chunk:
            # Print the entire chunk at once for efficiency and better rendering.
            if lock:
                async with lock:
                    print(painter.apply_color(char), end="", flush=True)
            else:
                print(painter.apply_color(char), end="", flush=True)
            
            # Append chunk to the buffer
            response_buffer[0] += char

            # Check for interruption condition (an even number of code fences)
            if response_buffer[0].count("```") >= 2 and response_buffer[0].count("```") % 2 == 0:
                parts = response_buffer[0].split("```")
                # Interrupt if the last captured code block is not empty.
                if len(parts) > 2 and parts[-2].strip():
                    if newline_on_interrupt:
                        print("\n")
                    raise StreamInterruptedException(response_buffer[0])

    return _callback

async def confirm_code_execution(args: argparse.Namespace, code_to_execute: str) -> bool:
    """
    Handles the confirmation process for code execution, supporting both auto and manual modes.
    """
    # This check for a fast-path bypass should only apply to pure bash/shell code.
    # If python is present, it must go through the full guard.
    python_codes: List[str] = get_extract_blocks()(code_to_execute, "python")

    if len(python_codes) == 0:
        # This list defines simple, non-destructive commands that can be permitted without a full LLM guard check.
        always_permitted_bash = ["ls ", "pwd ", "cd ", "echo ", "print ", "cat ", "head ", "tail ", "grep ", "sed ", "awk ", "sort "]
        
        # --- FIX: Check for both 'bash' and 'shell' to be consistent with the main loop ---
        bash_codes: List[str] = get_extract_blocks()(code_to_execute, ["bash", "shell"])
        bash_code = "\n".join(bash_codes)
        bash_code_lines = [line for line in bash_code.split("\n") if line.strip() and not line.strip().startswith("#")]

        # --- FIX: The core logic flaw was here. ---
        # If there are executable bash lines, check if they are all on the safe list.
        # If not, or if there are no lines, fall through to the main execution guard below.
        if bash_code_lines:
            is_entirely_safe = True
            collected_matching_commands = []
            for line in bash_code_lines:
                # Find if the line starts with any of the permitted commands.
                matching_commands = [cmd for cmd in always_permitted_bash if line.strip().startswith(cmd)]
                
                # A simple check: if a safe command is found and it's not part of a complex chain,
                # we consider it safe for the bypass. This is a heuristic.
                is_line_safe = len(matching_commands) > 0 and (line.count(" && ") + line.count(" || ") + line.count(";") + 1 == len(matching_commands))
                
                if is_line_safe:
                    collected_matching_commands.extend(matching_commands)
                    continue # This line is safe, check the next one.
                else:
                    # This line is not on the safe list, so the entire block is unsafe for the bypass.
                    is_entirely_safe = False
                    break # No need to check other lines.

            if is_entirely_safe:
                logging.info(colored(f"✅ Code execution permitted automatically (safe commands: '{', '.join(collected_matching_commands).strip()}')", "green"))
                return True
        # --- END FIX ---

    # If the fast-path check was not met (e.g., python code exists, or bash command was not on the safe list),
    # proceed to the full execution guard.
    if args.auto:
        execution_guard_chat: Chat = Chat(
            instruction_message="""You are a Code Execution Guardian. Your primary goal is to prevent unsafe or incomplete code execution while allowing legitimate operations.

**Safety Assessment Guidelines:**

SAFE operations include:
- Reading files and directories (ls, cat, head, tail, find, grep, etc.)
- Standard data retrieval from public APIs/services (curl to well-known services like ifconfig.me, httpbin.org, api.github.com, etc.)
- Basic system information commands (ps, top, df, free, uname, etc.)
- File modifications when comments indicate intentional and safe operations
- Standard development tools (git, npm, pip install from requirements.txt, etc.)
- Anything imported from utils.* (assumed safe)
- Scripts that only print/display information

UNSAFE operations include:
- File deletions without clear safety comments (rm -rf, especially with wildcards or system paths)
- System modifications (sudo commands, service management, user/permission changes)
- Unrestricted network operations to arbitrary hosts (except well-known public APIs)
- Running executables from untrusted sources
- Operations that could affect system stability or security

**Completeness Assessment:**
- Look for placeholder values like `YOUR_API_KEY`, `<REPLACE_ME>`, `TODO`, `FIXME`
- Identify unimplemented functions or incomplete logic
- Comments about future work are acceptable
- Scripts that only print text are always complete

**Process:**
1. **Brief Analysis:** Explain your safety and completeness reasoning in 1-2 sentences.
2. **Single Word Verdict:**
   - `no`: Unsafe (regardless of completeness)
   - `unfinished`: Safe but contains placeholders or incomplete logic  
   - `yes`: Safe and complete

Provide ONLY your brief analysis followed by exactly one word.""",
            debug_title="Auto Execution Guard"
        )
        if "bash" in code_to_execute and "python" in code_to_execute:
            analysis_prompt = f"Analyze this code for safe execution and completeness:\n{code_to_execute}"
        elif "python" in code_to_execute:
            analysis_prompt = f"Analyze this python code for safe execution and completeness:\n{code_to_execute}"
        else:
            analysis_prompt = f"Analyze these bash commands for safe execution and completeness:\n{code_to_execute}"
        execution_guard_chat.add_message(Role.USER, analysis_prompt)
        guard_preferred_models = []
        if args.local_exec_confirm:
            available_models = LlmRouter.get_models(force_local=True)
            guard_preferred_models = [model.model_key for model in available_models if not any(s == AIStrengths.STRONG for s in model.strengths)]
        safe_to_execute: str = await LlmRouter.generate_completion(execution_guard_chat, preferred_models=guard_preferred_models, hidden_reason="Assessing execution safety", force_local=args.local_exec_confirm, strengths=[])
        
        # Extract the analysis and verdict - get the LAST occurrence of the verdict
        verdict_match = None
        for match in re.finditer(r'\b(yes|no|unfinished)\b', safe_to_execute.lower()):
            verdict_match = match
        
        verdict = verdict_match.group(1) if verdict_match else "no" # Default to 'no' if unclear
        
        # Split at the last occurrence of the verdict
        if verdict_match:
            verdict_pos = verdict_match.start()
            analysis = safe_to_execute[:verdict_pos].strip()
        else:
            analysis = safe_to_execute.strip()
        
        if 'yes' in verdict:
            logging.info(colored("✅ Code guard permitted execution. Reasoning: ", "green") + colored(analysis, "light_green"))
            return True
        elif 'unfinished' in verdict:
            logging.warning(colored("⚠️ Code guard aborted execution, because of unfinished code.", "yellow"))
            completion_request = "The code you provided is unfinished. Please complete it properly with actual values and logic."
            args.message.insert(0, completion_request)
            logging.info(colored(f"💬 Added automatic follow-up request: {completion_request}", "blue"))
            return False
        else: # 'no' or default
            logging.error(colored("❌ Code execution aborted by auto-execution guard", "red"))
            
            # Timed input with 30-second countdown defaulting to 'n'  
            import threading
            import queue
            
            def timed_input(prompt, timeout=30):
                """Get user input with timeout, returns 'n' if timeout occurs"""
                q = queue.Queue()
                
                def get_input():
                    try:
                        response = input(colored(prompt, "cyan"))
                        q.put(response)
                    except:
                        q.put("n")  # Default on any error
                
                # Start input thread
                input_thread = threading.Thread(target=get_input, daemon=True)
                input_thread.start()
                
                # Countdown with queue checking
                for remaining in range(timeout, 0, -1):
                    print(f"\rAuto-declining in {remaining}s... ", end="", flush=True)
                    
                    # Check if input was received
                    try:
                        response = q.get_nowait()
                        print()  # New line after input
                        return response.lower().strip()
                    except queue.Empty:
                        time.sleep(1)
                
                print("\rAuto-declined (timeout)          ")
                return "n"
            
            user_response = timed_input("Do you want to manually execute this code? (y/n): ", 30)
            
            manual_permission = bool(user_response == "y")
            if manual_permission:
                logging.info(colored("✅ User permitted execution.", "green") + colored(analysis, "light_green"))
                return True
            logging.error(colored("❌ Code execution aborted", "red"))
            return False
    else:
        user_input = await get_user_input_with_bindings(args, None, colored("\n(Press Enter to confirm or 'n' to abort, press 'a' to toggle auto execution, 'l' for local auto execution)", "cyan"))
        if user_input.lower() == 'n':
            logging.error(colored("❌ Code execution aborted by user", "red"))
            return False
        elif user_input.lower() == 'a':
            args.auto = not args.auto
            logging.info(colored(f"# cli-agent: KeyBinding detected: Automatic execution toggled {'on' if args.auto else 'off'}.", "green"))
            return await confirm_code_execution(args, code_to_execute)
        elif user_input.lower() == 'l':
            args.auto = True
            args.local_exec_confirm = True
            logging.info(colored(f"# cli-agent: KeyBinding detected: Local auto execution toggled {'on' if args.local_exec_confirm else 'off'}.", "green"))
            return await confirm_code_execution(args, code_to_execute)
        else:
            logging.info(colored("✅ Code execution permitted", "green"))
    return True

async def select_best_branch(
    context_chat: Chat,
    assistant_responses: List[str],
) -> Tuple[int, str]:
    """
    Uses an LLM judge to select the best response from multiple MCT branches.
    
    Returns:
        Tuple[int, str]: (selected_index, judge_model_name)
    """
    # --- FIX: Calculate number of alternatives and new response index correctly ---
    num_alternatives = len(assistant_responses) - 1
    new_response_index = len(assistant_responses)
    
    # Handle grammar for singular/plural cases to make the prompt clearer
    alt_plural_s = '' if num_alternatives == 1 else 's'
    alt_index_range = f"1-{num_alternatives}" if num_alternatives > 1 else "1"
    alt_word = "Index" if num_alternatives == 1 else "Indexes"

    mct_branch_selector_chat: Chat = context_chat.deep_copy()
    mct_branch_selector_chat.set_instruction_message(f"""1. Review your original response (index 0) alongside the {num_alternatives} alternative{alt_plural_s}.
2. Consider factors like accuracy, helpfulness, clarity, and relevance.
3. Provide a brief comparative analysis.
4. End your response with: "Selected index: [number]".""")
    mct_branch_selector_chat.debug_title="MCT Branch Selection"
    mct_branch_selector_chat.add_message(Role.ASSISTANT, assistant_responses[0])

    selection_prompt = f"""Please cross-examine your response with these {num_alternatives} alternative response{alt_plural_s} from other agents. Please evaluate the responses and select the best one. Your own response has the Index 0. You MUST finish your response by writing the integer index of the top pick, in this case 0-{new_response_index-1}.

**Your task:**
1. Review your original response (index 0) alongside the {num_alternatives} alternative{alt_plural_s}.
2. Consider factors like accuracy, helpfulness, clarity, and relevance.
3. Provide a brief comparative analysis.
4. End your response with: "Selected index: [number]"

**Available options:**
- Index 0: Your original response
- {alt_word} {alt_index_range}: Alternative response{alt_plural_s}
- Index {new_response_index}: Generate a new response incorporating insights from all alternatives

**Output format:**
[Your comparative analysis]
Selected index: [your choice]"""
    
    # --- FIX: This loop was missing the actual response content, making comparison impossible ---
    for i, response in enumerate(assistant_responses[1:]):
        selection_prompt += f"\n\n# Index {i+1}\n{response}"
        
    mct_branch_selector_chat.add_message(Role.USER, selection_prompt)
    evaluator_model = []
    if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 0:
        evaluator_model = [g.SELECTED_LLMS[0]]
        logging.info(colored(f"Using {g.SELECTED_LLMS[0]} to evaluate responses from all models", "cyan"))
    text_stream_painter = TextStreamPainter()
    
    # Use the unified callback creator
    response_buffer_list = [""]
    interrupting_evaluator_callback = create_interruption_callback(
        response_buffer=response_buffer_list,
        painter=text_stream_painter,
        newline_on_interrupt=True
    )

    evaluator_response = ""
    try:
        await LlmRouter.generate_completion(mct_branch_selector_chat, evaluator_model, force_local=g.FORCE_LOCAL, generation_stream_callback=interrupting_evaluator_callback, exclude_reasoning_tokens=True)
        judge_llm = LlmRouter.last_used_model
        evaluator_response = response_buffer_list[0]
    except StreamInterruptedException as e:
        evaluator_response = e.response
        
    assistant_responses.append(evaluator_response)
    match: Optional[re.Match] = re.search(r'Selected index:\s*(\d+)', evaluator_response)
    if match:
        selected_branch_index: int = int(match.group(1))
        if 0 <= selected_branch_index < len(assistant_responses):
            return selected_branch_index,judge_llm
    logging.warning(colored("\n⚠️ No valid branch selection found. Defaulting to first branch.", "yellow"))
    return 0, ""

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
) -> bool | str:
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
            g.SELECTED_LLMS = await llm_selection(args, preselected_llms=None)
            continue
        elif user_input == "-a" or user_input == "--auto":
            args.auto = not args.auto
            logging.info(colored(f"# cli-agent: Automatic execution toggled {'on' if args.auto else 'off'}.", "green"))
            continue
        elif user_input == "-mct" or user_input == "--mct":
            args.mct = int(input(colored(f"# cli-agent: Enter a branch count (current: {args.mct}): ", "green")))
            logging.info(colored(f"# cli-agent: Monte Carlo Tree Search count set to {args.mct}", "green"))
            if context_chat:
                context_chat.debug_title = "MCTs Branching - Main Context Chat" if args.mct > 1 else "Main Context Chat"
            continue
        elif user_input == "-strong" or user_input == "--strong":
            args.strong = not args.strong
            g.FORCE_FAST = False
            g.LLM = "gemini-2.5-pro-exp-03-25"
            logging.info(colored(f"# cli-agent: Strong LLM mode toggled {'on' if args.strong else 'off'}.", "green"))
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
        elif user_input == "-s" or user_input == "--speak":
            args.speak = not args.speak
            logging.info(colored(f"# cli-agent: Text-to-speech mode toggled {'on' if args.speak else 'off'}.", "green"))
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
        
        # Load or initialize the system instruction from external Markdown file
        instruction_path = g.CLIAGENT_INSTRUCTION_FILE
        if not os.path.exists(instruction_path):
            default_inst = f'''# SYSTEM INSTRUCTION

You are a CLI agent with computational notebook access. You execute Python/shell code to solve tasks, maintaining persistent state across interactions.

## Core Workflow
1. **THINK & PLAN**: Reason through tasks before acting. Use the notebook to explore and validate assumptions
2. **TRACK PROGRESS**: Maintain a TODO list with 2-3 steps ahead, updating it continuously as you work
3. **EXECUTE**: Write complete, production-ready code with proper error handling and typing
4. **VERIFY**: Test outputs and iterate based on results

## Key Behaviors
- Execute with `python` or `bash` blocks
- Default workspace: {g.AGENTS_SANDBOX_DIR}
- Create new files rather than overwriting unless explicitly requested
- Handle errors gracefully and document limitations
- Complete full implementations without placeholders or test code

## Tools Available
- Persistent Python environment
- Shell commands
- Custom utilities (HuggingFaceSearch, HuggingFaceDownloader, OllamaManager, etc.)
- File system access

You approach tasks systematically, exploring thoroughly before implementing solutions. Every step is tracked in your evolving TODO list.'''
            with open(instruction_path, 'w') as f:
                f.write(default_inst)
        with open(instruction_path, 'r') as f:
            system_instruction = f.read()

        g.INITIAL_MCT_VALUE = args.mct
        if os.getenv("DEFAULT_FORCE_LOCAL") == get_local_ip():
            args.local = True
        if args.voice:
            args.auto = True
            logging.info(colored("# cli-agent: Voice mode enabled, automatically enabling auto execution mode", "green"))
        if args.sandbox:
            g.USE_SANDBOX = True
            if importlib.util.find_spec("paramiko") is None:
                logging.critical(colored("Error: 'paramiko' package is required for sandbox mode. Please install 'pip install paramiko'.", "red"))
                exit(1)
        
        stdout_buffer, stderr_buffer = "", ""
        def stdout_callback(text: str):
            nonlocal stdout_buffer
            # Add a newline before the first bit of output for better separation
            if not stdout_buffer and not stderr_buffer:
                print()
            print(text, end="")
            stdout_buffer += text
        def stderr_callback(text: str):
            nonlocal stderr_buffer
            # Add a newline before the first bit of output for better separation
            if not stdout_buffer and not stderr_buffer:
                print()
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
        
        if args.llm == "__select__":
            logging.info(colored("# cli-agent: Opening LLM selection...", "green"))
            g.SELECTED_LLMS = await llm_selection(args, preselected_llms=None)
            args.llm = None
        elif args.llm:
            g.SELECTED_LLMS = [args.llm]
        
        # Now that LLM selections are made, print the startup summary
        print_startup_summary(args)
        
        notebook = ComputationalNotebook(stdout_callback=stdout_callback, stderr_callback=stderr_callback, input_prompt_handler=input_callback)
        utils_manager = UtilsManager()
        util_names = utils_manager.get_util_names()
        if util_names:
            logging.info(f"UtilsManager: Loaded {len(util_names)} utilities.")
        else:
            logging.warning("UtilsManager: No utilities loaded.")

        if context_chat is None:
            context_chat = Chat(debug_title="Main Context Chat")
            inst = system_instruction
            if g.SELECTED_UTILS:
                inst += f"\n\nIMPORTANT: You MUST use the following utility tools: {', '.join(g.SELECTED_UTILS)}\n\n"
            if args.sandbox:
                inst += f"\nPlease try to stay within your sandbox directory at {g.AGENTS_SANDBOX_DIR}\n"
            context_chat.set_instruction_message(inst)

            # --- BEGIN: Auto-load preprompt files ---
            logging.info("Checking for preprompt files to load into context...")
            preprompt_files_to_load = ['CLAUDE.md', 'GEMINI.md', 'QWEN.md', 'readme.md', 'README.md']
            loaded_files_content = []
            # Use a set to handle case-insensitivity of readme.md and avoid duplicates
            loaded_filenames_lower = set()

            for filename in preprompt_files_to_load:
                # If a case-insensitive version of the file has been loaded, skip
                if filename.lower() in loaded_filenames_lower:
                    continue

                if os.path.exists(filename):
                    try:
                        with open(filename, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Only add files that have content
                            if len(content.strip())>3:
                                loaded_files_content.append((filename, content))
                                loaded_filenames_lower.add(filename.lower())
                                logging.info(f"Loaded '{filename}' into initial context.")
                    except Exception as e:
                        logging.warning(f"Could not read preprompt file '{filename}': {e}")
            
            if loaded_files_content:
                # Combine all file contents into a single user message for context efficiency
                combined_prompt = "# Initial Context Files\nThe following files were automatically read into the context to provide guidance and task context:\n\n"
                for filename, content in loaded_files_content:
                    combined_prompt += "---\n"
                    combined_prompt += f"## Content of: `{filename}`\n\n"
                    # Using markdown fences for better presentation in the prompt
                    combined_prompt += f"```markdown\n{content.strip()}\n```\n\n"
                
                context_chat.add_message(Role.USER, combined_prompt)
            # --- END: Auto-load preprompt files ---
            
            all_util_names = utils_manager.get_util_names()
            if not (args.voice or args.speak):
                if 'tts' in all_util_names:
                    logging.warning(colored("TTS utility disabled. Use -v or -s to enable.", "yellow"))
            
            try:
                # --- HARCODED TODOS HINT for KICKSTART ---
                todos_kickstart_hint = ""
                try:
                    all_todos = TodosUtil._load_todos()
                    if any(not todo.get('completed', False) for todo in all_todos):
                        todos_kickstart_hint = f"Welcome back. You have pending tasks to address.\n{TodosUtil._format_todos(all_todos)}\n\n---\n\n"
                except Exception as e:
                    logging.warning(colored(f"Could not check for pending to-dos for kickstart hint. Error: {e}", "red"))

                kickstart_code_1 = 'echo "OS: $(lsb_release -ds)" && echo "Desktop: $XDG_CURRENT_DESKTOP" && echo "Window Manager: $(ps -eo comm | grep -E \'^kwin|^mutter|^openbox|^i3|^dwm\' | head -1)"'
                kickstart_output_1 = subprocess.check_output(kickstart_code_1, shell=True, text=True, executable='/bin/bash').strip()

                kickstart_code_2_literal = """import os
print(f"Current working directory: {os.getcwd()}")
print(f"Total files in current directory: {len(os.listdir())}")
print(f"First 5 files in current directory: {os.listdir()[:5]}")"""
                cwd = os.getcwd()
                cwd_files = os.listdir()
                kickstart_output_2 = f"Current working directory: {cwd}\nTotal files in current directory: {len(cwd_files)}\nFirst 5 files in current directory: {cwd_files[:5]}"

                kickstart_code_3_literal = """import datetime
print(f"Current year and month: {datetime.datetime.now().strftime('%Y-%m')}")"""
                kickstart_output_3 = f"Current year and month: {datetime.datetime.now().strftime('%Y-%m')}"

                kickstart_preprompt = f"""{todos_kickstart_hint}Hi, I am going to run some things to show you how your computational notebook works.
Let's see the window manager and OS version of your user's system:
```bash
{kickstart_code_1}
```
<execution_output>
{kickstart_output_1}
</execution_output>

That succeeded, now let's see the current working directory and the first 5 files in it:
```python
{kickstart_code_2_literal}
```
<execution_output>
{kickstart_output_2}
</execution_output>

That succeeded, now let's check the current time:
```python
{kickstart_code_3_literal}
```
<execution_output>
{kickstart_output_3}
</execution_output>
"""
                context_chat.add_message(Role.USER, kickstart_preprompt)
            except Exception as e:
                logging.warning(f"Could not generate kickstart prompt. Skipping. Error: {e}")

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
        while True:
            LlmRouter().failed_models.clear()
            user_input: Optional[str] = None
            
            # --- FIX: Set LLM strengths based on --fast or --strong flags ---
            g.LLM_STRENGTHS = []
            if args.strong:
                g.LLM_STRENGTHS = [AIStrengths.STRONG]
            elif args.fast:
                g.LLM_STRENGTHS = [AIStrengths.SMALL]
            
            # Explicitly set global flags from args
            g.FORCE_LOCAL = args.local
            g.DEBUG_CHATS = args.debug_chats
            g.FORCE_FAST = args.fast
            g.LLM = args.llm
            g.FORCE_ONLINE = args.online
            g.MCT = args.mct
            
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
                    from utils.viewfile import ViewFile
                    local_paths, _ = extract_paths(user_input)
                    for path in local_paths:
                        expanded_path = os.path.expanduser(path)
                        if not os.path.exists(expanded_path):
                            continue
                        if os.path.isfile(expanded_path):
                            logging.info(colored(f"# cli-agent: Auto-viewing file: {path}", "green"))
                            view_result = json.loads(ViewFile.run(path=expanded_path))
                            if "result" in view_result:
                                user_input += f"\n\n# Content of: {path}\n```\n{view_result['result'].get('content', '')}\n```"
                        elif os.path.isdir(expanded_path):
                            logging.info(colored(f"# cli-agent: Auto-viewing directory: {path}", "green"))
                            tree_output = subprocess.check_output(['tree', '-L', '2', expanded_path]).decode('utf-8')
                            user_input += f"\n\n# Directory listing of: {path}\n```bash\n{tree_output}```"
                except (ImportError, FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
                    logging.debug("Path augmentation feature failed silently.")
                user_interrupt = False

            if LlmRouter.has_unconfirmed_data():
                LlmRouter.confirm_finetuning_data()

            action_counter, assistant_response = 0, ""
            text_stream_painter = TextStreamPainter()
            base64_images: List[str] = []
            if user_input and context_chat:
                # Start with the original user input
                augmented_input = user_input
                
                # --- HARCODED TODOS HINT ---
                # Always check for pending to-dos and add them as a high-priority hint.
                try:
                    all_todos = TodosUtil._load_todos()
                    if any(not todo.get('completed', False) for todo in all_todos):
                        todos_hint = TodosUtil._format_todos(all_todos)
                        # This structured format helps the agent parse the context clearly.
                        augmented_input = (
                            "# HIGH PRIORITY: PENDING TASKS\n"
                            "Address your existing to-do list before starting new work.\n"
                            f"{todos_hint}\n\n"
                            "---\n\n"
                            "# NEW REQUEST\n"
                            f"{augmented_input}"
                        )
                        logging.info(colored("\n📝 Pending to-dos found. Prepending hint to the context.", "magenta"))
                except Exception as e:
                    logging.warning(colored(f"Could not add todos hint. Error: {e}", "red"))

                # --- CONTEXTUAL TOOL SUGGESTIONS ---
                # Get relevant tools based on the *original* user input for better relevance.
                guidance_prompt = utils_manager.get_relevant_tools_prompt(user_input, top_k=3)
                if guidance_prompt:
                    # Append the guidance to the (potentially already augmented) input
                    augmented_input += f"\n\n{guidance_prompt}"
                    logging.info(colored(f"📝 Added contextual tool suggestions:\n{guidance_prompt}", "magenta"))

                context_chat.add_message(Role.USER, augmented_input)

            last_action_signature: Optional[str] = None
            stall_counter: int = 0
            MAX_STALLS: int = 2
            
            while True:
                try:
                    response_branches: List[str] = []
                    try:
                        if assistant_response:
                            logging.warning(colored("WARNING: Unhandled assistant response detected.", "yellow"))
                            context_chat.add_message(Role.ASSISTANT, assistant_response)
                            assistant_response = ""

                        models_to_use = g.SELECTED_LLMS or [m.model_key for m in LlmRouter.get_models(force_local=g.FORCE_LOCAL, strengths=g.LLM_STRENGTHS)]
                        num_branches = args.mct if args.mct > 1 else len(g.SELECTED_LLMS) if g.SELECTED_LLMS else 1
                        
                        if num_branches > 1:
                            print_lock = asyncio.Lock()
                            async def generate_branch(model_key: str, branch_index: int, lock: asyncio.Lock):
                                response_buffer_list = [""]
                                branch_update_callback = create_interruption_callback(
                                    response_buffer=response_buffer_list,
                                    painter=text_stream_painter,
                                    lock=lock
                                )
                                try:
                                    # --- FIX: Using explicit keyword arguments ---
                                    context_chat.debug_title = f"MCT Branching {branch_index+1}/{num_branches}"
                                    await LlmRouter.generate_completion(
                                        chat=context_chat,
                                        preferred_models=[model_key] if model_key else [],
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
                                        logging.error(colored(f"❌ Branch {branch_index+1} ({model_key}) failed: {e}", "red"))
                                        if model_key: LlmRouter().failed_models.add(model_key)
                                    return None
                            
                            tasks = []
                            for i in range(num_branches):
                                available_models = [m.model_key for m in LlmRouter.get_models(force_local=g.FORCE_LOCAL, strengths=g.LLM_STRENGTHS) if m.model_key not in LlmRouter().failed_models]
                                model_for_branch = g.SELECTED_LLMS[i] if g.SELECTED_LLMS and i < len(g.SELECTED_LLMS) else available_models[0] if available_models else None
                                
                                if model_for_branch is None:
                                    logging.warning(colored(f"❌ No available model for branch {i+1}, skipping", "yellow"))
                                    continue
                                    
                                tasks.append(generate_branch(model_for_branch, i, print_lock))
                            
                            if not tasks:
                                logging.error(colored("❌ No viable models available for MCT branching", "red"))
                                break
                                
                            branch_results = await asyncio.gather(*tasks)
                            response_branches = [res for res in branch_results if res and res.strip()]
                            print()
                        else:
                            model_for_branch = models_to_use[0] if models_to_use else None
                            context_chat.debug_title = f"Generation ({model_for_branch or 'auto'})"
                            
                            response_buffer_list = [""]
                            interruption_callback = create_interruption_callback(
                                response_buffer=response_buffer_list,
                                painter=text_stream_painter
                            )
                            try:
                                # --- FIX: Using explicit keyword arguments ---
                                await LlmRouter.generate_completion(
                                    chat=context_chat,
                                    preferred_models=[model_for_branch] if model_for_branch else [],
                                    force_preferred_model=True,
                                    temperature=temperature,
                                    base64_images=base64_images,
                                    generation_stream_callback=interruption_callback,
                                    strengths=g.LLM_STRENGTHS,
                                    thinking_budget=None,
                                    exclude_reasoning_tokens=True
                                )
                                if response_buffer_list[0].strip(): response_branches.append(response_buffer_list[0])
                            except StreamInterruptedException as e:
                                if e.response and e.response.strip(): response_branches.append(e.response)
                        base64_images = []
                    except KeyboardInterrupt:
                        logging.warning(colored("\n-=- User interrupted model generation -=-", "yellow"))
                        if args.message: args.message = []
                        break
                    except Exception as e:
                        if not isinstance(e, StreamInterruptedException):
                            LlmRouter.clear_unconfirmed_finetuning_data()
                            logging.error(colored(f"Error generating response: {e}", "red"), exc_info=args.debug)
                            break
                    
                    if response_branches:
                        if num_branches > 1 and len(response_branches) > 1:
                            try:
                                selected_branch_index, judge_llm = await select_best_branch(context_chat, response_branches)
                                assistant_response = response_branches[selected_branch_index]
                                logging.info(colored(f"✅ Picked branch {selected_branch_index} from {judge_llm}, appending response:", "green"))
                                print()
                                print(text_stream_painter.apply_color(assistant_response), flush=True) 
                            except Exception as e:
                                logging.error(colored(f"Error during MCT branch selection: {e}", "red"), exc_info=args.debug)
                                logging.warning(colored("\n⚠️ Defaulting to first branch.", "yellow"))
                                assistant_response = response_branches[0]
                        else:
                            # Single branch or only one successful branch
                            assistant_response = response_branches[0]
                            if num_branches > 1:
                                logging.info(colored(f"✅ Using single successful branch from {num_branches} attempts", "green"))
                    else:
                        logging.error(colored("All generation branches failed.", "red"))
                        break

                    current_action_signature = assistant_response
                    if last_action_signature and current_action_signature == last_action_signature:
                        stall_counter += 1
                        logging.warning(colored(f"Stall counter: {stall_counter}/{MAX_STALLS}", "yellow"))
                    else:
                        stall_counter = 0
                    last_action_signature = current_action_signature

                    if stall_counter >= MAX_STALLS:
                        logging.error(colored("! Agent appears to be stalled. Intervening.", "red"))
                        context_chat.add_message(Role.USER, "My last two attempts have failed. I need to stop, re-evaluate my strategy, and devise a new plan.")
                        stall_counter, last_action_signature = 0, None
                        break

                    shell_blocks = get_extract_blocks()(assistant_response, ["shell", "bash"])
                    python_blocks = get_extract_blocks()(assistant_response, ["python", "tool_code"])

                    if not python_blocks and not shell_blocks:
                        try:
                            all_todos = TodosUtil._load_todos()
                            if any(not todo.get('completed', False) for todo in all_todos):
                                auto_prompt = f"You have remaining todos. Here's your list: {TodosUtil._format_todos(all_todos)}"
                                logging.info(colored("\n📝 Pending to-dos found. Auto-prompting.", "magenta"))
                                context_chat.add_message(Role.USER, auto_prompt)
                        except Exception as e:
                            logging.warning(colored(f"Could not check for pending to-dos. Error: {e}", "red"))
                        
                        context_chat.add_message(Role.ASSISTANT, assistant_response)
                        assistant_response = ""
                        break

                    if g.SELECTED_UTILS:
                        used_tools = [tool for tool in g.SELECTED_UTILS if any(tool in code for code in python_blocks)]
                        if used_tools:
                            logging.info(colored(f"# cli-agent: Tools used and removed from required list: {', '.join(used_tools)}", "green"))
                            g.SELECTED_UTILS = [t for t in g.SELECTED_UTILS if t not in used_tools]

                    if args.voice or args.speak:
                        verbal_text = re.sub(r'```[^`]*```', '', assistant_response)
                        if python_blocks and shell_blocks: verbal_text += "I've implemented shell and python code."
                        elif python_blocks: verbal_text += "I've implemented python code."
                        elif shell_blocks: verbal_text += "I've implemented shell code."
                        TtsUtil.run(text=verbal_text)

                    formatted_code = ""
                    if shell_blocks: formatted_code += "```bash\n" + "\n".join(shell_blocks) + "\n```\n"
                    if python_blocks: formatted_code += "```python\n" + python_blocks[0] + "\n```"

                    context_chat.save_to_json()
                    if await confirm_code_execution(args, formatted_code):
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
                            logging.info(colored("\n✅ Code execution completed.", "cyan"))

                            tool_output = ""
                            if stdout_buffer.strip(): tool_output += f"```stdout\n{stdout_buffer.strip()}\n```\n"
                            if stderr_buffer.strip(): tool_output += f"```stderr\n{stderr_buffer.strip()}\n```\n"
                            tool_output = re.sub(r'\x1b\[[0-9;]*m', '', tool_output)

                            if len(tool_output) > 4000:
                                tool_output = tool_output[:3000] + "\n[...output truncated...]\n" + tool_output[-1000:]

                            if not tool_output.strip():
                                tool_output = "<execution_output>\nThe execution completed without output.\n</execution_output>"
                            else:
                                tool_output = f"<execution_output>\n{tool_output.strip()}\n</execution_output>"
                            
                            assistant_response_with_output = f"{assistant_response}\n{tool_output}\n<think>\n"
                            context_chat.add_message(Role.ASSISTANT, assistant_response_with_output)
                            assistant_response = ""
                            stdout_buffer, stderr_buffer = "", ""
                            action_counter += 1
                            continue
                        except Exception as e:
                            logging.error(colored(f"\n❌ Error executing code: {e}", "red"), exc_info=args.debug)
                            error_output = f"<execution_output>\n```error\n{traceback.format_exc()}\n```\n</execution_output>"
                            context_chat.add_message(Role.ASSISTANT, f"{assistant_response}\n{error_output}")
                            assistant_response, stdout_buffer, stderr_buffer = "", "", ""
                            break
                    else:
                        logging.warning(colored(" Execution cancelled by user.", "yellow"))
                        cancellation_notice = "<execution_output>\nCode execution cancelled by user.\n</execution_output>"
                        context_chat.add_message(Role.ASSISTANT, f"{assistant_response}\n{cancellation_notice}")
                        assistant_response, stdout_buffer, stderr_buffer = "", "", ""
                        break
                except KeyboardInterrupt:
                    logging.warning(colored("\n=== User interrupted execution (Ctrl+C) ===", "yellow"))
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
                        context_chat.add_message(Role.ASSISTANT, f"{assistant_response}\n{error_output}")
                    except Exception as context_e:
                        logging.error(colored(f"Failed to add error to context: {context_e}", "red"))
                    assistant_response, stdout_buffer, stderr_buffer = "", "", ""
                    break
            
            if context_chat:
                context_chat.save_to_json()
            if args.exit and not args.message:
                logging.info(colored("All automatic messages processed successfully. Exiting...", "green"))
                exit(0)
        logging.info(colored("\nCLI-Agent is shutting down.", "cyan"))

    except (asyncio.CancelledError, KeyboardInterrupt):
        logging.warning(colored("\nCLI-Agent was interrupted. Shutting down gracefully...", "yellow"))
    except Exception as e:
        if not isinstance(e, StreamInterruptedException):
            logging.critical(colored(f"\nCLI-Agent encountered a fatal error: {e}", "red"), exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(colored("\nCLI-Agent was interrupted by user. Shutting down...", "yellow"))
    except Exception as e:
        if not isinstance(e, StreamInterruptedException):
            print(colored(f"\nCLI-Agent encountered a fatal error during startup: {e}", "red"))
            traceback.print_exc()