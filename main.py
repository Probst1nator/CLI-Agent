#!/usr/bin/env python3

import datetime
import json
import logging
import os
import select
import time
import traceback
from typing import List, Optional, Tuple
from pyfiglet import figlet_format
from dotenv import load_dotenv
import pyperclip
from termcolor import colored
import argparse
import sys
import socket
import warnings
import asyncio
import re
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.widgets import CheckboxList, Frame, Label, RadioList
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
import base64
import tempfile
import subprocess

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")
# Suppress phonemizer warnings
warnings.filterwarnings("ignore", message="words count mismatch on*", module="phonemizer", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")  # Catch all phonemizer warnings

# Import core modules
from py_classes.cls_computational_notebook import ComputationalNotebook
from py_classes.cls_util_manager import UtilsManager
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_llm_router import Llm, LlmRouter
from py_classes.cls_chat import Chat, Role
from py_classes.utils.cls_utils_web_server import WebServer
from py_classes.globals import g
from py_classes.cls_text_stream_painter import TextStreamPainter

# Fix the import by using a relative or absolute import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Global timer for startup and operations
start_time = time.time()

def print_with_timer(message, color="white"):
    elapsed = time.time() - start_time
    print(colored(f"[{elapsed:.2f}s] {message}", color))

# Lazy load functions from py_methods.utils to avoid heavy imports at startup
def get_extract_blocks():
    from py_methods.utils import extract_blocks
    return extract_blocks

def get_pdf_or_folder_to_database():
    from py_methods.utils import pdf_or_folder_to_database
    return pdf_or_folder_to_database

def get_listen_microphone():
    from py_methods.utils import listen_microphone
    return listen_microphone

def get_take_screenshot():
    from py_methods.utils import take_screenshot
    return take_screenshot

def get_update_cmd_collection():
    from py_methods.utils import update_cmd_collection
    return update_cmd_collection

# Deprecated: TTS functionality moved to utils/tts.py
# def get_utils_audio():
#     from py_methods import utils_audio
#     return utils_audio

# Try importing with a direct import
try:
    from utils.viewimage import ViewImage
except ImportError:
    # Fallback to a direct import of the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("ViewImage", 
                                                 os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                             "utils", "viewimage.py"))
    viewimage_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewimage_module)
    ViewImage = viewimage_module.ViewImage

# Import the new TTS utility for internal use
try:
    from utils.tts import TtsUtil
except ImportError:
    # Handle case where TTS utility might not be available
    class TtsUtil:
        @staticmethod
        def run(text, **kwargs):
            print(colored(f"Warning: TtsUtil not found. Cannot speak: {text}", "red"))
            return json.dumps({"status": "error", "message": "TtsUtil not found"})
try:
    from utils.todos import TodosUtil
except ImportError:
    # Handle case where Todos utility might not be available
    class TodosUtil:
        @staticmethod
        def _load_todos(**kwargs):
            print(colored(f"Warning: TodosUtil not found.", "red"))
            return []
        @staticmethod
        def _format_todos(**kwargs):
            print(colored(f"Warning: TodosUtil not found.", "red"))
            return ""

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
# Disable CUDA warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # This will force CPU usage

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_local_ip():
    try:
        # Create a socket object and connect to an external server
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


def parse_cli_args() -> argparse.Namespace:
    """Setup and parse CLI arguments, ensuring the script's functionality remains intact."""
    
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
    parser.add_argument("-mct", "--mct", type=int, nargs='?', const=3, default=0,
                        help="Enable Monte Carlo Tree Search for acting, with an optional number of branches (default: 3).")
    parser.add_argument("-sbx", "--sandbox", action="store_true", default=False,
                        help="Use weakly sandboxed python execution. Sets g.USE_SANDBOX=True.")
    parser.add_argument("-o", "--online", action="store_true", default=False,
                        help="Force use of cloud AI.")
    
    parser.add_argument("-llm", "--llm", type=str, nargs='?', const="__select__", default=None,
                        help="Specify the LLM model key to use (e.g., 'gpt-4', 'gemini-pro'). Use without value to open selection menu.")
    
    parser.add_argument("--preload", action="store_true", default=False,
                        help="Preload systems like embeddings and other resources.")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Open a web interface for the chat")
    parser.add_argument("--debug-chats", action="store_true", default=False,
                        help="Enable debug windows for chat contexts without full debug logging. Sets g.DEBUG_CHATS=True.")
    parser.add_argument("--private_remote_wake_detection", action="store_true", default=False,
                        help="Use private remote wake detection")
    parser.add_argument("--local-exec-confirm", action="store_true", default=False,
                        help="Use local LLM for auto-execution confirmation instead of cloud models.")
    
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug logs")
    
    parser.add_argument("-e", "--exit", action="store_true", default=False,
                        help="Exit after all automatic messages have been parsed successfully")
    
    # Parse known arguments and capture any unrecognized ones
    args, unknown_args = parser.parse_known_args()

    if unknown_args or args.help:
        if not args.help:
            print(colored(f"Warning: Unrecognized arguments {' '.join(unknown_args)}.", "red"))
        parser.print_help()
        exit(1)
    
    return args

async def llm_selection(args: argparse.Namespace, preselected_llms: List[str] = None) -> List[str]:
    """
    Handles the LLM selection process, supporting multi-selection mode.
    
    Args:
        args: The parsed command line arguments containing auto mode settings
        preselected_llms: Optional list of LLMs to preselect. If None, loads from persistent storage.
        
    Returns:
        List of selected LLM model keys or special identifiers
    """
    # Show available LLMs using prompt_toolkit
    available_llms = Llm.get_available_llms(exclude_guards=True)
    
    # Create styled LLM choices
    llm_choices = []
    # Add "Any but local" option at the top
    llm_choices.append(("any_local", HTML('<provider>Any</provider> - <model>Any but local</model> - <pricing>Automatic selection</pricing>')))
    
    # Get Ollama model status for display
    from py_classes.ai_providers.cls_ollama_interface import OllamaClient
    ollama_status = {}
    try:
        ollama_status = OllamaClient.get_comprehensive_model_status(["localhost", "192.168.178.37"])
    except Exception:
        pass  # Continue without Ollama status if it fails
    
    for llm in available_llms:
        # Check if this is an Ollama model and get its download status
        status_indicator = ""
        if llm.provider.__class__.__name__ == "OllamaClient":
            model_base_name = llm.model_key.split(':')[0]  # Remove tag for lookup
            if model_base_name in ollama_status:
                if ollama_status[model_base_name]['downloaded']:
                    status_indicator = ' <downloaded>✓ Downloaded</downloaded>'
                else:
                    status_indicator = ' <notdownloaded>⬇ Available</notdownloaded>'
            else:
                status_indicator = ' <notdownloaded>⬇ Available</notdownloaded>'
        
        # Create HTML formatted text with colors
        styled_text = HTML(
            f'<provider>{llm.provider.__class__.__name__}</provider> - '
            f'<model>{llm.model_key}</model> - '
            f'<pricing>{f"${llm.pricing_in_dollar_per_1M_tokens}/1M tokens" if llm.pricing_in_dollar_per_1M_tokens else "Free"}</pricing> - '
            f'<context>Context: {llm.context_window}</context>'
            f'{status_indicator}'
        )
        llm_choices.append((llm.model_key, styled_text))
    
    # Define the style
    style = Style.from_dict({
        'model': 'ansicyan',
        'provider': 'ansigreen',
        'pricing': 'ansimagenta',  # Changed from harsh yellow to magenta
        'context': 'ansiblue',
        'downloaded': 'ansibrightgreen',
        'notdownloaded': 'ansibrightred',
    })
    
    # Use CheckboxList instead of RadioList to allow multiple selections
    # Set default selected values - use preselected_llms if provided, otherwise load from storage
    default_selected = []
    if preselected_llms is not None:
        default_selected = [llm for llm in preselected_llms if llm in [choice[0] for choice in llm_choices]]
    else:
        # Load previously saved selection
        saved_llms = load_selected_llms()
        default_selected = [llm for llm in saved_llms if llm in [choice[0] for choice in llm_choices]]
    
    checkbox_list = CheckboxList(
        values=llm_choices,
        default_values=default_selected
    )
    bindings = KeyBindings()

    @bindings.add("e")
    def _execute(event) -> None:
        # Extract the selected values
        selected = []
        for value, _ in llm_choices:
            if value in checkbox_list.current_values:
                selected.append(value)
        app.exit(result=selected)

    @bindings.add("a")
    def _abort(event) -> None:
        app.exit(result=None)

    instructions = Label(text="Use arrow keys to navigate, Space to select/deselect LLMs, 'e' to confirm or 'a' to abort")
    root_container = HSplit([
        Frame(title="Select LLMs to use", body=checkbox_list),
        instructions
    ])
    layout = Layout(root_container)
    app = Application(layout=layout, key_bindings=bindings, full_screen=False, style=style)
    
    # Use the current event loop instead of creating a new one
    try:
        selected_llms = await app.run_async()
        
        if selected_llms is not None and len(selected_llms) > 0:
            # Save the selection to persistent storage
            save_selected_llms(selected_llms)
            
            # Check if "Any but local" was selected
            if "any_local" in selected_llms:
                g.FORCE_LOCAL = True
                args.local = True
                args.llm = None
                print(colored(f"# cli-agent: 'Any but local' option selected. Local mode enabled.", "green"))
                return []
            
            # If multiple LLMs were selected, enable MCT mode automatically
            if len(selected_llms) > 1:
                args.mct = len(selected_llms)
                print(colored(f"# cli-agent: Multiple LLMs selected ({', '.join(selected_llms)}). MCT mode enabled automatically with {args.mct} branches.", "green"))
            else:
                # Single LLM selected - store in args.llm
                args.llm = selected_llms[0]
                print(colored(f"# cli-agent: LLM set to {args.llm}, type (--h) for info", "green"))
            
            return selected_llms
        else:
            print(colored(f"# cli-agent: No LLMs selected or selection cancelled", "yellow"))
            return []
    except asyncio.CancelledError:
        print(colored(f"# cli-agent: LLM selection was interrupted", "yellow"))
        return []
    except Exception as e:
        print(colored(f"# cli-agent: Error during LLM selection: {str(e)}", "red"))
        if args.debug:
            traceback.print_exc()
        return []

async def utils_selection(args: argparse.Namespace) -> List[str]:
    """
    Handles the Utils selection process, supporting multi-selection mode.
    
    Args:
        args: The parsed command line arguments

    Returns:
        List of selected util names or special identifiers
    """
    # Show available utils using prompt_toolkit
    utils_manager = UtilsManager()
    available_utils = utils_manager.get_util_names()
    
    # Create styled util choices
    util_choices = []
    
    # Add a special "Add New Tool" option at the top
    add_new_tool_id = "__add_new_tool__"
    util_choices.append((add_new_tool_id, HTML('<special>+ Add New Tool</special>')))
    
    for util_name in available_utils:
        # Create HTML formatted text with colors
        styled_text = HTML(
            f'<util>{util_name}</util>'
        )
        util_choices.append((util_name, styled_text))
    
    # Define the style
    style = Style.from_dict({
        'util': 'ansigreen',
        'special': 'ansiyellow bold',
    })
    
    checkbox_list = CheckboxList(
        values=util_choices
    )
    bindings = KeyBindings()

    @bindings.add("e")
    def _execute(event) -> None:
        # Fix: CheckboxList.current_values returns the selected values directly
        selected = []
        for value, _ in util_choices:
            if value in checkbox_list.current_values:
                selected.append(value)
        app.exit(result=selected)

    @bindings.add("a")
    def _abort(event) -> None:
        app.exit(result=None)

    instructions = Label(text="Use arrow keys to navigate, Space to select/deselect utils, 'e' to confirm or 'a' to abort")
    root_container = HSplit([
        Frame(title="Select Utils to Use", body=checkbox_list),
        instructions
    ])
    layout = Layout(root_container)
    app = Application(layout=layout, key_bindings=bindings, full_screen=False, style=style)
    
    # Use the current event loop instead of creating a new one
    try:
        selected_utils = await app.run_async()
        
        if selected_utils is not None and len(selected_utils) > 0:
            # Check if the special "Add New Tool" option was selected
            if add_new_tool_id in selected_utils:
                return [add_new_tool_id]
            
            utils_list = ", ".join(selected_utils)
            print(colored(f"# cli-agent: Selected utils: {utils_list}", "green"))
            return selected_utils
        else:
            print(colored(f"# cli-agent: No utils selected or selection cancelled", "yellow"))
            return []
    except asyncio.CancelledError:
        print(colored(f"# cli-agent: Utils selection was interrupted", "yellow"))
        return []
    except Exception as e:
        print(colored(f"# cli-agent: Error during utils selection: {str(e)}", "red"))
        if args.debug:
            traceback.print_exc()
        return []

async def confirm_code_execution(args: argparse.Namespace, code_to_execute: str) -> bool:
    """
    Handles the confirmation process for code execution, supporting both auto and manual modes.
    
    Args:
        args: The parsed command line arguments containing auto mode settings
        code_to_execute: The Python code to potentially execute
        
    Returns:
        bool: True if execution should proceed, False if aborted
    """
    python_codes: List[str] = get_extract_blocks()(code_to_execute, "python")
    
    if (len(python_codes) == 0):
        always_permitted_bash = ["ls ", "pwd ", "cd ", "echo ", "print ", "cat ", "head ", "tail ", "grep ", "sed ", "awk ", "sort "]
        # check if code to execute is executing solely always_permitted_commands
        bash_codes: List[str] = get_extract_blocks()(code_to_execute, "bash")
        bash_code = "\n".join(bash_codes)
        bash_code_lines = bash_code.split("\n")
        # cleanup empty lines and comments
        bash_code_lines = [line for line in bash_code_lines if line.strip() and not line.strip().startswith("#")]
        allowed = True
        collected_matching_commands = []
        for line in bash_code_lines:
            matching_commands = [cmd for cmd in always_permitted_bash if line.startswith(cmd)]
            if len(matching_commands) > 0 and (line.count(" && ") + line.count(" || ") + line.count(";") + 1 == len(matching_commands)):
                collected_matching_commands.extend(matching_commands)
                continue
            allowed = False
            break
        if allowed:
            print(colored(f"✅ Code execution permitted automatically (These commands are always allowed: '{', '.join(collected_matching_commands).strip()}')", "green"))
            return True
    
    if args.auto:  # Check if auto mode is enabled
        # Auto execution guard
        execution_guard_chat: Chat = Chat(
            instruction_message="""You are a Code Execution Guardian. Your primary goal is to prevent unsafe or incomplete code execution.

Priorities:
1.  **Safety First:** Identify any operations with potential negative side effects (e.g., unintended file/system modifications, risky shell commands, unrestricted network calls), modifications of files are allowed if the comments show that it is intentional and safe.
2.  **Completeness Second:** If safe, ensure that the script does not contain any placeholders (e.g., `YOUR_API_KEY`, `<REPLACE_ME>`), unimplemented logic or similar. Comments noting future work are allowed. Scripts that only print text are also always allowed.

Assume anything imported from utils.* is safe.

Process:
1.  **Reason Briefly:** First, explain your core reasoning for safety and completeness.
2.  **Verdict:** Conclude with EXACTLY ONE WORD:
    *   `no`: If the code is unsafe (regardless of completeness).
    *   `unfinished`: If the code is safe BUT contains placeholders or is incomplete.
    *   `yes`: If the code is safe AND complete.

Do not add any other text after this single-word verdict.""",
            debug_title="Auto Execution Guard"
        )
# """You are a Code Execution Guardian. Your primary goal is to prevent unsafe or incomplete code execution.

# Priorities:
# 1.  **Safety First:** Identify any operations with potential negative side effects (e.g., unintended file/system modifications, risky shell commands, unrestricted network calls), modifications of files are allowed if the comments show that it is intentional and safe.
# 2.  **Completeness Second:** If safe, ensure that the script does not contain any placeholders (e.g., `YOUR_API_KEY`, `<REPLACE_ME>`), unimplemented logic or similar. Comments noting future work are allowed. Scripts that only print text are always allowed.

# Assume anything imported from utils.* is safe.

# Process:
# 1.  **Reason Briefly:** First, explain your core reasoning for safety and completeness. This reasoning should be concise and direct.
# 2.  **Verdict (JSON):** After your reasoning, provide ONLY a JSON object in a markdown block (```json...```). This JSON object shall either be:
#     *   `{"execute": true}`
#     *   `{"execute": false, "reason": "the code does not implement xx properly or it is unsafe"}`

#     The 'reason' field is mandatory when "execute" is false, and should clearly state why the code is unsafe or unfinished (e.g., "contains placeholders", "attempts unauthorized file deletion").

# Do NOT include any other text after the JSON block.
# """,
        if ("bash" in code_to_execute and "python" in code_to_execute):
            analysis_prompt = f"Analyze this code for safe execution and completeness:\n```bash\n{code_to_execute}\n```"
        elif "python" in code_to_execute:
            analysis_prompt = f"Analyze this python code for safe execution and completeness:\n```python\n{code_to_execute}\n```"
        else:
            analysis_prompt = f"Analyze these bash commands for safe execution and completeness:\n{code_to_execute}"
        execution_guard_chat.add_message(Role.USER, analysis_prompt)
        # For auto execution guard, prefer faster models by excluding STRONG models
        guard_preferred_models = []
        if args.local_exec_confirm:
            # Get local models that are NOT strong
            available_models = LlmRouter.get_models(force_local=True)
            guard_preferred_models = [model.model_key for model in available_models if not any(s == AIStrengths.STRONG for s in model.strengths)]
        
        safe_to_execute: str = await LlmRouter.generate_completion(execution_guard_chat, preferred_models=guard_preferred_models, hidden_reason="Assessing execution safety", force_local=args.local_exec_confirm, strengths=[])
        if 'yes' in safe_to_execute.lower().strip():
            print(colored("✅ Code execution permitted", "green"))
            return True
        elif 'unfinished' in safe_to_execute.lower().strip():
            print(colored("⚠️ Code execution aborted by auto-execution guard, because it is unfinished", "yellow"))
            # Add a message to args.message to be automatically processed in the next loop
            completion_request = "The code you provided is unfinished. Please complete it properly with actual values and logic."
            
            # Append the completion request at the beginning of the list
            args.message.insert(0, completion_request)
            
            print(colored(f"💬 Added automatic follow-up request: {completion_request}", "blue"))
            return False
        else:
            text_stream_painter = TextStreamPainter()
            for char in safe_to_execute:
                print(text_stream_painter.apply_color(char), end="")
            print(colored("\n❌ Code execution aborted by auto-execution guard", "red"))
            return False
    else:
        # Manual confirmation
        user_input = await get_user_input_with_bindings(args, None, colored(" (Press Enter to confirm or 'n' to abort, press 'a' to toggle auto execution, 'l' for local auto execution)", "cyan"))
        if user_input.lower() == 'n':
            print(colored("❌ Code execution aborted by user", "red"))
            return False
        elif user_input.lower() == 'a':
            args.auto = not args.auto
            print(colored(f"# cli-agent: KeyBinding detected: Automatic execution toggled {'on' if args.auto else 'off'}, type (--h) for info", "green"))
            return await confirm_code_execution(args, code_to_execute)
        elif user_input.lower() == 'l':
            args.auto = True
            args.local_exec_confirm = True
            print(colored(f"# cli-agent: KeyBinding detected: Local auto execution toggled {'on' if args.local_exec_confirm else 'off'}, type (--h) for info", "green"))
            return await confirm_code_execution(args, code_to_execute)
        else:
            print(colored("✅ Code execution permitted", "green"))
    
    return True


async def select_best_branch(
    context_chat: Chat,
    assistant_responses: List[str],
) -> str:
    """
    Select the best branch from multiple full assistant responses.
    
    Args:
        assistant_responses: List of full assistant responses (including reasoning and code)
        user_input: The original user request
        
    Returns:
        Index of the selected branch
    """
    mct_branch_selector_chat: Chat = context_chat.deep_copy()
    mct_branch_selector_chat.debug_title="MCT Branch Selection"
    mct_branch_selector_chat.add_message(Role.ASSISTANT, assistant_responses[0])
    
    selection_prompt = f"I've also asked other agents to generate a reply to my question. Please compare the suggested responses to your own and pick your favorite by ending your response with the index of your preferred response. You can consider your previous response indexed as 0. If you would like to reply yourself, being more informed by the different perspectives, you will need to select your own response which is indexed by {len(assistant_responses)+1}"
    for i, response in enumerate(assistant_responses[1:]):
        selection_prompt += f"# Response {i}\n\n"
    
    mct_branch_selector_chat.add_message(Role.USER, selection_prompt)
    
    # If multiple LLMs are selected, use the first one for evaluation
    evaluator_model = []
    if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 0:
        evaluator_model = [g.SELECTED_LLMS[0]]
        print(colored(f"Using {g.SELECTED_LLMS[0]} to evaluate responses from all models", "cyan"))
    
    evaluator_response: str = await LlmRouter.generate_completion(
        mct_branch_selector_chat,
        evaluator_model,
    )

    assistant_responses.append(evaluator_response)
    
    # Extract the selected branch number
    match: Optional[re.Match] = re.search(r'\s*(\d+)', evaluator_response)
    if match:
        selected_branch_index: int = int(match.group(1))
        if 0 <= selected_branch_index < len(assistant_responses):
            return selected_branch_index
    
    # Default to first branch if no valid selection was made
    print(colored("\n⚠️ No valid branch selection found. Defaulting to first branch.", "yellow"))
    return 0

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
        except EOFError: # Handle Ctrl+D
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

    Returns:
        - The user's input string. (If empty the user will NOT be asked for input)
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
                # get user input from various sources if not already set (e.g., after screenshot)
                user_input = input(prompt)
            except KeyboardInterrupt: # Handle Ctrl+C as exit
                print(colored("\n# cli-agent: Exiting due to Ctrl+C.", "yellow"))
                exit()
        

        # USER INPUT HANDLING - BEGIN
        if user_input == "-r" or user_input == "--r":
            if not context_chat or len(context_chat.messages) < 2:
                print(colored("# cli-agent: No chat history found, cannot regenerate last response.", "red"))
                continue # Ask for input again
            print(colored("# cli-agent: KeyBinding detected: Regenerating last response, type (--h) for info", "green"))
            context_chat.messages.pop() # Remove last AI response
            user_input = ""

        elif user_input == "-l" or user_input == "--local":
            args.local = not args.local
            g.FORCE_LOCAL = args.local # Ensure global state is updated
            print(colored(f"# cli-agent: KeyBinding detected: Local mode toggled {'on' if args.local else 'off'}, type (--h) for info", "green"))
            continue

        elif user_input == "-llm" or user_input == "--llm":
            print(colored("# cli-agent: KeyBinding detected: Showing LLM selection, type (--h) for info", "green"))
            selected_llms = await llm_selection(args, preselected_llms=None)  # Load from persistent storage
            # Store the selected LLMs in globals for later use
            g.SELECTED_LLMS = selected_llms
            continue

        elif user_input == "-a" or user_input == "--auto":
            args.auto = not args.auto
            print(colored(f"# cli-agent: KeyBinding detected: Automatic execution toggled {'on' if args.auto else 'off'}, type (--h) for info", "green"))
            continue # Ask for input again

        elif user_input == "-mct" or user_input == "--mct":
            args.mct = 3 if args.mct == 0 else 0
            print(colored(f"# cli-agent: KeyBinding detected: Monte Carlo Tree Search toggled {'on' if args.mct else 'off'} (branches: {args.mct}), type (--h) for info", "green"))
            if context_chat:
                context_chat.debug_title = "MCTs Branching - Main Context Chat" if args.mct else "Main Context Chat"
            continue # Ask for input again

        elif user_input == "-strong" or user_input == "--strong":
            args.strong = not args.strong
            g.FORCE_FAST = False # Strong overrides fast
            g.LLM = "gemini-2.5-pro-exp-03-25"
            print(colored(f"# cli-agent: KeyBinding detected: Strong LLM mode toggled {'on' if args.strong else 'off'}, type (--h) for info", "green"))
            continue # Ask for input again

        elif user_input == "-f" or user_input == "--fast":
            args.fast = not args.fast
            g.FORCE_STRONG = False # Fast overrides strong
            print(colored(f"# cli-agent: KeyBinding detected: Fast LLM mode toggled {'on' if args.fast else 'off'}, type (--h) for info", "green"))
            continue
        
        elif user_input == "-v" or user_input == "--v":
            args.voice = not args.voice
            print(colored(f"# cli-agent: KeyBinding detected: Voice mode toggled {'on' if args.voice else 'off'}, type (--h) for info", "green"))
            continue
        
        elif user_input == "-speak" or user_input == "--speak":
            args.speak = not args.speak
            print(colored(f"# cli-agent: KeyBinding detected: Text-to-speech mode toggled {'on' if args.speak else 'off'}, type (--h) for info", "green"))
            continue

        elif user_input == "-img" or user_input == "--img" or user_input == "-screenshot" or user_input == "--screenshot" or args.image:
            print(colored("# cli-agent: KeyBinding detected: Taking screenshot with Spectacle, type (--h) for info", "green"))
            args.image = False
            base64_images = await handle_screenshot_capture(context_chat)
            continue

        elif user_input == "-p" or user_input == "--p":
            print(colored("# cli-agent: KeyBinding detected: Printing chat history, type (--h) for info", "green"))
            os.system('clear')
            print(colored("Chat history:", "green"))
            if context_chat:
                context_chat.print_chat()
            else:
                print(colored("No chat history available.", "yellow"))
            continue # Ask for input again

        elif user_input == "-m" or user_input == "--m":
            return handle_multiline_input() # Get multiline input
        
        elif user_input == "-o" or user_input == "--o" or user_input == "-online" or user_input == "--online":
            args.online = not args.online
            print(colored(f"# cli-agent: KeyBinding detected: Online mode toggled {'on' if args.online else 'off'}, type (--h) for info", "green"))
            continue
        
        elif user_input == "-e" or user_input == "--e" or user_input == "--exit":
            print(colored(f"# cli-agent: KeyBinding detected: Exiting...", "green"))
            exit(0)
        
        elif user_input == "-h" or user_input == "--h" or user_input == "--help":
            print(colored("# cli-agent: KeyBinding detected: Showing help", "green"))
            print(colored("\n=== CLI-Agent Interactive Keybindings ===", "cyan", attrs=["bold"]))
            print(colored("Available commands while chatting:", "yellow"))
            print(colored("  -r, --r           Regenerate last response", "white"))
            print(colored("  -l, --local       Toggle local/cloud LLM mode", "white"))
            print(colored("  -llm, --llm       Open LLM selection menu", "white"))
            print(colored("  -a, --auto        Toggle automatic code execution", "white"))
            print(colored("  -mct, --mct       Toggle Monte Carlo Tree Search", "white"))
            print(colored("  -strong, --strong Toggle strong LLM mode", "white"))
            print(colored("  -f, --fast        Toggle fast LLM mode", "white"))
            print(colored("  -v, --v           Toggle voice mode", "white"))
            print(colored("  -speak, --speak   Toggle text-to-speech", "white"))
            print(colored("  -img, --img       Take screenshot", "white"))
            print(colored("  -p, --p           Print chat history", "white"))
            print(colored("  -m, --m           Enter multiline input mode", "white"))
            print(colored("  -o, --o           Toggle online mode", "white"))
            print(colored("  -e, --e           Exit CLI-Agent", "white"))
            print(colored("  -h, --h           Show this help", "white"))
            print(colored("\nType any of these commands during chat to use them!", "green"))
            continue
        
        # USER INPUT HANDLING - END

        # No binding matched, return the input
        return user_input
    if args.image:
        args.image = False # Reset flag

async def handle_screenshot_capture(context_chat: Optional[Chat]) -> str:
    """
    Handles the screenshot capture process, including Spectacle, fallbacks, saving, and context update.
    Retries up to 3 times if no image is captured.

    Args:
        context_chat: The chat context to potentially add messages to.

    Returns:
        The user's input for further processing.
    """

    base64_images: List[str] = []
    screenshots_paths: List[str] = []
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        base64_images = [] # Reset for each attempt

        try:
            # Use Spectacle for region capture as the primary method
            print(colored("Attempting screenshot capture with Spectacle (region selection)...", "green"))

            # Create temp file path for spectacle to save the screenshot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_filename = temp_file.name

            # Call spectacle to capture a region (-r flag)
            # -b: no border, -n: no notification, -o: output file
            subprocess.run(['spectacle', '-rbno', temp_filename], check=True)

            # Check if the file exists and has content
            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                # Read the screenshot file
                with open(temp_filename, 'rb') as image_file:
                    png_data = image_file.read()

                # Convert to base64 and add to list
                base64_img = base64.b64encode(png_data).decode('utf-8')
                base64_images = [base64_img]
                # copy img to clipboard
                pyperclip.copy(base64_img)
                print(colored(f"Screenshot captured with Spectacle successfully. (+Copied to clipboard)", "green"))
            else:
                print(colored(f"No screenshot was captured with Spectacle or operation was cancelled on attempt {attempt}.", "yellow"))

            # Cleanup temp file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

        except subprocess.CalledProcessError:
            print(colored(f"Spectacle command failed or not found on attempt {attempt}. Is it installed?", "red"))
        except Exception as e:
            print(colored(f"An unexpected error occurred during screenshot capture attempt {attempt}: {str(e)}\n\n\n{traceback.print_exc()}", "red"))

        # If images were captured in this attempt, break the loop
        if base64_images:
            break

        # If this was not the last attempt and no images were captured, wait briefly before retrying
        if attempt < max_attempts and not base64_images:
            print(colored(f"Screenshot capture failed on attempt {attempt}. Retrying...", "yellow"))
            await asyncio.sleep(2) # Optional: brief pause before retry

    # After the loop (or break), process the results
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
                print(colored(f"Screenshot saved to {img_path}", "green"))
            except Exception as e:
                print(colored(f"Error saving screenshot {i+1}: {str(e)}", "red"))
    else:
        print(colored("No images were captured.", "red"))

    # Call the new handler function
    if not base64_images:
        print(colored("# cli-agent: No screenshot was captured or saved after multiple attempts.", "yellow"))
        return ""
    print(colored("Screenshot preprocesssing...", "green"))
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

    # Return the base64 images for use in the generate_completion call
    return base64_images

# Also, here's the enhanced sudo preprocessing function to integrate:
def preprocess_consecutive_sudo_commands(code: str) -> str:
    """Combine consecutive shell commands with sudo to reduce password prompts."""
    lines = code.strip().split('\n')
    processed_lines = []
    shell_commands = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('!'):
            shell_commands.append(line[1:])  # Remove ! prefix
        else:
            # Non-shell line encountered, flush any accumulated shell commands
            if shell_commands:
                if len(shell_commands) == 1:
                    processed_lines.append(f'!{shell_commands[0]}')
                else:
                    # Combine multiple shell commands with &&
                    combined = ' && '.join(shell_commands)
                    processed_lines.append(f'!{combined}')
                shell_commands = []
            processed_lines.append(line)
    
    # Handle any remaining shell commands
    if shell_commands:
        if len(shell_commands) == 1:
            processed_lines.append(f'!{shell_commands[0]}')
        else:
            combined = ' && '.join(shell_commands)
            processed_lines.append(f'!{combined}')
    
    return '\n'.join(processed_lines)

def extract_paths(user_input: str) -> Tuple[List[str], List[str]]:
    """
    Extracts local file paths, folder paths, and online URLs from a string.

    This function is designed to be robust by finding path-like strings that are
    either unquoted or enclosed in single or double quotes. It then cleans
    common trailing punctuation from the found paths.

    Args:
        user_input: The string to parse for paths and URLs.

    Returns:
        A tuple containing two lists of unique strings:
        - local_paths: A list of found local file/folder paths.
        - online_paths: A list of found online URLs.
    """
    # This pattern finds either:
    # 1. A quoted string: (["'])(.+?)\1 captures the content of matching quotes.
    # 2. An unquoted sequence of non-whitespace characters: (\S+)
    # This allows us to tokenize the input into potential paths.
    token_pattern = re.compile(r'(["\'])(.+?)\1|(\S+)')

    # A set of common punctuation to remove from the end of a path.
    # This prevents including sentence-ending characters in the path.
    trailing_chars_to_strip = '.,;:"\'!?)>]}'

    local_paths = set()
    online_paths = set()

    for match in token_pattern.finditer(user_input):
        # A match object will have either group 2 (quoted content) or group 3 (unquoted word).
        candidate = match.group(2) or match.group(3)

        if not candidate:
            continue

        # Clean the candidate path by stripping trailing punctuation.
        cleaned_candidate = candidate.rstrip(trailing_chars_to_strip)
        
        # Now, validate if the cleaned token is a URL or a local path.
        # Check for URL first.
        if cleaned_candidate.startswith(('http://', 'https://')):
            online_paths.add(cleaned_candidate)
        # Check for various forms of local paths (POSIX and Windows).
        # Added regex for Windows-style drive letters (e.g., C:\path or C:/path)
        elif cleaned_candidate.startswith(('/', './', '~/', 'file://')) or re.match(r'^[a-zA-Z]:[\\/]', cleaned_candidate):
            local_paths.add(cleaned_candidate)

    # Return the unique paths found, converted to lists.
    return list(local_paths), list(online_paths)

async def main() -> None:
    try:
        # Display cool ASCII header
        header = figlet_format("CLI-Agent", font="slant")
        print(colored(header, "cyan", attrs=["bold"]))
        print_with_timer("Starting CLI-Agent", "cyan")
        load_dotenv(g.CLIAGENT_ENV_FILE_PATH)
        
        # Load or initialize the system instruction from external Markdown file
        instruction_path = g.CLIAGENT_INSTRUCTION_FILE
        if not os.path.exists(instruction_path):
            default_inst = f'''# SYSTEM INSTRUCTION

You are a capable CLI agent with access to a computational notebook environment on the user's system. 
You can execute Python and shell code to solve tasks effectively. 
You have a tendency to reason about a task before replying to the user, in your reasoning you may use the computational notebook to gather, compute and reason about information for solving complex tasks.

## Core Capabilities
- Execute Python code with persistent state across interactions
- Run shell commands (bash) for system operations
- Access and manipulate files within the system
- Utilize custom utility tools when available
- Process and analyze data programmatically

## Operating Principles

### 1. UNDERSTAND & ASSESS
Listen carefully to understand both the request and its purpose. Determine whether Python code, shell commands, or both are needed. Break complex tasks into logical steps.

### 2. VERIFY & PREPARE
Before implementing, ensure required resources exist. Use the computational environment to validate assumptions. Check data reliability and system state. Only ask for clarification when genuinely necessary.

### 3. EXECUTE & CREATE
Plan before coding. The environment maintains state - variables, imports, and functions persist.
- Use ```python for Python code, ```bash for shell commands
- Include meaningful comments and clear output
- Handle edge cases gracefully
- Default workspace: {g.AGENTS_SANDBOX_DIR}

### 4. EVALUATE & ITERATE
Check results meet the original need. Learn from errors and iterate purposefully. Document limitations honestly.

## Key Guidelines
- **Explore thoroughly** before making changes
- **Create new files** rather than overwriting existing ones unless explicitly requested
- **Exercise caution** with system-critical operations
- **Provide clear output** showing what's happening
- **Complete tasks fully** with proper summaries

## Safety
- Protect user data and system integrity
- Request confirmation for potentially destructive operations
- Work within designated directories when possible
- Handle sensitive information appropriately

You approach each task with genuine interest in helping effectively. Every interaction is an opportunity to solve problems together.'''
            with open(instruction_path, 'w') as f:
                f.write(default_inst)
        with open(instruction_path, 'r') as f:
            system_instruction = f.read()

        args = parse_cli_args()

        # Override logging level if debug mode is enabled
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Override local arg by .env
        if (os.getenv("DEFAULT_FORCE_LOCAL") == get_local_ip()):
            args.local = True
        
        # Automatically enable auto mode when voice mode is enabled
        if args.voice:
            args.auto = True
            print(colored("# cli-agent: Voice mode enabled, automatically enabling auto execution mode", "green"))
        
        # Use sandboxed python execution
        if args.sandbox:
            g.USE_SANDBOX = True
        
            
            try:
                import paramiko
            except ImportError:
                exit(1)
        
        # Define streaming callbacks to display output in real-time
        stdout_buffer = ""
        stderr_buffer = ""
        
        def filter_cmd_output(text: str) -> str:
            return text

        def stdout_callback(text: str) -> None:
            nonlocal stdout_buffer
            text = filter_cmd_output(text)
            print(text, end="") # Print directly to console
            stdout_buffer += text

        def stderr_callback(text: str) -> None:
            nonlocal stderr_buffer
            text = filter_cmd_output(text)
            print(colored(text, "red"), end="") # Print errors in red
            stderr_buffer += text

        def input_callback(previous_output: str) -> bool | str:
            """
            Callback function to handle interactive input during code execution.
            
            Args:
                prompt: The prompt text from the execution environment
                previous_output: The output that has been generated so far
                
            Returns:
                bool | str: 
                    - True to continue without providing input (e.g., effectively pressing Enter or waiting)
                    - False to interrupt execution
                    - A string to provide as input and continue execution
            """
            # For now, disable LLM interaction to avoid async/sync issues
            # Just return True to continue waiting
            print(colored("⏳ Process seems idle, continuing to wait...", "yellow"))
            return True
        # ComputationalNotebook will be initialized after help display
            
        # Initialize web server early if GUI mode is enabled
        web_server = None
        if args.gui:
            web_server = WebServer()
            g.web_server = web_server  # Store in globals for print redirection
            web_server.start()  # Start with empty chat, will be updated later
        
        
        if args.preload:
            print(colored("Preloading resources...", "green"))
            print(colored("Generating atuin-command-history embeddings...", "green"))
            get_update_cmd_collection()()
            print(colored("Generating pdf embeddings for cli-agent directory...", "green"))
            get_pdf_or_folder_to_database()(g.CLIAGENT_ROOT_PATH)
            print(colored("Preloading complete.", "green"))
            exit(0)
        
        context_chat: Optional[Chat] = None
        if args.c or args.regenerate:
            try:
                context_chat = Chat.load_from_json()
                if args.regenerate:
                    print(colored("Loading previous chat for regeneration.", "green"))
                    # Check if there's chat history to regenerate
                    if not context_chat or len(context_chat.messages) < 2:
                        print(colored("# cli-agent: No sufficient chat history found, cannot regenerate last response.", "red"))
                        exit(1)
                    # If the last message is from assistant, remove it to regenerate
                    if context_chat.messages[-1][0] == Role.ASSISTANT:
                        context_chat.messages.pop()
                        print(colored("# cli-agent: Removed last assistant response, will regenerate.", "green"))
                    # If last message is from user, we'll generate the missing assistant response
                    elif context_chat.messages[-1][0] == Role.USER:
                        print(colored("# cli-agent: Last message is from user, will generate assistant response.", "green"))
                else:
                    print(colored("Continuing previous chat.", "green"))
            except FileNotFoundError:
                if args.regenerate:
                    print(colored("No previous chat found to regenerate. Exiting.", "red"))
                    exit(1)
                else:
                    print(colored("No previous chat found to continue. Starting a new chat.", "yellow"))
                    context_chat = None # Ensure it's None if load fails

        
        if args.mct and context_chat:
            context_chat.debug_title = "MCTs Branching - Main Context Chat"
            
        # Handle screenshot capture immediately if --img flag was provided
        if args.image:
            print(colored("# cli-agent: Taking screenshot with Spectacle due to --img flag...", "green"))
            base64_images = await handle_screenshot_capture(context_chat)
            args.image = False  # Reset the flag after handling
        
        # Handle LLM selection at startup if --llm was passed without value
        if args.llm == "__select__":
            print(colored("# cli-agent: --llm flag detected without value. Opening LLM selection...", "green"))
            selected_llms = await llm_selection(args, preselected_llms=None)  # Load from persistent storage
            g.SELECTED_LLMS = selected_llms
            args.llm = None  # Reset to None after selection
        elif args.llm:
            g.SELECTED_LLMS = [args.llm]
        
        # Initialize ComputationalNotebook after help display to avoid bash prompt appearing before banner
        notebook = ComputationalNotebook(stdout_callback=stdout_callback, stderr_callback=stderr_callback, input_prompt_handler=input_callback)
        
        # Initialize tool manager
        utils_manager = UtilsManager()
        
        # Print loaded utilities in a compact format
        util_names = utils_manager.get_util_names()
        if util_names:
            utils_str = ", ".join(util_names)
            print_with_timer(f"Loaded {len(util_names)} utilities: {utils_str}", "green")
        else:
            print_with_timer("No utilities loaded", "yellow")

        # Print help page by default at startup
        await get_user_input_with_bindings(args, context_chat, input_override="--h")

        if context_chat is None:
            context_chat = Chat(debug_title="Main Context Chat")
            inst = system_instruction
            
            # Add selected utils to instruction if any are selected
            if g.SELECTED_UTILS:
                inst += f"""

IMPORTANT: You MUST use the following utility tools that have been specifically requested:
{', '.join(g.SELECTED_UTILS)}

For any new code you write, be sure to make appropriate use of these selected utilities."""

            if (args.sandbox):
                inst += f"""
Please try to stay within your sandbox directory at {g.AGENTS_SANDBOX_DIR}
You may read from the whole system but if you need to save or modify any files, copy the file to your sandbox directory and do it there.
"""

            context_chat.set_instruction_message(inst)
            
            # Determine active utilities based on voice/speak flags
            all_util_names = utils_manager.get_util_names()
            active_util_names = all_util_names.copy()

            # Conditionally remove 'tts' if voice/speak modes are off
            if not (args.voice or args.speak):
                if 'tts' in active_util_names:
                    active_util_names.remove('tts')
                    print(colored("TTS utility disabled. Use -v or -s to enable.", "yellow"))
            
            # Generate the utils info string using only the active utilities
            active_utils_info = utils_manager.get_available_utils_info(util_names=active_util_names)
            
            kickstart_preprompt = f"""Hi, before starting off, let me show you some additional python utilities I coded for you to use if needed,
{active_utils_info}


Now, I am going to run somethings to show you how your computational notebook works.
Let's see the window manager and OS version of your users system::
```bash
echo "OS: $(lsb_release -ds)" && echo "Desktop: $XDG_CURRENT_DESKTOP" && echo "Window Manager: $(ps -eo comm | grep -E '^kwin|^mutter|^openbox|^i3|^dwm' | head -1)"
```
<execution_output>
{subprocess.check_output(['uname', '-a']).decode('utf-8')}
{subprocess.check_output(['lsb_release', '-a']).decode('utf-8')}
{subprocess.check_output(['echo', '$XDG_CURRENT_DESKTOP']).decode('utf-8')}
{subprocess.check_output('ps aux | grep -i kwin', shell=True).decode('utf-8')}
</execution_output>

That succeeded, now let's see the current working directory and the first 5 files in it:
```python
import os
print(f"Current working directory: {{os.getcwd()}}")
print(f"Total files in current directory: {{len(os.listdir())}}")
print(f"First 5 files in current directory: {{os.listdir()[:5]}}")
```
<execution_output>
Current working directory: {os.getcwd()}
Total files in current directory: {len(os.listdir())}
First 5 files in current directory: {os.listdir()[:5]}
</execution_output>

That succeeded, now let's check the current time:
```bash
import datetime
print(f"Current year and month: {{datetime.datetime.now().strftime('%Y-%m')}}")
```
<execution_output>
Current year and month: {datetime.datetime.now().strftime('%Y-%m')}
</execution_output>

"""
            context_chat.add_message(Role.USER, kickstart_preprompt)

        # Check if -m flag was provided without messages and prompt for multiline input
        if '-m' in sys.argv or '--message' in sys.argv:
            if not args.message:  # Empty list means -m was provided without arguments
                print(colored("# cli-agent: -m flag detected without messages. Entering multiline input mode.", "green"))
                multiline_input = handle_multiline_input()
                if multiline_input.strip():  # Only add if not empty
                    args.message.append(multiline_input)

        print_with_timer("Ready.", "magenta")

        user_interrupt: bool = False
        collected_local_paths: List[str] = []
        collected_online_paths: List[str] = []
        # Main loop
        while True:
            # Reset main loop variables
            user_input: Optional[str] = None
            async_task_to_await: Optional[asyncio.Task] = None
            g.LLM_STRENGTHS = [AIStrengths.STRONG] if args.strong else []
            g.FORCE_LOCAL = args.local
            g.DEBUG_CHATS = args.debug_chats
            g.FORCE_FAST = args.fast
            g.LLM = args.llm
            g.FORCE_ONLINE = args.online
            base64_images: List[str] = [] # Initialize here unconditionally
            thinking_budget = 2048 # in tokens
            if ((g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 1)):
                temperature = 0.85
            else:
                temperature = 0
            will_code = False # Initialize with default
            
            # IMPORTANT: stdout_buffer and stderr_buffer are reset after each execution
            # to prevent accumulation across agentic loop iterations. This fixes the race
            # condition where the agent would repeatedly attempt the same action because
            # it saw stale output from previous executions.

            # autosaving
            if context_chat:
                context_chat.save_to_json()

            if args.regenerate:
                # For regenerate mode, we don't need new user input
                user_input = ""
                print(colored("# cli-agent: Proceeding with regeneration...", "green"))
                # Reset the regenerate flag so it doesn't loop indefinitely
                args.regenerate = False
            elif args.voice:
                # Default voice handling
                user_input, _, wake_word_used = get_listen_microphone()(private_remote_wake_detection=args.private_remote_wake_detection)
            else:
                # Get input via the new wrapper function
                user_input = await get_user_input_with_bindings(args, context_chat, force_input=user_interrupt)
                local_paths, online_paths = extract_paths(user_input)

                # read local paths
                for path in local_paths:
                    if os.path.exists(path):
                        # is file -> read file in
                        if os.path.isfile(path):
                            with open(path, 'r') as file:
                                user_input += f"\n\n# {path}\n```\n{os.path.basename(path)}\n{file.read()}\n```"
                                print(colored(f"# cli-agent: Read local path: {path}", "green"))
                        # is folder -> run tree -L 3
                        elif os.path.isdir(path):
                            user_input += f"\n\n# {path}\n```\nbash\n{subprocess.check_output(['tree', '-L', '3', path]).decode('utf-8')}\n```"
                            print(colored(f"# cli-agent: Looked into local folder: {path}", "green"))
                user_interrupt = False


            if LlmRouter.has_unconfirmed_data():
                LlmRouter.confirm_finetuning_data()

            # AGENTIC IN-TURN LOOP - BEGIN

            action_counter = 0  # Initialize counter for consecutive actions
            response_buffer=""
            assistant_response = ""
            text_stream_painter = TextStreamPainter()

            # Only add user input if it's not empty (e.g. after -r or -s binding resulted in "")
            if user_input and context_chat: # Check context_chat exists
                context_chat.add_message(Role.USER, user_input)

            # --- Start Agentic Inner Loop ---
            last_action_signature: Optional[str] = None
            stall_counter: int = 0
            MAX_STALLS: int = 2  # Allow the same action twice before intervening
            
            while True:
                try:

                    def update_python_environment(chunk: str, print_char: bool = True) -> str:
                        nonlocal response_buffer
                        for char in chunk:
                            response_buffer += char
                            if print_char:
                                print(text_stream_painter.apply_color(char), end="", flush=True)
                            if response_buffer.count("```") % 2 == 0 and (response_buffer.count("```python") > 0 or response_buffer.count("```bash") > 0):
                                final_response = response_buffer
                                response_buffer = ""
                                return final_response
                        return None


                    response_branches: List[str] = []
                    try:
                        # Ensure last message is assistant before generating new one if needed
                        if assistant_response:
                            print(colored(f"WARNING: Assistant response was not handled, defensively adding to context", "yellow"))
                            if context_chat.messages[-1][0] == Role.USER:
                                context_chat.add_message(Role.ASSISTANT, assistant_response)
                            else:
                                context_chat.messages[-1] = (Role.ASSISTANT, assistant_response)
                            assistant_response = "" # Clear buffer after adding

                        for i in range(len(g.SELECTED_LLMS) if len(g.SELECTED_LLMS) > 1 else args.mct if args.mct else 1):
                            response_buffer = "" # Reset buffer for each branch
                            
                            branch_start_time = time.time()
                            
                            try:
                                # Stream monitoring wrapper for LLM calls
                                stream_monitor = {
                                    'last_char_time': time.time(),
                                    'stream_started': False,
                                    'total_chars': 0
                                }
                                
                                def monitored_stream_callback(chunk: str, print_char: bool = True) -> str:
                                    """Enhanced callback that monitors stream health"""
                                    nonlocal stream_monitor
                                    
                                    # Check for keyboard interrupt
                                    try:
                                        # Update stream monitoring
                                        if chunk:
                                            stream_monitor['last_char_time'] = time.time()
                                            stream_monitor['stream_started'] = True
                                            stream_monitor['total_chars'] += len(chunk)
                                        
                                        # Call the original callback
                                        return update_python_environment(chunk, print_char)
                                    except KeyboardInterrupt:
                                        print(colored("\n-=- User interrupted streaming (Ctrl+C) -=-", "yellow"))
                                        raise  # Re-raise to propagate the interrupt
                                
                                # If multi-LLM mode is active and there are selected LLMs
                                if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 1:
                                    # If this is the first iteration, just use the selected LLMs instead of temperature variations
                                    if i < len(g.SELECTED_LLMS):
                                        current_model = g.SELECTED_LLMS[i]
                                        if args.mct:
                                            context_chat.debug_title = f"MCT Branching ({i+1}/{len(g.SELECTED_LLMS)})"
                                        current_branch_response = await LlmRouter.generate_completion(
                                            context_chat,
                                            [current_model],
                                            force_preferred_model=True,
                                            temperature=temperature,
                                            base64_images=base64_images,
                                            generation_stream_callback=monitored_stream_callback,
                                            strengths=g.LLM_STRENGTHS,
                                            thinking_budget=thinking_budget,
                                            exclude_reasoning_tokens=True
                                        )
                                    else:
                                        # If we've already used all selected LLMs, use the first one with different temperatures
                                        current_branch_response = await LlmRouter.generate_completion(
                                            context_chat,
                                            [g.SELECTED_LLMS[0]],
                                            temperature=temperature,
                                            base64_images=base64_images,
                                            generation_stream_callback=monitored_stream_callback,
                                            strengths=g.LLM_STRENGTHS,
                                            thinking_budget=thinking_budget,
                                            exclude_reasoning_tokens=True
                                        )
                                else:
                                    # Standard single-LLM mode (or no specific LLM selection)
                                    current_branch_response = await LlmRouter.generate_completion(
                                        context_chat,
                                        g.SELECTED_LLMS if g.SELECTED_LLMS else [],
                                        temperature=temperature,
                                        base64_images=base64_images,
                                        generation_stream_callback=monitored_stream_callback,
                                        strengths=g.LLM_STRENGTHS,
                                        thinking_budget=thinking_budget,
                                        exclude_reasoning_tokens=True
                                    )
                                
                                # Check for successful completion and stream health
                                branch_duration = time.time() - branch_start_time
                                model_name = g.SELECTED_LLMS[i] if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 1 and i < len(g.SELECTED_LLMS) else "Unknown"
                                
                                # Analyze stream health
                                if not stream_monitor['stream_started']:
                                    print(colored(f"❌ Branch {i+1} failed: stream never started after {branch_duration:.1f}s (model: {model_name})", "red"))
                                elif current_branch_response and current_branch_response.strip():
                                    response_branches.append(current_branch_response)
                                else:
                                    # Stream started but result is empty - possible connection drop or early termination
                                    time_since_last_char = time.time() - stream_monitor['last_char_time']
                                    if time_since_last_char > 10 and stream_monitor['total_chars'] > 0:
                                        print(colored(f"⚠️ Branch {i+1} stream stalled: no new chars for {time_since_last_char:.1f}s, got {stream_monitor['total_chars']} chars (model: {model_name})", "yellow"))
                                    else:
                                        print(colored(f"❌ Branch {i+1} failed: returned empty response after {branch_duration:.1f}s (model: {model_name})", "red"))
                                    
                            except Exception as branch_error:
                                branch_duration = time.time() - branch_start_time
                                error_msg = str(branch_error)
                                model_name = g.SELECTED_LLMS[i] if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 1 and i < len(g.SELECTED_LLMS) else "Unknown"
                                
                                # Categorize the error type for better diagnostics
                                if ("ctrl+c" in error_msg.lower()):
                                    print(colored("=== User interrupted model generation (Ctrl+C) ===", "yellow"))
                                    break # Break from inner agentic loop
                                elif "connection" in error_msg.lower() or "connectionerror" in error_msg.lower():
                                    print(colored(f"🌐 Branch {i+1} connection failed after {branch_duration:.1f}s (model: {model_name}): {error_msg}", "red"))
                                elif "timeout" in error_msg.lower() or "asyncio.TimeoutError" in error_msg:
                                    print(colored(f"⏱️ Branch {i+1} timed out after {branch_duration:.1f}s (model: {model_name})", "red"))
                                elif "stream" in error_msg.lower():
                                    print(colored(f"📡 Branch {i+1} stream error after {branch_duration:.1f}s (model: {model_name}): {error_msg}", "red"))
                                else:
                                    print(colored(f"❌ Branch {i+1} failed after {branch_duration:.1f}s (model: {model_name}): {error_msg}", "red"))
                                
                                response_branches.append(f"Error: Branch {i+1} failed - {error_msg}")
                        
                        base64_images = [] # Clear images after use
                        context_chat.debug_title = "Main Context Chat"

                    except KeyboardInterrupt:
                        print(colored("\n-=- User interrupted model generation (Ctrl+C) -=-", "yellow"))
                        # Clear any queued messages and return control to the user
                        if args.message:
                            print(colored(f"Clearing {len(args.message)} queued message(s).", "yellow"))
                            args.message = []
                        break # Break from inner agentic loop

                    except Exception as e:
                        LlmRouter.clear_unconfirmed_finetuning_data()
                        print(colored(f"Error generating response: {str(e)}", "red"))
                        if args.debug:
                            traceback.print_exc()
                        break # Break inner loop

                    # --- MCT Branch Selection ---
                    if args.mct:
                        try:
                            selected_branch_index = await select_best_branch(context_chat, response_branches) # Use last user input
                            assistant_response = response_branches[selected_branch_index]
                            
                            # Show which model generated the selected response if in multi-LLM mode
                            if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 1 and selected_branch_index < len(g.SELECTED_LLMS):
                                model_name = g.SELECTED_LLMS[selected_branch_index]
                                print(colored(f"Selected branch {selected_branch_index} from model: {model_name}", "green"))
                            else:
                                print(colored(f"Selected branch: {selected_branch_index}", "green"))
                            print(colored("Adding to context:", "green"))                            
                            # Print the selected branch if we're in MCT mode (since we don't stream MCT branches)
                            text_stream_painter = TextStreamPainter()
                            for char in assistant_response:
                                print(text_stream_painter.apply_color(char), end="", flush=True)
                            print() # Add newline
                            
                            # Show which model generated the selected response if in multi-LLM mode
                            if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 1 and selected_branch_index < len(g.SELECTED_LLMS):
                                model_name = g.SELECTED_LLMS[selected_branch_index]
                                print(colored(f"Selected branch {selected_branch_index} from model: {model_name}", "green"))
                            else:
                                print(colored(f"Selected branch: {selected_branch_index}", "green"))
                                
                        except Exception as e:
                            print(colored(f"Error during MCT branch selection: {str(e)}", "red"))
                            if args.debug: traceback.print_exc()
                            print(colored("\n⚠️ Defaulting to first branch.", "yellow"))
                            assistant_response = response_branches[0]
                            
                            # Print the default branch
                            text_stream_painter = TextStreamPainter()
                            for char in assistant_response:
                                print(text_stream_painter.apply_color(char), end="", flush=True)
                            print() # Add newline
                    else:
                        # non mct case
                        assistant_response = response_branches[0]
                        
                    # --- STALL DETECTION LOGIC ---
                    current_action_signature = assistant_response
                    if last_action_signature and current_action_signature == last_action_signature:
                        stall_counter += 1
                        print(colored(f"Stall counter: {stall_counter}/{MAX_STALLS}", "yellow"))
                    else:
                        stall_counter = 0  # Reset counter if action is different

                    last_action_signature = current_action_signature

                    if stall_counter >= MAX_STALLS:
                        print(colored("! Agent appears to be stalled. Intervening.", "red"))
                        # Inject a user message to force re-evaluation
                        intervention_message = "My last two attempts have failed or made no progress. I need to stop and re-evaluate my entire strategy. I will analyze the situation from the beginning and devise a completely new plan of action. I will not repeat my previous failed attempts."
                        context_chat.add_message(Role.USER, intervention_message)
                        print(colored(f"  └─ Injected user message to force new strategy.", "red"))
                        stall_counter = 0  # Reset after intervention
                        last_action_signature = None
                        break  # Break inner loop to force LLM to process the new user message

                    # --- END STALL DETECTION ---
                        
                    # --- Code Extraction and Execution ---
                    shell_blocks = get_extract_blocks()(assistant_response, ["shell", "bash"])
                    python_blocks = get_extract_blocks()(assistant_response, ["python", "tool_code"])

                    # Handover to user if no python blocks are found
                    if len(python_blocks) == 0 and len(shell_blocks) == 0:
                        # No code found, assistant response is final for this turn.
                        # ADDED LOGIC: Check for pending todos before handing over to the user.
                        try:
                            all_todos = TodosUtil._load_todos()
                            incomplete_todos = [todo for todo in all_todos if not todo.get('completed', False)]

                            if incomplete_todos:
                                # There are pending todos. Create an automatic prompt.
                                next_todo = incomplete_todos[0]
                                current_list_formatted = TodosUtil._format_todos(all_todos)
                                auto_prompt = f"You have some remaining todos, please ensure they are taken care of. Here's your ordered to-do list: {current_list_formatted}"

                                print(colored(f"\n📝 Pending to-dos found. Auto-prompt: {auto_prompt}", "magenta"))
                                context_chat.add_message(Role.USER, auto_prompt)
                        except Exception as e:
                            # Don't crash the agent if the todo logic fails.
                            print(colored(f"Warning: Could not check for pending to-dos. Error: {e}", "red"))

                        # Original logic continues here
                        if context_chat.messages[-1][0] == Role.USER:
                            context_chat.add_message(Role.ASSISTANT, assistant_response)
                        else: # Update last assistant message (e.g. after regen)
                            context_chat.messages[-1] = (Role.ASSISTANT, assistant_response)
                        assistant_response = "" # Clear buffer as it's been added/printed
                        break # Break inner loop to get next user input
                    
                    # Check if any selected tools were used in the code response
                    if g.SELECTED_UTILS:
                        used_tools = []
                        for tool in g.SELECTED_UTILS:
                            # Check if the tool name appears in any of the code blocks
                            for code_block in python_blocks:
                                if tool in code_block:
                                    used_tools.append(tool)
                                    print(colored(f"# cli-agent: Tool '{tool}' was used and will be removed from required tools list.", "green"))
                                    break
                        
                        # Remove used tools from globals
                        if used_tools:
                            for tool in used_tools:
                                if tool in g.SELECTED_UTILS:
                                    g.SELECTED_UTILS.remove(tool)

                    if (args.voice or args.speak):
                        # remove all code blocks from the assistant response
                        verbal_text = re.sub(r'```[^`]*```', '', assistant_response)
                        if (len(python_blocks) > 0 and len(shell_blocks) > 0):
                            verbal_text += f"I've implemented some shell and python code, let's execute it."
                        elif (len(python_blocks) > 0):
                            verbal_text += f"I've implemented some python code, let's execute it."
                        elif (len(shell_blocks) > 0):
                            verbal_text += f"I've implemented some shell code, let's execute it."
                        
                        # Use the new TTS utility instead of the old utils_audio
                        TtsUtil.run(text=verbal_text)

                    # Join shell blocks into a single string
                    formatted_code = ""
                    if shell_blocks:
                        formatted_code += "```bash\n" + "\n".join(shell_blocks) + "\n```\n"
                    if python_blocks:
                        formatted_code += "```python\n" + python_blocks[0] + "\n```"


                    context_chat.save_to_json()
                    # Confirm code execution
                    if await confirm_code_execution(args, formatted_code):
                        print(colored("🔄 Executing code...", "cyan"))

                        try:
                            if (shell_blocks):
                                for shell_line in shell_blocks:
                                    l_shell_line = shell_line.strip()
                                    # Then in your main execution logic:
                                    if 'sudo ' in l_shell_line:
                                        # First, try to combine consecutive sudo commands
                                        l_shell_line = preprocess_consecutive_sudo_commands(l_shell_line)
                                        
                                        # Then apply sudo -A replacement for remaining sudo commands
                                        if ("sudo " in l_shell_line and not "sudo -A " in l_shell_line):
                                            l_shell_line = l_shell_line.replace("sudo ", "sudo -A ")
                                    notebook.execute(l_shell_line)
                            if (python_blocks):
                                notebook.execute(python_blocks[0], is_python_code=True)

                            print(colored("\n✅ Code execution completed.", "cyan"))

                            # Create a formatted output to add to the chat context
                            tool_output = ""
                            # Use the final captured stdout/stderr which might differ slightly if callbacks missed something
                            if stdout_buffer and stdout_buffer.strip():
                                tool_output += f"```stdout\n{stdout_buffer.strip()}\n```\n"
                            if stderr_buffer and stderr_buffer.strip():
                                tool_output += f"```stderr\n{stderr_buffer.strip()}\n```\n"
                            
                            tool_output = filter_cmd_output(tool_output)

                            # remove color codes
                            tool_output = re.sub(r'\x1b\[[0-9;]*m', '', tool_output)

                            # shorten to max 4000 characters, include the first 3000 and last 1000, splitting at the last newline
                            if len(tool_output) > 4000:
                                try:
                                    # Find the last newline in the first 3000 chars
                                    first_part = tool_output[:3000]
                                    first_index = first_part.rindex('\n') if '\n' in first_part else 3000
                                    # Find the first newline in the last 1000 chars (relative to the start of last_part)
                                    last_part = tool_output[-1000:]
                                    # Need to find newline relative to start of last_part, then adjust index relative to tool_output
                                    newline_in_last_part = last_part.find('\n') # Find first newline
                                    # Calculate the index relative to the full string where the truncated part starts
                                    split_index_in_full = len(tool_output) - 1000 + newline_in_last_part if newline_in_last_part != -1 else len(tool_output) - 1000

                                    tool_output = tool_output[:first_index] + "\n[...output truncated...]\n" + tool_output[split_index_in_full+1:] # +1 to exclude the newline itself

                                except ValueError: # Handle cases where newline isn't found
                                    tool_output = tool_output[:3000] + "\n[...output truncated...]" + tool_output[-1000:]


                            if not tool_output.strip():
                                tool_output = "<execution_output>\nThe execution completed without returning any output.\n</execution_output>"
                            else:
                                tool_output = f"<execution_output>\n{tool_output.strip()}\n</execution_output>"


                            # Append execution output to the assistant's response FOR THE CONTEXT
                            assistant_response_with_output = f"{assistant_response}\n{tool_output}\n<think>\n"

                            # Add the complete turn (Assistant response + execution) to context
                            context_chat.add_message(Role.ASSISTANT, assistant_response_with_output)

                            assistant_response = "" # Clear buffer as it's been processed

                            # Reset output buffers for next execution to prevent accumulation
                            stdout_buffer = ""
                            stderr_buffer = ""

                            action_counter += 1  # Increment action counter
                            
                            continue # Break inner loop after successful execution, continue to next agent turn

                        except Exception as e:
                            print(colored(f"\n❌ Error executing code: {str(e)}", "red"))
                            if args.debug:
                                traceback.print_exc()
                            # Add error to context before breaking
                            error_output = f"<execution_output>\n```error\n{traceback.format_exc()}\n```\n</execution_output>"
                            assistant_response_with_error = f"{assistant_response}\n{error_output}"
                            context_chat.add_message(Role.ASSISTANT, assistant_response_with_error)
                            assistant_response = "" # Clear buffer
                            
                            # Reset output buffers for next execution to prevent accumulation
                            stdout_buffer = ""
                            stderr_buffer = ""
                            
                            break # Break inner loop on execution error
                    else:
                        # Code execution denied by user
                        print(colored(" Execution cancelled by user.", "yellow"))
                        # Add assistant's plan and cancellation notice to chat history
                        cancellation_notice = "<execution_output>\nCode execution cancelled by user.\n</execution_output>"
                        assistant_response_with_cancellation = assistant_response + f"\n{cancellation_notice}"
                        context_chat.add_message(Role.ASSISTANT, assistant_response_with_cancellation)
                        assistant_response = "" # Clear buffer
                        
                        # Reset output buffers for next execution to prevent accumulation
                        stdout_buffer = ""
                        stderr_buffer = ""
                        
                        break # Break inner loop

                except KeyboardInterrupt:
                    print(colored("\n=== User interrupted execution (Ctrl+C) ===", "yellow"))
                    user_interrupt = True
                    break # Break from inner agentic loop

                except Exception as e:
                    LlmRouter.clear_unconfirmed_finetuning_data()
                    if ("ctrl+c" in str(e).lower()):
                        print(colored("=== User interrupted execution (Ctrl+C) ===", "yellow"))
                        user_interrupt = True
                        break # Break from inner agentic loop
                    
                    print(colored(f"An unexpected error occurred in the agent loop: {str(e)}", "red"))
                    if args.debug:
                        traceback.print_exc()
                    # Attempt to add error context before breaking
                    try:
                        error_output = f"<execution_output>\n```error\n{traceback.format_exc()}\n```\n</execution_output>"
                        assistant_response_with_error = assistant_response + f"\n{error_output}" # Append to potentially partial response
                        context_chat.add_message(Role.ASSISTANT, assistant_response_with_error)
                    except Exception as context_e:
                        print(colored(f"Failed to add error to context: {context_e}", "red"))
                    assistant_response = "" # Clear buffer
                    
                    # Reset output buffers for next execution to prevent accumulation
                    stdout_buffer = ""
                    stderr_buffer = ""
                    
                    break # Break inner loop
                
            # --- End of Agentic Inner Loop ---

            # save context once per turn (moved outside inner loop)
            if context_chat:
                context_chat.save_to_json()
                
            # Check if we should exit after all messages have been processed
            if args.exit and not args.message:
                print(colored("All automatic messages processed successfully. Exiting...", "green"))
                exit(0)

            print_with_timer("Turn completed.", "magenta")

        # End of outer while loop (Main loop)
        print(colored("\nCLI-Agent is shutting down.", "cyan"))


    except asyncio.CancelledError:
        print(colored("\nCLI-Agent was interrupted. Shutting down gracefully...", "yellow"))
    except KeyboardInterrupt:
        print(colored("\nCLI-Agent was interrupted by user. Shutting down...", "yellow"))
    except Exception as e:
        print(colored(f"\nCLI-Agent encountered an error: {str(e)}", "red"))
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(colored("\nCLI-Agent was interrupted by user. Shutting down...", "yellow"))
    except Exception as e:
        print(colored(f"\nCLI-Agent encountered an error: {str(e)}", "red"))
        traceback.print_exc()
                local_paths, online_paths = extract_paths(user_input)

                # read local paths
                for path in local_paths:
                    if os.path.exists(path):
                        # is file -> read file in
                        if os.path.isfile(path):
                            with open(path, 'r') as file:
                                user_input += f"\n\n# {path}\n```\n{os.path.basename(path)}\n{file.read()}\n```"
                                print(colored(f"# cli-agent: Read local path: {path}", "green"))
                        # is folder -> run tree -L 3
                        elif os.path.isdir(path):
                            user_input += f"\n\n# {path}\n```\nbash\n{subprocess.check_output(['tree', '-L', '3', path]).decode('utf-8')}\n```"
                            print(colored(f"# cli-agent: Looked into local folder: {path}", "green"))
                user_interrupt = False


            if LlmRouter.has_unconfirmed_data():
                LlmRouter.confirm_finetuning_data()

            # AGENTIC IN-TURN LOOP - BEGIN

            action_counter = 0  # Initialize counter for consecutive actions
            response_buffer=""
            assistant_response = ""
            text_stream_painter = TextStreamPainter()

            # Only add user input if it's not empty (e.g. after -r or -s binding resulted in "")
            if user_input and context_chat: # Check context_chat exists
                context_chat.add_message(Role.USER, user_input)

            # --- Start Agentic Inner Loop ---
            last_action_signature: Optional[str] = None
            stall_counter: int = 0
            MAX_STALLS: int = 2  # Allow the same action twice before intervening
            
            while True:
                try:

                    def update_python_environment(chunk: str, print_char: bool = True) -> str:
                        nonlocal response_buffer
                        for char in chunk:
                            response_buffer += char
                            if print_char:
                                print(text_stream_painter.apply_color(char), end="", flush=True)
                            if response_buffer.count("```") % 2 == 0 and (response_buffer.count("```python") > 0 or response_buffer.count("```bash") > 0):
                                final_response = response_buffer
                                response_buffer = ""
                                return final_response
                        return None


                    response_branches: List[str] = []
                    try:
                        # Ensure last message is assistant before generating new one if needed
                        if assistant_response:
                            print(colored(f"WARNING: Assistant response was not handled, defensively adding to context", "yellow"))
                            if context_chat.messages[-1][0] == Role.USER:
                                context_chat.add_message(Role.ASSISTANT, assistant_response)
                            else:
                                context_chat.messages[-1] = (Role.ASSISTANT, assistant_response)
                            assistant_response = "" # Clear buffer after adding

                        for i in range(len(g.SELECTED_LLMS) if len(g.SELECTED_LLMS) > 1 else args.mct if args.mct else 1):
                            response_buffer = "" # Reset buffer for each branch
                            
                            branch_start_time = time.time()
                            
                            try:
                                # Stream monitoring wrapper for LLM calls
                                stream_monitor = {
                                    'last_char_time': time.time(),
                                    'stream_started': False,
                                    'total_chars': 0
                                }
                                
                                def monitored_stream_callback(chunk: str, print_char: bool = True) -> str:
                                    """Enhanced callback that monitors stream health"""
                                    nonlocal stream_monitor
                                    
                                    # Check for keyboard interrupt
                                    try:
                                        # Update stream monitoring
                                        if chunk:
                                            stream_monitor['last_char_time'] = time.time()
                                            stream_monitor['stream_started'] = True
                                            stream_monitor['total_chars'] += len(chunk)
                                        
                                        # Call the original callback
                                        return update_python_environment(chunk, print_char)
                                    except KeyboardInterrupt:
                                        print(colored("\n-=- User interrupted streaming (Ctrl+C) -=-", "yellow"))
                                        raise  # Re-raise to propagate the interrupt
                                
                                # If multi-LLM mode is active and there are selected LLMs
                                if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 1:
                                    # If this is the first iteration, just use the selected LLMs instead of temperature variations
                                    if i < len(g.SELECTED_LLMS):
                                        current_model = g.SELECTED_LLMS[i]
                                        if args.mct:
                                            context_chat.debug_title = f"MCT Branching ({i+1}/{len(g.SELECTED_LLMS)})"
                                        current_branch_response = await LlmRouter.generate_completion(
                                            context_chat,
                                            [current_model],
                                            force_preferred_model=True,
                                            temperature=temperature,
                                            base64_images=base64_images,
                                            generation_stream_callback=monitored_stream_callback,
                                            strengths=g.LLM_STRENGTHS,
                                            thinking_budget=thinking_budget,
                                            exclude_reasoning_tokens=True
                                        )
                                    else:
                                        # If we've already used all selected LLMs, use the first one with different temperatures
                                        current_branch_response = await LlmRouter.generate_completion(
                                            context_chat,
                                            [g.SELECTED_LLMS[0]],
                                            temperature=temperature,
                                            base64_images=base64_images,
                                            generation_stream_callback=monitored_stream_callback,
                                            strengths=g.LLM_STRENGTHS,
                                            thinking_budget=thinking_budget,
                                            exclude_reasoning_tokens=True
                                        )
                                else:
                                    # Standard single-LLM mode (or no specific LLM selection)
                                    current_branch_response = await LlmRouter.generate_completion(
                                        context_chat,
                                        g.SELECTED_LLMS if g.SELECTED_LLMS else [],
                                        temperature=temperature,
                                        base64_images=base64_images,
                                        generation_stream_callback=monitored_stream_callback,
                                        strengths=g.LLM_STRENGTHS,
                                        thinking_budget=thinking_budget,
                                        exclude_reasoning_tokens=True
                                    )
                                
                                # Check for successful completion and stream health
                                branch_duration = time.time() - branch_start_time
                                model_name = g.SELECTED_LLMS[i] if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 1 and i < len(g.SELECTED_LLMS) else "Unknown"
                                
                                # Analyze stream health
                                if not stream_monitor['stream_started']:
                                    print(colored(f"❌ Branch {i+1} failed: stream never started after {branch_duration:.1f}s (model: {model_name})", "red"))
                                elif current_branch_response and current_branch_response.strip():
                                    response_branches.append(current_branch_response)
                                else:
                                    # Stream started but result is empty - possible connection drop or early termination
                                    time_since_last_char = time.time() - stream_monitor['last_char_time']
                                    if time_since_last_char > 10 and stream_monitor['total_chars'] > 0:
                                        print(colored(f"⚠️ Branch {i+1} stream stalled: no new chars for {time_since_last_char:.1f}s, got {stream_monitor['total_chars']} chars (model: {model_name})", "yellow"))
                                    else:
                                        print(colored(f"❌ Branch {i+1} failed: returned empty response after {branch_duration:.1f}s (model: {model_name})", "red"))
                                    
                            except Exception as branch_error:
                                branch_duration = time.time() - branch_start_time
                                error_msg = str(branch_error)
                                model_name = g.SELECTED_LLMS[i] if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 1 and i < len(g.SELECTED_LLMS) else "Unknown"
                                
                                # Categorize the error type for better diagnostics
                                if ("ctrl+c" in error_msg.lower()):
                                    print(colored("=== User interrupted model generation (Ctrl+C) ===", "yellow"))
                                    break # Break from inner agentic loop
                                elif "connection" in error_msg.lower() or "connectionerror" in error_msg.lower():
                                    print(colored(f"🌐 Branch {i+1} connection failed after {branch_duration:.1f}s (model: {model_name}): {error_msg}", "red"))
                                elif "timeout" in error_msg.lower() or "asyncio.TimeoutError" in error_msg:
                                    print(colored(f"⏱️ Branch {i+1} timed out after {branch_duration:.1f}s (model: {model_name})", "red"))
                                elif "stream" in error_msg.lower():
                                    print(colored(f"📡 Branch {i+1} stream error after {branch_duration:.1f}s (model: {model_name}): {error_msg}", "red"))
                                else:
                                    print(colored(f"❌ Branch {i+1} failed after {branch_duration:.1f}s (model: {model_name}): {error_msg}", "red"))
                                
                                response_branches.append(f"Error: Branch {i+1} failed - {error_msg}")
                        
                        base64_images = [] # Clear images after use
                        context_chat.debug_title = "Main Context Chat"

                    except KeyboardInterrupt:
                        print(colored("\n-=- User interrupted model generation (Ctrl+C) -=-", "yellow"))
                        # Clear any queued messages and return control to the user
                        if args.message:
                            print(colored(f"Clearing {len(args.message)} queued message(s).", "yellow"))
                            args.message = []
                        break # Break from inner agentic loop

                    except Exception as e:
                        LlmRouter.clear_unconfirmed_finetuning_data()
                        print(colored(f"Error generating response: {str(e)}", "red"))
                        if args.debug:
                            traceback.print_exc()
                        break # Break inner loop

                    # --- MCT Branch Selection ---
                    if args.mct:
                        try:
                            selected_branch_index = await select_best_branch(context_chat, response_branches) # Use last user input
                            assistant_response = response_branches[selected_branch_index]
                            
                            # Show which model generated the selected response if in multi-LLM mode
                            if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 1 and selected_branch_index < len(g.SELECTED_LLMS):
                                model_name = g.SELECTED_LLMS[selected_branch_index]
                                print(colored(f"Selected branch {selected_branch_index} from model: {model_name}", "green"))
                            else:
                                print(colored(f"Selected branch: {selected_branch_index}", "green"))
                            print(colored("Adding to context:", "green"))                            
                            # Print the selected branch if we're in MCT mode (since we don't stream MCT branches)
                            text_stream_painter = TextStreamPainter()
                            for char in assistant_response:
                                print(text_stream_painter.apply_color(char), end="", flush=True)
                            print() # Add newline
                            
                            # Show which model generated the selected response if in multi-LLM mode
                            if g.SELECTED_LLMS and len(g.SELECTED_LLMS) > 1 and selected_branch_index < len(g.SELECTED_LLMS):
                                model_name = g.SELECTED_LLMS[selected_branch_index]
                                print(colored(f"Selected branch {selected_branch_index} from model: {model_name}", "green"))
                            else:
                                print(colored(f"Selected branch: {selected_branch_index}", "green"))
                                
                        except Exception as e:
                            print(colored(f"Error during MCT branch selection: {str(e)}", "red"))
                            if args.debug: traceback.print_exc()
                            print(colored("\n⚠️ Defaulting to first branch.", "yellow"))
                            assistant_response = response_branches[0]
                            
                            # Print the default branch
                            text_stream_painter = TextStreamPainter()
                            for char in assistant_response:
                                print(text_stream_painter.apply_color(char), end="", flush=True)
                            print() # Add newline
                    else:
                        # non mct case
                        assistant_response = response_branches[0]
                        
                    # --- STALL DETECTION LOGIC ---
                    current_action_signature = assistant_response
                    if last_action_signature and current_action_signature == last_action_signature:
                        stall_counter += 1
                        print(colored(f"Stall counter: {stall_counter}/{MAX_STALLS}", "yellow"))
                    else:
                        stall_counter = 0  # Reset counter if action is different

                    last_action_signature = current_action_signature

                    if stall_counter >= MAX_STALLS:
                        print(colored("! Agent appears to be stalled. Intervening.", "red"))
                        # Inject a user message to force re-evaluation
                        intervention_message = "My last two attempts have failed or made no progress. I need to stop and re-evaluate my entire strategy. I will analyze the situation from the beginning and devise a completely new plan of action. I will not repeat my previous failed attempts."
                        context_chat.add_message(Role.USER, intervention_message)
                        print(colored(f"  └─ Injected user message to force new strategy.", "red"))
                        stall_counter = 0  # Reset after intervention
                        last_action_signature = None
                        break  # Break inner loop to force LLM to process the new user message

                    # --- END STALL DETECTION ---
                        
                    # --- Code Extraction and Execution ---
                    shell_blocks = get_extract_blocks()(assistant_response, ["shell", "bash"])
                    python_blocks = get_extract_blocks()(assistant_response, ["python", "tool_code"])

                    # Handover to user if no python blocks are found
                    if len(python_blocks) == 0 and len(shell_blocks) == 0:
                        # No code found, assistant response is final for this turn.
                        # ADDED LOGIC: Check for pending todos before handing over to the user.
                        try:
                            all_todos = TodosUtil._load_todos()
                            incomplete_todos = [todo for todo in all_todos if not todo.get('completed', False)]

                            if incomplete_todos:
                                # There are pending todos. Create an automatic prompt.
                                next_todo = incomplete_todos[0]
                                current_list_formatted = TodosUtil._format_todos(all_todos)
                                auto_prompt = f"You have some remaining todos, please ensure they are taken care of. Here's your ordered to-do list: {current_list_formatted}"

                                print(colored(f"\n📝 Pending to-dos found. Auto-prompt: {auto_prompt}", "magenta"))
                                context_chat.add_message(Role.USER, auto_prompt)
                        except Exception as e:
                            # Don't crash the agent if the todo logic fails.
                            print(colored(f"Warning: Could not check for pending to-dos. Error: {e}", "red"))

                        # Original logic continues here
                        if context_chat.messages[-1][0] == Role.USER:
                            context_chat.add_message(Role.ASSISTANT, assistant_response)
                        else: # Update last assistant message (e.g. after regen)
                            context_chat.messages[-1] = (Role.ASSISTANT, assistant_response)
                        assistant_response = "" # Clear buffer as it's been added/printed
                        break # Break inner loop to get next user input
                    
                    # Check if any selected tools were used in the code response
                    if g.SELECTED_UTILS:
                        used_tools = []
                        for tool in g.SELECTED_UTILS:
                            # Check if the tool name appears in any of the code blocks
                            for code_block in python_blocks:
                                if tool in code_block:
                                    used_tools.append(tool)
                                    print(colored(f"# cli-agent: Tool '{tool}' was used and will be removed from required tools list.", "green"))
                                    break
                        
                        # Remove used tools from globals
                        if used_tools:
                            for tool in used_tools:
                                if tool in g.SELECTED_UTILS:
                                    g.SELECTED_UTILS.remove(tool)

                    if (args.voice or args.speak):
                        # remove all code blocks from the assistant response
                        verbal_text = re.sub(r'```[^`]*```', '', assistant_response)
                        if (len(python_blocks) > 0 and len(shell_blocks) > 0):
                            verbal_text += f"I've implemented some shell and python code, let's execute it."
                        elif (len(python_blocks) > 0):
                            verbal_text += f"I've implemented some python code, let's execute it."
                        elif (len(shell_blocks) > 0):
                            verbal_text += f"I've implemented some shell code, let's execute it."
                        
                        # Use the new TTS utility instead of the old utils_audio
                        TtsUtil.run(text=verbal_text)

                    # Join shell blocks into a single string
                    formatted_code = ""
                    if shell_blocks:
                        formatted_code += "```bash\n" + "\n".join(shell_blocks) + "\n```\n"
                    if python_blocks:
                        formatted_code += "```python\n" + python_blocks[0] + "\n```"


                    context_chat.save_to_json()
                    # Confirm code execution
                    if await confirm_code_execution(args, formatted_code):
                        print(colored("🔄 Executing code...", "cyan"))

                        try:
                            if (shell_blocks):
                                for shell_line in shell_blocks:
                                    l_shell_line = shell_line.strip()
                                    # Then in your main execution logic:
                                    if 'sudo ' in l_shell_line:
                                        # First, try to combine consecutive shell commands
                                        l_shell_line = preprocess_consecutive_sudo_commands(l_shell_line)
                                        
                                        # Then apply sudo -A replacement for remaining sudo commands
                                        if ("sudo " in l_shell_line and not "sudo -A " in l_shell_line):
                                            l_shell_line = l_shell_line.replace("sudo ", "sudo -A ")
                                    notebook.execute(l_shell_line)
                            if (python_blocks):
                                notebook.execute(python_blocks[0], is_python_code=True)

                            print(colored("\n✅ Code execution completed.", "cyan"))

                            # Create a formatted output to add to the chat context
                            tool_output = ""
                            # Use the final captured stdout/stderr which might differ slightly if callbacks missed something
                            if stdout_buffer and stdout_buffer.strip():
                                tool_output += f"```stdout\n{stdout_buffer.strip()}\n```\n"
                            if stderr_buffer and stderr_buffer.strip():
                                tool_output += f"```stderr\n{stderr_buffer.strip()}\n```\n"
                            
                            tool_output = filter_cmd_output(tool_output)

                            # remove color codes
                            tool_output = re.sub(r'\x1b\[[0-9;]*m', '', tool_output)

                            # shorten to max 4000 characters, include the first 3000 and last 1000, splitting at the last newline
                            if len(tool_output) > 4000:
                                try:
                                    # Find the last newline in the first 3000 chars
                                    first_part = tool_output[:3000]
                                    first_index = first_part.rindex('\n') if '\n' in first_part else 3000
                                    # Find the first newline in the last 1000 chars (relative to the start of last_part)
                                    last_part = tool_output[-1000:]
                                    # Need to find newline relative to start of last_part, then adjust index relative to tool_output
                                    newline_in_last_part = last_part.find('\n') # Find first newline
                                    # Calculate the index relative to the full string where the truncated part starts
                                    split_index_in_full = len(tool_output) - 1000 + newline_in_last_part if newline_in_last_part != -1 else len(tool_output) - 1000

                                    tool_output = tool_output[:first_index] + "\n[...output truncated...]\n" + tool_output[split_index_in_full+1:] # +1 to exclude the newline itself

                                except ValueError: # Handle cases where newline isn't found
                                    tool_output = tool_output[:3000] + "\n[...output truncated...]" + tool_output[-1000:]


                            if not tool_output.strip():
                                tool_output = "<execution_output>\nThe execution completed without returning any output.\n</execution_output>"
                            else:
                                tool_output = f"<execution_output>\n{tool_output.strip()}\n</execution_output>"


                            # Append execution output to the assistant's response FOR THE CONTEXT
                            assistant_response_with_output = f"{assistant_response}\n{tool_output}\n<think>\n"

                            # Add the complete turn (Assistant response + execution) to context
                            context_chat.add_message(Role.ASSISTANT, assistant_response_with_output)

                            assistant_response = "" # Clear buffer as it's been processed

                            # Reset output buffers for next execution to prevent accumulation
                            stdout_buffer = ""
                            stderr_buffer = ""

                            action_counter += 1  # Increment action counter
                            
                            continue # Break inner loop after successful execution, continue to next agent turn

                        except Exception as e:
                            print(colored(f"\n❌ Error executing code: {str(e)}", "red"))
                            if args.debug:
                                traceback.print_exc()
                            # Add error to context before breaking
                            error_output = f"<execution_output>\n```error\n{traceback.format_exc()}\n```\n</execution_output>"
                            assistant_response_with_error = f"{assistant_response}\n{error_output}"
                            context_chat.add_message(Role.ASSISTANT, assistant_response_with_error)
                            assistant_response = "" # Clear buffer
                            
                            # Reset output buffers for next execution to prevent accumulation
                            stdout_buffer = ""
                            stderr_buffer = ""
                            
                            break # Break inner loop on execution error
                    else:
                        # Code execution denied by user
                        print(colored(" Execution cancelled by user.", "yellow"))
                        # Add assistant's plan and cancellation notice to chat history
                        cancellation_notice = "<execution_output>\nCode execution cancelled by user.\n</execution_output>"
                        assistant_response_with_cancellation = assistant_response + f"\n{cancellation_notice}"
                        context_chat.add_message(Role.ASSISTANT, assistant_response_with_cancellation)
                        assistant_response = "" # Clear buffer
                        
                        # Reset output buffers for next execution to prevent accumulation
                        stdout_buffer = ""
                        stderr_buffer = ""
                        
                        break # Break inner loop

                except KeyboardInterrupt:
                    print(colored("\n=== User interrupted execution (Ctrl+C) ===", "yellow"))
                    user_interrupt = True
                    break # Break from inner agentic loop

                except Exception as e:
                    LlmRouter.clear_unconfirmed_finetuning_data()
                    if ("ctrl+c" in str(e).lower()):
                        print(colored("=== User interrupted execution (Ctrl+C) ===", "yellow"))
                        user_interrupt = True
                        break # Break from inner agentic loop
                    
                    print(colored(f"An unexpected error occurred in the agent loop: {str(e)}", "red"))
                    if args.debug:
                        traceback.print_exc()
                    # Attempt to add error context before breaking
                    try:
                        error_output = f"<execution_output>\n```error\n{traceback.format_exc()}\n```\n</execution_output>"
                        assistant_response_with_error = assistant_response + f"\n{error_output}" # Append to potentially partial response
                        context_chat.add_message(Role.ASSISTANT, assistant_response_with_error)
                    except Exception as context_e:
                        print(colored(f"Failed to add error to context: {context_e}", "red"))
                    assistant_response = "" # Clear buffer
                    
                    # Reset output buffers for next execution to prevent accumulation
                    stdout_buffer = ""
                    stderr_buffer = ""
                    
                    break # Break inner loop
                
            # --- End of Agentic Inner Loop ---

            # save context once per turn (moved outside inner loop)
            if context_chat:
                context_chat.save_to_json()
                
            # Check if we should exit after all messages have been processed
            if args.exit and not args.message:
                print(colored("All automatic messages processed successfully. Exiting...", "green"))
                exit(0)

            print_with_timer("Turn completed.", "magenta")

        # End of outer while loop (Main loop)
        print(colored("\nCLI-Agent is shutting down.", "cyan"))


    except asyncio.CancelledError:
        print(colored("\nCLI-Agent was interrupted. Shutting down gracefully...", "yellow"))
    except KeyboardInterrupt:
        print(colored("\nCLI-Agent was interrupted by user. Shutting down...", "yellow"))
    except Exception as e:
        print(colored(f"\nCLI-Agent encountered an error: {str(e)}", "red"))
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(colored("\nCLI-Agent was interrupted by user. Shutting down...", "yellow"))
    except Exception as e:
        print(colored(f"\nCLI-Agent encountered an error: {str(e)}", "red"))
        traceback.print_exc()
