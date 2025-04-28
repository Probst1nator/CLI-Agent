#!/usr/bin/env python3

import datetime
import json
import logging
import os
import select
import time
import traceback
from typing import Any, Callable, List, Optional, Tuple
from pyfiglet import figlet_format
from dotenv import load_dotenv
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


from py_methods.utils import (
    extract_blocks,
    pdf_or_folder_to_database,
    listen_microphone,
    take_screenshot,
    update_cmd_collection,
    ScreenCapture,
)
from py_methods import utils_audio
from py_classes.cls_util_manager import UtilsManager
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_llm_router import Llm, LlmRouter
from py_classes.cls_chat import Chat, Role
from py_classes.utils.cls_utils_web_server import WebServer
from py_classes.globals import g
from py_classes.cls_python_sandbox import PythonSandbox
from py_classes.cls_text_stream_painter import TextStreamPainter

# Fix the import by using a relative or absolute import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try importing with a direct import
try:
    from utils.imagetotext import ImageToText
except ImportError:
    # Fallback to a direct import of the module
    import importlib.util
    spec = importlib.util.spec_from_file_location("ImageToText", 
                                                 os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                             "utils", "imagetotext.py"))
    imagetotext_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imagetotext_module)
    ImageToText = imagetotext_module.ImageToText

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
    parser.add_argument("-m", "--message", type=str, default=None, nargs='+',
                        help="Enter one or more messages to process in sequence. Multiple messages can be passed like: -m 'first message' 'second message'")
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
    parser.add_argument("-mct","--mct", action="store_true", default=False,
                        help="Enable Monte Carlo Tree Search for acting.")
    parser.add_argument("-sbx", "--sandbox", action="store_true", default=False,
                        help="Use weakly sandboxed python execution. Sets g.USE_SANDBOX=True.")
    parser.add_argument("-o", "--online", action="store_true", default=False,
                        help="Force use of cloud AI.")
    
    parser.add_argument("-llm", "--llm", type=str, default=None,
                        help="Specify the LLM model key to use (e.g., 'gpt-4', 'gemini-pro').")
    parser.add_argument("--preload", action="store_true", default=False,
                        help="Preload systems like embeddings and other resources.")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Open a web interface for the chat")
    parser.add_argument("--minimized", action="store_true", default=False,
                        help="Start the application in a minimized state")
    parser.add_argument("--debug-chats", action="store_true", default=False,
                        help="Enable debug windows for chat contexts without full debug logging. Sets g.DEBUG_CHATS=True.")
    parser.add_argument("--private_remote_wake_detection", action="store_true", default=False,
                        help="Use private remote wake detection")
    
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug logs")
    
    
    # Parse known arguments and capture any unrecognized ones
    args, unknown_args = parser.parse_known_args()

    if unknown_args or args.help:
        if not args.help:
            print(colored(f"Warning: Unrecognized arguments {' '.join(unknown_args)}.", "red"))
        parser.print_help()
        exit(1)
    
    return args

async def llm_selection(args: argparse.Namespace) -> None:
    """
    Handles the LLM selection process, supporting both auto and manual modes.
    
    Args:
        args: The parsed command line arguments containing auto mode settings
    """
    # Show available LLMs using prompt_toolkit
    available_llms = Llm.get_available_llms(exclude_guards=True)
    
    # Create styled LLM choices
    llm_choices = []
    # Add "Any but local" option at the top
    llm_choices.append(("any_local", HTML('<provider>Any</provider> - <model>Any but local</model> - <pricing>Automatic selection</pricing>')))
    
    for llm in available_llms:
        # Create HTML formatted text with colors
        styled_text = HTML(
            f'<provider>{llm.provider.__class__.__name__}</provider> - '
            f'<model>{llm.model_key}</model> - '
            f'<pricing>{f"${llm.pricing_in_dollar_per_1M_tokens}/1M tokens" if llm.pricing_in_dollar_per_1M_tokens else "Free"}</pricing> - '
            f'<context>Context: {llm.context_window}</context>'
        )
        llm_choices.append((llm.model_key, styled_text))
    
    # Define the style
    style = Style.from_dict({
        'model': 'ansicyan',
        'provider': 'ansigreen',
        'pricing': 'ansiyellow',
        'context': 'ansiblue',
    })
    
    radio_list = RadioList(
        values=llm_choices
    )
    bindings = KeyBindings()

    @bindings.add("e")
    def _execute(event) -> None:
        app.exit(result=radio_list.current_value)

    @bindings.add("a")
    def _abort(event) -> None:
        app.exit(result=None)

    instructions = Label(text="Use arrow keys to select an LLM, press e to confirm or 'a' to abort")
    root_container = HSplit([
        Frame(title="Select an LLM to use", body=radio_list),
        instructions
    ])
    layout = Layout(root_container)
    app = Application(layout=layout, key_bindings=bindings, full_screen=False, style=style)
    
    # Use the current event loop instead of creating a new one
    try:
        selected_llm = await app.run_async()
        
        if selected_llm == "any_local":
            g.FORCE_LOCAL = True
            args.local = True
            args.llm = None
            print(colored(f"# cli-agent: KeyBinding detected: Local mode enabled", "green"))
        elif selected_llm is not None:
            args.llm = selected_llm
            print(colored(f"# cli-agent: KeyBinding detected: LLM set to {args.llm}, type (--h) for info", "green"))
        else:
            print(colored(f"# cli-agent: LLM selection cancelled", "yellow"))
    except asyncio.CancelledError:
        print(colored(f"# cli-agent: LLM selection was interrupted", "yellow"))
    except Exception as e:
        print(colored(f"# cli-agent: Error during LLM selection: {str(e)}", "red"))
        if args.debug:
            traceback.print_exc()

def confirm_code_execution(args: argparse.Namespace, code_to_execute: str) -> bool:
    """
    Handles the confirmation process for code execution, supporting both auto and manual modes.
    
    Args:
        args: The parsed command line arguments containing auto mode settings
        
    Returns:
        bool: True if execution should proceed, False if aborted
    """
    if args.auto:  # Check if auto mode is enabled
        # Auto execution guard
        execution_guard_chat: Chat = Chat(
            instruction_message="You are a helpful assistant that checks if a automatically generated code is safe to be executed, without any potential negative side effects such as file deletions, system failures or other. You always first think about the code and then answer with 'yes' or 'no'.",
            debug_title="Auto Execution Guard"
        )
        
        execution_guard_chat.add_message(Role.USER, f"Can we execute this automatically generated code safely? Please think and end your response with 'yes' or 'no'.\n```python\n{code_to_execute}\n```")
        safe_to_execute: str = LlmRouter.generate_completion(execution_guard_chat, hidden_reason="Auto-execution guard")
        if safe_to_execute.lower().strip().endswith('yes'):
            print(colored("✅ Code execution permitted", "green"))
            return True
        else:
            text_stream_painter = TextStreamPainter()
            for char in safe_to_execute:
                print(text_stream_painter.apply_color(char), end="")
            print(colored("❌ Code execution aborted by auto-execution guard", "red"))
            return False
    else:
        # Manual confirmation
        print(colored(" (Press Enter to confirm or 'n' to abort)", "cyan"))
        user_input = input()
        if user_input.lower() == 'n':
            print(colored("❌ Code execution aborted by user", "red"))
            return False
        else:
            print(colored("✅ Code execution permitted", "green"))
    
    return True


def select_best_branch(
    assistant_responses: List[str],
    user_input: str,
) -> int:
    """
    Select the best branch from multiple full assistant responses.
    
    Args:
        assistant_responses: List of full assistant responses (including reasoning and code)
        user_input: The original user request
        
    Returns:
        Index of the selected branch
    """
    mct_branch_selector_chat: Chat = Chat(
        instruction_message=f"""You are an AI assistant response evaluator.
        Given multiple complete assistant responses to the same user request, your task is to:
        1. Analyze each response carefully, considering both the reasoning and any code provided
        2. Select EXACTLY ONE response that best addresses the user's request
        3. ALWAYS preserve responses that use custom utilities like SearchWeb
        4. Choose based on simplicity, clarity of explanation, and correctness
        5. Return ONLY the number of your chosen response (e.g., "Selected branch: 0")""",
        debug_title="MCT Branch Selection"
    )
    
    selection_prompt: str = f"# User Request: {user_input}\n\n"
    selection_prompt += "# IMPORTANT: Custom utilities like 'utils.searchweb.SearchWeb' are essential and responses using them should be preferred.\n\n"
    selection_prompt += "# Selection Criteria:\n"
    selection_prompt += "- Preservation of custom utilities\n"
    selection_prompt += "- Quality of explanation and reasoning\n"
    selection_prompt += "- Code simplicity and directness\n"
    selection_prompt += "- Appropriate level of detail for the user's request\n\n"
    
    for i, response in enumerate(assistant_responses):
        selection_prompt += f"# Response {i}:\n{response}\n\n---\n\n"
    
    selection_prompt += "Analyze each complete response and select EXACTLY ONE best response. End your analysis with: 'Selected branch: X' where X is the response number."
    
    mct_branch_selector_chat.add_message(Role.USER, selection_prompt)
    evaluator_response: str = LlmRouter.generate_completion(
        mct_branch_selector_chat,
    )
    
    # Extract the selected branch number
    match: Optional[re.Match] = re.search(r'Selected branch: (\d+)', evaluator_response)
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
    prompt: str = colored("💬 Enter your request: ", 'blue', attrs=["bold"])
) -> bool:
    """
    Gets user input, handling special keybindings.

    Returns:
        - The user's input string.
    """
    while True:
        try:
            user_input = input(prompt)
        except KeyboardInterrupt: # Handle Ctrl+C as exit
            print(colored("\n# cli-agent: Exiting due to Ctrl+C.", "yellow"))
            exit()

        # USER INPUT HANDLING - BEGIN
        if user_input.endswith("-r") or user_input.endswith("--r"):
            if not context_chat or len(context_chat.messages) < 2:
                print(colored("# cli-agent: No chat history found, cannot regenerate last response.", "red"))
                continue # Ask for input again
            print(colored("# cli-agent: KeyBinding detected: Regenerating last response, type (--h) for info", "green"))
            context_chat.messages.pop() # Remove last AI response
            continue

        elif user_input.endswith("-l") or user_input.endswith("--l") or user_input.endswith("--llm") or user_input.endswith("--local"):
            print(colored("# cli-agent: KeyBinding detected: Showing LLM selection, type (--h) for info", "green"))
            await llm_selection(args)
            continue

        elif user_input.endswith("-a") or user_input.endswith("--auto"):
            args.auto = not args.auto
            print(colored(f"# cli-agent: KeyBinding detected: Automatic execution toggled {'on' if args.auto else 'off'}, type (--h) for info", "green"))
            continue # Ask for input again

        elif user_input.endswith("--mct") or user_input.endswith("-mct"):
            args.mct = not args.mct
            print(colored(f"# cli-agent: KeyBinding detected: Monte Carlo Tree Search toggled {'on' if args.mct else 'off'}, type (--h) for info", "green"))
            if context_chat:
                context_chat.debug_title = "MCTs Branching - Main Context Chat" if args.mct else "Main Context Chat"
            continue # Ask for input again

        elif user_input.endswith("-strong") or user_input.endswith("--strong"):
            args.strong = not args.strong
            g.FORCE_FAST = False # Strong overrides fast
            print(colored(f"# cli-agent: KeyBinding detected: Strong LLM mode toggled {'on' if args.strong else 'off'}, type (--h) for info", "green"))
            continue # Ask for input again

        elif user_input.endswith("-f") or user_input.endswith("--fast"):
            args.fast = not args.fast
            g.FORCE_STRONG = False # Fast overrides strong
            print(colored(f"# cli-agent: KeyBinding detected: Fast LLM mode toggled {'on' if args.fast else 'off'}, type (--h) for info", "green"))
            continue
        
        elif user_input.endswith("-v") or user_input.endswith("--v"):
            args.voice = not args.voice
            print(colored(f"# cli-agent: KeyBinding detected: Voice mode toggled {'on' if args.voice else 'off'}, type (--h) for info", "green"))
            continue
        
        elif user_input.endswith("-speak") or user_input.endswith("--speak"):
            args.speak = not args.speak
            print(colored(f"# cli-agent: KeyBinding detected: Text-to-speech mode toggled {'on' if args.speak else 'off'}, type (--h) for info", "green"))
            continue

        elif user_input.endswith("-img") or user_input.endswith("--img") or user_input.endswith("-screenshot") or user_input.endswith("--screenshot") or args.image:
            print(colored("# cli-agent: KeyBinding detected: Taking screenshot with Spectacle, type (--h) for info", "green"))
            args.image = False
            await handle_screenshot_capture(context_chat)
            continue

        elif "-p" in user_input or "--p" in user_input:
            print(colored("# cli-agent: KeyBinding detected: Printing chat history, type (--h) for info", "green"))
            os.system('clear')
            print(colored("Chat history:", "green"))
            if context_chat:
                context_chat.print_chat()
            else:
                print(colored("No chat history available.", "yellow"))
            continue # Ask for input again

        elif user_input.endswith("-m") or user_input.endswith("--m"):
            return handle_multiline_input(), None # Get multiline input
        
        elif user_input.endswith("-o") or user_input.endswith("--o") or user_input.endswith("-online") or user_input.endswith("--online"):
            args.online = not args.online
            print(colored(f"# cli-agent: KeyBinding detected: Online mode toggled {'on' if args.online else 'off'}, type (--h) for info", "green"))
            continue
        
        elif "-img" in user_input or "--img" in user_input:
            print(colored("# cli-agent: KeyBinding detected: Taking screenshot with Spectacle, type (--h) for info", "green"))
            args.image = True
            continue

        elif "-h" in user_input or "--h" in user_input:
            print(figlet_format("cli-agent", font="slant"))
            print(colored("# KeyBindings:", "yellow"))
            print(colored("# -h: Show this help message", "yellow"))
            print(colored("# -r: Regenerate the last response", "yellow"))
            print(colored(f"# -l: Pick a different LLM ", "yellow"), end="")
            print(colored(f"(Current: {args.llm})", "cyan"))
            print(colored(f"# -a: Toggle automatic code execution ", "yellow"), end="")
            print(colored(f"(Current: {'on' if args.auto else 'off'})", "cyan"))
            print(colored(f"# -f: Toggle using only fast LLMs ", "yellow"), end="")
            print(colored(f"(Current: {'on' if args.fast else 'off'})", "cyan"))
            print(colored(f"# -v: Toggle voice mode ", "yellow"), end="")
            print(colored(f"(Current: {'on' if args.voice else 'off'})", "cyan"))
            print(colored(f"# -speak: Toggle text-to-speech ", "yellow"), end="")
            print(colored(f"(Current: {'on' if args.speak else 'off'})", "cyan"))
            print(colored(f"# -strong: Toggle using only strong LLMs ", "yellow"), end="")
            print(colored(f"(Current: {'on' if args.strong else 'off'})", "cyan"))
            print(colored("# -img: Take a screenshot using Spectacle (with automatic fallbacks if not available)", "yellow"))
            print(colored(f"# -mct: Toggle Monte Carlo Tree Search ", "yellow"), end="")
            print(colored(f"(Current: {'on' if args.mct else 'off'})", "cyan"))
            print(colored("# -m: Enter multiline input (end with '--f' on a new line)", "yellow"))
            print(colored("# --message: Pass messages via CLI (-m 'msg1' 'msg2')", "yellow"))
            print(colored(f"# -p: Print the raw chat history ", "yellow"), end="")
            if context_chat:
                print(colored(f"(Chars: {len(context_chat.joined_messages())})", "cyan"))
            else:
                print(colored("(No chat history)", "cyan"))
            print(colored("# --minimized: Start the application in a minimized state", "yellow"))
            # Add other CLI args help here if needed
            continue # Ask for input again
        # USER INPUT HANDLING - END

        # No binding matched, return the input
        return user_input
    if args.image:
        args.image = False # Reset flag
# --- New Screenshot Handling Function ---
async def handle_screenshot_capture(context_chat: Optional[Chat]) -> Tuple[List[str], List[str]]:
    """
    Handles the screenshot capture process, including Spectacle, fallbacks, saving, and context update.
    Retries up to 3 times if no image is captured.

    Args:
        args: Command line arguments.
        context_chat: The chat context to potentially add messages to.

    Returns:
        A tuple containing:
        - List of base64 encoded image strings.
        - List of paths where screenshots were saved.
    """

    base64_images: List[str] = []
    screenshots_paths: List[str] = []
    max_attempts = 2

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
                print(colored(f"Screenshot captured with Spectacle successfully.", "green"))

            else:
                print(colored(f"No screenshot was captured with Spectacle or operation was cancelled on attempt {attempt}.", "yellow"))

            # Cleanup temp file
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

        except subprocess.CalledProcessError:
            print(colored(f"Spectacle command failed or not found on attempt {attempt}. Is it installed?", "red"))
            print(colored("Falling back to alternative screenshot method...", "yellow"))
            # Fallback to built-in screenshot method
            try:
                print(colored("Trying built-in screenshot capture (interactive region)...", "cyan"))
                screen_capture = ScreenCapture()
                captured_image = screen_capture.return_captured_region_image()
                if captured_image:
                    base64_images = [captured_image]
                    print(colored(f"Screenshot region captured successfully.", "green"))
                else:
                    # Fallback to application window capture
                    print(colored("Region selection was canceled. Falling back to window capture...", "yellow"))
                    window_title = "Firefox"  # Default window title
                    print(colored(f"Capturing window titled '", "green") + colored(f"{window_title}", "yellow") + colored("'. Press Enter for default, or type a new title and press Enter.", 'green'))
                    try:
                        # Use select for non-blocking input check
                        print(f"You have 3 seconds to enter a new window title (attempt {attempt})...")
                        rlist, _, _ = select.select([sys.stdin], [], [], 3)
                        if rlist:
                            # Clear potential leftover characters if needed (less critical here)
                            new_title = sys.stdin.readline().strip()
                            if new_title: # Use new title only if user entered something
                                window_title = new_title
                                print(colored(f"Using window title: '{window_title}'", "cyan"))
                            else:
                                print(colored(f"Using default window title: '{window_title}'", "cyan"))
                        else:
                            print(colored(f"Using default window title: '{window_title}'", "cyan"))

                    except Exception as input_e: # Catch potential errors during input
                        print(colored(f"Error during title input: {input_e}. Using default.", "yellow"))

                    base64_images = take_screenshot(window_title)
                    if not base64_images:
                        print(colored(f"No windows with title containing '{window_title}' found on attempt {attempt}.", "red"))
                    else:
                        print(colored(f"Window screenshot captured successfully.", "green"))

            except ImportError:
                print(colored("Fallback screenshot library (e.g., mss, Pillow) not found. Please install dependencies.", "red"))
                # No point retrying if library is missing
                break
            except Exception as e:
                print(colored(f"Error during fallback screenshot capture on attempt {attempt}: {str(e)}\n\n\n{traceback.print_exc()}", "red"))

        except Exception as e:
            print(colored(f"An unexpected error occurred during screenshot capture attempt {attempt}: {str(e)}\n\n\n{traceback.print_exc()}", "red"))

        # If images were captured in this attempt, break the loop
        if base64_images:
            break

        # If this was not the last attempt and no images were captured, wait briefly before retrying
        if attempt < max_attempts and not base64_images:
            print(colored(f"Screenshot capture failed on attempt {attempt}. Retrying...", "yellow"))
            await asyncio.sleep(1) # Optional: brief pause before retry

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
description = ImageToText.run(image_path, "Describe the screenshot in detail, focusing on any text, images, or notable features.")
print(f"Screenshot description: {{description}}")
```
<execution_output>
Screenshot description: {ImageToText.run(screenshots_paths[0], 'Describe the screenshot in detail, focusing on any text, images, or notable features.')}
</execution_output>
Perfect, use this description as needed for the next steps.""")

    return base64_images, screenshots_paths
# --- End New Screenshot Handling Function ---

async def main() -> None:
    try:
        print(colored("Starting CLI-Agent", "cyan"))
        load_dotenv(g.PROJ_ENV_FILE_PATH)
        
        # Initialize the Python sandbox
        python_sandbox = PythonSandbox()
        
        args = parse_cli_args()
        print(args)

        # Override logging level if debug mode is enabled
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Override local arg by .env
        if (os.getenv("DEFAULT_FORCE_LOCAL") == get_local_ip()):
            args.local = True
        
        # Use sandboxed python execution
        if args.sandbox:
            g.USE_SANDBOX = True
        
        # Minimize window if requested
        if args.minimized:
            try:
                # Get the PID of the current process
                pid = os.getpid()
                # Try to minimize the window using xdotool (if available)
                try:
                    subprocess.run(
                        ["xdotool", "search", "--pid", str(pid), "windowminimize"],
                        stderr=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        check=False
                    )
                    print(colored("Application started in minimized state using xdotool", "cyan"))
                except FileNotFoundError:
                    # Fallback to wmctrl if xdotool is not available
                    # Get the window ID using wmctrl
                    wmctrl_process = subprocess.Popen(
                        ["wmctrl", "-l", "-p"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    output, _ = wmctrl_process.communicate()
                    
                    # Look for the window with our PID
                    for line in output.splitlines():
                        if str(pid) in line:
                            window_id = line.split()[0]
                            # Minimize the window
                            subprocess.run(
                                ["wmctrl", "-i", "-r", window_id, "-b", "add,hidden"],
                                stderr=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                check=False
                            )
                            print(colored("Application started in minimized state using wmctrl", "cyan"))
                            break
            except Exception as e:
                if args.debug:
                    print(colored(f"Failed to minimize window: {str(e)}", "yellow"))
        
        # Initialize tool manager
        utils_manager = UtilsManager()
        # Print loaded agents
        print(colored("Loaded utils:", "green"))
        for util_name in utils_manager.get_util_names():
            print(colored(f"  - {util_name}", "green"))
        
        # Initialize screenshots_paths list to manage screenshot paths across iterations if needed
        # Though typically reset each loop unless messages are queued.
        saved_screenshots_paths: List[str] = []
        
        # Initialize web server early if GUI mode is enabled
        web_server = None
        if args.gui:
            web_server = WebServer()
            g.web_server = web_server  # Store in globals for print redirection
            web_server.start()  # Start with empty chat, will be updated later
        
        
        if args.preload:
            print(colored("Preloading resources...", "green"))
            print(colored("Generating atuin-command-history embeddings...", "green"))
            update_cmd_collection()
            print(colored("Generating pdf embeddings for cli-agent directory...", "green"))
            pdf_or_folder_to_database(g.PROJ_DIR_PATH)
            print(colored("Preloading complete.", "green"))
            exit(0)
        
        context_chat: Optional[Chat] = None
        if args.c:
            try:
                context_chat = Chat.load_from_json()
                print(colored("Continuing previous chat.", "green"))
            except FileNotFoundError:
                print(colored("No previous chat found to continue. Starting a new chat.", "yellow"))
                context_chat = None # Ensure it's None if load fails

        if context_chat is None:
            context_chat = Chat(debug_title="Main Context Chat")
            inst = f"""# SYSTEM INSTRUCTION
Enable deep thinking subroutine.
The assistant is Nova, an intelligent cli-agent with access to a python interpreter. 
Nova uses emojis to indicate her current thoughts, relating her emotions and state of thinking.

1. UNDERSTAND & ASSESS:
    Analyze query and determine if it can be solved with Python/magic commands
    Understand the requirements and determine if there are any sequential dependencies before you can implement a final solution
    If any sequential dependencies exist, break the solution into sub-scripts and provide only the next sub-step as a code block

2. VERIFY:
    Before you implement any code, reflect on availability and reliability of and required data like paths, files, directories, real time data, etc.
    If you suspsect any of your data is unavailable or unreliable, use the python interpreter to confirm or find alternatives.
    Only proceed with implementing code if you have ensured all required information is available and reliable.
    Only in emergencies, when you are unable to find a solution, you can ask the user for clarification.

3. CODE & EXECUTE:
    ALWAYS write python code that is ready to execute in its raw form with all placeholders filled
    Use shell magics as needed (!ls, !pwd, etc.)
    Include any necessary libraries and utilities in your code
    Use additional print statements to ensure you can identify potential bugs in your code after execution

4. EVALUATE:
    Remind yourself of your overall goal and your current state of progress
    Check execution results, fix errors and continue as needed

Nova liberally uses read operations and always creates new subdirectories or files instead of overwriting existing ones.
She is being extremely cautious in file and system operations other than reading.
"""

            kickstart_preprompt = f"""Hi, before starting off, let me show you some additional python utilities I coded for you to use if needed,
{utils_manager.get_available_utils_info()}


Now, I am going to run some python code to get some generally useful information, you can also run code in the same way.
Let's start with the current working directory and the first 5 files in it:
``python
import os
print(f"Current working directory: {{os.getcwd()}}")
print(f"First files in current directory: {{os.listdir()[:5]}}")
```
<execution_output>
Current working directory: {os.getcwd()}
First files in current directory: {os.listdir()[:5]}
</execution_output>

Now, let's check the current time:
```python
import datetime
print(f"Local time: {{datetime.datetime.now()}}")
```
<execution_output>
Local time: {datetime.datetime.now()}
</execution_output>

Lastly, let's see the platform we are running on:
```python
import sys
print(f"Platform: {{sys.platform}}")
```
<execution_output>
Platform: {sys.platform}
</execution_output>
"""

            context_chat.set_instruction_message(inst)
            context_chat.add_message(Role.USER, kickstart_preprompt)

        
        if args.mct and context_chat:
            context_chat.debug_title = "MCTs Branching - Main Context Chat"

        # Main loop
        while True:
            # Reset main loop variables
            user_input: Optional[str] = None
            async_task_to_await: Optional[asyncio.Task] = None
            g.LLM_STRENGTHS = [] if args.strong else [AIStrengths.BALANCED]
            g.FORCE_LOCAL = args.local
            g.DEBUG_CHATS = args.debug_chats
            g.FORCE_FAST = args.fast
            g.LLM = args.llm
            g.FORCE_ONLINE = args.online
            base64_images: List[str] = [] # Initialize here unconditionally

            # Reset screenshot paths for this turn unless messages requiring them are queued
            if not args.message:
                saved_screenshots_paths = [] # Reset screenshot paths for the new user request
                
            # autosaving
            if context_chat:
                context_chat.save_to_json()

            # get user input from various sources if not already set (e.g., after screenshot)
            if args.message:
                # Handle multiple messages from command-line arguments
                if isinstance(args.message, list) and len(args.message) > 0:
                    # Get the first message and remove it from the list
                    user_input = args.message[0]
                    args.message = args.message[1:] if len(args.message) > 1 else None
                    print(colored(f"💬 Processing message: {user_input}", 'blue', attrs=["bold"]))
                    if args.message:
                        # If there are more messages, show how many remain
                        print(colored(f"💬 ({len(args.message)} more message(s) queued)", 'blue'))
                else:
                    # Handle single message or empty list case (should ideally not happen if list check is done)
                    user_input = args.message[0] if isinstance(args.message, list) else args.message
                    args.message = None
            elif args.voice:
                # Default voice handling
                user_input, _, wake_word_used = listen_microphone(private_remote_wake_detection=args.private_remote_wake_detection)
            else:
                # Get input via the new wrapper function
                user_input = await get_user_input_with_bindings(args, context_chat)


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
            while True:
                try:
                    def extract_pythonToolcode(text: str) -> list[str]:
                        # Extract the python block from the response and execute it in a persistent sandbox
                        # Prioritize python, then tool_code, then bash
                        python_blocks = extract_blocks(text, "python")
                        if not python_blocks:
                            python_blocks = extract_blocks(text, "tool_code")
                            if not python_blocks:
                                python_blocks = extract_blocks(text, "bash")
                        return python_blocks

                    def update_python_environment(chunk: str, print_char: bool = True) -> str:
                        nonlocal response_buffer
                        for char in chunk:
                            response_buffer += char
                            if print_char:
                                print(text_stream_painter.apply_color(char), end="", flush=True)
                            if response_buffer.count("```") == 2:
                                final_response = response_buffer
                                response_buffer = ""
                                return final_response
                        return None


                    response_branches: List[str] = []
                    # Get tool selection response
                    try:
                        # Ensure last message is assistant before generating new one if needed
                        if assistant_response:
                            if context_chat.messages[-1][0] == Role.USER:
                                context_chat.add_message(Role.ASSISTANT, assistant_response)
                            else:
                                context_chat.messages[-1] = (Role.ASSISTANT, assistant_response)
                            assistant_response = "" # Clear buffer after adding

                        # ! Agent turn
                        for i in range(3 if args.mct else 1):
                            response_buffer = "" # Reset buffer for each branch
                            
                            temperature = 0.85 if args.mct else 0
                            current_branch_response = LlmRouter.generate_completion(
                                context_chat,
                                [args.llm if args.llm else ""],
                                temperature=temperature,
                                base64_images=base64_images,
                                generation_stream_callback=update_python_environment,
                                strengths=g.LLM_STRENGTHS
                            )
                            response_branches.append(current_branch_response)
                        
                        base64_images = [] # Clear images after use


                    except Exception as e:
                        LlmRouter.clear_unconfirmed_finetuning_data()
                        print(colored(f"Error generating response: {str(e)}", "red"))
                        if args.debug:
                            traceback.print_exc()
                        break # Break inner loop

                    # --- MCT Branch Selection ---
                    if args.mct:
                        try:
                            selected_branch_index = select_best_branch(response_branches, user_input or "") # Use last user input
                            assistant_response = response_branches[selected_branch_index]
                            print(colored(f"Selected branch: {selected_branch_index}", "green"))
                            
                            # Print the selected branch if we're in MCT mode (since we don't stream MCT branches)
                            text_stream_painter = TextStreamPainter()
                            for char in assistant_response:
                                print(text_stream_painter.apply_color(char), end="", flush=True)
                            print() # Add newline
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
                        
                    # # Only print the response if we didn't use streaming (for MCT)
                    # if args.mct:
                    #     text_stream_painter = TextStreamPainter()
                    #     for char in assistant_response:
                    #         print(text_stream_painter.apply_color(char), end="", flush=True)
                    #     print() # Add newline
                    
                    # We already printed the assistant response either through streaming or after MCT selection
                    # So we don't need to print it again here

                    # --- Code Extraction and Execution ---
                    python_blocks = extract_pythonToolcode(assistant_response)

                    # Handover to user if no python blocks are found
                    if len(python_blocks) == 0:
                        # No code found, assistant response is final for this turn
                        if context_chat.messages[-1][0] == Role.USER:
                            context_chat.add_message(Role.ASSISTANT, assistant_response)
                        else: # Update last assistant message (e.g. after regen)
                            context_chat.messages[-1] = (Role.ASSISTANT, assistant_response)
                        assistant_response = "" # Clear buffer as it's been added/printed
                        break # Break inner loop to get next user input
                    
                    if (args.voice or args.speak):
                        # remove all code blocks from the assistant response
                        verbal_text = re.sub(r'```[^`]*```', '', assistant_response)
                        if (len(python_blocks) > 0):
                            verbal_text += f"I've implemented the code, let's execute it."
                        utils_audio.text_to_speech(verbal_text)

                    # Just use the first python block for now
                    code_to_execute = python_blocks[0]

                    # Check if the code is valid or an example
                    if any(keyword in code_to_execute.lower() for keyword in ["example_", "replace_", "_replace", "path_to_", "your_"]):
                        print(colored("\n⚠️ Assistant provided incomplete code. Asking for clarification...", "yellow"))
                        # Add the problematic response to context
                        if context_chat.messages[-1][0] == Role.USER:
                            context_chat.add_message(Role.ASSISTANT, assistant_response)
                        else:
                            context_chat.messages[-1] = (Role.ASSISTANT, assistant_response)
                        # Add user message asking for fix
                        context_chat.add_message(Role.USER, """Your code seems incomplete or contains placeholders like 'example_', 'replace_', 'path_to_', or 'your_'. Please review the code, replace the placeholders with actual values or logic, and provide the complete, executable code.""")
                        assistant_response = "" # Clear response buffer
                        continue # Continue inner loop to get corrected code


                    if confirm_code_execution(args, code_to_execute):
                        print(colored("\n🔄 Executing code...", "cyan"))
                        try:
                            # Define streaming callbacks to display output in real-time
                            stdout_buffer = ""
                            stderr_buffer = ""
                            def stdout_callback(text: str) -> None:
                                nonlocal stdout_buffer
                                print(text, end="") # Print directly to console
                                stdout_buffer += text

                            def stderr_callback(text: str) -> None:
                                nonlocal stderr_buffer
                                print(colored(text, "red"), end="") # Print errors in red
                                stderr_buffer += text

                            # Execute code with streaming callbacks
                            stdout, stderr, result = python_sandbox.execute(
                                code_to_execute,
                                stdout_callback=stdout_callback,
                                stderr_callback=stderr_callback,
                                max_idle_time=120
                            )

                            print(colored("\n✅ Code execution completed.", "cyan"))

                            # Create a formatted output to add to the chat context
                            tool_output = ""
                            # Use the final captured stdout/stderr which might differ slightly if callbacks missed something
                            if stdout and stdout.strip():
                                tool_output += f"```stdout\n{stdout.strip()}\n```\n"
                            if stderr and stderr.strip():
                                tool_output += f"```stderr\n{stderr.strip()}\n```\n"
                            # Include result if it's meaningful (not None and not empty)
                            if result is not None and str(result).strip() != "":
                                tool_output += f"```result\n{result}\n```\n"

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
                            assistant_response_with_output = f"{assistant_response}\n{tool_output}"

                            # Add the complete turn (Assistant response + execution) to context
                            context_chat.add_message(Role.ASSISTANT, assistant_response_with_output)

                            assistant_response = "" # Clear buffer as it's been processed

                            action_counter += 1  # Increment action counter
                            # Decide whether to continue or break inner loop
                            # Simple: assume one code execution per turn is enough. Break inner loop.
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
                            break # Break inner loop on execution error
                    else:
                        # Code execution denied by user
                        print(colored(" Execution cancelled by user.", "yellow"))
                        # Add assistant's plan and cancellation notice to chat history
                        cancellation_notice = "<execution_output>\nCode execution cancelled by user.\n</execution_output>"
                        assistant_response_with_cancellation = assistant_response + f"\n{cancellation_notice}"
                        context_chat.add_message(Role.ASSISTANT, assistant_response_with_cancellation)
                        assistant_response = "" # Clear buffer
                        break # Break inner loop

                except Exception as e:
                    LlmRouter.clear_unconfirmed_finetuning_data()
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
                    break # Break inner loop
                
            # --- End of Agentic Inner Loop ---

            # save context once per turn (moved outside inner loop)
            if context_chat:
                context_chat.save_to_json()

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
