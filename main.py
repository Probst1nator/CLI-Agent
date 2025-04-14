#!/usr/bin/env python3

import datetime
import json
import logging
import os
import select
import time
import traceback
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


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")
# Suppress phonemizer warnings
warnings.filterwarnings("ignore", message="words count mismatch on*", module="phonemizer", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")  # Catch all phonemizer warnings


from py_classes.cls_util_manager import UtilsManager
from py_methods.utils import (
    extract_blocks,
    pdf_or_folder_to_database,
    listen_microphone,
    take_screenshot,
    text_to_speech,
    update_cmd_collection,
)
from py_classes.cls_llm_router import Llm, LlmRouter
from py_classes.cls_chat import Chat, Role
from py_classes.utils.cls_utils_web_server import WebServer
from py_classes.globals import g
from py_classes.cls_python_sandbox import PythonSandbox

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
    parser.add_argument("-a", "--auto", nargs='?', const=5, type=int, default=None,
                        help="""Automatically execute safe commands after specified delay in seconds. Unsafe commands still require confirmation.""", metavar="DELAY")
    parser.add_argument("-c", action="store_true", default=False,
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-l", "--local", action="store_true", default=False,
                        help="Use the local Ollama backend for processing.")
    parser.add_argument("-m", "--message", type=str, default=None,
                        help="Enter your first message instantly.")
    parser.add_argument("-r", "--regenerate", action="store_true", default=False,
                        help="Regenerate the last response.")
    parser.add_argument("-v", "--voice", action="store_true", default=False,
                        help="Enable microphone input and text-to-speech output.")
    parser.add_argument("-s", "--speak", action="store_true", default=False,
                        help="Text-to-speech output.")
    parser.add_argument("-img", "--image", action="store_true", default=False,
                        help="Take a screenshot and generate a response based on the contents of the image.")
    parser.add_argument("--preload", action="store_true", default=False,
                        help="Preload systems like embeddings and other resources.")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Open a web interface for the chat")
    parser.add_argument("--debug-chats", action="store_true", default=False,
                        help="Enable debug windows for chat contexts without full debug logging")
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
    llm_choices.append(("any_but_local", HTML('<provider>Any</provider> - <model>Any but local</model> - <pricing>Automatic selection</pricing>')))
    
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
    selected_llm = await app.run_async()
    
    if selected_llm == "any_but_local":
        g.FORCE_LOCAL = False
        args.local = False
        args.llm = None
        print(colored(f"# cli-agent: KeyBinding detected: Local mode disabled, using automatic LLM selection", "green"))
    else:
        args.llm = selected_llm
        print(colored(f"# cli-agent: KeyBinding detected: Local toggled {args.local}, LLM set to {args.llm}, type (--h) for info", "green"))

def confirm_code_execution(args: argparse.Namespace) -> bool:
    """
    Handles the confirmation process for code execution, supporting both auto and manual modes.
    
    Args:
        args: The parsed command line arguments containing auto mode settings
        
    Returns:
        bool: True if execution should proceed, False if aborted
    """
    if args.auto is not None:  # Check if auto mode is enabled
        # Auto-execution with countdown
        for i in range(5, 0, -1):
            print(colored(f" in {i}", "cyan"), end="", flush=True)
            for _ in range(10):  # Check 10 times per second
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    print(colored("\n❌ Code execution aborted by user", "red"))
                    return False
                time.sleep(0.1)
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                return False
            print("\b" * (len(str(i)) + 4), end="", flush=True)  # Clear the countdown
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

async def main() -> None:
    print(colored("Starting CLI-Agent", "cyan"))
    load_dotenv(g.PROJ_ENV_FILE_PATH)
    
    args = parse_cli_args()
    print(args)

    # Display all chat windows
    if args.debug_chats:
        g.DEBUG_CHATS = True
    
    # Override logging level if debug mode is enabled
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize tool manager
    utils_manager = UtilsManager()
    # Print loaded agents
    print(colored("Loaded utils:", "green"))
    for util_name in utils_manager.get_util_names():
        print(colored(f"  - {util_name}", "green"))
        
    # Initialize web server early if GUI mode is enabled
    web_server = None
    if args.gui:
        web_server = WebServer()
        g.web_server = web_server  # Store in globals for print redirection
        web_server.start()  # Start with empty chat, will be updated later
    
    if args.local or (os.getenv("DEFAULT_FORCE_LOCAL") == get_local_ip()):
        args.local = True
        g.FORCE_LOCAL = True
    
    if args.preload:
        print(colored("Preloading resources...", "green"))
        print(colored("Generating atuin-command-history embeddings...", "green"))
        update_cmd_collection()
        print(colored("Generating pdf embeddings for cli-agent directory...", "green"))
        pdf_or_folder_to_database(g.PROJ_DIR_PATH)
        print(colored("Preloading complete.", "green"))
        exit(0)
    
    user_input: str = ""
    context_chat: Chat
    
    if args.c:
        context_chat = Chat.load_from_json()
        context_chat.title = "Main Context Chat"
    else:
        inst = f"""# SYSTEM INSTRUCTION
Enable deep thinking subroutine.
The assistant is Nova, an intelligent cli-agent with access to a python interpreter. 
Nova uses emojis to indicate her current thoughts, relating her emotions and state of thinking.

1. UNDERSTAND & ASSESS:
    Analyze query and determine if it can be solved with Python/magic commands
    If not resolvable, break task into sub-tasks and attempt pythonic solutions step by step

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
She is being extremely cautious in file and system operations.
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
        context_chat = Chat(debug_title="Main Context Chat")
        context_chat.set_instruction_message(inst)
        context_chat.add_message(Role.USER, kickstart_preprompt)
    
    if (args.voice or args.speak) and context_chat and len(context_chat.messages) > 0:
        # tts last response (when continuing)
        last_response = context_chat.messages[-1][1]
        text_to_speech(last_response)
        print(colored(last_response, 'magenta'))
    
    # Main loop
    while True:
        # save the context_chat to a json file
        if context_chat:
            context_chat.save_to_json()
        
        # screen capture
        if args.image:
            args.image = False
            window_title = "Firefox" # Default window title
            print(colored(f"Capturing screenshots of '", "green") + colored(f"{window_title}", "yellow") + colored("' press any key to enter another title.", 'green'))
            try:
                for remaining in range(3, 0, -1):
                    sys.stdout.write("\r" + colored(f"Proceeding in {remaining} seconds... ", 'yellow'))
                    sys.stdout.flush()
                    time.sleep(1)
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        raise KeyboardInterrupt
                sys.stdout.write("\n")
            except KeyboardInterrupt:
                window_title = input(colored("\nEnter the title of the window you want to capture: ", 'blue'))
                
            base64_images = take_screenshot(window_title)
            if not base64_images:
                print(colored(f"# cli-agent: No images were returned.", "red"))
                continue
            for i, base64_image in enumerate(base64_images):
                print(colored(f"# cli-agent: Converting Image ({i}/{len(base64_images)}) into words...", "green"))
                image_response_str = LlmRouter.generate_completion("Put words to the contents of the image for a blind user.", base64_images=[base64_image])
                prompt_context_augmentation += f'\n\n```vision_{i}\n{image_response_str}\n```'
        
        if LlmRouter.has_unconfirmed_data():
            LlmRouter.confirm_finetuning_data()
        
        # get user input from various sources
        if args.message:
            user_input = args.message
            args.message = None
        elif args.voice:
            # Default voice handling
            user_input, _, wake_word_used = listen_microphone(private_remote_wake_detection=args.private_remote_wake_detection)
        else:
            user_input = input(colored("💬 Enter your request: ", 'blue', attrs=["bold"]))
        
        # USER INPUT HANDLING - BEGIN
        if user_input.endswith("-r") or user_input.endswith("--r") and context_chat:
            if len(context_chat.messages) < 2:
                print(colored(f"# cli-agent: No chat history found, cannot regenerate last response.", "red"))
                continue
            print(colored(f"# cli-agent: KeyBinding detected: Regenerating last response, type (--h) for info", "green"))
            if context_chat.messages[-1][0] == Role.USER:
                context_chat.messages.pop()
            context_chat.messages.pop()
            
        if user_input.endswith("-l") or user_input.endswith("--l") or user_input.endswith("--llm") or user_input.endswith("--local"):
            user_input = user_input[:-3]
            print(colored(f"# cli-agent: KeyBinding detected: Showing LLM selection, type (--h) for info", "green"))
            await llm_selection(args)
            user_input = ""
            continue
        
        if user_input.endswith("-a") or user_input.endswith("--auto"):
            user_input = user_input[:-3]
            args.auto = not args.auto
            print(colored(f"# cli-agent: KeyBinding detected: Automatic execution toggled {args.auto}, type (--h) for info", "green"))
            continue
        
        if user_input.endswith("-s") or user_input.endswith("--s") or user_input.endswith("--screenshot") :
            user_input = user_input[:-3]
            print(colored(f"# cli-agent: KeyBinding detected: Taking screenshot, type (--h) for info", "green"))
            args.image = True
            continue
        
        if "-p" in user_input or "--p" in user_input:
            print(colored(f"# cli-agent: KeyBinding detected: Printing chat history, type (--h) for info", "green"))
            os.system('clear')
            print(colored("Chat history:", "green"))
            context_chat.print_chat()
            continue
        
        if user_input.endswith("-m") or user_input.endswith("--m"):
            print(colored("Enter your multiline input. Type '--f' on a new line when finished.", "blue"))
            lines = []
            while True:
                line = input()
                if line == "--f":
                    break
                lines.append(line)
            user_input = "\n".join(lines)
        
        if "-h" in user_input or "--h" in user_input:
            user_input = user_input[:-3]
            print(figlet_format("cli-agent", font="slant"))
            print(colored(f"""# cli-agent: KeyBindings:
# cli-agent: -h: Show this help message
# cli-agent: -r: Regenerate the last response
# cli-agent: -l: Pick a different LLM
# cli-agent: -a: Toggle auto mode
# cli-agent: -s: Take a screenshot
# cli-agent: -p: Print the raw chat history
""", "yellow"))
            continue
        # USER INPUT HANDLING - END
        

        # AGENTIC IN-TURN LOOP - BEGIN
        action_counter = 0  # Initialize counter for consecutive actions
        perform_exit: bool = False
        incomplete_assistant_text=""

        context_chat.add_message(Role.USER, user_input)
        
        while True:
            try:

                def update_python_environment(chunk: str) -> bool:
                    nonlocal incomplete_assistant_text
                    incomplete_assistant_text += chunk
                    if incomplete_assistant_text.count("```") == 2:
                        return True
                    return False

                # Get tool selection response
                try:
                    if incomplete_assistant_text:
                        if context_chat.messages[-1][0] == Role.USER:
                            context_chat.add_message(Role.ASSISTANT, incomplete_assistant_text)
                        else:
                            context_chat.messages[-1] = (Role.ASSISTANT, incomplete_assistant_text)
                        incomplete_assistant_text = ""
                    # ! Agent turn
                    response = LlmRouter.generate_completion(context_chat, [args.llm if args.llm else ""], callback=update_python_environment)
                except Exception as e:
                    LlmRouter.clear_unconfirmed_finetuning_data()
                    print(colored(f"Error generating tool selection response: {str(e)}", "red"))
                    if args.debug:
                        traceback.print_exc()
                    break
                
                # Extract the python block from the response and execute it in a persistent sandbox
                python_blocks = extract_blocks(incomplete_assistant_text, "python")
                if not python_blocks:
                    python_blocks = extract_blocks(incomplete_assistant_text, "tool_code")
                    if not python_blocks:
                        python_blocks = extract_blocks(incomplete_assistant_text, "bash")
                
                if python_blocks:
                    # Check if the code is valid or an example
                    if any(keyword in python_blocks[0].lower() for keyword in ["example", "replace", "path_to_", "your_"]):
                        if incomplete_assistant_text:
                            if context_chat.messages[-1][0] == Role.USER:
                                context_chat.add_message(Role.ASSISTANT, incomplete_assistant_text)
                            else:
                                context_chat.messages[-1] = (Role.ASSISTANT, incomplete_assistant_text)
                            incomplete_assistant_text = ""
                        context_chat.add_message(Role.USER, """Your code is incomplete, please check where you used any of these keywords and replace/gather the correct information yourself: ["example", "replace", "path_to_", "your_"]""")
                        continue
                        
                    
                    print(colored("\n🔄 Starting code execution...", "cyan"), end="")
                    if confirm_code_execution(args):
                        # if bash_blocks:
                        #     bash_to_execute = bash_blocks[0]
                        #     os.system(bash_to_execute)
                        
                        if python_blocks:
                            # Initialize the Python sandbox if not already done
                            if not hasattr(g, 'python_sandbox'):
                                g.python_sandbox = PythonSandbox(g.FORCE_LOCAL)
                            
                            code_to_execute = python_blocks[0]  # Take the first Python block
                            
                            try:
                                # Define streaming callbacks to display output in real-time
                                def stdout_callback(text: str) -> None:
                                    print(text, end="")
                                
                                def stderr_callback(text: str) -> None:
                                    print(text, end="")
                                
                                # Execute code with streaming callbacks
                                stdout, stderr, result = g.python_sandbox.execute(
                                    code_to_execute,
                                    stdout_callback=stdout_callback,
                                    stderr_callback=stderr_callback,
                                    max_idle_time=120
                                )
                                
                                # # Display result if any (after streaming is done)
                                # if result is not None and result != "":
                                #     print(colored(f"\n🔄 Result:", "cyan"))
                                #     print(result)
                                
                                # Always print completion message
                                print(colored("\n✅ Code execution completed", "cyan"))
                                
                                # Create a formatted output to add to the chat context
                                tool_output = ""
                                if stdout.strip():
                                    tool_output += f"```stdout\n{stdout.strip()}\n```\n"
                                if stderr.strip():
                                    tool_output += f"```stderr\n{stderr.strip()}\n```\n"
                                if result is not None and result != "":
                                    tool_output += f"```result\n{result}\n```\n"
                                
                                # remove color codes
                                tool_output = re.sub(r'\x1b\[[0-9;]*m', '', tool_output)
                                
                                # shorten to max 4000 characters, include the first 3000 and last 1000, splitting at the last newline
                                if len(tool_output) > 4000:
                                    # Find the last newline in the middle section
                                    first_split = tool_output[:3000]
                                    first_index = first_split.rfind('\n')
                                    second_split = tool_output[-1000:]
                                    second_index = second_split.find('\n')
                                    
                                    tool_output = tool_output[:first_index] + "\n[...output truncated...]\n" + second_split[second_index:]
                                
                                if not tool_output:
                                    tool_output = "The execution returned no output."
                                incomplete_assistant_text += f"\n<execution_output>\n{tool_output}</execution_output>\n"
                                
                                action_counter += 1  # Increment action counter
                            except Exception as e:
                                print(colored(f"\n❌ Error executing code: {str(e)}", "red"))
                                if args.debug:
                                    traceback.print_exc()
                                break
                    else:
                        # Code execution denied by user
                        break
                else:
                    # No blocks to execute found, end the loop and handover to user
                    if incomplete_assistant_text:
                        if context_chat.messages[-1][0] == Role.USER:
                            context_chat.add_message(Role.ASSISTANT, incomplete_assistant_text)
                        else:
                            context_chat.messages[-1] = (Role.ASSISTANT, incomplete_assistant_text)
                        incomplete_assistant_text = ""
                    break
            
            except Exception as e:
                LlmRouter.clear_unconfirmed_finetuning_data()
                print(colored(f"An unexpected error occurred: {str(e)}", "red"))
                if args.debug:
                    traceback.print_exc()
                break
        # AGENTIC TOOL USE - END
        
        if perform_exit:
            exit(0)
        
        # save context once per turn
        if context_chat:
            context_chat.save_to_json()

if __name__ == "__main__":
    asyncio.run(main())
