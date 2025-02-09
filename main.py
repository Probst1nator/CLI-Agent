#!/usr/bin/env python3

import json
import logging
import os
import select
import time
import traceback
from typing import List, Tuple
from pyfiglet import figlet_format
from dotenv import load_dotenv
from termcolor import colored
import argparse
import sys
import socket
import warnings
import asyncio


from py_methods.cmd_execution import select_and_execute_commands
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2:")
# Suppress phonemizer warnings
warnings.filterwarnings("ignore", message="words count mismatch on*", module="phonemizer", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="phonemizer")  # Catch all phonemizer warnings

from py_methods.tooling import extract_blocks, pdf_or_folder_to_database, listen_microphone, remove_blocks, take_screenshot, text_to_speech, update_cmd_collection
from py_classes.cls_llm_router import LlmRouter
from py_classes.cls_chat import Chat, Role
from py_classes.cls_web_server import WebServer
from py_classes.cls_tool_manager import ToolManager
from py_classes.globals import g

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
    
    parser.add_argument("--auto", nargs='?', const=5, type=int, default=None,
                        help="""Automatically execute safe commands after specified delay in seconds. Unsafe commands still require confirmation.""", metavar="DELAY")
    parser.add_argument("-c", action="store_true", default=False,
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-e", "--edit", nargs='?', const="", type=str, default=None, metavar="FILEPATH",
                        help="Edit either the file at the specified path or the contents of the clipboard.")
    parser.add_argument("-h", "--help", action="store_true", default=False,
                        help="Display this help")
    parser.add_argument("-i", "--intelligent", action="store_true", default=False,
                        help="Use the current most intelligent model for the agent.")
    parser.add_argument("-l", "--local", action="store_true", default=False,
                        help="Use the local Ollama backend for processing.")
    parser.add_argument("-o", "--online", action="store_true", default=False,
                        help="Use online backends for processing.")
    parser.add_argument("-m", "--message", type=str, default=None,
                        help="Enter your first message instantly.")
    parser.add_argument("-p", "--presentation", nargs='?', const="", type=str, default=None, metavar="TOPIC",
                        help="Interactively create a presentation.")    
    parser.add_argument("-r", "--regenerate", action="store_true", default=False,
                        help="Regenerate the last response.")
    parser.add_argument("-v", "--voice", action="store_true", default=False,
                        help="Enable microphone input and text-to-speech output.")
    parser.add_argument("-spe", "--speak", action="store_true", default=False,
                        help="Text-to-speech output.")
    parser.add_argument("-img", "--image", action="store_true", default=False,
                        help="Take a screenshot and generate a response based on the contents of the image.")
    parser.add_argument("--llm", nargs='?', type=str, default="",
                        help='Specify model to use. Supported backends: Groq, Ollama, OpenAI. \nDefault: "llama3.2:3b", Examples: ["llama3.2:3b", "llama3.1:8b", "claude3.5", "gpt-4o"]')
    parser.add_argument("--preload", action="store_true", default=False,
                        help="Preload systems like embeddings and other resources.")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Open a web interface for the chat")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debug mode")
    parser.add_argument("--debug-chats", action="store_true", default=False,
                        help="Enable debug windows for chat contexts without full debug logging")
    parser.add_argument("--majority", action="store_true", default=False,
                        help="Use majority voting for responses")
    
    # Parse known arguments and capture any unrecognized ones
    args, unknown_args = parser.parse_known_args()

    if unknown_args or args.help:
        if not args.help:
            print(colored(f"Warning: Unrecognized arguments {' '.join(unknown_args)}.", "red"))
        parser.print_help()
        exit(1)
    
    return args


async def main() -> None:
    print("Environment path: ", g.PROJ_ENV_FILE_PATH)
    load_dotenv(g.PROJ_ENV_FILE_PATH)
    
    args = parse_cli_args()
    print(args)
    
    # Override logging level if debug mode is enabled
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    elif args.debug_chats:
        logger.debug("Chat debug windows enabled")
    
    # Store args in globals
    g.set_args(args)
    
    # Initialize tool manager
    tool_manager = ToolManager()
    if args.debug:
        logger.debug("Tool manager initialized")
    print(colored("\nLoaded tools:", "green"))
    for tool_name in tool_manager.tools.keys():
        print(colored(f"  - {tool_name}", "green"))
    
    # Initialize web server early if GUI mode is enabled
    web_server = None
    if args.gui:
        web_server = WebServer()
        web_server.start(Chat(debug_title="Web Interface Chat"))  # Start with empty chat, will be updated later
        if os.getenv("DEFAULT_FORCE_LOCAL") == get_local_ip() and not args.online:
            args.local = True
            if not args.llm:
                args.llm = "mistral-nemo:12b"
    
    if os.getenv("DEFAULT_FORCE_LOCAL") == get_local_ip() and not args.online:
        args.local = True
        if not args.llm:
            args.llm = "mistral-nemo:12b"
    
    if args.preload:
        print(colored("Preloading resources...", "green"))
        print(colored("Generating atuin-command-history embeddings...", "green"))
        update_cmd_collection()
        print(colored("Generating pdf embeddings for cli-agent directory...", "green"))
        pdf_or_folder_to_database(g.PROJ_DIR_PATH, force_local=args.local)
        print(colored("Preloading complete.", "green"))
        exit(0)
    
    user_input: str = ""
    context_chat: Chat
    
    if args.c:
        context_chat = Chat.load_from_json()
    else:
        context_chat = Chat(debug_title="Main Context Chat")
    
    # Update web server with the actual chat context
    if web_server:
        web_server.chat = context_chat
    
    if (args.voice or args.speak) and context_chat and len(context_chat.messages) > 0:
        # tts last response (when continuing)
        last_response = context_chat.messages[-1][1]
        text_to_speech(last_response)
        print(colored(last_response, 'magenta'))

    if args.edit and args.fixpy and args.presentation:
        from py_agents.assistants import python_error_agent, code_assistant, presentation_assistant
        if args.edit != None: # code edit mode
            pre_chosen_option = ""
            if (args.auto):
                pre_chosen_option = "1"
            code_assistant(context_chat, args.edit, pre_chosen_option)    
        
        if args.fixpy != None:
            python_error_agent(context_chat, args.fixpy)

        if args.presentation != None:
            presentation_assistant(args, context_chat, args.presentation)
    
    prompt_context_augmentation: str = ""
    prompt_context_augmentation: str = ""
    previous_model_key: str | None = None
    
    # remove empty context chat for few_shot_inits
    if len(context_chat.messages) == 0:
        context_chat = None
    
    # Main loop
    while True:
        # remove temporary context augmentation from the last user message
        if context_chat:
            if len(context_chat.messages) > 0:
                if context_chat.messages[-1][0] == Role.USER:
                    context_chat.messages[-1] = (Role.USER, context_chat.messages[-1][1].replace(prompt_context_augmentation, "").strip())
        
        # save the context_chat to a json file
        if context_chat:
            context_chat.save_to_json()
        
        # cli args regenerate last message
        if args.regenerate:
            args.regenerate = False
            if context_chat and len(context_chat.messages) > 1:
                if context_chat.messages[-1][0] == Role.USER:
                    context_chat.messages.pop()
                    user_input = context_chat.messages.pop()[1]
                    print(colored(f"# cli-agent: Regenerating last response.", "green"))
                    print(colored(user_input, "blue"))
        
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
                image_response_str = LlmRouter.generate_completion("Put words to the contents of the image for a blind user.", base64_images=[base64_image], force_local=args.local, silent_reasoning=False)
                prompt_context_augmentation += f'\n\n```vision_{i}\n{image_response_str}\n```'
        
        # get user input from various sources
        if args.message:
            user_input = args.message
            args.message = None
        elif args.voice:
            # Default voice handling
            user_input, _, wake_word_used = listen_microphone()
        else:
            user_input = input(colored("Enter your request: ", 'blue', attrs=["bold"]))
        
        # USER INPUT HANDLING - BEGIN
        if user_input.endswith('--q'):
            print(colored("Exiting...", "red"))
            break
        
        if user_input.endswith("--r") and context_chat:
            if len(context_chat.messages) < 2:
                print(colored(f"# cli-agent: No chat history found, cannot regenerate last response.", "red"))
                continue
            context_chat.messages.pop()
            user_input = context_chat.messages.pop()[1]
            print(colored(f"# cli-agent: KeyBinding detected: Regenerating last response, type (--h) for info", "green"))
        
        if user_input.endswith("--l"):
            user_input = user_input[:-3]
            args.local = not args.local
            print(colored(f"# cli-agent: KeyBinding detected: Local toggled {args.local}, type (--h) for info", "green"))
            continue
        
        if user_input.endswith("--auto"):
            user_input = user_input[:-3]
            args.auto = not args.auto
            print(colored(f"# cli-agent: KeyBinding detected: Auto mode toggled {args.auto}, type (--h) for info", "green"))
            continue
        
        if user_input.endswith("--img"):
            user_input = user_input[:-3]
            print(colored(f"# cli-agent: KeyBinding detected: Starting ScreenCapture, type (--h) for info", "green"))
            args.image = True
            continue
        
        if user_input.startswith("--llm"):
            user_input = input(colored("Enter the llm to use (phi3.5:3.8b, claude3.5, gpt-4o), or leave empty for automatic: ", 'blue'))
            if user_input:
                args.llm = user_input
            else:
                args.llm = None
            user_input = ""
            print(colored(f"# cli-agent: KeyBinding detected: LLM set to {args.llm}, type (--h) for info", "green"))
            continue
        
        if user_input.endswith("--maj"):
            args.majority = True    
            print(colored(f"# cli-agent: KeyBinding detected: Running majority response assistant, type (--h) for info", "green"))
            continue
        
        if user_input.endswith("--i"):
            previous_model_key = args.llm
            args.llm = g.CURRENT_MOST_INTELLIGENT_MODEL_KEY
            if previous_model_key:
                args.llm = previous_model_key
                previous_model_key = None
                print(colored(f"# cli-agent: KeyBinding detected: Disabled the current most intelligent model, now using: {args.llm}, type (--h) for info", "green"))
            else:    
                print(colored(f"# cli-agent: KeyBinding detected: Enabled the current most intelligent model: {args.llm}, type (--h) for info", "green"))
        
        if "--debug" in user_input:
            print(colored(f"# cli-agent: KeyBinding detected: Debug information:", "green"))
            context_chat.print_chat()
        
        if user_input.endswith("--m"):
            print(colored("Enter your multiline input. Type '--f' on a new line when finished.", "blue"))
            lines = []
            while True:
                line = input()
                if line == "--f":
                    break
                lines.append(line)
            user_input = "\n".join(lines)
        
        if "--h" in user_input:
            user_input = user_input[:-3]
            print(figlet_format("cli-agent", font="slant"))
            print(colored(f"""# cli-agent: KeyBinding detected: Display help message:
# cli-agent: KeyBindings:
# cli-agent: --h: Shows this help message.
# cli-agent: --r: Regenerates the last response.
# cli-agent: --l: Toggles local llm host mode.
# cli-agent: --auto: Toggles auto mode.
# cli-agent: --img: Take a screenshot.
# cli-agent: --maj: Run the majority response assistant.
# cli-agent: --llm: Set the language model to use. (Examples: "phi3.5:3.8b", "claude3.5", "gpt-4o")
# cli-agent: --debug: Enable full debug mode with logging
# cli-agent: --debug-chats: Enable debug windows for chat contexts only
# cli-agent: Type 'quit' to exit the program.
""", "yellow"))
            continue
        # USER INPUT HANDLING - END
        
        # AGENT INITIALIZEATION - BEGIN
        if not context_chat:
            context_chat = Chat("You are a scalable agentic AI assistant.")
        # AGENT INITIALIZEATION - END
        
        # Add user message to both context and web interface
        context_chat.add_message(Role.USER, user_input)
        if web_server and web_server.chat:
            web_server.add_message_to_chat(Role.USER, user_input)

        # AGENTIC IN-TURN LOOP - BEGIN
        action_counter = 0  # Initialize counter for consecutive actions
        MAX_ACTIONS = 10    # Maximum number of consecutive actions before forcing a reply
        perform_exit: bool = False

        while True:
            try:
                # Check if we've hit the maximum number of consecutive actions
                if action_counter >= MAX_ACTIONS:
                    # Ask user if to continue or not
                    user_input = input(colored(f"Warning: Agent has performed {MAX_ACTIONS} consecutive actions without replying. Do you want to continue? (Y/n) ", "yellow")).lower()
                    if (user_input == "" or user_input == "y" or "yes" in user_input or "sure" in user_input or "ja" in user_input):
                        MAX_ACTIONS += 3  # Increase the maximum number of consecutive actions
                    else:
                        context_chat.add_message(Role.USER, f"You have performed {MAX_ACTIONS} actions without replying and are being interrupted by the user. Please summarize your progress and respond intelligently to the user.")
                        break

                # Create tool use context
                tool_use_context_chat = context_chat.deep_copy()
                tool_use_context_chat.debug_title = "Tool Use Context Chat"
                if tool_use_context_chat.messages[-1][0] == Role.USER:
                    tool_use_context_chat.messages[-1] = (Role.USER, f"Raw User input: {tool_use_context_chat.messages[-1][1]}\n")
                
                # Get all available tools and their prompts
                tools_prompt = tool_manager.get_tools_prompt()
                
                # Add tool selection guidance
                tool_use_context_chat.add_message(Role.USER, f"""You are an AI assistant with access to several tools. Your primary role is to provide direct, helpful responses while using tools only when strictly necessary. Follow this decision process to response to the user's request. The following tools are available:

{tools_prompt}

These have been provided to you to respond to the user's request.

MANDATORY DECISION TREE:
1. Have I just executed a tool in my previous turn?
YES → Consider:
- Did the tool provide sufficient information to give a complete response?
- Do I need to use the sequential tool to complete the task?
- Should I summarize the tool results for the user?
NO → Continue to 2

2. Does the request require ANY of the following?
- Real-time data (weather, news, ongoing developments)
- Domain specific information (code libraries, facts, expert information)
- System operations, computations or visualizations
- Multiple steps or operations that need to be coordinated
YES → Select appropriate tool (use sequential tool for multi-step operations)
NO → Continue to 3

3. Can I provide a COMPLETE and RELIABLY ACCURATE response using just "reply"?
YES → Use "reply"
NO → Select appropriate tool

4. Which tool best solves the core need?
- For single operations, select the most direct tool
- For multi-step operations, use the sequential tool
- Aim for user satisfaction
{f'''
CONTEXT-AWARE BEHAVIOR:
- You are a voice assistant
- Your name is Nova
- Keep responses concise
- Use conversational tone
''' if args.voice or args.speak else ""}
ERROR PREVENTION:
1. Before tool selection:
- Validate necessity
- Check for simpler solutions
- Verify user intent
- For multi-step operations, use sequential tool instead of chaining individual tools

ANTI-PATTERNS (STRICTLY AVOID):
1. DO NOT use "reply" for:
- Current weather, prices, or any real-time data
- System operations
- Computations
- Visualizations

2. DO NOT chain individual tools directly:
- Use the sequential tool for multi-step operations
- Each operation sequence should be contained within a single sequential tool call
- Break complex operations into logical steps within the sequential tool

RESPONSE FORMAT:
1. Always respond only in the JSON example format
2. Start with a clear reasoning about the user's request inside the "reasoning" field
3. Populate all required tool-specific fields
4. Example:
{{
    "reasoning": "This question can be answered directly without tools because it asks about a historical fact that doesn't require current data.",
    "tool": "reply",
    "reply": "direct response"
}}

FINAL VALIDATION CHECKLIST:
1. Am I certain I have the required information?
2. Can I justify any tool usage beyond "reply"?
3. Can I provide a reliable and complete response without tools?
4. If multiple steps are needed, am I using the sequential tool appropriately?

Remember: You are a helpful assistant first, and a tool user second. Always use tools when needed for accuracy or fulfillment of the user's request.""")

                # Get tool selection response
                try:
                    tool_use_response = LlmRouter.generate_completion(tool_use_context_chat, [] if args.local else [args.llm if args.llm else "llama-3.1-8b-instant"], force_local=args.local, silent_reasoning=True)
                except Exception as e:
                    print(colored(f"Error generating tool selection response: {str(e)}", "red"))
                    if args.debug:
                        traceback.print_exc()
                    break

                # Parse tool selection response
                tool_call_json_list = []  # Initialize list before parsing attempts
                agent_tool_calls = []  # Initialize agent_tool_calls before try block
                try:
                    # First try to find a simple JSON object ending at first }
                    import re
                    simple_json_pattern = r'\{[^}]*\}'
                    simple_json_match = re.search(simple_json_pattern, tool_use_response)
                    if simple_json_match:
                        try:
                            # Try to parse it as JSON
                            simple_json = simple_json_match.group()
                            # Clean the JSON string - replace unescaped newlines and normalize whitespace
                            simple_json = re.sub(r'(?<!\\)\n', '\\n', simple_json)
                            simple_json = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', simple_json)  # Remove control characters
                            parsed_json = json.loads(simple_json)
                            tool_call_json_list = [parsed_json]  # Store the parsed object directly
                        except json.JSONDecodeError:
                            # If simple pattern fails, continue with more complex parsing
                            pass
                    
                    # If simple pattern didn't work, try more complex parsing
                    if not tool_call_json_list:
                        # Use the framework's extract_blocks function
                        tool_use_reponse_blocks = extract_blocks(tool_use_response)
                        # First try to find JSON blocks in code blocks and parse them
                        tool_call_json_list = [json.loads(block[1]) for block in tool_use_reponse_blocks if block[0] in ["json", "first{}"]]

                except Exception as e:
                    print(colored(f"Unexpected error parsing tool selection response: {str(e)}", "red"))
                    if args.debug:
                        traceback.print_exc()
                    break

                if not tool_call_json_list:
                    print(colored("No valid tool calls found in response", "red"))
                    break

                # Assign parsed tools outside try block
                agent_tool_calls = tool_call_json_list

                print(colored(f"Selected tools: {[tool.get('tool', '') for tool in agent_tool_calls]}", "green"))
                
                # Notify web interface about tool selection immediately
                if web_server and web_server.chat:
                    # First show the raw tool selection
                    web_server.add_message_to_chat(Role.ASSISTANT, f"🛠️ Tool selection:\n```json\n{tool_call_json_list[0]}\n```")
                    # Then show the formatted version
                    for tool_call in agent_tool_calls:
                        selected_tool = tool_call.get('tool', '').strip()
                        reasoning = tool_call.get('reasoning', 'No specific reasoning provided.')
                        if selected_tool and selected_tool != "reply":
                            web_server.add_message_to_chat(Role.ASSISTANT, f"🛠️ Using tool: {selected_tool}\n{reasoning}")
                        elif selected_tool == "reply" and 'reply' in tool_call:
                            web_server.add_message_to_chat(Role.ASSISTANT, tool_call['reply'])
                
                should_continue: bool = False
                
                for tool_call in agent_tool_calls:
                    try:
                        selected_tool = tool_call.get('tool', '').strip()
                        reasoning = tool_call.get('reasoning', 'No specific reasoning provided.')

                        if selected_tool and selected_tool != "reply":
                            context_chat.add_message(Role.ASSISTANT, reasoning)
                        
                        print(colored(f"Using tool: {selected_tool}", "green"))
                        print(colored(f"Reasoning: {reasoning}", "cyan"))

                        try:
                            tool = tool_manager.get_tool(selected_tool)()
                        except KeyError:
                            print(colored(f"Tool {selected_tool} not found", "red"))
                            context_chat.add_message(Role.ASSISTANT, f"Error: Tool {selected_tool} not found")
                            continue
                        except Exception as e:
                            print(colored(f"Error initializing tool {selected_tool}: {str(e)}", "red"))
                            context_chat.add_message(Role.ASSISTANT, f"Error initializing {selected_tool}: {str(e)}")
                            continue

                        try:
                            result = await tool.execute(tool_call)
                        except Exception as e:
                            print(colored(f"Error executing tool {selected_tool}: {str(e)}", "red"))
                            if args.debug:
                                traceback.print_exc()
                            context_chat.add_message(Role.ASSISTANT, f"Error executing {selected_tool}: {str(e)}")
                            continue
                        
                        # Process tool result
                        if result.get("status") == "error":
                            error_msg = result.get("error", "Unknown error")
                            print(colored(f"Tool execution error: {error_msg}", "red"))
                            
                            # Create error feedback prompt
                            error_feedback = f"""The previous tool call failed with error: {error_msg}

Previous attempt:
{json.dumps(tool_call, indent=2)}

Please provide a corrected response that fixes this error. Remember:
1. Check parameter names and structure
2. Ensure all required parameters are provided
3. Validate parameter values
4. Follow the tool's example format exactly"""

                            context_chat.add_message(Role.USER, error_feedback)
                            
                            try:
                                error_response = LlmRouter.generate_completion(context_chat, [] if args.local else [args.llm if args.llm else "llama-3.1-8b-instant"], force_local=args.local, silent_reasoning=True)
                                error_response_json = json.loads(error_response)
                                result = await tool.execute(error_response_json)
                            except Exception as e:
                                print(colored(f"Error correcting tool execution: {str(e)}", "red"))
                                if args.debug:
                                    traceback.print_exc()
                                context_chat.add_message(Role.ASSISTANT, f"Error executing {selected_tool}: Failed to correct the error")
                                continue

                            if result.get("status") == "error":
                                # If still failing, give up and inform user
                                error_message = f"Error executing {selected_tool}: {result.get('error')}"
                                context_chat.add_message(Role.ASSISTANT, error_message)
                                if web_server and web_server.chat:
                                    web_server.add_message_to_chat(
                                        Role.ASSISTANT, 
                                        f"❌ Tool execution failed even after correction:\n```\n{error_message}\n\nOriginal attempt:\n{json.dumps(tool_call, indent=2)}\n\nCorrected attempt:\n{json.dumps(error_response_json, indent=2)}\n```"
                                    )
                                should_continue = False
                            else:
                                # Success with corrected parameters
                                if selected_tool == "reply":
                                    context_chat.add_message(Role.ASSISTANT, result["reply"])
                                    if args.voice or args.speak:
                                        text_to_speech(remove_blocks(result["reply"], ["md"]))
                                    should_continue = False
                                elif selected_tool == "goodbye":
                                    if "reply" in result:
                                        context_chat.add_message(Role.ASSISTANT, result["reply"])
                                        if web_server and web_server.chat:
                                            web_server.add_message_to_chat(Role.ASSISTANT, result["reply"])
                                        if args.voice or args.speak:
                                            text_to_speech(remove_blocks(result["reply"], ["md"]))
                                    should_continue = False
                                    perform_exit = True
                                else:
                                    context_chat.add_message(Role.ASSISTANT, f"Tool {selected_tool} executed successfully after correction: {json.dumps(result, indent=2)}")
                                    if web_server and web_server.chat:
                                        web_server.add_message_to_chat(Role.ASSISTANT, f"✅ {selected_tool.capitalize()} tool executed successfully after correction")
                                    should_continue = True
                        else:
                            if selected_tool == "reply":
                                context_chat.add_message(Role.ASSISTANT, result["reply"])
                                if args.voice or args.speak:
                                    text_to_speech(remove_blocks(result["reply"], ["md"]))
                                should_continue = False
                            elif selected_tool == "goodbye":
                                if "reply" in result:
                                    context_chat.add_message(Role.ASSISTANT, result["reply"])
                                    if web_server and web_server.chat:
                                        web_server.add_message_to_chat(Role.ASSISTANT, result["reply"])
                                    if args.voice or args.speak:
                                        text_to_speech(remove_blocks(result["reply"], ["md"]))
                                should_continue = False
                                perform_exit = True
                            elif selected_tool == "web_search":
                                # Add tool response to chat context
                                context_chat.add_message(Role.ASSISTANT, f"Tool {selected_tool} executed: {json.dumps(result, indent=2)}")
                                if web_server and web_server.chat:
                                    web_server.add_message_to_chat(Role.ASSISTANT, f"✅ {selected_tool.capitalize()} tool executed successfully")
                                # Speak the web search results directly
                                if args.voice or args.speak:
                                    search_summary = LlmRouter.generate_completion(f"Summarize these web search results concisely:\n{json.dumps(result, indent=2)}", force_local=args.local, silent_reasoning=True)
                                    text_to_speech(remove_blocks(search_summary, ["md"]))
                                should_continue = False
                            elif selected_tool == "python":
                                context_chat.add_message(Role.ASSISTANT, f"Tool {selected_tool} executed: {json.dumps(result, indent=2)}")
                                if web_server and web_server.chat:
                                    web_server.add_message_to_chat(Role.ASSISTANT, f"✅ {selected_tool.capitalize()} tool executed successfully")
                                should_continue = False
                            else:
                                # Add tool response to chat context
                                context_chat.add_message(Role.ASSISTANT, f"Tool {selected_tool} executed: {json.dumps(result, indent=2)}")
                                if web_server and web_server.chat:
                                    web_server.add_message_to_chat(Role.ASSISTANT, f"✅ {selected_tool.capitalize()} tool executed successfully")
                                should_continue = True
                                action_counter += 1

                    except Exception as e:
                        print(colored(f"Unexpected error during tool execution: {str(e)}", "red"))
                        if args.debug:
                            traceback.print_exc()
                        context_chat.add_message(Role.ASSISTANT, f"An unexpected error occurred: {str(e)}")
                        should_continue = False
                
                if not should_continue:
                    break

            except Exception as e:
                print(colored(f"An unexpected error occurred: {str(e)}", "red"))
                if args.debug:
                    traceback.print_exc()
                break

        # AGENTIC TOOL USE - END

        # Check if we already have a complete response
        has_complete_response = (
            any(tool.get('tool') == 'reply' for tool in agent_tool_calls) or  # Reply tool was used
            any(tool.get('tool') == 'web_search' for tool in agent_tool_calls)  # Web search was performed
        )
        
        # Only generate final response if we don't have a complete response yet
        if not has_complete_response:
            print(colored("# # # RESPONSE # # #", "green"))
            llm_response = LlmRouter.generate_completion(context_chat, [args.llm], force_local=args.local)
            context_chat.add_message(Role.ASSISTANT, llm_response)
            
            if (args.voice or args.speak):
                spoken_response = remove_blocks(llm_response, ["md"])
                text_to_speech(spoken_response)
                        

        if perform_exit:
            exit(0)
        
        # save context once per turn
        if context_chat:
            context_chat.save_to_json()


def run_bash_cmds(bash_blocks: List[str], args) -> Tuple[str, str]:
    command_guard_prompt = f"The following command must follow these guidelines:\n1. After execution it must exit fully automatically.\n2. It must not modify the operating system in major ways, although it is allowed to install trusted apt packages and updated software.\nRespond only with 'Safe' or 'Unsafe'\n\nCommand: ",
    safe_bash_blocks: List[str] = []
    for bash_block in bash_blocks:
        print(colored(bash_block, 'magenta'))
        execute_actions_guard_response = LlmRouter.generate_completion(f"{command_guard_prompt}{bash_blocks}", ["llama-guard"], force_local=args.local, silent_reason="command guard")
        
        execute_actions_automatically: bool = not "unsafe" in execute_actions_guard_response.lower()
        if "S8" in execute_actions_guard_response or "S7" in execute_actions_guard_response : # Ignore: S7 - Privacy, S8 - Intellectual Property
            execute_actions_automatically = True
            
        if execute_actions_automatically:
            safe_bash_blocks.append(bash_block)
        else:
            pass
    
    if len(safe_bash_blocks) > 0 and len(safe_bash_blocks) != len(bash_blocks):
        bash_blocks = safe_bash_blocks
        execute_actions_automatically = True
    
    if (args.auto is None and not execute_actions_automatically) or args.safe:
        if args.voice or args.speak:
            confirmation_response = "Do you want me to execute these steps? (Yes/no)"
            print(colored(confirmation_response, 'yellow'))
            text_to_speech(confirmation_response)
            user_input = listen_microphone(10)[0]
        else:
            user_input = input(colored("Do you want me to execute these steps? (Y/n) ", 'yellow')).lower()
        if not (user_input == "" or user_input == "y" or "yes" in user_input or "sure" in user_input or "ja" in user_input):
            bash_blocks = safe_bash_blocks
    else:
        if not execute_actions_automatically:
            print(colored(f"Command will be executed in {args.auto} seconds, press Ctrl+C to abort.", 'yellow'))
            try:
                for remaining in range(args.auto, 0, -1):
                    sys.stdout.write("\r" + colored(f"Executing in {remaining} seconds... ", 'yellow'))
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\n")  # Ensure we move to a new line after countdown
            except KeyboardInterrupt:
                print(colored("\nExecution aborted by the user.", 'red'))
    return select_and_execute_commands(bash_blocks, args.auto is not None or execute_actions_automatically) 

def extract_reply_content(tool_response: dict) -> str:
    """Extract reply content from a tool response."""
    if tool_response.get('tool') == 'reply' or tool_response.get('tool') == 'python':
        return tool_response.get('reply', '')
    return ''

if __name__ == "__main__":
    asyncio.run(main())
