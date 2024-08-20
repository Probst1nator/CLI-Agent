#!/usr/bin/env python3

import hashlib
from typing import List, Literal, Optional, Tuple
from pyfiglet import figlet_format
import speech_recognition as sr
from dotenv import load_dotenv
from termcolor import colored
import pyperclip
import argparse
import chromadb
import pyaudio
import json
import json
import json
import time
import sys
import os
import os
import re
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from assistants import code_agent, code_assistant, majority_response_assistant, presentation_assistant, search_folder_assistant
from classes.cls_pptx_presentation import PptxPresentation
from tooling import extract_pdf_content, list_files_recursive, run_python_script, select_and_execute_commands, listen_microphone, remove_blocks, split_string_into_chunks, text_to_speech, ScreenCapture
from classes.cls_web_scraper import search_and_scrape, get_github_readme
from classes.ai_providers.cls_ollama_interface import OllamaClient
from classes.cls_llm_router import AIStrengths, LlmRouter
from classes.cls_few_shot_factory import FewShotProvider
from classes.cls_chat import Chat, Role


def extract_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Extract code blocks encased by ``` from a text.
    This function handles various edge cases, including:
    - Nested code blocks
    - Incomplete code blocks
    - Code blocks with or without language specifiers
    - Whitespace and newline variations
    Args:
        text (str): The input text containing code blocks.
    Returns:
        List[Tuple[str, str]]: A list of tuples containing the block type (language) and the block content. If no language is specified, the type will be an empty string.
    """
    
    def find_matching_end(start: int) -> Optional[int]:
        """Find the matching end of a code block, handling nested blocks."""
        stack = 1
        for i in range(start + 3, len(text)):
            if text[i:i+3] == '```':
                if i == 0 or text[i-1] in ['\n', '\r']:
                    stack -= 1
                    if stack == 0:
                        return i
            elif text[i:i+3] == '```' and (i+3 == len(text) or text[i+3] in ['\n', '\r']):
                stack += 1
        return None

    blocks: List[Tuple[str, str]] = []
    start = 0

    while True:
        start = text.find('```', start)
        if start == -1:
            break

        # Check if it's the start of a line
        if start > 0 and text[start-1] not in ['\n', '\r']:
            start += 3
            continue

        end = find_matching_end(start)
        if end is None:
            # Unclosed block, treat the rest of the text as a block
            end = len(text)

        # Extract the block content
        block_content = text[start+3:end].strip()

        # Determine the language (if specified)
        first_newline = block_content.find('\n')
        if first_newline != -1:
            language = block_content[:first_newline].strip()
            content = block_content[first_newline+1:].strip()
        else:
            # Single line block or no language specified
            language = ''
            content = block_content

        blocks.append((language, content))
        start = end + 3

    return blocks


ColorType = Literal['black', 'grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 
                    'light_grey', 'dark_grey', 'light_red', 'light_green', 'light_yellow', 
                    'light_blue', 'light_magenta', 'light_cyan', 'white']

def recolor(text: str, start_string_sequence: str, end_string_sequence: str, color: ColorType = 'red') -> str:
    """
    Returns the response with different colors, with text between
    start_string_sequence and end_string_sequence colored differently.
    Handles multiple instances of such sequences.

    :param text: The entire response string to recolor.
    :param start_string_sequence: The string sequence marking the start of the special color zone.
    :param end_string_sequence: The string sequence marking the end of the special color zone.
    :param color: The color to use for text within the special color zone.
    :return: The colored response string.
    """
    last_end_index = 0
    colored_response = ""
    while True:
        start_index = text.find(start_string_sequence, last_end_index)
        if start_index == -1:
            colored_response += colored(text[last_end_index:], 'light_blue')
            break

        end_index = text.find(end_string_sequence, start_index + len(start_string_sequence))
        if end_index == -1:
            colored_response += colored(text[last_end_index:], 'light_blue')
            break

        colored_response += colored(text[last_end_index:start_index], 'light_blue')
        colored_response += colored(text[start_index:end_index + len(end_string_sequence)], color)
        last_end_index = end_index + len(end_string_sequence)

    return colored_response


def parse_cli_args() -> argparse.Namespace:
    """Setup and parse CLI arguments, ensuring the script's functionality remains intact."""
    
    parser = argparse.ArgumentParser(
        description="AI CLI-Agent with backend options and more.",
        add_help=False  # Disable automatic help to manually handle unrecognized args
    )
    
    parser.add_argument("-m", "--message", type=str,
                        help="Enter your first message instantly.")
    parser.add_argument("-l", "--local", action="store_true",
                        help="Use the local Ollama backend for language processing.")
    parser.add_argument("--llm", type=str, nargs='?', const='phi3:medium-128k', default='',
                        help='Specify model to use. Supported backends: Groq, Ollama, OpenAI. Examples: ["phi3:medium-128k", "phi3:3.8b", "llama3.1"]')
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Enable microphone input and text-to-speech.")
    parser.add_argument("-s", "--speak", action="store_true",
                        help="Enable text-to-speech for agent responses.")
    parser.add_argument("-o", "--optimize", action="store_true", default=True,
                        help="Enable optimizations.")
    parser.add_argument("-c", action="store_true",
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-w", action="store_true",
                        help="Use web search to enhance responses.")
    parser.add_argument("-a", "--auto", nargs='?', const=10, default=None, type=int,
                        help="""Enable autonomous execution without user confirmation. 
Because this is dangerous, any generated command is only executed after a delay of %(const)s seconds, by default.
Add a custom integer to change this delay.""", metavar="DELAY")
    parser.add_argument("-e", "--edit", type=str, nargs='?', default=None,
                        help="Edit either the file at the specified path or the contents of the clipboard.")
    parser.add_argument("-p", "--presentation", nargs='?', default=None, type=str,
                        help="Interactively create a presentation.")    
    parser.add_argument("-u", "--utilise", nargs='?', default=None, type=str,
                        help="Intelligently use a given path/file. (Examples: --utilise 'path/to/file.py' or --utilise 'file.xx' or --utilise 'folder')")
    parser.add_argument("-f", "--find", nargs='?', default=None, type=str,
                        help="Search the directory for something.")
    parser.add_argument("-ma", "--majority", nargs='?', default=None, type=str,
                        help="Generate a response based on the majority of all local models.")
    parser.add_argument("-h", "--help", action="store_true",
                        help="Display this help")
    parser.add_argument("-fp", "--fixpy", type=str,
                        help="Execute the Python file at the specified path and iterate if an error occurs.")
    
    # Parse known arguments and capture any unrecognized ones
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args or args.help:
        if not args.help:
            print(colored(f"Warning: Unrecognized arguments {' '.join(unknown_args)}.", "red"))
        parser.print_help()
        exit(1)
    
    return args



def main():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    # print(env_path)
    load_dotenv(env_path)
    # print(os.path.exists(env_path))
    # print(open(env_path).read())
    # print([item[0] for item in os.environ.items()])
    
    
    args = parse_cli_args()
    print(args)
    working_dir = os.getcwd()
    vscode_path = os.path.join(working_dir, ".vscode")
    config_path = os.path.join(vscode_path, "cli-agent.json")
    context_chat = Chat()
    next_prompt = ""
    
    if args.c:
        context_chat = Chat.load_from_json()
    
    
    if args.interactive:
        # setup microphone
        pyaudio_instance = pyaudio.PyAudio()
        default_microphone_info = pyaudio_instance.get_default_input_device_info()
        microphone_device_index = default_microphone_info['index']
        r = sr.Recognizer()
        source = sr.Microphone(device_index=microphone_device_index)
        if len(context_chat.messages) > 0:
            # tts last response
            last_response = context_chat.messages[-1][1]
            text_to_speech(last_response)
            print(colored(last_response, 'magenta'))
    
    if args.edit != None: # code edit mode
        code_assistant(args, context_chat, args.edit)    
    
    if args.fixpy != None:
        code_agent(args, context_chat, args.fixpy)

    if args.presentation != None:
        presentation_assistant(args, context_chat, args.presentation)
    
    if args.find != None:
        search_folder_assistant(args, context_chat, args.find)
    
    if args.majority != None:
        majority_response_assistant(args, context_chat, args.majority)
    
    prompt_context_augmentation: str = ""
    temporary_prompt_context_augmentation: str = ""
    
    while True:
        if args.message:
            next_prompt = args.message
            args.message = None
        elif args.interactive:
            next_prompt = ""
            while not next_prompt:
                with source:
                    r.adjust_for_ambient_noise(source, 2)
                next_prompt = listen_microphone(source, r)[0]
        else:
            next_prompt = input(colored("Enter your request: ", 'blue', attrs=["bold"]))
        
        if next_prompt.lower().endswith('--q'):
            print(colored("Exiting...", "red"))
            break
        
        if next_prompt.endswith("--r"):
            if len(context_chat.messages) < 2:
                print(colored(f"# cli-agent: No chat history found, cannot regenerate last response.", "red"))
                continue
            context_chat.messages.pop()
            next_prompt = context_chat.messages.pop()[1]
            print(colored(f"# cli-agent: KeyBinding detected: Regenerating last response, type (--h) for info", "green"))
            
        if next_prompt.startswith("--p"):
            next_prompt = next_prompt[:-3]
            print(colored(f"# cli-agent: KeyBinding detected: Starting ScreenCapture, type (--h) for info [NOT IMPLEMENTED]", "green"))
            # screenCapture = ScreenCapture()
            # region_image_base64 = screenCapture.return_captured_region_image()
            # fullscreen_image_base64 = screenCapture.return_fullscreen_image()
            # session.generate_completion("Put words to the contents of the image for a blind user.", "gpt-4o", )
        
        if next_prompt.endswith("--l"):
            next_prompt = next_prompt[:-3]
            args.local = not args.local
            print(colored(f"# cli-agent: KeyBinding detected: Local toggled {args.local}, type (--h) for info", "green"))
            continue
        
        if next_prompt.startswith("--llm"):
            next_prompt = next_prompt.replace("--llm ", "")
            args.llm = next_prompt
            next_prompt = ""
            print(colored(f"# cli-agent: KeyBinding detected: LLM set to {args.llm}, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--a"):
            next_prompt = next_prompt[:-3]
            args.auto = not args.auto
            print(colored(f"# cli-agent: KeyBinding detected: Autonomous command execution toggled {args.auto}, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--w"):
            args.w = True
            print(colored(f"# cli-agent: KeyBinding detected: Websearch enabled, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--m"):
            print(colored("Enter your multiline input. Type '--f' on a new line when finished.", "blue"))
            lines = []
            while True:
                line = input()
                if line == "--f":
                    break
                lines.append(line)
            next_prompt = "\n".join(lines)
            
        
        if next_prompt.endswith("--h"):
            next_prompt = next_prompt[:-3]
            
            print(figlet_format("cli-agent", font="slant"))
            print(colored(f"""# cli-agent: KeyBinding detected: Display help message:
# cli-agent: KeyBindings:
# cli-agent: --r: Regenerates the last response.
# cli-agent: --p: Add a screenshot to the next prompt.
# cli-agent: --l: Toggles local llm host mode.
# cli-agent: --a: Toggles autonomous command execution.
# cli-agent: --m: Multiline input mode.
# cli-agent: --w: Perform a websearch before answering.
# cli-agent: --h: Shows this help message.
# cli-agent: Type 'quit' to exit the program.
""", "yellow"))
            continue
            
        if args.w:
            recent_context_str = context_chat.get_messages_as_string(-3)
            query = FewShotProvider.few_shot_TextToQuery(recent_context_str)
            results = search_and_scrape(query, 2)
            temporary_prompt_context_augmentation += f"\n\n```web_search_result\n{''.join(results)}\n```"
            args.w = False
        
        
        if "https://github.com/" in next_prompt and next_prompt.count("/") >= 4:
            github_repo_url = re.search("(?P<url>https?://[^\s]+)", next_prompt).group("url")
            github_readme = get_github_readme(github_repo_url)
            prompt_context_augmentation += f"\n\nHere's the readme from the github repo:\n```md\n{github_readme}\n```"
        
        next_prompt +=  temporary_prompt_context_augmentation  # appending relevant content to generate better responses
        next_prompt +=  prompt_context_augmentation  # appending relevant content to generate better responses
        
        if len(context_chat.messages) > 1:
            context_chat.add_message(Role.USER, next_prompt)
            llm_response = LlmRouter.generate_completion(context_chat, [args.llm], force_local=args.local)
            context_chat.add_message(Role.ASSISTANT, llm_response)
        else:
            llm_response, context_chat = FewShotProvider.few_shot_CmdAgent(next_prompt, [args.llm], force_local=args.local)
        
        if (args.speak or args.interactive):
            spoken_response = remove_blocks(llm_response, ["md"])
            text_to_speech(spoken_response)
        
        # remove temporary context augmentation from the last user message
        context_chat.messages[-1] = (Role.USER, context_chat.messages[-1][1].replace(temporary_prompt_context_augmentation, ""))
        if os.path.exists(vscode_path):
            with open(config_path, 'w') as file:
                json_content: dict[str,str] = {}
                json_content["history"] = context_chat._to_dict()
                json.dump(json_content, file, indent=2)

        reponse_blocks = extract_blocks(llm_response)
        
        bash_blocks = [block[1] for block in reponse_blocks if block[0] == "bash"]
        if not bash_blocks:
            continue  # or other logic to handle non-command responses
        
        if args.auto is None and not args.interactive:
            # text_to_speech("Do you want me to execute these steps?")
            # if args.interactive:
            #     print(colored("Do you want me to execute these steps? (Yes/no) ", 'yellow'))
            #     user_input = listen_microphone(source, r)[0]
            # else:
            user_input = input(colored("Do you want me to execute these steps? (Y/n) ", 'yellow')).lower()
            if not (user_input == "" or user_input == "y" or "yes" in user_input or "sure" in user_input or "ja" in user_input):
                continue
        else:
            if args.auto is None:
                args.auto = 10
            print(colored(f"Command will be executed in {args.auto} seconds, press Ctrl+C to abort.", 'yellow'))
            try:
                for remaining in range(args.auto, 0, -1):
                    sys.stdout.write("\r" + colored(f"Executing in {remaining} seconds... ", 'yellow'))
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\n")  # Ensure we move to a new line after countdown
            except KeyboardInterrupt:
                print(colored("\nExecution aborted by the user.", 'red'))
                continue  # Skip the execution of commands and start over
        
        temporary_prompt_context_augmentation, execution_summarization = select_and_execute_commands(bash_blocks, args.auto is not None) 
        print(recolor(execution_summarization, "\t#", "successfully", "green"))
        
if __name__ == "__main__":
    main()