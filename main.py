#!/usr/bin/env python3

import hashlib
from pathlib import Path
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

from assistants import code_agent, code_assistant, git_message_generator, majority_response_assistant, presentation_assistant, search_folder_assistant
from classes.cls_pptx_presentation import PptxPresentation
from tooling import extract_blocks, extract_pdf_content, list_files_recursive, recolor, run_python_script, select_and_execute_commands, listen_microphone, remove_blocks, split_string_into_chunks, text_to_speech, ScreenCapture, update_cmd_collection
from classes.cls_web_scraper import search_and_scrape, get_github_readme
from classes.ai_providers.cls_ollama_interface import OllamaClient
from classes.cls_llm_router import AIStrengths, LlmRouter
from classes.cls_few_shot_factory import FewShotProvider
from classes.cls_chat import Chat, Role
from globals import g


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
    parser.add_argument("--llm", nargs='?', const='phi3:medium-128k', type=str,
                        help='Specify model to use. Supported backends: Groq, Ollama, OpenAI. Examples: ["phi3:medium-128k", "phi3:3.8b", "llama3.1"]')
    parser.add_argument("-stt", "--speech_to_text", action="store_true",
                        help="Enable microphone input and text-to-speech. (Wip: please split this up)")
    parser.add_argument("-s", "--speak", action="store_true",
                        help="Enable text-to-speech for agent responses.")
    parser.add_argument("-c", action="store_true",
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-w", action="store_true",
                        help="Use web search to enhance responses.")
    parser.add_argument("-a", "--auto", nargs='?', const=10, type=int,
                        help="""Enable autonomous execution without user confirmation. 
Because this is dangerous, any generated command is only executed after a delay of %(const)s seconds, by default.
Add a custom integer to change this delay.""", metavar="DELAY")
    parser.add_argument("-e", "--edit", nargs='?', const="", type=str,
                        help="Edit either the file at the specified path or the contents of the clipboard.")
    parser.add_argument("-p", "--presentation", nargs='?', const="", type=str,
                        help="Interactively create a presentation.")    
    parser.add_argument("-u", "--utilise", nargs='?', const="", type=str,
                        help="Intelligently use a given path/file. (Examples: --utilise 'path/to/file.py' or --utilise 'file.xx' or --utilise 'folder')")
    parser.add_argument("-f", "--find", nargs='?', const="", type=str,
                        help="Search the directory for something.")
    parser.add_argument("-ma", "--majority", nargs='?', const="", type=str,
                        help="Generate a response based on the majority of all local models.")
    parser.add_argument("-fp", "--fixpy", type=str,
                        help="Execute the Python file at the specified path and iterate if an error occurs.")
    parser.add_argument("--preload", action="store_true",
                        help="Preload systems like embeddings and other resources.")
    parser.add_argument("--git_message_generator", nargs='?', const="", type=str,
                        help="Will rework all messages done by the user on the current branch. Enter the projects theme for better results.")
    parser.add_argument("-h", "--help", action="store_true",
                        help="Display this help")
    
    # Parse known arguments and capture any unrecognized ones
    args, unknown_args = parser.parse_known_args()
    
    if unknown_args or args.help:
        if not args.help:
            print(colored(f"Warning: Unrecognized arguments {' '.join(unknown_args)}.", "red"))
        parser.print_help()
        exit(1)
    
    return args



def main():
    print("Environment path: ", g.PROJ_ENV_FILE_PATH)
    load_dotenv(g.PROJ_ENV_FILE_PATH)
    
    args = parse_cli_args()
    print(args)
    
    if args.preload:
        print(colored("Preloading resources...", "green"))
        print(colored("Generating atuin-command-history embeddings...", "green"))
        update_cmd_collection()
        exit(0)
    
    context_chat = Chat()
    next_prompt = ""
    
    if args.c:
        context_chat = Chat.load_from_json()
    
    if args.speech_to_text:
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
    
    if args.git_message_generator:
        git_message_generator(args, context_chat, args.git_message_generator)
    
    
    prompt_context_augmentation: str = ""
    temporary_prompt_context_augmentation: str = ""
    
    while True:
        if args.message:
            next_prompt = args.message
            args.message = None
        elif args.speech_to_text:
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
# cli-agent: --llm: Set the language model to use. (Examples: "phi3:medium-128k", "claude3.5", "gpt-4o")
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
        
        if (args.speak or args.speech_to_text):
            spoken_response = remove_blocks(llm_response, ["md"])
            text_to_speech(spoken_response)
        
        # remove temporary context augmentation from the last user message
        context_chat.messages[-1] = (Role.USER, context_chat.messages[-1][1].replace(temporary_prompt_context_augmentation, ""))

        # save the context_chat to a json file
        if os.path.exists(g.PROJ_VSCODE_DIR_PATH):
            with open(g.PROJ_CONFIG_FILE_PATH, 'w') as file:
                json_content: dict[str,str] = {}
                json_content["history"] = context_chat._to_dict()
                json.dump(json_content, file, indent=2)

        reponse_blocks = extract_blocks(llm_response)
        
        bash_blocks = [block[1] for block in reponse_blocks if block[0] == "bash"]
        if not bash_blocks:
            continue  # or other logic to handle non-command responses
        
        if args.auto is None and not args.speech_to_text:
            # text_to_speech("Do you want me to execute these steps?")
            # if args.speech_to_text:
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