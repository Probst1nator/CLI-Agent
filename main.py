#!/usr/bin/env python3

from typing import Any, Dict, List, Literal, Optional, Tuple
from pyfiglet import figlet_format
import speech_recognition as sr
from dotenv import load_dotenv
from termcolor import colored
from json import dump, load
from pathlib import Path
from typing import Dict
import pyperclip
import argparse
import logging
import pyaudio
import json
import json
import json
import time
import sys
import os
import re

from classes.cls_pptx_presentation import PptxPresentation
from tooling import run_python_script, select_and_execute_commands, listen_microphone, remove_blocks, text_to_speech, ScreenCapture
from classes.cls_web_scraper import search_and_scrape, get_github_readme
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

def extract_single_snippet(response: str, allow_no_end: bool = False) -> str:
    start_index = response.find("```")
    if start_index != -1:
        start_line_end = response.find("\n", start_index)
        end_index = response.rfind("```")
        if end_index != -1 and end_index > start_index:
            # Find the end of the start line and the start of the end line
            end_line_start = response.rfind("\n", start_index, end_index)
            if start_line_end != -1 and end_line_start != -1:
                return response[start_line_end + 1:end_line_start]
        elif allow_no_end:
            return response[start_line_end + 1:]
    return ""


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
    parser.add_argument("--llm", type=str, nargs='?', const='phi3:3.8b', default='',
                        help='Specify model to use. Supported backends: Groq, Ollama, OpenAI. Examples: ["phi3:3.8b", "llama3.1"]')
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
    parser.add_argument("-fp", "--fixpy", type=str,
                        help="Execute the Python file at the specified path and iterate if an error occurs.")
    
    # Parse known arguments and capture any unrecognized ones
    args, unknown_args = parser.parse_known_args()

    # If there are unrecognized arguments, notify the user
    if unknown_args and '-h' not in unknown_args:
        print(colored(f"Warning: Unrecognized arguments {' '.join(unknown_args)}.", "red"))
        parser.print_help()
        exit(1)
    
    return args



def code_assistant(args: argparse.Namespace, context_chat: Chat, snippets: str):
    while True:
        next_prompt = ""
        if args.auto:
            abstract_code_overview = LlmRouter.generate_completion("Please explain the below code step by step, provide a short abstract overview of its stages.\n\n" + snippets,  preferred_model_keys=["llama-3.1-70b-versatile"], strength=AIStrengths.STRONG, force_free=True)
            if len(abstract_code_overview)/4 >= 2048:
                abstract_code_overview = LlmRouter.generate_completion(f"Summarize this code analysis, retaining the most important features and minimal details:\n{abstract_code_overview}",  preferred_model_keys=["llama-3.1-70b-versatile"], strength=AIStrengths.STRONG, force_free=True)
            next_prompt = "Provide the code below in full while adding xml doc comments. Ensure all existing comments remain unchanged or, if appropriate, rephrased minimally. You *must* not modify the code itself at ALL, provide it in full. Focus mainly on adding xml docs to classes and methods."
            next_prompt += f"\nTo reduce the inherent complexity and enhance your understanding of the code, I've created an abstract overview of it, please use it to improve your contextual understanding of the code: \n{abstract_code_overview}"
        else:
            print(colored("Please choose an option:", 'cyan', attrs=["bold"]))
            print(colored("1. Add xml-docs", 'yellow'))
            print(colored("2. Refactor", 'yellow'))
            print(colored("3. Explain", 'yellow'))
            # print(colored("4. Self-Supervise", 'yellow'))
            print(colored("Write the prompt yourself", 'yellow') + " " + colored("(Use --m for multiline input)", 'grey'))
            user_input = input(colored("Enter your choice: ", 'blue'))
            if user_input == "1":
                abstract_code_overview = LlmRouter.generate_completion("Please explain the code step by step, provide an abstract high level overview of its stages.\n\n" + snippets,  preferred_model_keys=["llama-3.1-70b-versatile"], strength=AIStrengths.STRONG, force_free=True)
                if len(abstract_code_overview)/4 >= 2048:
                    abstract_code_overview = LlmRouter.generate_completion(f"Summarize this code analysis, retaining the most important features and minimal details:\n{abstract_code_overview}",  preferred_model_keys=["llama-3.1-70b-versatile"], strength=AIStrengths.STRONG, force_free=True)
                next_prompt = "Provide the code in full while adding xml doc comments. Ensure all existing comments remain unchanged or, if appropriate, rephrased minimally. You *must* not modify the code itself at ALL, provide it in full. Focus mainly on adding xml docs to classes and methods."
                next_prompt += f"\nTo reduce the codes inherent complexity and enhance your understanding, I've created an abstract overview of the code, please use it to improve your contextual understanding of the code: \n{abstract_code_overview}"
            elif user_input == "2":
                next_prompt = "Please refactor the code, ensuring it remains functionally equivalent. You may change the code structure, variable names, and comments."
            elif user_input == "3":
                next_prompt = "Please explain the code, providing a concise explanation of the code's functionality and use."
            else:
                next_prompt = user_input

            if next_prompt == "--m":
                print(colored("Enter your multiline input. Type '--f' on a new line when finished.", "blue"))
                lines = []
                while True:
                    line = input()
                    if line == "--f":
                        break
                    lines.append(line)
                next_prompt = "\n".join(lines)      
        args.auto = False
        next_prompt = FewShotProvider.few_shot_rephrase(next_prompt, ["llama-3.1-70b-versatile", "gpt-4o", "claude-3-5-sonnet"])

        # only add the code to the context chat once
        if not any(snippets in message for message in context_chat.messages):
            next_prompt += f"\n\n{snippets}"
            
        context_chat.add_message(Role.USER, next_prompt)
        
        response = LlmRouter.generate_completion(context_chat, preferred_model_keys=["llama-3.1-70b-versatile", "gpt-4o", "claude-3-5-sonnet"], strength=AIStrengths.STRONG)
        snippet = extract_single_snippet(response, allow_no_end=True)
        context_chat.add_message(Role.ASSISTANT, response)
        if (len(snippet) > 0):
            pyperclip.copy(snippet)
            print(colored("Snippet copied to clipboard.", 'green'))
        elif (args.auto):
            print(colored("Something went wrong, no snippet could be extracted.", "red"))


def code_agent(args: argparse.Namespace, context_chat: Chat):
    latest_script_path = args.fixpy
    fix_iteration = 0
    while True:
        print(colored(f"Executing Python file at: {latest_script_path}\n" + "# " * 10, 'green'))
        py_script = ""
        with open(latest_script_path, 'r') as file:
            py_script = file.read()
        output, error = run_python_script(latest_script_path)
        analysis_amalgam = ""
        user_input_insights = ""
        if error:
            print(colored(f"Error: {error}", 'red'))
            
            if len(context_chat.messages) == 0: # first iteration
                context_chat.add_message(Role.USER, FewShotProvider.few_shot_rephrase(f"Please analyze the following error step by step and inspect how it can be fixed in the appended script, please do not suggest a fixed implementation instead focus on understanding and explaining the issue step by step.") + f"\n```error\n{error}\n\n```python\n{py_script}\n```")
            elif user_input_insights:
                context_chat.add_message(Role.USER, FewShotProvider.few_shot_rephrase(f"{user_input_insights}\nAgain, do not suggest a fixed implementation instead for now, solely focus on understanding and explaining the issue step by step.") + f"\n```error\n{error}```")
            else: # default case
                context_chat.add_message(Role.USER, FewShotProvider.few_shot_rephrase(f"Reflect on your past steps in the light of this new error, what did you miss? Only reflect, combine and infer for now. Do not provide the full reimplementation yet!") + f"\n```error\n{error}")
            error_analysis = LlmRouter.generate_completion(context_chat)
            context_chat.add_message(Role.ASSISTANT, error_analysis)
            
            
            analysis_amalgam += f"Analysis {fix_iteration}: {error_analysis}\n"
            context_chat.add_message(Role.USER, "Seems reasonable. Now, please provide the fixed script in full.")
            script_fix = LlmRouter.generate_completion(context_chat, preferred_model_keys=["llama-3.1-70b-versatile", "gpt-4o", "claude-3-5-sonnet"])
            context_chat.add_message(Role.ASSISTANT, script_fix)
            fixed_script = extract_single_snippet(script_fix)
            
            latest_script_path = args.fixpy.replace(".py", f"_patchV{fix_iteration}.py")
            with open(latest_script_path, 'w') as file:
                file.write(fixed_script)
                print(colored(f"Iteration {fix_iteration}: Patched script written to {latest_script_path}", 'yellow'))
            fix_iteration += 1
            if fix_iteration > 3:
                print(colored(f"fixpy summary" + 10 * "# " + f"\n{analysis_amalgam}", 'light_magenta'))
                user_input = input(colored(f"3 Unsuccessful iterations, continue? (Y/n).", 'yellow')).lower()
                if user_input != "y" and user_input != "":
                    pyperclip.copy(f"```issue_report\n{analysis_amalgam}\n```\n\n```python\n{fixed_script}\n```")
                    print(colored(f"A summary of the analysis has been copied to the clipboard.", 'green'))
                    exit(1)
                user_input_insights = input(colored("Do you have any additional insight to enlighten the agent before we continue? (Press enter or type your insights): ", 'yellow'))
                fix_iteration = 0
            continue
        else:
            print(colored(f"Execution success!\n```output\n{output}\n```", 'green'))
            user_input = input(colored("Do you wish to overwrite the original script with the successfully executed version? (Y/n) ", 'yellow')).lower()
            if user_input == "y" or user_input == "":
                with open(args.fixpy, 'w') as file:
                    file.write(fixed_script)
                    print(colored(f"Script overwritten with patched version.", 'green'))
            user_input = input(colored("Do you wish to remove the other deprecated patched versions? (Y/n) ", 'yellow')).lower()
            if user_input == "y" or user_input == "":
                for i in range(fix_iteration):
                    os.remove(args.fixpy.replace(".py", f"_patchV{i}.py"))
                    print(colored(f"Removed deprecated patched version {i}.", 'green'))
            exit(0)

def presentation_assistant(args: argparse.Namespace, context_chat: Chat, user_input:str = ""):
    if not user_input:
        print(colored("Please enter your presentation topic(s) and supplementary data if present. *Multiline mode* Type '--f' on a new line when finished.", "magenta"))
        lines = []
        while True:
            line = input()
            if line == "--f":
                break
            lines.append(line)
        user_input = "\n".join(lines)
    
    
    rephrased_user_input = FewShotProvider.few_shot_rephrase(user_input, preferred_model_keys=[args.llm], force_local=args.local)
    decomposition_prompt = FewShotProvider.few_shot_rephrase("Please decompose the following into 3-6 subtopics and provide step by step explanations + a very short discussion:", preferred_model_keys=[args.llm], force_local=args.local)
    presentation_details = LlmRouter.generate_completion(f"{decomposition_prompt}: '{rephrased_user_input}'", strength=AIStrengths.STRONG, use_cache=False, preferred_model_keys=[args.llm], force_local=args.local)
    
    chat, response = FewShotProvider.few_shot_textToPresentation(presentation_details, preferred_model_keys=[args.llm], force_local=args.local)
    while True:
        while True:
            try:
                presentation_json = response.split("```")[1].split("```")[0]
                presentation = PptxPresentation.from_json(presentation_json)
                presentation.save()
                break
            except Exception as e:
                chat.add_message(Role.USER, "Your json object did not follow the expected format, please try again.\nError: " + str(e))
                response = LlmRouter.generate_completion(chat, strength=AIStrengths.STRONG, use_cache=False, preferred_model_keys=[args.llm], force_local=args.local)
                chat.add_message(Role.ASSISTANT, response)
                
        print(colored("Presentation saved.", 'green'))
        print(colored("Please choose an option:", 'cyan', attrs=["bold"]))
        print(colored("1. Add details", 'yellow'))
        print(colored("2. Regenerate", 'yellow'))
        print(colored("3. Add Images, this may take a while...", 'yellow'))
        print(colored("Write the prompt yourself", 'yellow') + " " + colored("(Use --m for multiline input)", 'grey'))
        user_input = input(colored("Enter your choice: ", 'blue'))
        if user_input == "1":
            add_details_prompt = FewShotProvider.few_shot_rephrase(f"Please think step by step to add relevant/ missing details to the following topic: {presentation_details}", preferred_model_keys=[args.llm])
            suggested_details = LlmRouter.generate_completion(f"{add_details_prompt} {presentation_details}", strength=AIStrengths.STRONG, preferred_model_keys=[args.llm], force_local=args.local)
            next_prompt = f"Please add the following details to the presentation: \n{suggested_details}"
        elif user_input == "2":
            next_prompt = "I am unhappy with your suggested presentation, please try again."
        else:
            next_prompt = user_input
            
        next_prompt = FewShotProvider.few_shot_rephrase(next_prompt, preferred_model_keys=[args.llm], force_local=args.local)
        chat.add_message(Role.USER, next_prompt)
        response = LlmRouter.generate_completion(chat, strength=AIStrengths.STRONG, preferred_model_keys=[args.llm], force_local=args.local)

def main():
    load_dotenv()
    args = parse_cli_args()
    print(args)
    working_dir = os.getcwd()
    vscode_path = os.path.join(working_dir, ".vscode")
    config_path = os.path.join(vscode_path, "cli-agent.json")
    context_chat = Chat()
    next_prompt = ""
    
    if os.path.exists(vscode_path):
        log_file_path = os.path.join(vscode_path, 'cli-agent.log')
    else:
        usr_dir = os.path.expanduser('~/.local/share')
        logs_path = os.path.join(usr_dir,'cli-agent','logs')
        os.makedirs(logs_path, exist_ok=True)
        log_file_path = os.path.join(logs_path, 'cli-agent.log')
    logging.basicConfig(level=logging.CRITICAL, filename=log_file_path)
    print(colored(f"Log is being written to {log_file_path}", 'yellow'))
    
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
        
        if args.edit:
            with open(args.edit, 'r') as file:
                file_ending = os.path.splitext(args.edit)[1]
                snippets = f"```{file_ending}\n{file.read()}\n```"
        try:
            print(colored("Add clipboard? Press (ctrl+c) to add.","yellow"))
            for remaining in range(3, 0, -1):
                sys.stdout.write("\r" + colored(f"Ignoring clipboard in {remaining}s... ", 'yellow'))
                sys.stdout.flush()
                time.sleep(1)
            sys.stdout.write("\n")  # Ensure we move to a new line after countdown
        except KeyboardInterrupt:
            clipboard_content = pyperclip.paste()
            snippets += f"```userClipboard\n{clipboard_content}\n```"

        code_assistant(args, context_chat, snippets)
        
            
    if args.fixpy:
        code_agent(args, context_chat)

    if args.presentation:
        presentation_assistant(args, context_chat, args.presentation)
        
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
            query = FewShotProvider.few_shot_TextToQuery(next_prompt)
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