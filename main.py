#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Literal

from dotenv import load_dotenv
from termcolor import colored

from interface.cls_chat import Chat, Role
from interface.cls_few_shot_factory import FewShotProvider
from interface.cls_llm_router import AIStrengths, LlmRouter
from interface.cls_web_scraper import WebScraper
from tooling import run_python_script, select_and_execute_commands, ScreenCapture
import pyperclip
import json
from json import dump, load
from typing import Dict
import json


def extract_llm_snippets(response: str) -> Dict[str, List[str]]:
    bash_snippets: List[str] = []
    other_snippets: List[str] = []
    all_blocks = response.split("```")  # Split by the start of any code block

    bash_blocks = response.split("```bash")
    
    # Extract bash snippets
    if len(bash_blocks) > 1:
        multiline_command = ""
        is_multiline: bool = False
        for snippet_block in bash_blocks[1:]:
            script = snippet_block.split("```")[0]
            
            for line in script.split("\n"):
                line = line.strip()
                if is_multiline or ("echo -e " in line and line.count('"') == 1) or ("<<" in line and "EOF" in line):
                    # Start of a multiline command or continuation
                    is_multiline = True
                    multiline_command += line + "\n"
                    if line.strip() == "EOF" :
                        bash_snippets.append(multiline_command.strip())
                        multiline_command = ""
                        is_multiline = False
                else:
                    trimmed_snippet = line.strip()
                    if trimmed_snippet and not trimmed_snippet.startswith("#"):
                        bash_snippets.append(trimmed_snippet)
    
    # Identify and extract other snippets
    for i in range(1, len(all_blocks), 2):  # Iterate over code blocks, skipping non-code text
        text_snippets = all_blocks[i].split("\n", 1)
        if len(text_snippets) > 1:
            # Determine if the block is neither bash nor python
            for line in text_snippets[1].split("\n"):
                trimmed_snippet = line.strip()
                if trimmed_snippet:
                    other_snippets.append(trimmed_snippet)

    return {"bash": bash_snippets, "other": other_snippets}

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



def parse_cli_args():
    """Setup and parse CLI arguments, ensuring the script's functionality remains intact."""
    parser = argparse.ArgumentParser(
        description="AI CLI-Agent with backend options and more.",
        add_help=False  # Disable automatic help to manually handle unrecognized args
    )
    parser.add_argument("-m", "--message", type=str,
                        help="Enter your first message instantly.")
    parser.add_argument("-l", "--local", action="store_true",
                        help="Use the local Ollama backend for language processing.")
    parser.add_argument("-llm", type=str, nargs='?', const='llama3',
                        help='Specify model to use. Supported backends: Groq, Ollama, OpenAI. Examples: ["dolphin-mixtral","phi3", "llama3]')
    # parser.add_argument("--speak", action="store_true",
    #                     help="Enable text-to-speech for agent responses.")
    parser.add_argument("-i", "--intelligent", action="store_true",
                        help="Use most intelligent available model(s) for processing.")
    parser.add_argument("-o", "--optimize", action="store_true", default=True,
                        help="Enable optimizations.")
    parser.add_argument("-c", action="store_true",
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-a", "--auto", nargs='?', const=10, default=None, type=int,
                        help="""Enables autonomous execution without user confirmation. 
Because this is dangerous, any generated command is only executed after a delay of %(const)s seconds, by default.
Add a custom integer to change this delay.""", metavar="DELAY")
    parser.add_argument("-e", "--edit", type=str, nargs='?', default=None,
                        help="Edits either the file at the specified path, or the contents of the clipboard.")
    
    parser.add_argument("-fp", "--fixpy", type=str,
                        help="Executes the Python file at the specified path and iterates if an error occurs.")
    
    parser.add_argument("-sc", "--saved_chat", type=str,
                        help="Uses the saved_chat as few_shot_prompt.")

    # Parse known arguments and capture any unrecognized ones
    args, unknown_args = parser.parse_known_args()

    # If there are unrecognized arguments, notify the user
    if unknown_args:
        if not "-h" in unknown_args:
            print(colored(f"Warning: Unrecognized arguments {' '.join(unknown_args)}.", "red"))
        # Optionally, you can display help message here
        parser.print_help()
        exit(1)
    
    if not args.llm:
        args.llm = ""

    return args



def main():
    load_dotenv()
    args = parse_cli_args()
    
    print(args)
    instruction: str = None
    
    workspace = os.getcwd()
    vscode_path = os.path.join(workspace, ".vscode")
    config_path = os.path.join(vscode_path, "cli-agent.json")
    print(config_path)
    if (os.path.exists(config_path)):
        with open(config_path, 'r') as file:
            config = json.load(file)
            instruction = config["instruction"]
    else:
        os.makedirs(vscode_path, exist_ok=True)
        instruction = "The assistant provides code completion, error analysis, and code fixes. It can also execute commands and provide explanations. The assistant interacts with the terminal, edits code, and more. The user can type '--h' for help."
        config = {"instruction": instruction}
        with open(config_path, 'w') as file:
            json.dump(config, file, indent="\t")
    
    
    
    if not os.getenv('GROQ_API_KEY') and not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("No Groq- (free!) or other-Api key was found in the '.env' file. Falling back to Ollama locally...")
        args.local = True
    
    if not args.local and args.intelligent:
        if os.getenv('ANTHROPIC_API_KEY'):
            args.llm = "claude-3-5-sonnet"
        elif os.getenv('OPENAI_API_KEY'):
            args.llm = "gpt-4o"
    
    
    
    
    context_chat = Chat(instruction)
    next_prompt = ""
    
    if args.c:
        context_chat = Chat.load_from_json()
        
    if args.edit != None: # code edit mode
        if True: # use clipboard content editing
            clipboard_content = pyperclip.paste()
            if len(clipboard_content) > 10:
                snippets = f"```code\n{clipboard_content}\n```"
                
            # print(colored(f"Editing content at: {args.edit}\n" + "# " * 10, 'green'))

            while True:
                next_prompt = ""
                if args.auto:
                    next_prompt = "Provide the code below in full while adding xml doc comments. Ensure all existing comments remain unchanged or, if appropriate, rephrased minimally. You *must* not modify the code itself at ALL, provide it in full. Focus mainly on adding xml docs to classes and methods:"
                else:
                    next_prompt = input(colored("(--m for multiline) Enter your code-related request: ", 'blue', attrs=["bold"]))
                    
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
                next_prompt = FewShotProvider.few_shot_rephrase(next_prompt, "gpt-4o")
                context_chat.add_message(Role.USER, f"{next_prompt}\n\n{snippets}")
                response = LlmRouter.generate_completion(context_chat, model_key="gpt-4o")
                snippet = extract_single_snippet(response, allow_no_end=True)
                context_chat.add_message(Role.ASSISTANT, response)
                if (len(snippet) > 0):
                    pyperclip.copy(snippet)
                    print(colored("Snippet copied to clipboard.", 'green'))
                elif (args.auto):
                    print(colored("Something went wrong, no snippet could be extracted.", "red"))
            
    if args.fixpy:
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
                
                if fix_iteration > 3:
                    print(colored(f"fixpy summary" + 10 * "# " + f"\n{analysis_amalgam}", 'light_magenta'))
                    user_input = input(colored(f"3 Unsuccessful iterations, continue? (Y/n).", 'yellow')).lower()
                    if user_input != "y" and user_input != "":
                        pyperclip.copy(f"```issue_report\n{analysis_amalgam}\n```\n\n```python\n{fixed_script}\n```")
                        print(colored(f"A summary of the analysis has been copied to the clipboard.", 'green'))
                        exit(1)
                    user_input_insights = input(colored("Do you have any additional insight to enlighten the agent before we continue? (Press enter or type your insights): ", 'yellow'))
                    fix_iteration = 0
                
                if len(context_chat.messages) == 0: # first iteration
                    context_chat.add_message(Role.USER, FewShotProvider.few_shot_rephrase(f"Please analyze the following error step by step and inspect how it can be fixed in the appended script, please do not suggest a fixed implementation instead focus on understanding and explaining the issue step by step.") + f"\n```error\n{error}\n\n```python\n{py_script}\n```")
                elif user_input_insights:
                    context_chat.add_message(Role.USER, FewShotProvider.few_shot_rephrase(f"{user_input_insights}\nAgain, do not suggest a fixed implementation instead for now, solely focus on understanding and explaining the issue step by step.") + f"\n```error\n{error}```")
                else: # default case
                    context_chat.add_message(Role.USER, FewShotProvider.few_shot_rephrase(f"Reflect on your past steps in the light of this new error, what did you miss? Only reflect, combine and infer for now. Do not provide the full reimplementation yet!") + f"\n```error\n{error}")
                error_analysis = LlmRouter.generate_completion(context_chat)
                context_chat.add_message(Role.ASSISTANT, error_analysis)
                
                # FewShotProvider.selfSupervised_few_shot("Does the error ocurr in the file at path: '{args.fixpy}'?", "The response must be 'yes' or 'no'.")
                
                
                
                analysis_amalgam += f"Analysis {fix_iteration}: {error_analysis}\n"
                context_chat.add_message(Role.USER, "Seems reasonable. Now, please provide the fixed script in full.")
                script_fix = LlmRouter.generate_completion(context_chat, "gpt-4o")
                context_chat.add_message(Role.ASSISTANT, script_fix)
                fixed_script = extract_single_snippet(script_fix)
                
                latest_script_path = args.fixpy.replace(".py", f"_patchV{fix_iteration}.py")
                with open(latest_script_path, 'w') as file:
                    file.write(fixed_script)
                    print(colored(f"Iteration {fix_iteration}: Patched script written to {latest_script_path}", 'yellow'))
                fix_iteration += 1
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
            
        

    prompt_context_augmentation: str = ""
    alt_llm = None
    
    while True:
        if args.message:
            next_prompt = args.message
            args.message = None
        else:
            next_prompt = input(colored("Enter your request: ", 'blue', attrs=["bold"]))
        
        if next_prompt.lower().endswith('--q'):
            print(colored("Exiting...", "red"))
            break
        
        
        # Check if the -i parameter is activated
        if next_prompt.endswith("--r"):
            if len(context_chat.messages) < 2:
                print(colored(f"# cli-agent: No chat history found, cannot regenerate last response.", "red"))
                continue
            context_chat.messages.pop()
            next_prompt = context_chat.messages.pop()[1]
            print(colored(f"# cli-agent: KeyBinding detected: Regenerating last response, type (--h) for info", "green"))
            
        if next_prompt.startswith("--p"):
            next_prompt = next_prompt[:-3]
            print(colored(f"# cli-agent: KeyBinding detected: Starting ScreenCapture, type (--h) for info", "green"))
            # screenCapture = ScreenCapture()
            # region_image_base64 = screenCapture.return_captured_region_image()
            # fullscreen_image_base64 = screenCapture.return_fullscreen_image()
            # session.generate_completion("Put words to the contents of the image for a blind user.", "gpt-4o", )
        
        if next_prompt.endswith("--openai"):
            next_prompt = next_prompt[:-3]
            args.intelligent = not args.intelligent
            if (args.intelligent and not args.local):
                alt_llm = args.llm
                args.llm = "gpt-4o"
            elif (args.alt_llm):
                args.llm = alt_llm
                alt_llm = None
            print(colored(f"# cli-agent: KeyBinding detected: Intelligence toggled {args.intelligent}, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--i"):
            next_prompt = next_prompt[:-3]
            args.intelligent = not args.intelligent
            if (args.intelligent and not args.local):
                alt_llm = args.llm
                args.llm = "claude-3-5-sonnet"
            elif (args.alt_llm):
                args.llm = alt_llm
                alt_llm = None
            print(colored(f"# cli-agent: KeyBinding detected: Intelligence toggled {args.intelligent}, type (--h) for info", "green"))
            continue
        
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
        
        if next_prompt.startswith("--f"):
            print(colored(f"# cli-agent: KeyBinding detected: Gathering intel: Sorry, not implemented! WIP, type (--h) for info", "yellow"))
            # search_string = input("Please enter your search string: ")
            # next_prompt = next_prompt[:-3]
            # print(colored(f"# cli-agent: KeyBinding detected: Gathering intel please hold...", "green"))
            # next_prompt = f"I'd like to know more about the term {search_string} in the context of this directory. Please use the following summary to explain it to me:"
            # prompt_context_augmentation += gather_intel(search_string)
        
        if next_prompt.endswith("--a"):
            next_prompt = next_prompt[:-3]
            args.auto = not args.auto
            print(colored(f"# cli-agent: KeyBinding detected: Autonomous command execution toggled {args.auto}, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--s"):
            next_prompt = next_prompt[:-3]
            sliced_chat = context_chat[-3,-2]
            sliced_chat.save_to_json("saved_few_shots.json", True)
            print(colored(f"# cli-agent: KeyBinding detected: Append most recent prompt->response pair to saved_chat for use with --saved_chat, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--so"):
            next_prompt = next_prompt[:-3]
            assert isinstance(context_chat, Chat) # does not fix the blue squiggly line but can't hurt
            context_chat.save_to_json("saved_few_shots.json", False)
            print(colored(f"# cli-agent: KeyBinding detected: Saved chat for use with --saved_chat, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--o"):
            next_prompt = next_prompt[:-3]
            args.optimize = not args.optimize
            print(colored(f"# cli-agent: KeyBinding detected: Optimizer mode toggled {args.optimize}, type (--h) for info", "green"))
            continue
        
        # if next_prompt.endswith("--saved_chat"):
        #     next_prompt = next_prompt[:-3]
        #     args.saved_chat = not args.saved_chat
        #     context_chat = None
        #     print(colored(f"# cli-agent: KeyBinding detected: Saved_chat mode toggled {args.saved_chat}, type (--h) for info", "green"))
        #     continue
        
        if next_prompt.endswith("--h"):
            next_prompt = next_prompt[:-3]
            print(colored(f"""# cli-agent: KeyBinding detected: Display help message:
# cli-agent: KeyBindings:
# cli-agent: --r: Regenerates the last response.
# cli-agent: --p: Add a screenshot to the next prompt.
# cli-agent: --l: Toggles local llm host mode.
# cli-agent: --i: Use the most intelligent model (Claude 3.5 Sonnet).
# cli-agent: --openai: Use gpt-4o.
# cli-agent: --f: Gather understanding of the search string given the working directory as context.
# cli-agent: --a: Toggles autonomous command execution.
# cli-agent: --s: Saves the most recent prompt->response pair.
# cli-agent: --so: Save the current chat for use with --saved_chat.
# cli-agent: --o: Toggles llm optimizer.
# cli-agent: --h: Shows this help message.
# cli-agent: Type 'quit' to exit the program.
"""))
            continue
            
        if prompt_context_augmentation:
            next_prompt +=  f"\n\n```\n{prompt_context_augmentation}\n```" # append contents from previous iteration to the end of the new prompt
            
        # Assuming FewShotProvider and extract_llm_commands are defined as per the initial setup
        if False:
            term = FewShotProvider.few_shot_TextToTerm(next_prompt)
            scraper = WebScraper()
            texts = scraper.search_and_extract_texts(term, 3)
            print(texts)
            summarization = ". ".join(session.generate_completion(
                f"Summarize the most relevant information accurately and densely:\n```txt\n{texts}\n```", 
                "mixtral").split(". ")[1:])
            print(summarization)
        
        if len(context_chat.messages) > 0:
            context_chat.add_message(Role.USER, next_prompt)
            llm_response = LlmRouter.generate_completion(context_chat, args.llm, force_local=args.local)
            context_chat.add_message(Role.ASSISTANT, llm_response)
        else:
            llm_response, context_chat = FewShotProvider.few_shot_CmdAgent(next_prompt, args.llm, force_local=args.local, optimize=args.optimize)
        
        context_chat.save_to_json()

        snippets = extract_llm_snippets(llm_response)
        
        if not snippets["bash"]:
            continue  # or other logic to handle non-command responses
        
        if args.auto is None:
            user_input = input(colored("Do you want me to execute these steps? (Y/n) ", 'yellow')).lower()
            if not (user_input == "" or user_input == "y"):
                continue
        else:
            try:
                print(colored(f"Command will be executed in {args.auto} seconds, press Ctrl+C to abort.", 'yellow'))
                for remaining in range(args.auto, 0, -1):
                    sys.stdout.write("\r" + colored(f"Executing in {remaining} seconds... ", 'yellow'))
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\n")  # Ensure we move to a new line after countdown
            except KeyboardInterrupt:
                print(colored("\nExecution aborted by the user.", 'red'))
                continue  # Skip the execution of commands and start over
        
        prompt_context_augmentation, execution_summarization = select_and_execute_commands(snippets["bash"], args.auto is not None) 
        print(recolor(execution_summarization, "\t#", "successfully", "green"))
        # # Execute commands extracted from the llm_response
        # user_request = ""
        # for snippet in snippets["bash"]:
        #     print(colored(f"Executing command: {snippet}\n" + "# " * 10, 'green'))
        #     user_request += run_command(snippet)
        #     print(colored(user_request, 'green'))
        
if __name__ == "__main__":
    main()


