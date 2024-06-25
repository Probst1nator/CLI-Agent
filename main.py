#!/usr/bin/env python3
# sudo ln -s /home/prob/repos/CLI-Agent/main.py /usr/local/bin/cli-agent

import argparse
import os
import sys
import time
from typing import Dict, List, Literal

from dotenv import load_dotenv
from termcolor import colored

from interface.cls_chat import Chat, Role
from interface.cls_few_shot_factory import FewShotProvider
from interface.cls_ollama_client import OllamaClient
from interface.cls_web_scraper import WebScraper
from tooling import fetch_search_results, select_and_execute_commands, ScreenCapture, gather_intel

# def setup_sandbox():
#     sandbox_dir = "./sandbox/"
#     if os.path.exists(sandbox_dir):
#         shutil.rmtree(sandbox_dir)
#     os.mkdir(sandbox_dir)

# setup_sandbox()


def extract_llm_snippets(response: str) -> Dict[str, List[str]]:
    bash_snippets: List[str] = []
    python_snippets: List[str] = []
    other_snippets: List[str] = []
    all_blocks = response.split("```")  # Split by the start of any code block

    bash_blocks = response.split("```bash")
    python_blocks = response.split("```python")
    
    
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
                    if '"> ' in line or'" > ' in line or line.strip() == "EOF" :
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


# def recolor_response(response: str, start_string_sequence:str, end_string_sequence:str, color: str = "red"):
#     """
#     Prints the response with different colors, with text between
#     start_string_sequence and end_string_sequence colored differently.
#     Handles multiple instances of such sequences.

#     :param response: The entire response string to recolor.
#     :param start_string_sequence: The string sequence marking the start of the special color zone.
#     :param end_string_sequence: The string sequence marking the end of the special color zone.
#     """
#     last_end_index = 0
#     while True:
#         start_index = response.find(start_string_sequence, last_end_index)
#         if start_index == -1:
#             break  # No more start sequences found

#         # Adjust the search for the end by adding the length of the start sequence
#         # to ensure we're searching beyond its overlap with the end sequence
#         adjusted_start_for_end_search = start_index + len(start_string_sequence)
#         end_index = response.find(end_string_sequence, adjusted_start_for_end_search)
        
#         if end_index == -1:
#             break  # No corresponding end sequence found

#         # The actual end_index should include the length of the end string to capture it fully
#         end_index += len(end_string_sequence)

#         # Print the part of the response before the current start sequence in light blue
#         print(colored(response[last_end_index:start_index], 'light_blue'), end="")
        
#         # Then, print the part from the start to the end sequence in red
#         print(colored(response[start_index:end_index], color), end="")

#         # Update last_end_index to search for the next sequence after the current end
#         last_end_index = end_index


#     # Print any remaining text after the last end sequence in light blue
#     return colored(response[last_end_index:], 'light_blue')


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
    parser.add_argument("-l", "--local", action="store_true",
                        help="Use the local Ollama backend for language processing.")
    parser.add_argument("-llm", type=str, nargs='?', const='llama3',
                        help='Specify model to use. Supported backends: Groq, Ollama, OpenAI. Examples: ["dolphin-mixtral","phi3", "llama3]')
    # parser.add_argument("--speak", action="store_true",
    #                     help="Enable text-to-speech for agent responses.")
    parser.add_argument("-i", "--intelligent", action="store_true",
                        help="Use most intelligent available model(s) for processing.")
    parser.add_argument("-o", "--optimize", action="store_true",
                        help="Enable optimizations.")
    parser.add_argument("-c", action="store_true",
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-a", "--auto", nargs='?', const=10, default=None, type=int,
                        help="""Enables autonomous command execution without user confirmation. 
Because this is dangerous, any generated command is only executed after a delay of %(const)s seconds, by default.
Add a custom integer to change this delay.""", metavar="DELAY")
    parser.add_argument("-e", "--experimental", action="store_true",
                        help="Experimental")

    # Parse known arguments and capture any unrecognized ones
    args, unknown_args = parser.parse_known_args()

    # If there are unrecognized arguments, notify the user
    if unknown_args:
        if not "-h" in unknown_args:
            print(colored(f"Warning: Unrecognized arguments {' '.join(unknown_args)}.", "red"))
        # Optionally, you can display help message here
        parser.print_help()
        exit(1)

    return args

def main():
    load_dotenv()
    args = parse_cli_args()
    
    print(args)
    
    if not os.getenv('GROQ_API_KEY') and not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("No Groq- (free!) or other-Api key was found in the '.env' file. Falling back to Ollama locally...")
        args.local = True
    
    if not args.local and args.intelligent:
        if os.getenv('ANTHROPIC_API_KEY'):
            args.llm = "claude-3-5-sonnet"
        elif os.getenv('OPENAI_API_KEY'):
            args.llm = "gpt-4o"
        
        
    session = OllamaClient()
    context_chat = None
    next_prompt = ""
    
    if args.c:
        context_chat = Chat.load_from_json()

    prompt_context_augmentation: str = ""
    alt_llm = None
    
    while True:
        next_prompt = input(colored("Enter your request: ", 'blue', attrs=["bold"]))
        if next_prompt.lower().endswith('quit'):
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
            screenCapture = ScreenCapture()
            region_image_base64 = screenCapture.return_captured_region_image()
            fullscreen_image_base64 = screenCapture.return_fullscreen_image()
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
        
        if next_prompt.startswith("--f"):
            search_string = input("Please enter your search string: ")
            next_prompt = next_prompt[:-3]
            print(colored(f"# cli-agent: KeyBinding detected: Gathering intel please hold...", "green"))
            next_prompt = f"I'd like to know more about the term {search_string} in the context of this directory. Please use the following summary to explain it to me:"
            prompt_context_augmentation += gather_intel(search_string)
        
        if next_prompt.endswith("--a"):
            next_prompt = next_prompt[:-3]
            args.auto = not args.auto
            print(colored(f"# cli-agent: KeyBinding detected: Autonomous command execution toggled {args.auto}, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--s"):
            next_prompt = next_prompt[:-3]
            sliced_chat = context_chat[-3,-2]
            sliced_chat.save_to_json("saved_few_shots.json", True)
            print(colored(f"# cli-agent: KeyBinding detected: Saved most recent prompt->response pair, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--so"):
            next_prompt = next_prompt[:-3]
            assert isinstance(context_chat, Chat) # does not fix the blue squiggly line but can't hurt
            context_chat.save_to_json("saved_few_shots.json", False)
            print(colored(f"# cli-agent: KeyBinding detected: Wrote chat to saved prompt->response pairs, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--o"):
            next_prompt = next_prompt[:-3]
            args.optimize = not args.optimize
            print(colored(f"# cli-agent: KeyBinding detected: Optimizer mode toggled {args.optimize}, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--e"):
            next_prompt = next_prompt[:-3]
            args.experimental = not args.experimental
            context_chat = None
            print(colored(f"# cli-agent: KeyBinding detected: Experimental mode toggled {args.experimental}, type (--h) for info", "green"))
            continue
        
        if next_prompt.endswith("--h"):
            next_prompt = next_prompt[:-3]
            print(colored(f"""# cli-agent: KeyBinding detected: Display help message:
# cli-agent: KeyBindings:
# cli-agent: --r: Regenerates the last response.
# cli-agent: --p: Add a screenshot to the next prompt.
# cli-agent: --l: Toggles local llm host mode.
# cli-agent: --i: Use the most intelligent model (Claude 3.5 Sonnet).
# cli-agent: --openai: Use GPT4o.
# cli-agent: --f: Gather understanding of the search string given the working directory as context.
# cli-agent: --a: Toggles autonomous command execution.
# cli-agent: --s: Saves the most recent prompt->response pair.
# cli-agent: --so: Overwrite the saved prompt->response pairs with this chat.
# cli-agent: --o: Toggles llm optimizer.
# cli-agent: --e: Toggles experimental mode.
# cli-agent: --h: Shows this help message.
# cli-agent: Type 'quit' to exit the program.
"""))
            continue
            
        if prompt_context_augmentation:
            next_prompt +=  f"\n\n'''\n{prompt_context_augmentation}\n'''" # append contents from previous iteration to the end of the new prompt
            
        # Assuming FewShotProvider and extract_llm_commands are defined as per the initial setup
        if False:
            term = FewShotProvider.few_shot_TextToTerm(next_prompt)
            scraper = WebScraper()
            texts = scraper.search_and_extract_texts(term, 3)
            print(texts)
            summarization = ". ".join(session.generate_completion(
                f"Summarize the most relevant information accurately and densely:\n'''txt\n{texts}\n'''", 
                "mixtral").split(". ")[1:])
            print(summarization)
        
        if context_chat:
            context_chat.add_message(Role.USER, next_prompt)
            llm_response = session.generate_completion(context_chat, args.llm, local=args.local, stream=True)
            context_chat.add_message(Role.ASSISTANT, llm_response)
        else:
            if (args.experimental):
                llm_response, context_chat = FewShotProvider.few_shot_CmdAgentExperimental(next_prompt, args.llm, local=args.local, optimize=args.optimize, stream=True)
            else:
                llm_response, context_chat = FewShotProvider.few_shot_CmdAgent(next_prompt, args.llm, local=args.local, optimize=args.optimize, stream=True)
            # llm_response, context_chat = FewShotProvider.few_shot_FunctionCallingAgent(user_request, args.llm, local=args.local, stream=True)
        
        context_chat.save_to_json()

            
        snippets = extract_llm_snippets(llm_response)
        
        if not snippets["bash"]:
            continue  # or other logic to handle non-command responses
        
        if args.auto is None:
            user_input = input(colored("Do you want me to execute these steps? (Y/n) ", 'yellow'))
            if not (user_input == "" or user_input.lower() == "y"):
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


