#!/usr/bin/env python3
# sudo ln -s /home/prob/repos/CLI-Agent/main.py /usr/local/bin/cli-agent

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from termcolor import colored
from prompt_toolkit.widgets import RadioList, Frame, CheckboxList
from typing import List
from typing import List, Callable
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit
from prompt_toolkit.widgets import Frame, CheckboxList

from interface.cls_chat import Chat, Role
from interface.cls_few_shot_factory import FewShotProvider
from interface.cls_ollama_client import OllamaClient
from tooling import run_command
from prompt_toolkit.widgets import Frame, CheckboxList, Label
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
        for snippet_block in bash_blocks[1:]:
            snippet_text = snippet_block.split("```")[0]
            for snippet in snippet_text.split("\n"):
                trimmed_snippet = snippet.strip()
                if trimmed_snippet:
                    bash_snippets.append(trimmed_snippet)
    
    # Extract python snippets
    if len(python_blocks) > 1:
        for snippet_block in python_blocks[1:]:
            snippet_text = snippet_block.split("```")[0]
            for snippet in snippet_text.split("\n"):
                trimmed_snippet = snippet.strip()
                if trimmed_snippet:
                    python_snippets.append(trimmed_snippet)

    # Identify and extract other snippets
    for i in range(1, len(all_blocks), 2):  # Iterate over code blocks, skipping non-code text
        snippet_text = all_blocks[i].split("\n", 1)
        if len(snippet_text) > 1:
            # Determine if the block is neither bash nor python
            if not all_blocks[i].startswith("bash") and not all_blocks[i].startswith("python"):
                for snippet in snippet_text[1].split("\n"):
                    trimmed_snippet = snippet.strip()
                    if trimmed_snippet:
                        other_snippets.append(trimmed_snippet)

    return {"bash": bash_snippets, "python": python_snippets, "other": other_snippets}


def recolor_response(response: str, start_string_sequence:str, end_string_sequence:str, color: str = "red"):
    """
    Prints the response with different colors, with text between
    start_string_sequence and end_string_sequence colored differently.
    Handles multiple instances of such sequences.

    :param response: The entire response string to recolor.
    :param start_string_sequence: The string sequence marking the start of the special color zone.
    :param end_string_sequence: The string sequence marking the end of the special color zone.
    """
    last_end_index = 0
    while True:
        start_index = response.find(start_string_sequence, last_end_index)
        if start_index == -1:
            break  # No more start sequences found

        # Adjust the search for the end by adding the length of the start sequence
        # to ensure we're searching beyond its overlap with the end sequence
        adjusted_start_for_end_search = start_index + len(start_string_sequence)
        end_index = response.find(end_string_sequence, adjusted_start_for_end_search)
        
        if end_index == -1:
            break  # No corresponding end sequence found

        # The actual end_index should include the length of the end string to capture it fully
        end_index += len(end_string_sequence)

        # Print the part of the response before the current start sequence in light blue
        print(colored(response[last_end_index:start_index], 'light_blue'), end="")
        
        # Then, print the part from the start to the end sequence in red
        print(colored(response[start_index:end_index], color), end="")

        # Update last_end_index to search for the next sequence after the current end
        last_end_index = end_index


    # Print any remaining text after the last end sequence in light blue
    return colored(response[last_end_index:], 'light_blue')


def recolor(text: str, start_string_sequence: str, end_string_sequence: str, color: str = "red") -> str:
    """
    Returns the response with different colors, with text between
    start_string_sequence and end_string_sequence colored differently.
    Handles multiple instances of such sequences.

    :param response: The entire response string to recolor.
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
            # Append the rest of the response if no more start sequences found
            colored_response += colored(text[last_end_index:], 'light_blue')
            break

        end_index = text.find(end_string_sequence, start_index + len(start_string_sequence))
        if end_index == -1:
            # Append the rest of the response if no corresponding end sequence found
            colored_response += colored(text[last_end_index:], 'light_blue')
            break

        # Append text before the current start sequence
        colored_response += colored(text[last_end_index:start_index], 'light_blue')

        # Append the special color zone text
        colored_response += colored(text[start_index:end_index + len(end_string_sequence)], color)

        # Update last_end_index for the next iteration
        last_end_index = end_index + len(end_string_sequence)

    return colored_response

def parse_cli_args():
    """Setup and parse CLI arguments, ensuring the script's functionality remains intact."""
    parser = argparse.ArgumentParser(
        description="Enhanced AI CLI-Agent with backend options and more.",
        add_help=False  # Disable automatic help to manually handle unrecognized args
    )
    parser.add_argument("-local", action="store_true",
                        help="Use the local backend for the Ollama language model processing.")
    parser.add_argument("-llm", type=str, nargs='?', const='dolphin-mistral',
                    help='Specify the Ollama model to use. '
                    'Examples: ["dolphin-mixtral","dolphin-mistral","dolphin-phi"]')
    # parser.add_argument("--speak", action="store_true",
    #                     help="Enable text-to-speech for agent responses.")
    parser.add_argument("-c", action="store_true",
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("-cg", action="store_true",
                        help="Continue a global conversation, shared across sessions.")
    parser.add_argument("-f", nargs='?', const=10, default=None, type=int,
                    help="Enables fully automatic command execution without user confirmation. "
                    "Because this is dangerous, any generated command is only executed after a being shown for 10 seconds, by default. "
                    "Add a custom integer to change this delay.")

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


def select_and_execute_commands(commands: List[str]) -> str:
    checkbox_list = CheckboxList(
        values=[(cmd, cmd) for i, cmd in enumerate(commands)],default_values=[cmd for cmd in commands]
    )
    bindings = KeyBindings()

    @bindings.add("q")
    def _quit(event) -> None:
        """Trigger command execution if "Execute Commands" is selected."""
        app.exit(result=checkbox_list.current_values )

    # Instruction message
    instructions = Label(text="Press 'q' to continue.")

    # Define the layout with the instructions
    root_container = HSplit([
        Frame(title="Select commands to execute, in order", body=checkbox_list),
        instructions  # Add the instructions to the layout
    ])
    layout = Layout(root_container)

    # Create the application
    app = Application(layout=layout, key_bindings=bindings, full_screen=False)

    # Run the application and get the selected option(s)
    selected_commands = app.run()
    
    # Execute selected commands and collect their outputs
    outputs = [run_command(cmd) for cmd in selected_commands if cmd in commands]  # Ensure "Execute Commands" is not executed
    
    return "\n".join(outputs)


data_dir = os.path.expanduser('~/.local/share') + "/cli-agent"
os.makedirs(data_dir, exist_ok=True)

def main():
    load_dotenv()
    args = parse_cli_args()
    if not os.getenv('GROQ_API_KEY'):
        print("No Groq Key was found in the .env file. Falling back to Ollama.")
        args.local = True
    # Initialize session and context_chat based on the previous script's structure
    session = OllamaClient()
    context_chat = None
    user_request = ""
    
    if args.cg:
        context_chat = Chat.load_from_json(f"{data_dir}/global_chat.json")
    elif (args.c):
        context_chat = Chat.load_from_json(f"{data_dir}/last_chat.json")

    while True:
        user_request += input(colored("Enter your request: ", 'yellow'))
        if user_request.lower() == 'quit':
            print(colored("Exiting...", "red"))
            break

        # Assuming FewShotProvider and extract_llm_commands are defined as per the initial setup
        if (context_chat):
            context_chat.add_message(Role.USER, user_request)
            llm_response = session.generate_completion(context_chat, args.llm, local=args.local, stream=True)
            context_chat.add_message(Role.ASSISTANT, llm_response)
        else:
            llm_response, context_chat = FewShotProvider.few_shot_SuggestAgentStrategy(user_request, args.llm, local=args.local, stream=True)
        
        if args.cg:
            context_chat.save_to_json(f"{data_dir}/global_chat.json")
        else:
            context_chat.save_to_json(f"{data_dir}/last_chat.json")
            
        
            
            
        snippets = extract_llm_snippets(llm_response)
        
        # llm_response = recolor_response(llm_response, "```bash", "```", "red")
        # print(llm_response)
        
        if not snippets["bash"]:
            continue  # or other logic to handle non-command responses
        
        if args.f is None:
            user_input = input(colored("Do you want me to execute these steps? (Y/n) ", 'yellow'))
            if user_input.lower() == "n":
                continue
        else:
            try:
                print(colored(f"Command will be executed in {args.f} seconds, press Ctrl+C to abort.", 'yellow'))
                for remaining in range(args.f, 0, -1):
                    sys.stdout.write("\r" + colored(f"Executing in {remaining} seconds... ", 'yellow'))
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\n")  # Ensure we move to a new line after countdown
            except KeyboardInterrupt:
                print(colored("\nExecution aborted by the user.", 'red'))
                continue  # Skip the execution of commands and start over
        
        
        user_request = select_and_execute_commands(snippets["bash"] + snippets["other"])
        print(recolor(user_request, "```cmd_output","```", "green"))
        # # Execute commands extracted from the llm_response
        # user_request = ""
        # for snippet in snippets["bash"]:
        #     print(colored(f"Executing command: {snippet}\n" + "# " * 10, 'green'))
        #     user_request += run_command(snippet)
        #     print(colored(user_request, 'green'))
            
        user_request += "\n\n"
        
if __name__ == "__main__":
    main()


