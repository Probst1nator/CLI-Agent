#!/usr/bin/env python3
# sudo ln -s /home/prob/repos/CLI-Agent/main.py /usr/local/bin/cli-agent

import argparse
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List

from termcolor import colored

from interface.cls_chat import Chat, Role
from interface.cls_few_shot_factory import FewShotProvider
from interface.cls_ollama_client import OllamaClient
from tooling import run_command

# def setup_sandbox():
#     sandbox_dir = "./sandbox/"
#     if os.path.exists(sandbox_dir):
#         shutil.rmtree(sandbox_dir)
#     os.mkdir(sandbox_dir)

# setup_sandbox()


def extract_llm_commands(response: str) -> List[str]:
    cmds: List[str] = []
    if "```bash" not in response:
        return cmds
    for cmd_block in response.split("```bash")[1:]:  # Skip the first chunk as it's before the first marker
        cmd_text = cmd_block.split("```")[0]  # Extract command text before the closing marker
        for cmd in cmd_text.split("\n"):
            trimmed_cmd = cmd.strip()
            if trimmed_cmd:  # Add command if it's not empty or whitespace
                cmds.append(trimmed_cmd)
    return cmds




def parse_cli_args():
    """Setup and parse CLI arguments, ensuring the script's functionality remains intact."""
    parser = argparse.ArgumentParser(
        description="Enhanced AI CLI-Agent with backend options and more.",
        add_help=False  # Disable automatic help to manually handle unrecognized args
    )
    parser.add_argument("--local", action="store_true",
                        help="Use the local backend for the Ollama language model processing.")
    parser.add_argument("--llm", type=str, default="mixtral",
                        help="Specify the Ollama model to use.", 
                        choices=["dolphin-mixtral","dolphin-mistral","dolphin-phi","dolphincoder", "mixtral"])
    # parser.add_argument("--speak", action="store_true",
    #                     help="Enable text-to-speech for agent responses.")
    parser.add_argument("-c", action="store_true",
                        help="Continue the last conversation, retaining its context.")
    parser.add_argument("--cg", action="store_true",
                        help="Continue a global conversation, shared across sessions.")
    # Parse known arguments and capture any unrecognized ones
    args, unknown_args = parser.parse_known_args()

    # If there are unrecognized arguments, notify the user
    if unknown_args:
        print(colored(f"Warning: Unrecognized arguments {' '.join(unknown_args)}.", "red"))
        # Optionally, you can display help message here
        parser.print_help()

    return args

script_directory = "/home/prob/repos/CLI-Agent/"
print(script_directory)
def main():
    args = parse_cli_args()
    # Initialize session and context_chat based on the previous script's structure
    session = OllamaClient()
    context_chat = None
    user_request = ""
    
    if args.cg:
        context_chat = Chat.load_from_json(f"{script_directory}/cache/global_chat.json")
    elif (args.c):
        context_chat = Chat.load_from_json(f"{script_directory}/cache/last_chat.json")
    
    while True:
        user_request += input(colored("Enter your request: ", 'yellow'))
        if user_request.lower() == 'quit':
            print(colored("Exiting...", "red"))
            break

        # Assuming FewShotProvider and extract_llm_commands are defined as per the initial setup
        if (context_chat):
            context_chat.add_message(Role.USER, user_request)
            llm_response = session.generate_completion(context_chat, args.llm, local=args.local)
            context_chat.add_message(Role.ASSISTANT, llm_response)
        else:
            llm_response, context_chat = FewShotProvider.few_shot_SuggestAgentStrategy(user_request, args.llm, local=args.local)
        
        if args.cg:
            context_chat.save_to_json(f"{script_directory}/cache/global_chat.json")
        else:
            context_chat.save_to_json(f"{script_directory}/cache/last_chat.json")
            
            
        print(colored(llm_response, 'light_blue'))

        cmds = extract_llm_commands(llm_response)
        if not cmds:
            continue  # or other logic to handle non-command responses

        user_input = input(colored("Do you want me to execute these steps? (Y/n) ", 'yellow'))
        if user_input.lower() == "n":
            continue

        # Execute commands extracted from the llm_response
        user_request = ""
        for cmd in cmds:
            print(colored(f"Executing command: {cmd}\n" + "# " * 10, 'green'))
            user_request += run_command(cmd)
            print(colored(user_request, 'green'))
            
        user_request += "\n\n"
        
if __name__ == "__main__":
    main()
