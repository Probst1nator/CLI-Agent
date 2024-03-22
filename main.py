#!/usr/bin/env python3
# sudo ln -s /home/prob/repos/CLI-Agent/main.py /usr/local/bin/cli-agent

import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List

from termcolor import colored

from interface.cls_chat import Chat, Role
from interface.cls_chat_session import ChatSession
from interface.cls_few_shot_factory import FewShotProvider
from interface.cls_ollama_client import OllamaClient


def setup_sandbox():
    sandbox_dir = "./sandbox/"
    if os.path.exists(sandbox_dir):
        shutil.rmtree(sandbox_dir)
    os.mkdir(sandbox_dir)

setup_sandbox()

def run_command(command: str) -> Dict[str, Any]:
    print(colored(command, 'magenta'))  # Highlight the command being executed

    output_lines = []  # List to accumulate output lines

    try:
        with subprocess.Popen(command, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) as process:
            # Stream output in real-time
            while True:
                output_line = process.stdout.readline()  # type: ignore
                if not output_line and process.poll() is not None:
                    break
                sys.stdout.write(colored(output_line, 'red'))
                sys.stdout.flush()
                output_lines.append(output_line)

            # Wait for the process to terminate and capture remaining output, if any
            remaining_output, error = process.communicate()

            # It's possible, though unlikely, that new output is generated between the last readline and communicate call
            if remaining_output:
                output_lines.append(remaining_output)
                sys.stdout.write(colored(remaining_output, 'red'))
                sys.stdout.flush()

            # Combine all captured output lines into a single string
            final_output = ''.join(output_lines)

            return {
                'success': True if process.returncode == 0 else False,
                'output': final_output,
                'error': error,
                'exit_code': process.returncode
            }

    except subprocess.CalledProcessError as e:
        # If a command fails, this block will be executed
        return {
            'success': False,
            'output': e.stdout,
            'error': e.stderr,
            'exit_code': e.returncode
        }

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


if __name__ == "__main__":
    session = OllamaClient()
    # user_request = "How do i install python?"
    while True:
        user_request = ""
        while (len(user_request)==0):
            user_request = input(colored("Enter your request: ", 'yellow'))
        llm_response, context_chat = FewShotProvider.few_shot_SuggestAgentStrategy(user_request, "mixtral")
        print(colored(llm_response, 'light_blue'))
        cmds = extract_llm_commands(llm_response)
        results = []
        while True:
            while (len(cmds)==0):
                if (len(results)>0):
                    user_request = input(colored("USER: ", 'yellow'))
                    context_chat.add_message(Role.USER, "\n\n".join(results) + "\n\n\n" + user_request)
                else:
                    user_request = input(colored("USER: ", 'yellow'))
                    context_chat.add_message(Role.USER, user_request)
                llm_response = session.generate_completion(context_chat, "mixtral")
                context_chat.add_message(Role.ASSISTANT, llm_response)
                print(colored(llm_response, 'light_blue'))
                cmds = extract_llm_commands(llm_response)
                
            user_input = input(colored("Do you want me to execute these steps? (Y/n) ", 'yellow'))
            if user_input.lower() == "n":
                user_input = input(colored("Continue conversation or type (n) to exit: ", 'yellow')).lower()
                if ("n" == user_input):
                    exit(0)
                context_chat.add_message(Role.USER, user_input)
                llm_response = session.generate_completion(context_chat, "mixtral")
                context_chat.add_message(Role.ASSISTANT, llm_response)
                print(colored(llm_response, 'light_blue'))
                cmds = extract_llm_commands(llm_response)
                continue
            
            print(colored("executing...", 'green'))
            results: List[str] = []
            for cmd in cmds:
                result = run_command(cmd)
                # Conditional checks on result can be implemented here as needed
                result_formatted = cmd + "\n\n"
                if (result["output"]):
                    result_formatted = f"'''cmd_output{result['output']}'''\n"
                if (result["error"]):
                    result_formatted = f"'''cmd_error{result['error']}'''\n"
                results.append(result_formatted)
            cmds = []
            # print("\n\n".join(results))
            llm_response = session.generate_completion(context_chat, "mixtral")
            context_chat.add_message(Role.ASSISTANT, llm_response)