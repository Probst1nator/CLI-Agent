import os
import subprocess
import sys
from typing import List, Tuple
import chromadb
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.widgets import CheckboxList, Frame, Label
from py_classes.globals import g

from py_classes.ai_providers.cls_ollama_interface import OllamaClient

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pyperclip
from termcolor import colored


def run_command(command: str, verbose: bool = True, max_output_length:int = 16000) -> Tuple[str,str]:
    """
    Run a shell command and capture its output, truncating if necessary.
    
    Args:
        command (str): The shell command to execute.
        verbose (bool): Whether to print the command and its output.
        max_output_length (int): Maximum length of the output to return.
        
    Returns:
        Tuple[str, str]: A tuple containing the formatted result and raw output.
    """
    
    output_lines = []  # List to accumulate output lines

    try:
        if (verbose):
            print(colored(command, 'light_green'))
        with subprocess.Popen(command, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) as process:
            if process.stdout is not None:
                output_string = ""
                for line in process.stdout:
                    output_string += line
                    if verbose:
                        print(line, end='')  # Print lines as they are received

            # Wait for the process to terminate and capture remaining output, if any
            remaining_output, error = process.communicate()

            # It's possible, though unlikely, that new output is generated between the last readline and communicate call
            if remaining_output:
                output_lines.append(remaining_output)

            # Combine all captured output lines into a single string
            final_output = ''.join(output_lines)
            
            if len(final_output) > max_output_length:
                half_length = max_output_length // 2
                final_output = final_output[:half_length] + "\n\n...Response truncated due to length.\n\n" + final_output[-half_length:]

            result = {
                'output': final_output,
                'error': error,
                'exit_code': process.returncode
            }
            
            # Conditional checks on result can be implemented here as needed
            result_formatted = command
            if (result["output"]):
                result_formatted += f"\n{result['output']}"
            if (result["error"] and result["exit_code"] != 0):
                result_formatted += f"\n{result['error']}"
            if (not result["output"] and result["exit_code"] == 0):
                result_formatted += "\t# Command executed successfully"

            return result_formatted, output_string
    except subprocess.CalledProcessError as e:
        # If a command fails, this block will be executed
        result = {
            'output': e.stdout,
            'error': e.stderr,
            'exit_code': e.returncode
        }
        # Conditional checks on result can be implemented here as needed
        result_formatted = command
        if (result["output"]):
            result_formatted += f"\n{result['output']}"
        if (result["error"]):
            result_formatted += f"\n{result['error']}"

        return result_formatted, ""


def select_and_execute_commands(commands: List[str], skip_user_confirmation: bool = False, verbose: bool = True) -> Tuple[str,str]:
    """
    Allow the user to select and execute a list of commands.

    Args:
        commands (List[str]): The list of commands to choose from.
        skip_user_confirmation (bool): If True, execute all commands without user confirmation.
        verbose (bool): Whether to print command outputs.

    Returns:
        Tuple[str, str]: Formatted result and execution summarization.
    """
    if not skip_user_confirmation:
        checkbox_list = CheckboxList(
            values=[(cmd, cmd) for i, cmd in enumerate(commands)],
            default_values=[cmd for cmd in commands]
        )
        bindings = KeyBindings()

        @bindings.add("e")
        def _execute(event) -> None:
            """Trigger command execution if "Execute Commands" is selected."""
            app.exit(result=checkbox_list.current_values)

        @bindings.add("c")
        def _copy_and_quit(event) -> None:
            """Copy selected commands and quit."""
            selected_commands = " && ".join(checkbox_list.current_values)
            pyperclip.copy(selected_commands)
            app.exit(result=["exit"])
            
        @bindings.add("a")
        def _abort(event) -> None:
            """Abort the command selection process."""
            app.exit(result=[])

        # Instruction message
        instructions = Label(text="Press 'e' to execute commands or 'c' to copy selected commands and quit. ('a' to abort)")

        # Define the layout with the instructions
        root_container = HSplit([
            Frame(title="Select commands to execute, in order", body=checkbox_list),
            instructions  # Add the instructions to the layout
        ])
        layout = Layout(root_container)

        # Create the application
        app:Application = Application(layout=layout, key_bindings=bindings, full_screen=False)

        # Run the application and get the selected option(s)
        selected_commands = app.run()
        if selected_commands == ["exit"]:
            print(colored("Selected commands copied to clipboard.", "light_green"))
            sys.exit(0)
    else:
        selected_commands = commands
    
    client = chromadb.PersistentClient(g.PROJ_VSCODE_DIR_PATH)
    collection = client.get_or_create_collection(name="commands")

    # Execute selected commands and collect their outputs
    results = []
    
    # Execute selected commands and collect their outputs
    formatted_results: List[str] = []
    for cmd in selected_commands:
        if cmd in commands:
            # Save to recent actions
            g.remember_recent_action(cmd)
            
            result, output = run_command(cmd, verbose)
            results.append(result)
            
            if not collection.get(cmd):
                cmd_embedding = OllamaClient.generate_embedding(cmd, "bge-m3")
                if not cmd_embedding:
                    break
                collection.add(
                    ids=[cmd],
                    embeddings=cmd_embedding,
                    documents=[cmd]
                )
            
            formatted_results.append(f"```cmd\n{result}\n```\n```cmd_log\n{output}\n```")
    
    execution_summarization = "```terminal_response\n" + "\n".join(results) + "\n```"
    
    return "\n\n".join(formatted_results), execution_summarization