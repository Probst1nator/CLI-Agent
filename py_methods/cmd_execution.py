import os
import subprocess
import sys
from typing import List, Tuple, Dict
import chromadb
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.widgets import CheckboxList, Frame, Label
from py_classes.cls_llm_router import LlmRouter
from py_classes.globals import g
from py_classes.ai_providers.cls_ollama_interface import OllamaClient

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pyperclip
from termcolor import colored

def run_command(command: str, verbose: bool = True) -> Dict[str, str]:
    """
    Run a shell command and capture its output, optimized for chatbot interaction.
    
    Args:
        command (str): The shell command to execute.
        verbose (bool): Whether to print the command and its output.
        max_output_length (int): Maximum length of the output to return.
        
    Returns:
        Dict[str, str]: A dictionary containing the command execution results.
    """
    try:
        if verbose:
            print(colored(command, 'light_green'))
        
        # Execute the command in the user's current working directory
        process = subprocess.Popen(command, shell=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=os.getcwd())
        stdout, stderr = process.communicate()
        
        output = stdout if stdout else stderr
        truncated = False
        
        if verbose:
            print(output)
        
        return {
            "command": command,
            "output": output.strip(),
            "exit_code": process.returncode,
            "truncated": truncated
        }
    except Exception as e:
        return {
            "command": command,
            "output": f"Error executing command: {str(e)}",
            "exit_code": -1,
            "truncated": False
        }

def format_command_result(result: Dict[str, str]) -> str:
    """
    Format the command execution result for chatbot output.
    
    Args:
        result (Dict[str, str]): The result from run_command.
        
    Returns:
        str: Formatted result string.
    """
    status = "✅ Success" if result["exit_code"] == 0 else "❌ Failed"
    truncation_note = "\n\nNote: Output was truncated." if result["truncated"] else ""
    
    formatted_result = f"{status} (Exit code: {result['exit_code']})\n\n"
    formatted_result += f"```\n$ {result['command']}\n{result['output']}\n```"
    formatted_result += truncation_note
    
    return formatted_result

def select_and_execute_commands(commands: List[str], skip_user_confirmation: bool = False, verbose: bool = True) -> Tuple[str, str]:
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
            app.exit(result=checkbox_list.current_values)

        @bindings.add("c")
        def _copy_and_quit(event) -> None:
            selected_commands = " && ".join(checkbox_list.current_values)
            pyperclip.copy(selected_commands)
            app.exit(result=["exit"])
            
        @bindings.add("a")
        def _abort(event) -> None:
            app.exit(result=[])

        instructions = Label(text="Press 'e' to execute commands or 'c' to copy selected commands and quit. ('a' to abort)")
        root_container = HSplit([
            Frame(title="Select commands to execute, in order", body=checkbox_list),
            instructions
        ])
        layout = Layout(root_container)
        app = Application(layout=layout, key_bindings=bindings, full_screen=False)
        selected_commands = app.run()
        
        if selected_commands == ["exit"]:
            print(colored("Selected commands copied to clipboard.", "light_green"))
            sys.exit(0)
    else:
        selected_commands = commands
    
    client = chromadb.PersistentClient(g.PROJ_VSCODE_DIR_PATH)
    collection = client.get_or_create_collection(name="commands")

    results = []
    summary = []
    
    for cmd in selected_commands:
        # execute_actions_guard_response = LlmRouter.generate_completion(
        #     f"The following command must not require any user interaction at all after execution, it must exit fully automatically, if any of these don't apply respond with: 'Unsafe'\n\n{cmd}",
        #     ["llama-guard3:1b"], force_local=True, silent_reasoning=True, silent_reason="command guard"
        # )
        # if "unsafe" in execute_actions_guard_response.lower():
        #     results.append(f"Skipped unsafe command: {cmd}")
        #     summary.append(f"Command '{cmd}' was skipped because it potentially requires user interaction. If needed, execute it again but in a way that doesn't require user interaction.")
        #     continue
        
        g.remember_recent_action(cmd)
        
        result = run_command(cmd, verbose)
        formatted_result = format_command_result(result)
        results.append(formatted_result)
        
        status = "succeeded" if result["exit_code"] == 0 else "failed"
        summary.append(f"Command '{cmd}' {status} (Exit code: {result['exit_code']})")
        
        if not collection.get(cmd):
            cmd_embedding = OllamaClient.generate_embedding(cmd, "bge-m3")
            if cmd_embedding:
                collection.add(
                    ids=[cmd],
                    embeddings=cmd_embedding,
                    documents=[cmd]
                )
    
    formatted_output = "\n\n".join(results)
    summary_output = "\n".join(summary)
    
    return formatted_output, summary_output

# Example usage
if __name__ == "__main__":
    commands_to_run = [
        "echo Hello, World!",
        "ls -l /nonexistent",
        "python --version"
    ]
    
    detailed_output, summary = select_and_execute_commands(commands_to_run)
    print("Summary:")
    print(summary)
    print("\nDetailed Output:")
    print(detailed_output)