import subprocess
import sys
from typing import Any, Dict, List

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.widgets import CheckboxList, Frame, Label
import pyperclip
from termcolor import colored
import time


def run_command(command: str, verbose: bool = True) -> str:
    output_lines = []  # List to accumulate output lines

    try:
        if (verbose):
            print(colored(command, 'light_green'))
        with subprocess.Popen(command, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) as process:
            if process.stdout is not None:
                if verbose:
                    for line in process.stdout:
                        print(line, end='')  # Print lines as they are received

            # Wait for the process to terminate and capture remaining output, if any
            remaining_output, error = process.communicate()

            # It's possible, though unlikely, that new output is generated between the last readline and communicate call
            if remaining_output:
                output_lines.append(remaining_output)

            # Combine all captured output lines into a single string
            final_output = ''.join(output_lines)

            result = {
                'output': final_output,
                'error': error,
                'exit_code': process.returncode
            }
            
            # Conditional checks on result can be implemented here as needed
            result_formatted = command
            if (result["output"]):
                # result_formatted += f"\n```cmd_output\n{result['output']}```"
                result_formatted += f"\n{result['output']}"
            if (result["error"] and result["exit_code"] != 0):
                # result_formatted += f"\n```cmd_error\n{result['error']}```"
                result_formatted += f"\n{result['error']}"
            if (not result["output"] and result["exit_code"] == 0):
                result_formatted += "\t# Command executed successfully"

            return result_formatted
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
            # result_formatted += f"\n```cmd_output\n{result['output']}```"
            result_formatted += f"\n{result['output']}"
        if (result["error"]):
            # result_formatted += f"\n```cmd_error\n{result['error']}```"
            result_formatted += f"\n{result['error']}"

        return result_formatted



def select_and_execute_commands(commands: List[str], skip_user_confirmation: bool = False, verbose: bool = True) -> str:
    if not skip_user_confirmation:
        checkbox_list = CheckboxList(
            values=[(cmd, cmd) for i, cmd in enumerate(commands)],
            default_values=[cmd for cmd in commands]
        )
        bindings = KeyBindings()

        @bindings.add("q")
        def _quit(event) -> None:
            """Trigger command execution if "Execute Commands" is selected."""
            app.exit(result=checkbox_list.current_values)

        @bindings.add("c")
        def _copy_and_quit(event) -> None:
            """Copy selected commands and quit."""
            selected_commands = " && ".join(checkbox_list.current_values)
            pyperclip.copy(selected_commands)
            app.exit(result=["exit"])
            # time.sleep(0.2)
            # print("\n\n\nSelected commands copied to clipboard.")
            # sys.exit(0)  # Quit the whole Python process

        # Instruction message
        instructions = Label(text="Press 'q' to execute commands or 'c' to copy selected commands and quit.")

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

    # Execute selected commands and collect their outputs
    outputs = [run_command(cmd, verbose) for cmd in selected_commands if cmd in commands]  # Ensure "Execute Commands" is not executed
    
    return "```bash_response\n" + "\n".join(outputs) + "\n```"



def fetch_search_results(query: str) -> List[str]:
    # Build the URL for DuckDuckGo search
    url = f"https://duckduckgo.com/?q={query}"
    
    # Execute w3m command to fetch and dump search results
    try:
        result = subprocess.run(['w3m', '-dump', url], text=True, capture_output=True)
        return filter_top_results(result.stdout)
    except subprocess.SubprocessError as e:
        print(f"Failed to execute w3m: {e}")
        return []

def filter_top_results(results: str, num_results: int = 5) -> List[str]:
    results_arr: list[str] = []
    for i in range(1,num_results+1):
        start_i = results.index(f"\n{i}. ")
        end_i = results.index(f"\n{i+1}. ")
        results_arr.append(results[start_i:end_i])
    
    return results_arr

def get_first_site_content(url: str) -> str:
    # Fetch and return the content of the first site
    try:
        result = subprocess.run(['w3m', '-dump', url], text=True, capture_output=True)
        return result.stdout
    except subprocess.SubprocessError as e:
        print(f"Failed to load URL {url}: {e}")
        return ""
    
class cls_tooling:
    saved_block_delimiters: str = ""
    color_red: bool = False

    def apply_color(self, string: str, return_remaining: bool = False):
        last_red: bool = False
        if "`" in string:
            self.saved_block_delimiters += string
            string = ""
            if self.saved_block_delimiters.count("`") == 3:
                self.color_red = not self.color_red
                string = self.saved_block_delimiters
                self.saved_block_delimiters = ""
                last_red = True
        else:
            string = self.saved_block_delimiters + string
            self.saved_block_delimiters = ""
        # elif len(self.saved_block_delimiters) >= 3:
        #     string = self.saved_block_delimiters
        #     self.saved_block_delimiters = ""
        if (return_remaining):
            string = self.saved_block_delimiters + string
            self.saved_block_delimiters = ""
        if self.color_red or last_red:
            string = colored(string, "light_red")
        else:
            string = colored(string, "magenta")
        return string

