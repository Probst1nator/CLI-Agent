import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Tuple

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.widgets import CheckboxList, Frame, Label
import pyperclip
from termcolor import colored
import time

from interface.cls_chat import Chat, Role
from interface.cls_ollama_client import OllamaClient

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
                final_output = final_output[:half_length] + "\n\n...Output truncated due to length.\n\n" + final_output[-half_length:]

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

    # Execute selected commands and collect their outputs
    results = []
    
    # Execute selected commands and collect their outputs
    formatted_results: List[str] = []
    for cmd in selected_commands:
        if cmd in commands:
            result, output = run_command(cmd, verbose)
            results.append(result)
            formatted_results.append(f"```cmd\n{result}\n```\n```cmd_log\n{output}\n```")
    
    execution_summarization = "```execution_summarization\n" + "\n".join(results) + "\n```"
    
    return "\n\n".join(formatted_results), execution_summarization


def fetch_search_results(query: str) -> List[str]:
    """
    Fetch search results from DuckDuckGo for a given query.
    
    Args:
        query (str): The search query.
        
    Returns:
        List[str]: The filtered top results.
    """
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
    """
    Filter the top search results.
    
    Args:
        results (str): The raw search results.
        num_results (int): Number of top results to return.
        
    Returns:
        List[str]: List of top search results.
    """
    results_arr: list[str] = []
    for i in range(1,num_results+1):
        start_i = results.index(f"\n{i}. ")
        end_i = results.index(f"\n{i+1}. ")
        results_arr.append(results[start_i:end_i])
    return results_arr

def remove_ansi_escape_sequences(text: str) -> str:
    """
    Remove ANSI escape sequences from the text.
    
    Args:
        text (str): The text containing ANSI escape sequences.
        
    Returns:
        str: Cleaned text without ANSI escape sequences.
    """
    ansi_escape = re.compile(r'''
        (?:\x1B[@-_])|                  # ESC followed by a character between @ and _
        (?:\x1B\[0-9;]*[ -/]*[@-~])|   # ESC [ followed by zero or more digits or semicolons, then a character between @ and ~
        (?:\x1B\][0-9]*;?[ -/]*[^\a]*\a)  # ESC ] followed by zero or more digits, an optional semicolon, any non-BEL characters, and a BEL
    ''', re.VERBOSE)
    return ansi_escape.sub('', text)

def read_from_terminal(num_lines: int, file_path: str = "/tmp/terminal_output.txt") -> List[str]:
    """
    Read the last `num_lines` from the file at `file_path`, removing ANSI sequences.
    
    Args:
        num_lines (int): Number of lines to read from the end of the file.
        file_path (str): Path to the file to read from.
        
    Returns:
        List[str]: List of cleaned lines.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Remove ANSI sequences from each line
        cleaned_lines = [remove_ansi_escape_sequences(line) for line in lines[-num_lines:]]
        return cleaned_lines


import tkinter as tk
from PIL import ImageGrab, Image, ImageTk
import os
import base64

class ScreenCapture:
    def __init__(self) -> None:
        """
        Initialize the ScreenCapture class with paths to save images and start the capture process.
        """
        self.user_cli_agent_dir = os.path.expanduser('~/.local/share') + "/cli-agent"
        os.makedirs(self.user_cli_agent_dir, exist_ok=True)
        
        self.fullscreen_image_path = os.path.join(self.user_cli_agent_dir, "fullscreen.png")
        self.captured_region_path = os.path.join(self.user_cli_agent_dir, "captured_region.png")
        
        self.capture_fullscreen()

        self.root = tk.Tk()
        self.root.title("Draw Rectangle to Capture Region")
        self.root.attributes('-fullscreen', True)

        self.image = Image.open(self.fullscreen_image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)

        self.canvas = tk.Canvas(self.root, cursor="cross", bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.root.mainloop()

    def on_button_press(self, event: tk.Event) -> None:
        """
        Handle mouse button press event to start drawing a rectangle.
        
        Args:
            event (tk.Event): The event object.
        """
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_mouse_drag(self, event: tk.Event) -> None:
        """
        Handle mouse drag event to update the rectangle's size.
        
        Args:
            event (tk.Event): The event object.
        """
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event: tk.Event) -> None:
        """
        Handle mouse button release event to finalize the rectangle and capture the region.
        
        Args:
            event (tk.Event): The event object.
        """
        end_x, end_y = (event.x, event.y)
        self.capture_region(self.start_x, self.start_y, end_x, end_y)
        self.root.destroy()

    def capture_region(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        """
        Capture and save a region of the screen.
        
        Args:
            start_x (int): Starting x-coordinate.
            start_y (int): Starting y-coordinate.
            end_x (int): Ending x-coordinate.
            end_y (int): Ending y-coordinate.
        """
        cropped_image = self.image.crop((start_x, start_y, end_x, end_y))
        cropped_image.save(self.captured_region_path)
        print(f"Region captured and saved as '{self.captured_region_path}'")

    def capture_fullscreen(self) -> None:
        """
        Capture and save the entire screen.
        """
        screen = ImageGrab.grab()
        screen.save(self.fullscreen_image_path)
        print(f"Fullscreen captured and saved as '{self.fullscreen_image_path}'")

    def return_fullscreen_image(self) -> str:
        """
        Return the base64-encoded fullscreen image.
        
        Returns:
            str: Base64-encoded fullscreen image.
        """
        with open(self.fullscreen_image_path, "rb") as fullscreen_file:
            fullscreen_image = base64.b64encode(fullscreen_file.read()).decode("utf-8")
        return fullscreen_image

    def return_captured_region_image(self) -> str:
        """
        Return the base64-encoded captured region image.
        
        Returns:
            str: Base64-encoded captured region image.
        """
        with open(self.captured_region_path, "rb") as region_file:
            captured_region_image = base64.b64encode(region_file.read()).decode("utf-8")
        return captured_region_image


def search_files_for_term(search_term: str) -> List[Tuple[str, str]]:
    """
    Search for a given term in all files within the current working directory and its subdirectories.
    
    Args:
        search_term (str): The term to search for.
        
    Returns:
        List[Tuple[str, str]]: A list of tuples containing the relative file paths and the matching content.
    """
    result: List[Tuple[str, str]] = []
    # Compile the regex pattern for case insensitive search
    pattern = re.compile(re.escape(search_term), re.IGNORECASE)
    
    # Get the current working directory
    working_directory = os.getcwd()
    for root, dirs, files in os.walk(working_directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file name contains any blacklisted substrings
            blacklisted_substrings = ["pycache", ".git", ".venv", "node_modules", ".idea", ".vscode", ".ipynb_checkpoints", "env"]
            if any(substring in file_path for substring in blacklisted_substrings):
                continue
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                try:
                    content = f.read()
                    if pattern.search(content):
                        relative_path = os.path.relpath(file_path, working_directory)
                        result.append((relative_path, content))
                except Exception as e:
                    # Handle exceptions if any file cannot be read
                    print(f"Error reading {file_path}: {e}")
    return result

def gather_intel(search_term: str) -> str:
    """
    Gather intelligence on a search term by analyzing its occurrences in files.
    
    Args:
        search_term (str): The term to gather intelligence on.
        
    Returns:
        str: The gathered intelligence.
    """
    path_contents: List[tuple[str,str]] = search_files_for_term(search_term)
    if (len(path_contents)):
        return f"No files containing the term '{search_term}' could be found."
    path_contents.sort(key=lambda x: len(x[1]), reverse=True) # Sort by length of content, descending
    print(f"Found {len(path_contents)} files with the search term.")
    filtered_path_contents = [(path, content) for path, content in path_contents if len(content) < 10000]
    print(f"Filtered to {len(filtered_path_contents)} files with the search term.")
    if len(filtered_path_contents) > 5:
        filtered_path_contents = filtered_path_contents[:5]
    print(f"Refiltered to {len(filtered_path_contents)} files with the search term.")
    
    session = OllamaClient()
    chat = Chat("In this conversation the user provides content which is incrementally reviewed and understood by the assistant. The assistant provides detailed, yet concise, responses.")   
    for path, content in filtered_path_contents:
        print(f"Path: {path}")
        # print(f"Content: {content}")
        fileending = path.split(".")[-1]
        prompt = f"Please infer the likely context of the given file: {path}\n'''{fileending}\”{content}\n'''"
        chat.add_message(Role.USER, prompt)
        response = session.generate_completion(chat, "llama3-gradient", local=True)
        chat.add_message(Role.ASSISTANT, response)
        chat.add_message(Role.USER, f"Please explain all ocurrences of '{search_term}' in the file. Work ocurrence by ocurrence and provide a contextual explanation.")
        response = session.generate_completion(chat, "llama3-gradient", local=True)
        chat.add_message(Role.ASSISTANT, response)
        chat.messages.pop(-3)
        chat.messages.pop(-3)

    chat.add_message(Role.USER, f"Explain '{search_term}' in detail.")
    intel = session.generate_completion(chat, "mixtral", f"Sure! Baed on our conversation")
    
    return intel