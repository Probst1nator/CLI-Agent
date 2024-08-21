import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Any, List, Literal, Optional, Tuple
import sqlite3
import os
from typing import List, Tuple
import chromadb
from gtts import gTTS
import numpy as np
from librosa import *
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.widgets import CheckboxList, Frame, Label
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pyperclip
from termcolor import colored
from pynput import keyboard
from speech_recognition import Recognizer, AudioSource, AudioData
from pydub import AudioSegment
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

from classes.ai_providers.cls_openai_interface import OpenAIChat
from classes.cls_chat import Chat, Role
from classes.ai_providers.cls_ollama_interface import OllamaClient
from logger import logger

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

    # Execute selected commands and collect their outputs
    results = []
    
    # Execute selected commands and collect their outputs
    formatted_results: List[str] = []
    for cmd in selected_commands:
        if cmd in commands:
            result, output = run_command(cmd, verbose)
            results.append(result)
            formatted_results.append(f"```cmd\n{result}\n```\n```cmd_log\n{output}\n```")
    
    execution_summarization = "```terminal_response\n" + "\n".join(results) + "\n```"
    
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

def wip_gather_intel(search_term: str) -> str:
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
        response = session.generate_response(chat, "llama3-gradient", force_local=True)
        chat.add_message(Role.ASSISTANT, response)
        chat.add_message(Role.USER, f"Please explain all ocurrences of '{search_term}' in the file. Work ocurrence by ocurrence and provide a contextual explanation.")
        response = session.generate_response(chat, "llama3-gradient", force_local=True)
        chat.add_message(Role.ASSISTANT, response)
        chat.messages.pop(-3)
        chat.messages.pop(-3)

    chat.add_message(Role.USER, f"Explain '{search_term}' in detail.")
    # intel = session.generate_completion(chat, "mixtral", f"Sure! Based on our conversation")
    intel = ""
    return intel


def run_python_script(script_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Executes a Python script using the provided file path and returns the script's output and any errors.

    Args:
        script_path (str): The file path of the Python script to execute.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing:
            - The script's output if execution is successful, otherwise None.
            - The error message if an error occurred, otherwise None.
    """
    try:
        # Run the script and capture the output and errors
        result = subprocess.run(['python3', script_path], capture_output=True, text=True)
        # Check if the script executed successfully
        if result.returncode == 0:
            return result.stdout, None
        else:
            return None, result.stderr
    except Exception as e:
        return None, str(e)

def on_press(key):
    if key == keyboard.Key.esc:
        pygame.mixer.music.stop()

def text_to_speech(text: str, lang_key: str = 'en', enable_keyboard_interrupt: bool = True, speed: float = 1.3, pitch_shift: float = 0.2):
    """
    Convert the assistant's response to speech, adjust speed and pitch, then play it.
    Args:
    text (str): The text to convert to speech.
    lang_key (str, optional): The language of the text. Defaults to 'en'.
    enable_keyboard_interrupt (bool, optional): Whether to enable keyboard interrupt. Defaults to True.
    speed (float, optional): The speed of the speech. Defaults to 1.2.
    pitch_shift (float, optional): The pitch shift in semitones. Positive values increase pitch, negative values decrease it. Defaults to 0.
    Returns:
    None
    """
    if text == "":
        print(colored("No text to convert to speech.", "red"))
        return

    tts_file = 'tts_response.mp3'
    modified_tts_file = 'modified_tts_response.mp3'

    # Convert text to speech and save to a file
    tts = gTTS(text=text, lang=lang_key)
    tts.save(tts_file)

    # Load the audio file
    sound = AudioSegment.from_mp3(tts_file)

    # Change speed
    faster_sound = sound.speedup(playback_speed=speed)

    # Change pitch
    if pitch_shift != 0:
        # Extract raw audio data as an array of samples
        samples = np.array(faster_sound.get_array_of_samples())
        
        # Resample the audio to change pitch
        pitch_factor = 2 ** (pitch_shift / 12)  # Convert semitones to a multiplication factor
        new_sample_rate = int(faster_sound.frame_rate * pitch_factor)
        pitched_samples = samples.astype(np.float64)
        pitched_samples = np.interp(
            np.linspace(0, len(pitched_samples), int(len(pitched_samples) / pitch_factor)),
            np.arange(len(pitched_samples)),
            pitched_samples
        ).astype(np.int16)
        
        # Create a new AudioSegment with the pitched samples
        pitched_sound = AudioSegment(
            pitched_samples.tobytes(),
            frame_rate=new_sample_rate,
            sample_width=faster_sound.sample_width,
            channels=faster_sound.channels
        )
        
        # Export the pitched sound
        pitched_sound.export(modified_tts_file, format="mp3")
    else:
        # If no pitch change, just export the speed-adjusted sound
        faster_sound.export(modified_tts_file, format="mp3")

    # Initialize pygame mixer and play the modified file
    pygame.mixer.init()
    pygame.mixer.music.load(modified_tts_file)
    pygame.mixer.music.play()

    if enable_keyboard_interrupt:
        print(colored("Press 'Esc' to interrupt the speech.", "yellow"))
        # Create a new thread to listen for keyboard events
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    if enable_keyboard_interrupt:
        # Stop the keyboard listener
        listener.stop()

    # Clean up the files
    os.remove(tts_file)
    os.remove(modified_tts_file)


def listen_microphone(
    source: AudioSource,
    r: Recognizer,
    max_duration: Optional[int] = 40,
    language: str = ""
) -> Tuple[str, str]:
    """
    Listen to the microphone and return transcribed text and language.
    Args:
        source (AudioSource): The audio source to listen from.
        r (Recognizer): The speech recognizer object.
        max_duration (Optional[int]): The maximum duration to listen. Defaults to 15.
        language (str): The language to use for transcription. Defaults to "".

    Returns:
        Tuple[str, str]: A tuple containing (transcribed text from the audio, language).
    """
    print(colored("Listening to microphone...", "yellow"))
    with source:
        audio: AudioData = r.listen(source, timeout=max_duration, phrase_time_limit=max_duration/2)
    print(colored("Not listening anymore...", "yellow"))
    transcription, language = OpenAIChat.transcribe_audio(audio, language=language)
    # Print the recognized text
    print("Microphone transcription: " + colored(transcription, "green"))
    
    return transcription, language

def remove_blocks(text: str, except_types: Optional[List[str]] = None) -> str:
    """
    Remove all code blocks from the text, except for specified types.

    Args:
        text (str): The input text containing code blocks.
        except_types (Optional[List[str]]): List of code block types to keep (e.g., ['md', 'text']).

    Returns:
        str: The text with code blocks removed (except for specified types).
    """
    if except_types is None:
        except_types = []
    
    pattern = r'```(?P<lang>\w*)\n(?P<content>.*?)```'
    
    def replacement(match: re.Match) -> str:
        lang = match.group('lang').lower()
        if lang in except_types:
            return match.group(0)  # Keep the entire match
        return ''  # Remove the code block
    
    return re.sub(pattern, replacement, text, flags=re.DOTALL)

def clean_pdf_text(text):
    # Step 1: Handle unicode characters (preserving special characters)
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    # Step 2: Remove excessive newlines and spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    # Step 3: Join split words
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Step 4: Separate numbers and text
    text = re.sub(r'(\d+)([A-Za-zÄäÖöÜüß])', r'\1 \2', text)
    text = re.sub(r'([A-Za-zÄäÖöÜüß])(\d+)', r'\1 \2', text)
    # Step 5: Add space after periods if missing
    text = re.sub(r'\.(\w)', r'. \1', text)
    # Step 6: Capitalize first letter after period and newline
    text = re.sub(r'(^|\. )([a-zäöüß])', lambda m: m.group(1) + m.group(2).upper(), text)
    # Step 7: Format Euro amounts
    text = re.sub(r'(\d+)\s*Euro', r'\1 Euro', text)
    # Step 8: Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    return text.strip()

def extract_pdf_content(file_path: str) -> Tuple[str, List[Any]]:
    def extract_text(pdf_path):
        resource_manager = PDFResourceManager()
        fake_file_handle = StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams(all_texts=True))
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        
        with open(pdf_path, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
            
            text = fake_file_handle.getvalue()
        
        converter.close()
        fake_file_handle.close()
        
        return text

    text_content = extract_text(file_path)
    text_content = clean_pdf_text(text_content)
    
    # Note: pdfminer.six doesn't have a built-in method for image extraction
    # You might need to use a different library like PyMuPDF for image extraction
    image_content: List[Any] = []
    
    return text_content, image_content

def nuextract_template(text: str, template: dict[str,Any]):
    return f"### Template:\n{json.dumps(template)}\n### Text:\n{text}"


def list_files_recursive(path: str, max_depth: int = 1) -> List[str]:
    result: List[str] = []
    
    def explore(current_path: str, current_depth: int) -> None:
        if current_depth > max_depth:
            return

        items: List[Tuple[str, str]] = []

        for item in os.listdir(current_path):
            item_path = os.path.join(current_path, item)
            if os.path.isfile(item_path):
                items.append(("file", item_path))
            elif os.path.isdir(item_path):
                items.append(("dir", item_path))

        # Add files at this level to the result
        result.extend([path for item_type, path in items if item_type == "file"])

        # Explore subdirectories
        for item_type, item_path in items:
            if item_type == "dir":
                explore(item_path, current_depth + 1)

    explore(path, 0)
    return result


def split_string_into_chunks(
    input_string: str, 
    max_chunk_size: int = 2000,
    delimiter: str = "\n",
) -> List[str]:
    """
    Splits a string into chunks with a specified maximum size, overlap, and minimum remaining size.
    Ensures that chunks are split at newline characters to maintain semantic coherence.

    Args:
        input_string (str): The input string to split.
        max_chunk_size (int, optional): The maximum size of each chunk. Defaults to 10000.
        min_remaining_size (int, optional): The minimum size of the last chunk. Will extend backwards overlap_size if last chunk is too small. Defaults to 4000.

    Returns:
        List[str]: A list of chunks.
    """
    chunks = []
    start_index = 0
    input_length = len(input_string)
    overlap_size = int(max_chunk_size/2)

    while start_index < input_length:
        # Calculate the tentative end index for the current chunk
        tentative_end_index = start_index + max_chunk_size
        
        # Ensure the end index does not exceed the string length
        if tentative_end_index >= input_length:
            chunks.append(input_string[start_index:])
            break
        
        # Find the last newline character before the tentative end index
        end_index = input_string.rfind(delimiter, start_index, tentative_end_index)
        
        # If no newline is found, set the end index to the max_chunk_size limit
        if end_index == -1 or end_index <= start_index:
            end_index = tentative_end_index
        
        # Add the chunk to the list
        chunks.append(input_string[start_index:end_index])
        
        # Move the start index for the next chunk, considering overlap
        start_index = end_index - overlap_size

    return chunks

def get_atuin_history(limit: int = 10) -> List[str]:
    """
    Retrieve the last N commands from Atuin's history database.

    :param limit: Number of commands to retrieve (default 10)
    :return: List of tuples containing (index, command)
    """
    db_path = os.path.expanduser('~/.local/share/atuin/history.db')

    if not os.path.exists(db_path):
        return []

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            query = """
            SELECT command
            FROM history
            ORDER BY timestamp DESC
            LIMIT ?
            """
            cursor.execute(query, (limit,))
            result_tuples = cursor.fetchall()
            results = []
            for result_tuple in result_tuples:
                results.append(result_tuple[0])
            return results
    except sqlite3.Error as e:
        logger.error(f"Querying {db_path} caused an error: {e}")
        return []


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
