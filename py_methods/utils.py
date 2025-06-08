from collections import defaultdict
from datetime import datetime
import hashlib
import os
import pickle
import re
import subprocess
import tempfile
from typing import Any, List, Literal, Optional, Tuple, Dict, Union, Set, cast, TypedDict
import sqlite3
import os
from typing import List, Tuple
import chromadb
from gtts import gTTS
import numpy as np
import pyaudio
import logging
import json
from termcolor import colored

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from py_classes.ai_providers.cls_groq_interface import GroqAPI
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pynput import keyboard
from speech_recognition import Microphone, Recognizer, WaitTimeoutError
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

from py_classes.ai_providers.cls_openai_interface import OpenAIAPI
from py_classes.ai_providers.cls_ollama_interface import OllamaClient
from py_classes.globals import g

import tkinter as tk
from PIL import ImageGrab, Image, ImageTk
import os
import base64

# Define the typed structures for tool calls
class ToolCallParameters(TypedDict, total=False):
    """Type for tool call parameters with optional fields"""
    message: Optional[str]
    command: Optional[str]
    file_path: Optional[str]
    raw_content: Optional[str]
    content_prompt: Optional[str]
    queries: Optional[Union[str, List[str]]]
    # Add other common parameters as needed


class ToolCall(TypedDict):
    """Type for tool calls"""
    tool: str
    reasoning: str
    parameters: ToolCallParameters
    positional_parameters: Optional[List[Any]]

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
    ansi_escape = re.compile(r"""
        (?:\x1B[@-_])|                  # ESC followed by a character between @ and _
        (?:\x1B\[0-9;]*[ -/]*[@-~])|   # ESC [ followed by zero or more digits or semicolons, then a character between @ and ~
        (?:\x1B\][0-9]*;?[ -/]*[^\a]*\a)  # ESC ] followed by zero or more digits, an optional semicolon, any non-BEL characters, and a BEL
    """, re.VERBOSE)
    return ansi_escape.sub('', text)



r: Recognizer = None
def calibrate_microphone(calibration_duration: int = 1) -> Microphone:
    """
    Calibrate the microphone for ambient noise.

    Returns:
        sr.Microphone: The calibrated microphone.
    """
    global r, source
    if not r:
        pyaudio_instance = pyaudio.PyAudio()
        default_microphone_info = pyaudio_instance.get_default_input_device_info()
        microphone_device_index = default_microphone_info["index"]
        r = Recognizer()
        source = Microphone(device_index=microphone_device_index)
    
    print(
        colored(f"Calibrating microphone for {calibration_duration} seconds", "yellow")
    )
    with source as source:
        r.adjust_for_ambient_noise(source, calibration_duration)
    r.energy_threshold *= 2
    
    return source

def listen_microphone(
    max_listening_duration: Optional[int] = 60, private_remote_wake_detection: bool = False
) -> Tuple[str, str, bool|str]:
    """
    Listen to the microphone, save to a temporary file, and return transcription.
    Args:
    max_duration (Optional[int], optional): The maximum duration to listen. Defaults to 15.
    language (str): The language of the audio (optional).
    Returns:
    Tuple[str, str, bool|str]: (transcribed text from the audio, language, used wake word)
    """
    from py_classes.cls_pyaihost_interface import PyAiHost
    global r, source
    if not r:
        calibrate_microphone()
    transcription: str = ""
    
    while not transcription or transcription.strip().lower() == "you":
        print(colored("Listening to microphone...", "yellow"))

        try:
            # Listen for speech until it seems to stop or reaches the maximum duration
            PyAiHost.play_notification()
            used_wake_word = PyAiHost.wait_for_wake_word(private_remote_wake_detection)
            print(colored("Listening closely...", "yellow"))
            PyAiHost.play_notification()
            while True:
                start_time = time.time()
                with source:
                    audio = r.listen(
                        source, timeout=max_listening_duration, phrase_time_limit=max_listening_duration/2
                    )
                listen_duration = time.time() - start_time
                
                PyAiHost.play_notification()

                if listen_duration > 0.5:
                    break
                
                # If we spent more than 90% of the max duration listening, the microphone might need recalibration
                if listen_duration > max_listening_duration * 0.9:
                    r = None # Recalibrate the microphone
                

            print(colored("Processing sounds...", "yellow"))

            # Create a temporary file to store the audio data
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            ) as temp_audio_file:
                temp_audio_file.write(audio.get_wav_data())
                temp_audio_file_path = temp_audio_file.name

                # Transcribe the audio from the temporary file
                transcription = ""
                detected_language = ""
                
                if g.FORCE_LOCAL:
                    transcription, detected_language = PyAiHost.transcribe_audio(temp_audio_file_path)
                else:
                    # Try cloud-based transcription with multiple fallbacks
                    cloud_success = False
                    
                    # First try Groq API
                    try:
                        transcription, detected_language = GroqAPI.transcribe_audio(temp_audio_file_path)
                        print(colored("✅ Groq cloud transcription successful", "green"))
                        cloud_success = True
                    except Exception as groq_error:
                        print(colored(f"⚠️ Groq transcription failed: {str(groq_error)}", "yellow"))
                    
                    # If Groq failed, try OpenAI as secondary cloud option
                    if not cloud_success:
                        try:
                            # Convert file to AudioData format for OpenAI API
                            import speech_recognition as sr
                            r = sr.Recognizer()
                            with sr.AudioFile(temp_audio_file_path) as source:
                                audio_data = r.record(source)
                            
                            from py_classes.ai_providers.cls_openai_interface import OpenAIAPI
                            transcription, detected_language = OpenAIAPI.transcribe_audio(audio_data)
                            print(colored("✅ OpenAI cloud transcription successful", "green"))
                            cloud_success = True
                        except Exception as openai_error:
                            print(colored(f"⚠️ OpenAI transcription failed: {str(openai_error)}", "yellow"))
                    
                    # If all cloud services failed, fallback to local
                    if not cloud_success:
                        print(colored("🔄 All cloud services failed. Falling back to local Whisper transcription...", "blue"))
                        try:
                            # Fallback to local transcription
                            transcription, detected_language = PyAiHost.transcribe_audio(temp_audio_file_path)
                            print(colored("✅ Local transcription successful", "green"))
                        except Exception as local_error:
                            print(colored(f"❌ Local transcription also failed: {str(local_error)}", "red"))
                            transcription, detected_language = "", ""

                print("Whisper transcription: " + colored(transcription, "green"))

        except WaitTimeoutError:
            print(colored("Listening timed out. No speech detected.", "red"))
        except Exception as e:
            print(colored(f"An error occurred: {str(e)}", "red"))
        finally:
            # Clean up the temporary file
            if "temp_audio_file_path" in locals():
                os.remove(temp_audio_file_path)
    return transcription, detected_language, used_wake_word


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

def clean_pdf_text(text: str):
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

import os
import hashlib
import pickle
from io import StringIO
from typing import List, Union
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from termcolor import colored
import json

def get_cache_file_path(file_path: str, cache_key: str) -> str:
    cache_dir = os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, "pdf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    last_modified = os.path.getmtime(file_path)
    full_cache_key = hashlib.md5(f"{file_path}_{last_modified}".encode()).hexdigest()
    return os.path.join(cache_dir, f"{cache_key}_{full_cache_key}.pickle")

def load_from_cache(cache_file: str) -> Union[str, List[str], None]:
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_to_cache(cache_file: str, content: Union[str, List[str]]) -> None:
    with open(cache_file, 'wb') as f:
        pickle.dump(content, f)

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    resource_manager = PDFResourceManager()
    page_contents = []
    
    with open(pdf_path, 'rb') as fh:
        pages = list(PDFPage.get_pages(fh, caching=True, check_extractable=True))
        for i, page in enumerate(pages):
            fake_file_handle = StringIO()
            converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams(all_texts=True))
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            
            page_interpreter.process_page(page)
            
            text = fake_file_handle.getvalue()
            page_contents.append(clean_pdf_text(text))
            
            converter.close()
            fake_file_handle.close()
            
            print(colored(f"{i+1}/{len(pages)}. Extracted page from '{pdf_path}'", "green"))
    
    return page_contents

def extract_pdf_content_page_wise(file_path: str) -> List[str]:
    cache_file = get_cache_file_path(file_path, "page_wise_text")
    cached_content = load_from_cache(cache_file)
    
    if cached_content is not None:
        return cached_content
    
    page_contents = extract_text_from_pdf(file_path)
    
    save_to_cache(cache_file, page_contents)
    return page_contents

def extract_pdf_content(file_path: str) -> str:
    cache_file = get_cache_file_path(file_path, "full_text")
    cached_content = load_from_cache(cache_file)
    
    if cached_content is not None:
        return cached_content
    
    page_contents = extract_text_from_pdf(file_path)
    full_content = "\n".join(page_contents)
    
    save_to_cache(cache_file, full_content)
    return full_content


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
            cursor.run(query, (limit,))
            result_tuples = cursor.fetchall()
            results = []
            for result_tuple in result_tuples:
                results.append(result_tuple[0])
            return results
    except sqlite3.Error as e:
        logger.error(f"Querying {db_path} caused an error: {e}")
        return []


def extract_blocks(text: str, language: Optional[str|List[str]] = None) -> List[str]:
    """
    Extract code blocks of specific languages from text.

    Args:
        text (str): The text to extract code blocks from.
        language (Optional[str|List[str]]): The specific language(s) to extract 
                                                  (e.g., 'python', ['python', 'javascript']). 
                                                  If None, all code blocks are extracted.

    Returns:
        List[str]: List of extracted code block contents. If language(s) are specified,
                  only blocks of those language(s) are included.
    """
    pattern = r'```(?P<lang>\w*)\n(?P<content>.*?)```'
    matches = re.finditer(pattern, text, re.DOTALL)
    blocks = []

    for match in matches:
        block_lang = match.group('lang').strip().lower()
        content = match.group('content').strip()

        # Determine if the block language matches the specified language(s)
        is_match = False
        if language is None:
            is_match = True # Extract all blocks if no language is specified
        elif isinstance(language, str) and language.lower() == block_lang:
            is_match = True # Extract blocks matching the single specified language
        elif isinstance(language, list):
            # Extract blocks matching any language in the list
            if block_lang in [lang.lower() for lang in language]:
                is_match = True

        if is_match:
            blocks.append(content)

    return blocks

def extract_blocks_with_metadata(text: str, include_context: bool = False) -> List[Union[Tuple[str, str], Tuple[str, str, str]]]:
    """
    Extract code blocks encased by ``` from a text and the first curly brace block.
    This function handles various edge cases, including:
    - Nested code blocks
    - Incomplete code blocks
    - Code blocks with or without language specifiers
    - Whitespace and newline variations
    - JSON blocks and other language-specific blocks
    - Special 'first{}' case that extracts first curly brace block

    Args:
    text (str): The input text containing code blocks.
    include_context (bool): Whether to include context/reasoning before the code block.

    Returns:
    List[Union[Tuple[str, str], Tuple[str, str, str]]]: A list of tuples.
    If include_context is False: [(language, content), ...] 
    If include_context is True: [(language, content, text_without_blocks), ...]
    If no language is specified, the type will be an empty string.
    'first{}' type contains the first curly brace block found.
    """
    blocks = []
    text_without_blocks = text
    
    # Extract first {} block
    first_brace_pattern = r'\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    first_brace_match = re.search(first_brace_pattern, text)
    if first_brace_match:
        # Include the braces in the content
        brace_content = '{' + first_brace_match.group(1) + '}'
        if include_context:
            # Remove the brace content from text_without_blocks
            start_pos = first_brace_match.start()
            end_pos = first_brace_match.end()
            text_without_blocks = text_without_blocks[:start_pos] + text_without_blocks[end_pos:]
            blocks.append(('first{}', brace_content, text_without_blocks))
        else:
            blocks.append(('first{}', brace_content))
    
    # Extract code blocks with their positions
    code_pattern = r'```(\w*)\n([\s\S]*?)```'
    code_matches = list(re.finditer(code_pattern, text, re.MULTILINE))
    
    # Create a copy of the text and remove all code blocks from it
    text_without_blocks = text
    for match in reversed(code_matches):  # Process in reverse to maintain correct positions
        start_pos = match.start()
        end_pos = match.end()
        text_without_blocks = text_without_blocks[:start_pos] + text_without_blocks[end_pos:]
    
    # Process code blocks
    for match in code_matches:
        language = match.group(1).strip()
        content = match.group(2).strip()
        
        if include_context:
            blocks.append((language, content, text_without_blocks))
        else:
            blocks.append((language, content))
    
    return blocks

def extract_json(text: str, required_keys: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """
    Extract and validate JSON from text that may contain explanatory content or markdown.
    Handles various formats including code blocks and direct JSON content.
    
    Args:
        text (str): The text to extract JSON from
        required_keys (Optional[List[str]]): List of keys that must be present in the JSON
    
    Returns:
        Optional[Dict[str, Any]]: The extracted and validated JSON object, or None if extraction fails
    """
    try:
        # First try to extract from code blocks
        blocks = extract_blocks_with_metadata(text)
        
        def clean_json_content(content: str) -> str:
            """Helper function to clean JSON content before parsing."""
            content = re.sub(r'(?<!\\)\\n', r'\\n', content)  # Fix newlines
            content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content)  # Remove control chars
            content = re.sub(r',\s*([}\]])', r'\1', content)  # Remove trailing commas
            content = re.sub(r'(?<=\w)"(?=\w)', '\\"', content)  # Fix unescaped quotes
            return content
            
        def validate_json_dict(parsed: Any) -> Optional[Dict[str, Any]]:
            """Helper function to validate parsed JSON."""
            if not isinstance(parsed, dict):
                return None
            if required_keys and not all(key in parsed for key in required_keys):
                return None
            return parsed
        
        # Look for JSON in code blocks first
        for block_type, content in blocks:
            if block_type.lower() in ['json', '']:
                try:
                    cleaned_content = clean_json_content(content)
                    parsed = json.loads(cleaned_content)
                    if result := validate_json_dict(parsed):
                        return result
                except (json.JSONDecodeError, re.error):
                    continue
        
        # If no valid JSON found in code blocks, try the first {} block
        for block_type, content in blocks:
            if block_type == 'first{}':
                try:
                    cleaned_content = clean_json_content(content)
                    parsed = json.loads(cleaned_content)
                    if result := validate_json_dict(parsed):
                        return result
                except (json.JSONDecodeError, re.error):
                    continue
        
        # If still no valid JSON, try more aggressive extraction
        cleaned_text = re.sub(r'\s+', ' ', text)
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        matches = re.finditer(json_pattern, cleaned_text)
        
        for match in matches:
            try:
                cleaned_content = clean_json_content(match.group())
                parsed = json.loads(cleaned_content)
                if result := validate_json_dict(parsed):
                    return result
            except (json.JSONDecodeError, re.error):
                continue
        
        return None
        
    except Exception as e:
        print(colored(f"Error extracting JSON: {str(e)}", "red"))
        return None

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


def update_cmd_collection():
    client = chromadb.PersistentClient(g.CLIAGENT_PERSISTENT_STORAGE_PATH)
    collection = client.get_or_create_collection(name="commands")
    all_commands = get_atuin_history(200)
    if all_commands:
        for command in all_commands:
            if not collection.get(command)['documents']:
                cmd_embedding = OllamaClient.generate_embedding(command, "bge-m3")
                if not cmd_embedding:
                    break
                collection.add(
                    ids=[command],
                    embeddings=cmd_embedding,
                    documents=[command]
                )

def create_rag_prompt(results: chromadb.QueryResult, user_query: str) -> str:
    if not results['documents'] or not results['metadatas']:
        return "The knowledge database seems empty, please report this to the user as this is likely a bug. A system-supervisor should be informed."
    # Group documents by source
    source_groups = defaultdict(list)
    for document, metadata in zip(*results["documents"], *results["metadatas"]):
        source_groups[metadata['file_path']].append(document)
    # Create the retrieved context string
    retrieved_context = ""
    for source, documents in source_groups.items():
        retrieved_context += f"## SOURCE: {source}\n"
        for document in documents:
            retrieved_context += f"### CONTENT:\n{document}\n"
        retrieved_context += "\n"  # Add an extra newline between sources
    retrieved_context = retrieved_context.strip()
    
    prompt = f"""# QUESTION:\n{user_query}
# CONTEXT:\n{retrieved_context}"""

    return prompt

def get_joined_pdf_contents(pdf_or_folder_path: str) -> str:
    all_contents = []

    if os.path.isfile(pdf_or_folder_path):
        if pdf_or_folder_path.lower().endswith('.pdf'):
            text_content = extract_pdf_content(pdf_or_folder_path)
            # if ("steffen" in text_content.lower()):
            all_contents.append(clean_pdf_text(text_content))
    elif os.path.isdir(pdf_or_folder_path):
        for root, _, files in os.walk(pdf_or_folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    text_content = extract_pdf_content(file_path)
                    # if ("steffen" in text_content.lower()):
                    all_contents.append(clean_pdf_text(text_content))
    else:
        raise ValueError(f"The path {pdf_or_folder_path} is neither a file nor a directory.")

    return "\n\n".join(all_contents)

import subprocess
import time
import base64
from typing import List
import tempfile
import os
import re

def take_screenshot(title: str = 'Firefox', verbose: bool = False) -> List[str]:
    """
    Captures screenshots of all windows with the specified title on Linux
    using xwininfo and import, and returns them as a list of base64 encoded strings.
    
    Args:
    title (str): The title of the windows to capture. Defaults to 'Firefox'.
    verbose (bool): If True, print detailed error messages. Defaults to False.
    
    Returns:
    List[str]: A list of base64 encoded strings of the captured screenshots.
    """
    try:
        # Find windows matching the title
        windows_info = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True, stderr=subprocess.DEVNULL)
        window_ids = re.findall(f'(0x[0-9a-f]+).*{re.escape(title)}', windows_info, re.IGNORECASE)
        
        if not window_ids:
            print(f"No windows with title containing '{title}' found.")
            return []

        base64_images: List[str] = []
        captured_count = 0
        error_count = 0

        for window_id in window_ids:
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_filename = temp_file.name

                # Capture the screenshot using import
                subprocess.run(['import', '-window', window_id, temp_filename], 
                               check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

                # Read the screenshot file
                with open(temp_filename, 'rb') as image_file:
                    png_data = image_file.read()

                # Remove the temporary file
                os.unlink(temp_filename)
                
                # Convert to base64 and add to list
                base64_img = base64.b64encode(png_data).decode('utf-8')
                base64_images.append(base64_img)
                
                captured_count += 1
                if verbose:
                    print(f"Captured screenshot of window: {window_id}")

            except subprocess.CalledProcessError:
                error_count += 1
                if verbose:
                    print(f"Error capturing window {window_id}")
                continue  # Skip this window and continue with the next

        print(f"Successfully captured {captured_count} screenshots.")
        if error_count > 0:
            print(f"Failed to capture {error_count} windows.")

        return base64_images

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []


import re
from lxml import etree
from zxcvbn import zxcvbn
from typing import Set

def clean_and_reduce_html(html: str) -> str:
    """
    Clean and reduce HTML by removing unnecessary attributes and tags,
    and shortening high-entropy or long attribute values.
    
    Args:
    html (str): The input HTML string to clean and reduce.
    
    Returns:
    str: The cleaned and reduced HTML string.
    """
    allowed_attributes: Set[str] = {
        'aria-label', 'role', 'type', 'placeholder', 'name', 'title', 'class', 'href', 'alt', 'src'
    }
    allowed_tags: Set[str] = {'a', 'button', 'input', 'select', 'textarea', 'label', 'img', 'div', 'span'}
    max_attribute_length: int = 100

    def clean_element(element: etree._Element):
        if element.tag.lower() not in allowed_tags:
            element.getparent().remove(element)
            return

        # Clean attributes
        for attr in list(element.attrib.keys()):
            if attr not in allowed_attributes:
                del element.attrib[attr]
            else:
                value = element.attrib[attr]
                if len(value) > max_attribute_length:
                    element.attrib[attr] = value[:max_attribute_length] + '...'
                elif is_high_entropy(value):
                    del element.attrib[attr]

        # Recursively clean child elements
        for child in element:
            clean_element(child)

    def is_high_entropy(value: str, threshold: float = 3.0) -> bool:
        """Check if a string has high entropy using zxcvbn."""
        result = zxcvbn(value)
        return result['entropy'] > threshold

    def remove_css_selectors(html: str) -> str:
        """Remove CSS selectors from class names."""
        def remove_selectors(match):
            classes = match.group(1).split()
            cleaned_classes = [c for c in classes if not re.match(r'^[a-z]+[A-Z]', c)]  # Remove camelCase classes
            return f'class="{" ".join(cleaned_classes)}"'

        return re.sub(r'class="([^"]*)"', remove_selectors, html)

    # Main cleaning process
    parser = etree.HTMLParser()
    tree = etree.fromstring(html, parser)
    clean_element(tree)
    cleaned_html = etree.tostring(tree, encoding='unicode', method='html')
    
    # Remove CSS selectors
    cleaned_html = remove_css_selectors(cleaned_html)
    
    # Additional reduction steps could be added here
    # For example, removing unnecessary whitespace, comments, etc.

    return cleaned_html


def extract_tool_code(text: str) -> Optional[ToolCall]:
    """
    Extract and parse a tool code call from text.
    Handles the new format with tool_code blocks and variables.
    If no tool_code block is found, attempts to find tool code patterns anyway.
    
    Returns:
        Optional[ToolCall]: The parsed tool call or None if no valid tool call was found
    """
    try:
        # Extract code blocks with tool_code language tag
        blocks = extract_blocks_with_metadata(text)
        tool_code_content = None
        
        # First try to get content from tool_code blocks
        for block_type, content in blocks:
            if block_type.lower() == 'tool_code':
                tool_code_content = content
                break
        
        # Fallback: If no tool_code block was found, check for tool code patterns in raw text
        if tool_code_content is None:
            # Look for a pattern like something.run(...) in the text
            tool_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*run\s*\('
            tool_match = re.search(tool_pattern, text)
            if tool_match:
                # Extract everything from the start of the match to a reasonable endpoint
                start_idx = tool_match.start()
                # Find a reasonable endpoint by looking for the closing parenthesis with proper nesting
                end_idx = start_idx
                paren_level = 0
                in_string = False
                string_char = None
                
                for i in range(start_idx, len(text)):
                    char = text[i]
                    
                    if not in_string:
                        if char == '"' or char == "'":
                            in_string = True
                            string_char = char
                        elif char == '(':
                            paren_level += 1
                        elif char == ')':
                            paren_level -= 1
                            if paren_level == 0:
                                end_idx = i + 1  # Include the closing parenthesis
                                break
                    elif char == string_char and (i == 0 or text[i-1] != '\\'):
                        in_string = False
                
                if end_idx > start_idx:
                    # Look for variable declarations before this
                    var_section_start = max(0, text[:start_idx].rfind('\n\n'))
                    if var_section_start == -1:
                        var_section_start = 0
                    
                    # Include any variables declared before the tool code
                    tool_code_content = text[var_section_start:end_idx].strip()
        
        if tool_code_content:
            # Extract TODOs
            todos: List[str] = []
            todo_pattern = r'#\s*TODO:\s*(.*?)(?=$|\n)'
            todo_matches = re.finditer(todo_pattern, tool_code_content, re.IGNORECASE | re.MULTILINE)
            for todo_match in todo_matches:
                todos.append(todo_match.group(1).strip())
            
            # Extract variable declarations
            variables: Dict[str, str] = {}
            var_pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*?)(?=$|\n)'
            var_matches = re.finditer(var_pattern, tool_code_content, re.MULTILINE)
            for var_match in var_matches:
                var_name = var_match.group(1).strip()
                var_value = var_match.group(2).strip()
                # Clean up the value (remove quotes if it's a string)
                if (var_value.startswith('"') and var_value.endswith('"')) or \
                   (var_value.startswith("'") and var_value.endswith("'")):
                    var_value = var_value[1:-1]
                elif var_value in variables:  # Handle variable references in variable declarations
                    var_value = variables[var_value]
                variables[var_name] = var_value
            
            # Find the tool name and start of parameters
            tool_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*run\s*\('
            tool_match = re.search(tool_pattern, tool_code_content)
            if not tool_match:
                return None
                
            tool_name = tool_match.group(1)
            params_start = tool_match.end()
            
            # Find the matching closing parenthesis
            params_end = -1
            paren_level = 1
            in_string = False
            string_char = None
            
            for i in range(params_start, len(tool_code_content)):
                char = tool_code_content[i]
                
                if not in_string:
                    if char == '"' or char == "'":
                        in_string = True
                        string_char = char
                    elif char == '(':
                        paren_level += 1
                    elif char == ')':
                        paren_level -= 1
                        if paren_level == 0:
                            params_end = i
                            break
                elif char == string_char and tool_code_content[i-1] != '\\':
                    in_string = False
            
            if params_end == -1:
                return None
            
            params_str = tool_code_content[params_start:params_end]
            
            # Create the base structure
            result: ToolCall = {
                "tool": tool_name,
                "reasoning": "No specific reasoning provided.",
                "parameters": {}
            }
            
            if todos:
                result["todos"] = todos  # type: ignore
            
            # Split parameters at top-level commas
            params = []
            start = 0
            depth = 0
            in_string = False
            string_char = None
            
            for i, char in enumerate(params_str):
                if not in_string:
                    if char == '"' or char == "'":
                        in_string = True
                        string_char = char
                    elif char in '([{':
                        depth += 1
                    elif char in ')]}':
                        depth -= 1
                    elif char == ',' and depth == 0:
                        params.append(params_str[start:i].strip())
                        start = i + 1
                elif char == string_char and params_str[i-1] != '\\':
                    in_string = False
            
            # Add the last parameter
            if start < len(params_str):
                params.append(params_str[start:].strip())
            
            # Keep track of positional parameter index
            pos_param_idx = 0
            
            # Create typed parameters object
            parameters: ToolCallParameters = {}
            
            # Process each parameter
            for param in params:
                param = param.strip()
                if not param:
                    continue
                    
                # Check if it's a keyword argument or positional
                equals_pos = -1
                in_string = False
                string_char = None
                
                for i, char in enumerate(param):
                    if not in_string:
                        if char == '"' or char == "'":
                            in_string = True
                            string_char = char
                        elif char == '=':
                            equals_pos = i
                            break
                    elif char == string_char and param[i-1] != '\\':
                        in_string = False
                
                # Handle positional vs keyword arguments
                if equals_pos == -1:
                    # This is a positional parameter
                    param_name = str(pos_param_idx)
                    param_value_str = param
                    pos_param_idx += 1
                else:
                    # This is a keyword parameter
                    param_name = param[:equals_pos].strip()
                    param_value_str = param[equals_pos+1:].strip()
                
                # Handle variable references in parameters
                if param_value_str in variables:
                    parameters[param_name] = variables[param_value_str]
                    continue
                
                # Handle string values
                if (param_value_str.startswith('"') and param_value_str.endswith('"')) or \
                   (param_value_str.startswith("'") and param_value_str.endswith("'")):
                    parameters[param_name] = param_value_str[1:-1]
                else:
                    # Try to convert to appropriate type
                    try:
                        if '.' in param_value_str:
                            param_value = float(param_value_str)
                        else:
                            param_value = int(param_value_str)
                    except ValueError:
                        if param_value_str.lower() == 'true':
                            param_value = True
                        elif param_value_str.lower() == 'false':
                            param_value = False
                        elif param_value_str.lower() == 'none':
                            param_value = None
                        else:
                            param_value = param_value_str
                            
                    parameters[param_name] = param_value
            
            # Assign parameters to result
            result["parameters"] = parameters
            
            # Convert positional parameters to a list if any found
            positional_params = []
            i = 0
            while str(i) in parameters:
                positional_params.append(parameters[str(i)])
                del parameters[str(i)]  # type: ignore
                i += 1
                
            if positional_params:
                result["positional_parameters"] = positional_params
            
            return result
        
        return None
        
    except Exception as e:
        print(colored(f"Error extracting tool code: {str(e)}", "red"))
        return None