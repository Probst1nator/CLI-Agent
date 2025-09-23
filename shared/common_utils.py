from collections import defaultdict
import os
import re
import subprocess
import tempfile
from typing import List, Optional, Tuple
import sqlite3
import chromadb
# import pyaudio  # Moved to lazy import in functions that need it
import logging
from termcolor import colored

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.providers.cls_groq_interface import GroqAPI
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from speech_recognition import Microphone, Recognizer, WaitTimeoutError
# Moved pdfminer imports to lazy loading in functions that need them

from core.providers.cls_ollama_interface import OllamaClient
from core.globals import g

import os
import base64



r: Recognizer = None
def calibrate_microphone(calibration_duration: int = 1) -> Microphone:
    """
    Calibrate the microphone for ambient noise.

    Returns:
        sr.Microphone: The calibrated microphone.
    """
    global r, source
    if not r:
        import pyaudio  # Lazy import
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

                            from core.providers.cls_openai_interface import OpenAIAPI
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

# PDF processing functions moved to shared.utils.pdf_utils
# Import PDF utilities from the centralized module

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


def extract_blocks(text: str, language: Optional[str | List[str]] = None) -> List[str]:
    """
    Extracts content from markdown code blocks (```) or XML-style tags (<tag>).

    Args:
        text (str): The text to extract blocks from.
        language (Optional[str | List[str]]): A language identifier (e.g., 'python')
            or a tag name (e.g., 'tool_code') to extract. Can be a single
            string or a list of strings. If None, all blocks are extracted.

    Returns:
        List[str]: A list of the extracted content strings.
    """
    import xml.etree.ElementTree as ET

    blocks = []

    # Normalize the requested language(s) into a set for efficient lookup.
    requested_languages = set()
    if isinstance(language, str):
        requested_languages = {language.lower()}
    elif isinstance(language, list):
        requested_languages = {lang.lower() for lang in language}

    # Handle markdown code blocks first (unchanged logic)
    markdown_pattern = r"```(?P<lang>\w*?)\s*\n(?P<content>.*?)```"
    for match in re.finditer(markdown_pattern, text, re.DOTALL):
        block_identifier = match.group("lang").strip().lower()
        content = match.group("content").strip()

        if not requested_languages or block_identifier in requested_languages:
            blocks.append(content)

    # Handle XML blocks using proper XML parsing
    xml_pattern = r"<(?P<tag>[a-zA-Z0-9_]+)[^>]*>.*?</(?P=tag)>"
    for match in re.finditer(xml_pattern, text, re.DOTALL):
        xml_block = match.group(0)
        tag_name = match.group("tag").lower()

        # Skip if we have specific language requirements and this tag doesn't match
        if requested_languages and tag_name not in requested_languages:
            continue

        try:
            # Parse the XML block using ElementTree
            root = ET.fromstring(xml_block)

            # Extract the inner XML content (everything between opening and closing tags)
            # This preserves the XML structure for tools that expect XML content
            if list(root):
                # If there are child elements, reconstruct the inner XML
                inner_content = ""
                for child in root:
                    inner_content += ET.tostring(child, encoding='unicode').strip()
                # Also include any text content directly in the root
                if root.text and root.text.strip():
                    inner_content = root.text.strip() + inner_content
                content = inner_content
            else:
                # If no child elements, just get the text content
                content = root.text or ""

            blocks.append(content)

        except ET.ParseError as e:
            # Detailed error message with line and column information
            error_msg = f"XML parsing failed for tag '<{tag_name}>' at line {e.lineno}, column {e.offset}: {e.msg}"

            # Log the error for debugging
            import logging
            logging.warning(f"{error_msg}\nMalformed XML block: {xml_block[:200]}...")

            # Continue processing other blocks instead of failing completely
            continue

    return blocks






def update_cmd_collection():
    client = chromadb.PersistentClient(g.CLIAGENT_PERSISTENT_STORAGE_PATH)
    collection = client.get_or_create_collection(name="commands")
    all_commands = get_atuin_history(200)
    if all_commands:
        for command in all_commands:
            if not collection.get(command)['documents']:
                cmd_embedding = OllamaClient.generate_embedding(command)
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


import time
from typing import List
import os

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


