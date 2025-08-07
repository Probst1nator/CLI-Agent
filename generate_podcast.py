import argparse
import os
import re
import struct
import time
import tempfile # Added for temporary file/directory management
import sys
import asyncio
from dotenv import load_dotenv
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.widgets import CheckboxList, Frame, Label
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from py_classes.enum_ai_strengths import AIStrengths

# Google imports
try:
    from google import genai
    from google.genai import types
    from google.genai.types import SpeechConfig
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None
    types = None

from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_classes.globals import g
import pyperclip
from termcolor import colored
from typing import Dict, List, Tuple
import numpy as np

# Add pydub for MP3 conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

# Add PyMuPDF for PDF text extraction
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from py_methods.dia_helper import get_dia_model

load_dotenv(g.CLIAGENT_ENV_FILE_PATH)

DEFAULT_PODCAST_SAVE_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "podcast_generations")
PODCAST_SAVE_LOCATION = DEFAULT_PODCAST_SAVE_LOCATION
GOOGLE_TTS_MODEL_NAME = "gemini-2.5-flash-preview-tts"

async def show_interactive_options() -> Tuple[bool, bool, bool, bool]:
    """
    Show interactive checkbox options using prompt_toolkit with auto-confirm after 10 seconds.
    Returns (multiple_podcasts, single_model, multiple_model, full_local) as booleans.
    Default: multiple_podcasts=False, single_model=True, multiple_model=True, full_local=False
    
    Controls:
    - Arrow keys: Navigate between options
    - Space: Toggle selected option
    - 'e': Confirm selection
    - 'a': Abort selection
    - Auto-confirms after 10 seconds if no input
    """
    
    # Create styled podcast option choices
    option_choices = [
        ("multiple_podcasts", HTML('<option>Multiple podcasts</option> - <description>Generate separate podcasts for subtopics</description>')),
        ("single_model", HTML('<option>Single model generation</option> - <description>Use one LLM with analysis for podcast creation</description>')),
        ("multiple_model", HTML('<option>Multiple model conversation</option> - <description>Use different LLMs for dialogue</description>')),
        ("full_local", HTML('<option>Full local processing</option> - <description>Use Dia TTS locally</description>'))
    ]
    
    # Define the style
    style = Style.from_dict({
        'option': 'ansicyan',
        'description': 'ansigreen',
    })
    
    # Set default selected values (both single and multiple model enabled by default)
    default_selected = ["single_model", "multiple_model"]
    
    checkbox_list = CheckboxList(
        values=option_choices,
        default_values=default_selected
    )
    
    # Track if user has interacted
    user_interacted = False
    
    bindings = KeyBindings()

    @bindings.add("e")
    def _execute(event) -> None:
        nonlocal user_interacted
        user_interacted = True
        # Extract the selected values
        selected = []
        for value, _ in option_choices:
            if value in checkbox_list.current_values:
                selected.append(value)
        app.exit(result=selected)

    @bindings.add("a")
    def _abort(event) -> None:
        nonlocal user_interacted
        user_interacted = True
        app.exit(result=None)
    
    # Add bindings for any other key to mark user interaction
    @bindings.add('<any>')
    def _any_key(event) -> None:
        nonlocal user_interacted
        user_interacted = True

    # Create a countdown label that updates
    countdown_text = "Auto-confirming in 10.0s... (Press any key to interact)"
    instructions = Label(text=f"Use arrow keys to navigate, Space to select/deselect options, 'e' to confirm or 'a' to abort\n{countdown_text}")
    
    root_container = HSplit([
        Frame(title="🎧 Podcast Generation Options", body=checkbox_list),
        instructions
    ])
    layout = Layout(root_container)
    app = Application(layout=layout, key_bindings=bindings, full_screen=False, style=style)
    
    # Create timeout task
    async def timeout_task():
        nonlocal user_interacted
        start_time = time.time()
        while time.time() - start_time < 10.0:
            if user_interacted:
                return  # User interacted, stop countdown
            
            remaining = 10.0 - (time.time() - start_time)
            countdown_text = f"Auto-confirming in {remaining:.1f}s... (Press any key to interact)"
            instructions.text = f"Use arrow keys to navigate, Space to select/deselect options, 'e' to confirm or 'a' to abort\n{countdown_text}"
            
            # Invalidate the app to refresh display
            try:
                app.invalidate()
            except:
                pass  # App might not be running yet
            
            await asyncio.sleep(0.1)
        
        # Timeout reached, auto-confirm with current selection
        if not user_interacted:
            selected = []
            for value, _ in option_choices:
                if value in checkbox_list.current_values:
                    selected.append(value)
            app.exit(result=selected)
    
    try:
        # Create tasks explicitly
        timeout_task_obj = asyncio.create_task(timeout_task())
        app_task_obj = asyncio.create_task(app.run_async())
        
        # Wait for either the app to complete or timeout
        done, pending = await asyncio.wait(
            [app_task_obj, timeout_task_obj], 
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel any pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Get the result from the completed app task
        selected_options = None
        for task in done:
            if not task.cancelled():
                try:
                    selected_options = task.result()
                except:
                    pass
        
        if selected_options is not None:
            # Convert selection to boolean tuple
            multiple_podcasts = "multiple_podcasts" in selected_options
            single_model = "single_model" in selected_options
            multiple_model = "multiple_model" in selected_options
            full_local = "full_local" in selected_options
            
            # Show final configuration
            print(colored("🎧 FINAL PODCAST CONFIGURATION", "cyan", attrs=['bold']))
            print(colored("="*50, "cyan"))
            print(colored(f"☑ Multiple podcasts: {'ENABLED' if multiple_podcasts else 'DISABLED'}", "green" if multiple_podcasts else "red"))
            print(colored(f"☑ Single model: {'ENABLED' if single_model else 'DISABLED'}", "green" if single_model else "red"))
            print(colored(f"☑ Multiple model: {'ENABLED' if multiple_model else 'DISABLED'}", "green" if multiple_model else "red"))
            print(colored(f"☑ Full local: {'ENABLED' if full_local else 'DISABLED'}", "green" if full_local else "red"))
            print(colored("="*50, "cyan"))
            print()
            
            return multiple_podcasts, single_model, multiple_model, full_local
        else:
            print(colored("# cli-agent: Selection cancelled, using defaults", "yellow"))
            return False, True, True, False  # Default values
            
    except Exception as e:
        print(colored(f"# cli-agent: Error during option selection: {str(e)}", "red"))
        print(colored("# cli-agent: Using default options", "yellow"))
        return False, True, True, False  # Default values

def save_binary_file(file_name, data):
    # This function is fine, as it's used for the final binary WAV before pydub
    # or for saving the final MP3 if we were to save it directly as binary (which we aren't).
    # The temporary WAV will be handled by tempfile.
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wb") as f:
        f.write(data)

def save_audio_files(audio_data, sample_rate: int, file_name_base: str) -> str:
    """
    Save audio data as an MP3 file.
    A temporary WAV file is created in the system's temporary directory and then deleted.
    Returns the path to the MP3 file, or an empty string on failure.
    """
    os.makedirs(PODCAST_SAVE_LOCATION, exist_ok=True) # Ensure final output dir exists

    temp_wav_file_path = None
    final_mp3_file_path = os.path.join(PODCAST_SAVE_LOCATION, f"{file_name_base}.mp3")

    try:
        # Step 1: Create and save data as a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file_obj:
            temp_wav_file_path = temp_wav_file_obj.name
        # temp_wav_file_obj is now closed, but the file still exists at temp_wav_file_path

        if isinstance(audio_data, np.ndarray):
            import soundfile as sf
            sf.write(temp_wav_file_path, audio_data, sample_rate)
        else: # Binary data (Google TTS)
            wav_binary_data = create_wav_file(audio_data, sample_rate=sample_rate, bits_per_sample=16, num_channels=1)
            with open(temp_wav_file_path, "wb") as f: # Use standard open for binary write after NamedTemporaryFile
                f.write(wav_binary_data)
        
        # Step 2: Convert temporary WAV to MP3
        if not PYDUB_AVAILABLE:
            print(colored("Warning: pydub not available. Cannot create MP3 file. Install with 'pip install pydub'", "yellow"))
            return ""

        mp3_saved_successfully = False
        try:
            audio = AudioSegment.from_wav(temp_wav_file_path)
            audio.export(final_mp3_file_path, format="mp3", bitrate="192k")
            print(colored(f"MP3 file saved to: {final_mp3_file_path}", "green"))
            mp3_saved_successfully = True
        except Exception as e:
            print(colored(f"Error converting WAV to MP3 ({temp_wav_file_path} -> {final_mp3_file_path}): {e}", "red"))
            if os.path.exists(final_mp3_file_path):
                try: os.remove(final_mp3_file_path)
                except OSError: pass
        
        return final_mp3_file_path if mp3_saved_successfully else ""

    except Exception as e:
        print(colored(f"Error during temporary WAV creation or processing: {e}", "red"))
        return ""
    finally:
        # Step 3: Clean up temporary WAV file if it was created
        if temp_wav_file_path and os.path.exists(temp_wav_file_path):
            try:
                os.remove(temp_wav_file_path)
            except OSError as e_del:
                print(colored(f"Error deleting temporary WAV file {temp_wav_file_path}: {e_del}", "red"))


def _is_speaker_line(line: str) -> bool:
    return bool(re.match(r"^\s*[\w\s.-]+\s*\([\w\s.-]+\):", line))

def _is_quota_exhausted_error(error_message: str) -> bool:
    """Check if an error message indicates Google TTS quota exhaustion."""
    return "429" in error_message and "RESOURCE_EXHAUSTED" in error_message

def _split_podcast_text_into_chunks(text: str, max_chars: int = 600) -> list[str]:
    chunks = []
    current_chunk_lines = []
    input_lines = text.splitlines()
    if not input_lines: return []
    for i, line_to_add in enumerate(input_lines):
        temp_prospective_lines = current_chunk_lines + [line_to_add]
        prospective_char_count = len("\n".join(temp_prospective_lines))
        must_split_before_adding_current_line = False
        if current_chunk_lines:
            if prospective_char_count > max_chars: must_split_before_adding_current_line = True
            elif _is_speaker_line(line_to_add): must_split_before_adding_current_line = True
        if must_split_before_adding_current_line:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
        current_chunk_lines.append(line_to_add)
    if current_chunk_lines: chunks.append("\n".join(current_chunk_lines))
    return [chunk for chunk in chunks if chunk.strip()]


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file using PyMuPDF.
    Returns the extracted text as a string."""
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF (fitz) is required for PDF support. Install with: pip install PyMuPDF")
    
    try:
        doc = fitz.open(pdf_path)
        text_content = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content += page.get_text() + "\n"
        
        doc.close()
        return text_content.strip()
    
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {e}")


def generate_podcast(podcast_dialogue: str, title: str, use_local_dia: bool = False) -> str:
    safe_title_for_filename = re.sub(r'[^\w_-]+', '', title.replace(' ', '_')).strip('_')
    os.makedirs(PODCAST_SAVE_LOCATION, exist_ok=True)
    os.makedirs(os.path.join(PODCAST_SAVE_LOCATION, "transcripts"), exist_ok=True)

    # save text to file
    with open(os.path.join(PODCAST_SAVE_LOCATION, "transcripts", f"{safe_title_for_filename}.txt"), "w") as f:
        f.write(podcast_dialogue)

    if use_local_dia:
        text_chunks = _split_podcast_text_into_chunks(podcast_dialogue, max_chars=600)
        print(colored(f"[{title}] Successfully split into {len(text_chunks)} chunks for Dia.", "blue"))
    else:
        text_chunks = _split_podcast_text_into_chunks(podcast_dialogue, max_chars=4500)
        print(colored(f"[{title}] Successfully split into {len(text_chunks)} chunks for Google TTS.", "blue"))

    if use_local_dia:
        print(colored(f"[{title}] Attempting to use local Dia TTS model.", "magenta"))
        try:
            import soundfile as sf
        except ImportError:
            print(colored("Error: 'soundfile' library not found for Dia. Please install it with 'pip install soundfile'", "red"))
            return ""

        print(colored(f"[{title}] NOTE: Dia TTS requires a compatible GPU and may take time to load the model.", "yellow"))
        dia_model = get_dia_model()
        if dia_model is None:
            print(colored("Failed to initialize Dia model. Falling back to Google TTS if available...", "red"))
            if GOOGLE_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
                 return generate_podcast(podcast_dialogue, title, use_local_dia=False)
            else:
                print(colored("Google TTS fallback not possible (unavailable or API key missing).", "red"))
                return ""
        print(colored(f"[{title}] Dia model loaded.", "blue"))

        mp3_final_path = ""
        # Use TemporaryDirectory for Dia's audio chunks
        with tempfile.TemporaryDirectory() as temp_dia_chunks_dir:
            audio_chunk_files = [] # Store paths to temporary chunk WAV files within temp_dia_chunks_dir
            total_chunks = len(text_chunks)
            for i, chunk in enumerate(text_chunks): 
                if not chunk.strip(): continue
                print(colored(f"[{title}] Dia: Generating audio for chunk {i+1}/{total_chunks} ({len(chunk)} chars)...", "green"))
                try:
                    output_waveform = dia_model.generate(chunk)
                    dia_sample_rate = 44100 
                    if output_waveform is None or output_waveform.size == 0:
                        print(colored(f"[{title}] Dia: No audio data for chunk {i+1}.", "yellow"))
                        continue
                    
                    temp_chunk_filename = f"temp_dia_chunk_{i+1:03d}.wav"
                    temp_chunk_filepath = os.path.join(temp_dia_chunks_dir, temp_chunk_filename)
                    sf.write(temp_chunk_filepath, output_waveform, dia_sample_rate)
                    audio_chunk_files.append(temp_chunk_filepath)
                    print(colored(f"[{title}] Dia: Chunk {i+1} audio generated and saved to temp dir.", "cyan"))
                except Exception as e:
                    print(colored(f"[{title}] Dia: Error for chunk {i+1}: {e}", "red"))
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not audio_chunk_files:
                print(colored(f"[{title}] Dia: No audio chunks generated.", "red"))
                return "" # Exits the 'with' block, temp_dia_chunks_dir is cleaned up

            print(colored(f"[{title}] Dia: Merging {len(audio_chunk_files)} audio chunks from temp dir...", "blue"))
            try:
                merged_audio_dia_list = []
                final_sample_rate_dia = None
                for af_path in audio_chunk_files: # af_path is a path within temp_dia_chunks_dir
                    audio_data, sr = sf.read(af_path)
                    if final_sample_rate_dia is None: final_sample_rate_dia = sr
                    elif sr != final_sample_rate_dia:
                        print(colored(f"[{title}] Dia: Warning sample rate mismatch. Using {final_sample_rate_dia} Hz.", "yellow"))
                    merged_audio_dia_list.append(audio_data)
                
                final_audio_dia_np = np.concatenate(merged_audio_dia_list)
                
                file_extension_for_numbering = ".mp3"
                highest_num = 0
                file_pattern = re.compile(rf"(\d+)_.*{re.escape(file_extension_for_numbering)}$")
                # Ensure PODCAST_SAVE_LOCATION exists before listing its directory
                if os.path.exists(PODCAST_SAVE_LOCATION):
                    for f_name in os.listdir(PODCAST_SAVE_LOCATION):
                        match = file_pattern.match(f_name)
                        if match:
                            num = int(match.group(1))
                            if num > highest_num: highest_num = num
                episode_number = highest_num + 1
                file_name_base = f"{episode_number:03d}_{safe_title_for_filename}"
                
                mp3_final_path = save_audio_files(final_audio_dia_np, final_sample_rate_dia, file_name_base)
                
                if mp3_final_path:
                    print(colored(f"[{title}] Dia podcast processing complete. Final MP3 available.", "green"))
                else:
                    print(colored(f"[{title}] Dia podcast MP3 generation failed.", "red"))
                
            except Exception as e:
                print(colored(f"[{title}] Dia: Error merging or saving audio: {e}", "red"))
            # TemporaryDirectory will clean up audio_chunk_files automatically when this 'with' block exits
        return mp3_final_path

    else: # Google TTS Implementation
        print(colored(f"[{title}] Using Google GenAI TTS model: {GOOGLE_TTS_MODEL_NAME} for sequential chunk processing.", "magenta"))
        if not GOOGLE_AVAILABLE:
            print(colored("Error: Google dependencies not available. Install with 'pip install google-genai'.", "red"))
            return ""
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            print(colored("Error: GEMINI_API_KEY environment variable not set.", "red"))
            return ""
        try:
            client = genai.Client(api_key=gemini_api_key)
        except Exception as e:
            print(colored(f"Failed to initialize Google GenAI Client: {e}", "red"))
            return ""

        generate_content_config = types.GenerateContentConfig(
            response_modalities=["audio"],
            speech_config=SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(speaker="Chloe", voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore"))),
                        types.SpeakerVoiceConfig(speaker="Liam", voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="iapetus"))),
                    ]
                ),
            ),
        )
        all_audio_data_bytes = b""
        auto_retry_seconds = 30
        max_retries = 3
        total_chunks = len(text_chunks)
        for i, l_chunk in enumerate(text_chunks):
            chunk = f"TTS the following conversation between Liam and Chloe, do not TTS ANYTHING that is in brackets, those are instructions guiding the tone and delivery of the dialogue:\n{l_chunk}"
            if not chunk.strip(): continue
            print(colored(f"[{title}] Google TTS: Generating audio for chunk {i+1}/{total_chunks} ({len(chunk)} chars)...", "green"))
            retry_count = 0
            chunk_audio_data = b""
            while retry_count <= max_retries:
                try:
                    response_stream = client.models.generate_content_stream(model=f"models/{GOOGLE_TTS_MODEL_NAME}", contents=chunk, config=generate_content_config)
                    current_chunk_audio = b""
                    for response_part in response_stream:
                        if not (response_part.candidates and response_part.candidates[0].content and response_part.candidates[0].content.parts): continue
                        api_part_data = response_part.candidates[0].content.parts[0]
                        if hasattr(api_part_data, 'inline_data') and api_part_data.inline_data and hasattr(api_part_data.inline_data, 'data'):
                            current_chunk_audio += api_part_data.inline_data.data
                        elif hasattr(api_part_data, 'text') and api_part_data.text:
                            print(colored(f"ERROR: [{title}] API returned text instead of audio for chunk {i+1}: {api_part_data.text}", "red"))
                            current_chunk_audio = b""; break
                    if current_chunk_audio:
                        chunk_audio_data = current_chunk_audio
                        print(colored(f"[{title}] Google TTS: Chunk {i+1} audio generated ({len(chunk_audio_data)} bytes).", "cyan")); break
                except Exception as e:
                    error_message = str(e)
                    if "stop" in error_message.lower() or "candidate" in error_message.lower():
                        print(colored(f"[{title}] Generation stopped for chunk {i+1}: {e}", "red")); break
                    elif _is_quota_exhausted_error(error_message):
                        print(colored(f"[{title}] Google TTS quota exhausted for chunk {i+1}: {e}", "red"))
                        # Check if local Dia fallback is possible
                        try:
                            print(colored(f"[{title}] Attempting fallback to local Dia TTS due to quota exhaustion...", "yellow"))
                            return generate_podcast(podcast_dialogue, title, use_local_dia=True)
                        except Exception as fallback_error:
                            print(colored(f"[{title}] Local Dia fallback failed: {fallback_error}", "red"))
                            print(colored(f"[{title}] Google TTS quota exhausted and no local fallback available.", "red"))
                            print(colored(f"[{title}] Please wait for quota reset or use --local flag for Dia TTS.", "yellow"))
                            return ""  # Return empty to signal failure
                    else: 
                        print(colored(f"[{title}] Error generating audio for chunk {i+1}: {e}", "red"))
                
                retry_count += 1
                # Don't retry on quota exhaustion errors
                if _is_quota_exhausted_error(error_message):
                    print(colored(f"[{title}] Skipping retries due to quota exhaustion.", "yellow"))
                    break
                elif retry_count <= max_retries:
                    print(colored(f"Retrying chunk {i+1} (attempt {retry_count}/{max_retries}) in {auto_retry_seconds} seconds...", "yellow")); time.sleep(auto_retry_seconds)
                else: 
                    print(colored(f"[{title}] Failed to process chunk {i+1} after {max_retries} attempts. Skipping.", "red")); break
            if chunk_audio_data: all_audio_data_bytes += chunk_audio_data
            else: print(colored(f"[{title}] No audio data for chunk {i+1}. Continuing.", "yellow"))

        if not all_audio_data_bytes:
            print(colored(f"[{title}] No audio data was generated for any chunks (Google TTS).", "red")); return ""
        print(colored(f"[{title}] Successfully gathered audio data from {total_chunks} chunks for Google TTS.", "green"))

        file_extension_for_numbering = ".mp3"
        highest_num = 0
        file_pattern = re.compile(rf"(\d+)_.*{re.escape(file_extension_for_numbering)}$")
        if os.path.exists(PODCAST_SAVE_LOCATION): # Ensure dir exists before listing
            for f_name in os.listdir(PODCAST_SAVE_LOCATION):
                match = file_pattern.match(f_name)
                if match:
                    num = int(match.group(1))
                    if num > highest_num: highest_num = num
        episode_number = highest_num + 1
        file_name_base = f"{episode_number:03d}_{safe_title_for_filename}"
        
        mp3_final_path = save_audio_files(all_audio_data_bytes, 24000, file_name_base)
        if mp3_final_path: print(colored(f"[{title}] Google TTS podcast processing complete.", "green"))
        else: print(colored(f"[{title}] Google TTS podcast MP3 generation failed.", "red"))
        return mp3_final_path

def _clean_speaker_response(response: str, speaker_name: str) -> str:
    """Remove speaker name prefix from response if present."""
    response = response.strip()
    
    # More comprehensive cleaning - remove various speaker prefix formats
    import re
    
    # Pattern to match various speaker formats at the beginning
    patterns = [
        rf"^{re.escape(speaker_name)}\s*:\s*",  # "Name: "
        rf"^\*\*{re.escape(speaker_name)}\*\*\s*:\s*",  # "**Name**: "
        rf"^\*{re.escape(speaker_name)}\*\s*:\s*",  # "*Name*: "
        rf"^{re.escape(speaker_name)}\s*\([^)]+\)\s*:\s*",  # "Name (description): "
        rf"^\[{re.escape(speaker_name)}\]\s*:\s*",  # "[Name]: "
        rf"^{re.escape(speaker_name)}\s*-\s*",  # "Name - "
    ]
    
    for pattern in patterns:
        response = re.sub(pattern, "", response, flags=re.IGNORECASE)
    
    return response.strip()

async def _generate_llm_conversation(content: str, llms: List[str], force_local: bool = False) -> Tuple[str, str]:
    """Generate a conversation between two LLMs about the given content.
    Returns (dialogue, title)"""
    
    # First, generate a title for the content
    print(colored("Generating title for LLM conversation...", "blue"))
    try:
        title_response = await LlmRouter.generate_completion(
            f"Generate a very, descriptive title (3-5 words) for a podcast discussion about the following content. "
            f"The title should be suitable for a filename (no special characters). "
            f"Respond with just the title, nothing else.\n\n{content[:1000]}...",  # Limit content for title generation
            force_local=force_local
        )
        extracted_title = re.sub(r'[^\w\s-]', '', title_response.strip()).strip()
        extracted_title = re.sub(r'\s+', '_', extracted_title)
        if not extracted_title or len(extracted_title) > 50:
            extracted_title = "LLM_Conversation_Podcast"
    except Exception as e:
        print(colored(f"Error generating title: {e}", "red"))
        extracted_title = "LLM_Conversation_Podcast"
    
    print(colored(f"Generated Title: {extracted_title}", "magenta"))
    
    # Set up the conversation
    if llms is None or len(llms) == 0:
        # Use automatic model selection with qwen2.5vl:3b as default
        chloe_llm = "qwen2.5vl:3b"
        liam_llm = None  # Will use auto-selected small model
        chloe_model_info = "Chloe (qwen2.5vl:3b)"
        liam_model_info = "Liam (auto-selected small model)"
    else:
        chloe_llm = llms[0]
        liam_llm = llms[1] if len(llms) > 1 else None
        chloe_model_info = f"Chloe ({chloe_llm})"
        liam_model_info = f"Liam ({liam_llm})" if liam_llm else "Liam (auto-selected small model)"
    
    print(colored(f"Setting up conversation between {chloe_model_info} and {liam_model_info}", "blue"))
    
    # Initialize conversation context
    conversation_context = f"""You are participating in a podcast conversation about the following content:

{content}

You should engage in a natural, flowing discussion that explores the key insights, concepts, and implications of this content. Keep your responses conversational and engaging, as this will be converted to audio.

Include natural speech patterns and occasional parenthetical cues for emotional tone like (laughs), (thoughtfully), (enthusiastically), etc.

Focus on making the content accessible and interesting to listeners."""

    # Initialize chat objects for both LLMs
    chloe_chat = Chat()
    liam_chat = Chat()
    
    # Set up initial system messages
    chloe_system_prompt = f"""You are Chloe, an intelligent and creative student in a podcast conversation with Liam. {conversation_context}

CRITICAL: Provide ONLY ONE response per turn. Do not continue speaking or create multiple paragraphs or responses.

- You are a learning student
- You always attempt to judge your understanding of the topic by very explicitly making assumptions and inferring solutions
- Keep responses to 1-5 sentences maximum
- Use the name Liam when addressing your conversation partner
- Stop after your single response and wait for Liam to reply"""

    liam_system_prompt = f"""You are Liam, a self-taught established expert in a podcast conversation with Chloe. {conversation_context}

CRITICAL: Provide ONLY ONE response per turn. Do not continue speaking or create multiple paragraphs or responses.

You should:
- Understand your partner, clarify ideas, and expand the educational content of the dialogue
- Reframe questions carefully and think out loud when you do
- Avoid agreeing multiple times consecutively, instead enrich the conversation with your own insights, examples or mental exercises
- Keep responses to 1-4 sentences maximum
- Use the name Chloe when addressing your conversation partner
- Stop after your single response and wait for Chloe to reply"""

    # Determine who starts based on whether using automatic selection or only 1 LLM provided
    only_one_llm_provided = llms is not None and len(llms) == 1
    automatic_selection = llms is None or len(llms) == 0
    liam_starts = only_one_llm_provided or automatic_selection
    
    # Add system prompts as USER messages to set up the context
    if liam_starts:
        # Liam starts when only 1 LLM provided OR when using automatic selection
        liam_chat.add_message(Role.USER, f"{liam_system_prompt}\n\nProvide ONLY a single, short welcome message (1-2 sentences) to start the podcast and introduce the topic. Do not continue speaking or add multiple paragraphs. End your response and wait for Chloe to reply.")
        chloe_chat.add_message(Role.USER, f"{chloe_system_prompt}\n\nWait for Liam to start, then respond naturally to continue the conversation.")
        if automatic_selection:
            print(colored("Liam (auto-selected small model) will start the conversation.", "yellow"))
        else:
            print(colored("Liam (specified model) will start the conversation.", "yellow"))
    else:
        # Chloe starts when 2 specific LLMs provided
        chloe_chat.add_message(Role.USER, f"{chloe_system_prompt}\n\nProvide ONLY a single, short welcome message (1-2 sentences) to start the podcast and introduce the topic. Do not continue speaking or add multiple paragraphs. End your response and wait for Liam to reply.")
        liam_chat.add_message(Role.USER, f"{liam_system_prompt}\n\nWait for Chloe to start, then respond naturally to continue the conversation.")
        print(colored("Chloe will start the conversation.", "yellow"))
    
    # Start the conversation
    dialogue_lines = []
    
    print(colored("Starting multi llm podcast conversation...", "green"))
    
    for exchange in range(100):
        try:
            liam_chat.debug_title = f"Podcast exchange {exchange + 1}"
            chloe_chat.debug_title = f"Podcast exchange {exchange + 1}"
            if exchange > 15:
                response = await LlmRouter.generate_completion("Decide if the conversation is finished. If you are highly certain it has finished conclusively (like the podcast actors waving goodbye), respond with 'finished'. If it is not, respond with 'continue'.", strengths=[AIStrengths.SMALL], force_local=force_local)
                if "finished" in response.lower():
                    print(colored("Conversation was detected as finished.", "green"))
                    break
            
            if liam_starts:
                # Liam starts when only 1 LLM provided OR automatic selection
                # Liam's turn first
                if liam_llm:
                    liam_response = await LlmRouter.generate_completion(liam_chat, preferred_models=[liam_llm], force_local=force_local)
                else:
                    liam_response = await LlmRouter.generate_completion(liam_chat, strengths=[AIStrengths.SMALL], force_local=force_local)
                liam_chat.add_message(Role.ASSISTANT, liam_response)
                cleaned_liam_response = _clean_speaker_response(liam_response, "Liam")
                dialogue_lines.append(f"Liam: {cleaned_liam_response}")
                
                # Add Liam's response as USER message to Chloe's chat
                chloe_chat.add_message(Role.USER, liam_response)
                
                # Chloe's turn
                if chloe_llm:
                    chloe_response = await LlmRouter.generate_completion(chloe_chat, preferred_models=[chloe_llm], temperature=1.3, force_local=force_local)
                else:
                    chloe_response = await LlmRouter.generate_completion(chloe_chat, strengths=[AIStrengths.SMALL], temperature=1.3, force_local=force_local)
                chloe_chat.add_message(Role.ASSISTANT, chloe_response)
                cleaned_chloe_response = _clean_speaker_response(chloe_response, "Chloe")
                dialogue_lines.append(f"Chloe: {cleaned_chloe_response}")
                
                # Add Chloe's response as USER message to Liam's chat
                liam_chat.add_message(Role.USER, chloe_response)
            else:
                # Chloe starts when 2 specific LLMs provided
                # Chloe's turn first
                if chloe_llm:
                    chloe_response = await LlmRouter.generate_completion(chloe_chat, preferred_models=[chloe_llm], force_local=force_local)
                else:
                    chloe_response = await LlmRouter.generate_completion(chloe_chat, strengths=[AIStrengths.SMALL], force_local=force_local)
                chloe_chat.add_message(Role.ASSISTANT, chloe_response)
                cleaned_chloe_response = _clean_speaker_response(chloe_response, "Chloe")
                dialogue_lines.append(f"Chloe: {cleaned_chloe_response}")
                
                # Add Chloe's response as USER message to Liam's chat
                liam_chat.add_message(Role.USER, chloe_response)
                
                # Liam's turn
                if liam_llm:
                    liam_response = await LlmRouter.generate_completion(liam_chat, preferred_models=[liam_llm], force_local=force_local)
                else:
                    liam_response = await LlmRouter.generate_completion(liam_chat, strengths=[AIStrengths.SMALL], force_local=force_local)
                liam_chat.add_message(Role.ASSISTANT, liam_response)
                cleaned_liam_response = _clean_speaker_response(liam_response, "Liam")
                dialogue_lines.append(f"Liam: {cleaned_liam_response}")
                
                # Add Liam's response as USER message to Chloe's chat
                liam_chat.add_message(Role.USER, liam_response)
            
        except KeyboardInterrupt:
            print(colored("\nConversation interrupted by user (Ctrl+C). Ending podcast generation...", "yellow"))
            break
        except Exception as e:
            # Check if the error is due to user interruption
            if "interrupted by user" in str(e).lower() or "ctrl+c" in str(e).lower():
                print(colored("\nConversation interrupted by user (Ctrl+C). Ending podcast generation...", "yellow"))
                break
            
            print(colored(f"Error in exchange {exchange + 1}: {e}", "red"))
            # Try to continue with a generic response
            if liam_starts:
                # Liam speaks first when only 1 LLM provided
                if exchange % 2 == 0:  # Liam's turn (first speaker)
                    fallback_response = "I completely agree. Let's explore that further."
                    dialogue_lines.append(f"Liam: {fallback_response}")
                    chloe_chat.add_message(Role.USER, fallback_response)
                else:  # Chloe's turn
                    fallback_response = "That's a fascinating point. (thoughtfully)"
                    dialogue_lines.append(f"Chloe: {fallback_response}")
                    liam_chat.add_message(Role.USER, fallback_response)
            else:
                # Chloe speaks first when 2 LLMs provided
                if exchange % 2 == 0:  # Chloe's turn (first speaker)
                    fallback_response = "That's a fascinating point. (thoughtfully)"
                    dialogue_lines.append(f"Chloe: {fallback_response}")
                    liam_chat.add_message(Role.USER, fallback_response)
                else:  # Liam's turn
                    fallback_response = "I completely agree. Let's explore that further."
                    dialogue_lines.append(f"Liam: {fallback_response}")
                    chloe_chat.add_message(Role.USER, fallback_response)
            continue
    
    final_dialogue = "\n".join(dialogue_lines)
    print(colored("LLM conversation generation complete!", "green"))
    
    return final_dialogue, extracted_title


def create_wav_file(pcm_data: bytes, sample_rate: int = 24000, bits_per_sample: int = 16, num_channels: int = 1) -> bytes:
    data_size = len(pcm_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    riff_chunk_id = b"RIFF"; riff_chunk_size = 36 + data_size; wave_format = b"WAVE"
    fmt_chunk_id = b"fmt "; fmt_chunk_size = 16; audio_format = 1
    data_chunk_id = b"data"; data_chunk_size = data_size
    wav_header = struct.pack("<4sI4s4sIHHIIHH4sI", riff_chunk_id, riff_chunk_size, wave_format, fmt_chunk_id, fmt_chunk_size, audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample, data_chunk_id, data_chunk_size)
    return wav_header + pcm_data

async def analyze_content_for_subtopics(content: str) -> List[Dict[str, str]]:
    """Analyze content to identify distinct subtopics and extract relevant sections.
    Returns list of dictionaries with 'title' and 'content' keys."""
    
    print(colored("Analyzing content to identify distinct subtopics...", "blue"))
    
    try:
        analysis_prompt = f"""Analyze the following content and identify distinct, separable subtopics that could each warrant their own focused discussion or podcast episode.

For each subtopic you identify:
1. Provide a descriptive title (3-6 words, suitable for a filename)
2. Extract the most relevant content sections that relate to that subtopic
3. Ensure each subtopic has enough substance for a meaningful discussion (at least 200-300 words of content)

Format your response as a JSON array where each object has:
- "title": "Descriptive Title"
- "content": "Relevant extracted content for this subtopic"

Only identify subtopics that are truly distinct and substantial. If the content is better suited as a single topic, return only one subtopic.

Content to analyze:
{content}

Respond with ONLY the JSON array, no additional text."""

        response = await LlmRouter.generate_completion(analysis_prompt)
        
        # Try to extract JSON from the response
        import json
        try:
            # Look for JSON array in the response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                subtopics = json.loads(json_str)
                
                # Validate the structure
                valid_subtopics = []
                for subtopic in subtopics:
                    if isinstance(subtopic, dict) and 'title' in subtopic and 'content' in subtopic:
                        if len(subtopic['content'].strip()) >= 200:  # Minimum content length
                            valid_subtopics.append({
                                'title': subtopic['title'].strip(),
                                'content': subtopic['content'].strip()
                            })
                        else:
                            print(colored(f"Skipping subtopic '{subtopic.get('title', 'Unknown')}' - insufficient content", "yellow"))
                
                if valid_subtopics:
                    print(colored(f"Identified {len(valid_subtopics)} substantial subtopics", "green"))
                    for i, subtopic in enumerate(valid_subtopics, 1):
                        print(colored(f"  {i}. {subtopic['title']} ({len(subtopic['content'])} chars)", "cyan"))
                    return valid_subtopics
                else:
                    print(colored("No substantial subtopics identified, treating as single topic", "yellow"))
                    return [{'title': 'Complete_Discussion', 'content': content}]
            else:
                raise ValueError("No JSON array found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(colored(f"Could not parse subtopics from LLM response: {e}", "yellow"))
            print(colored("Treating content as single topic", "yellow"))
            return [{'title': 'Complete_Discussion', 'content': content}]
            
    except Exception as e:
        print(colored(f"Error analyzing content for subtopics: {e}", "red"))
        print(colored("Falling back to single topic processing", "yellow"))
        return [{'title': 'Complete_Discussion', 'content': content}]

async def generate_multiple_podcasts(content: str, llms: List[str] = None, use_local_dia: bool = False, use_multiple_model: bool = False) -> List[str]:
    """Generate multiple podcasts for different subtopics identified in the content.
    Returns list of generated MP3 file paths."""
    
    subtopics = await analyze_content_for_subtopics(content)
    
    if len(subtopics) == 1:
        print(colored("Content identified as single topic, generating one podcast", "blue"))
        # Use the existing single podcast generation
        if use_multiple_model:
            dialogue, title = await _generate_llm_conversation(content, llms, force_local=use_local_dia)
        else:
            dialogue, title = await process_raw_content_to_dialogue(content, llms)
        mp3_path = generate_podcast(dialogue, title, use_local_dia)
        return [mp3_path] if mp3_path else []
    
    print(colored(f"Generating {len(subtopics)} separate podcasts for identified subtopics...", "magenta"))
    
    generated_files = []
    quota_exhausted = False
    
    for i, subtopic in enumerate(subtopics, 1):
        if quota_exhausted:
            print(colored(f"\n--- Skipping remaining subtopics ({i}/{len(subtopics)}) due to quota exhaustion ---", "yellow"))
            break
            
        print(colored(f"\n--- Processing subtopic {i}/{len(subtopics)}: {subtopic['title']} ---", "magenta"))
        
        try:
            # Generate dialogue for this subtopic
            if use_multiple_model:
                dialogue, title = await _generate_llm_conversation(subtopic['content'], llms, force_local=use_local_dia)
            else:
                dialogue, title = await process_raw_content_to_dialogue(subtopic['content'], llms)
            
            # Use the subtopic title, but add a prefix to distinguish it
            subtopic_title = f"Subtopic_{i:02d}_{subtopic['title']}"
            
            # Generate the podcast
            mp3_path = generate_podcast(dialogue, subtopic_title, use_local_dia)
            
            if mp3_path:
                generated_files.append(mp3_path)
                print(colored(f"✓ Completed subtopic {i}: {mp3_path}", "green"))
            else:
                print(colored(f"✗ Failed to generate podcast for subtopic {i}: {subtopic['title']}", "red"))
                # Check if this was a quota exhaustion failure
                if not use_local_dia:
                    print(colored("💡 Hint: Use --local flag to enable local Dia TTS fallback for remaining subtopics", "cyan"))
                    quota_exhausted = True  # Stop processing remaining subtopics
                
        except Exception as e:
            error_message = str(e)
            if _is_quota_exhausted_error(error_message):
                print(colored(f"✗ Google TTS quota exhausted while processing subtopic {i} ({subtopic['title']}): {e}", "red"))
                quota_exhausted = True
                if not use_local_dia:
                    print(colored("💡 Restart with --local flag to use Dia TTS for remaining subtopics", "cyan"))
            else:
                print(colored(f"✗ Error processing subtopic {i} ({subtopic['title']}): {e}", "red"))
            continue
    
    if generated_files:
        print(colored(f"\n🎉 Successfully generated {len(generated_files)} podcasts!", "green"))
        if quota_exhausted:
            print(colored(f"⚠️  Note: Processing stopped due to Google TTS quota exhaustion after {len(generated_files)} successful podcasts", "yellow"))
        print(colored("Generated files:", "green"))
        for file_path in generated_files:
            print(colored(f"  📄 {file_path}", "cyan"))
        
        if quota_exhausted:
            print(colored("\n💡 To process remaining subtopics:", "cyan"))
            print(colored("   • Wait for Google TTS quota reset (typically 24 hours)", "cyan"))
            print(colored("   • Or use --local flag to enable local Dia TTS fallback", "cyan"))
            print(colored("   • Or upgrade to a paid Google API plan for higher quotas", "cyan"))
    else:
        print(colored("❌ No podcasts were successfully generated", "red"))
        if quota_exhausted:
            print(colored("This was due to Google TTS quota exhaustion. Consider using --local flag for Dia TTS.", "yellow"))
    
    return generated_files

async def process_raw_content_to_dialogue(content: str, llms: List[str] = None, force_local: bool = False) -> Tuple[str, str]:
    """Process raw text content through analysis and dialogue generation.
    Returns (dialogue, title)"""
    if llms:
        print(colored(f"Using LLMs for conversation: {llms}", "magenta"))
        if len(llms) < 1 or len(llms) > 2:
            print(colored("Error: Provide 1 LLM (other speaker uses default) or 2 LLMs for conversation", "red"))
            exit(1)
        return await _generate_llm_conversation(content, llms, force_local=force_local)
    else:
        print(colored("Analyzing content for summary and title...", "blue"))
    try:
        analysisResponse = await LlmRouter.generate_completion(
            f"For the following text snippet, please highlight key insights and lay out the fundamentals of the topic of concern. "
            f"Explore analogies to other fields and highlight related concepts to manifest a constructive framework for the topic.\n"
            f"Here's the raw unfiltered text snippet:\n{content}",
            force_local=force_local
        )
        titleResponseText = await LlmRouter.generate_completion(
            f"I am going to provide you with a text and you should generate a descriptive title for whatever content it's about. "
            f"Ensure it is a short fit for a file name. Please think about the contents first and then like this:\n"
            f"Title: <title>\nThis is the text:\n{analysisResponse}",
            force_local=force_local
        )
    except Exception as e:
        print(colored(f"Error during LLM processing for analysis/title: {e}", "red"))
        import traceback
        traceback.print_exc()
        exit(1)

    # Extract and clean title
    title_match = re.search(r"Title:\s*(.*)", titleResponseText, re.IGNORECASE)
    extracted_title = title_match.group(1).strip() if title_match else titleResponseText.splitlines()[0].strip()
    MAX_TITLE_LEN = 50 
    if len(extracted_title) > MAX_TITLE_LEN: 
        extracted_title = extracted_title[:MAX_TITLE_LEN]
    extracted_title = re.sub(r'[^\w\s-]', '', extracted_title).strip()
    extracted_title = re.sub(r'\s+', '_', extracted_title)
    if not extracted_title: 
        extracted_title = "Untitled_Podcast"

    print(colored(f"Generated Title: {extracted_title}", "magenta"))
    print(colored("Generating podcast dialogue...", "blue"))
    
    podcastGenPrompt = f"""Use a deepthinking subroutine before providing your final response to this query. Create a meditative and joyful expert discussion between an intelligent and very creative and educated student and an self-taught established expert.
The discussion should revolve around powerful insights and intuitions that enable easy understanding and retention of the topic.
To guide the emotional tone and style of the speech, include concise parenthetical cues like (laughs), (stutters), (happy), (sadly), (whispering), or (enthusiastically) directly within the dialogue.
IMPORTANT: These cues are instructions for how the dialogue should be *spoken* by the text-to-speech system, and should NOT be spoken aloud themselves. They should only guide the voice's delivery. Because of this, do not use brackets for anything else.
For example, if Chloe says something funny, write: Chloe: That's absolutely hilarious! (laughs)
If Liam is hesitant: Liam: I, uh, (stutters) I'm not entirely sure about that.
If Chloe is expressing joy: Chloe: I'm so glad we covered this topic! (happy)
To provide the dialogue please use the following delimiters and this exact format:
```txt
Chloe: Welcome!
Liam: It's great to be here, lets tackle the topic of {extracted_title}.
...
You don't need to use my example introduction, you can create a more fitting one for the topic.
The topic of {extracted_title} has been automatically reviewed and the following insights were distilled and are provided to enhance the clarity and depth of the exchange:
{analysisResponse}"""
    
    try:
        podcastGenChat = Chat()
        podcastGenChat.add_message(Role.USER, podcastGenPrompt)
        # podcastGenChat.add_message(Role.ASSISTANT, "<think>")
        podcastDialogueResponse = await LlmRouter.generate_completion(podcastGenChat, force_local=force_local)
    except Exception as e: 
        print(colored(f"Error during LLM processing for podcast dialogue: {e}", "red"))
        exit(1)
    
    # Extract dialogue from response
    from py_methods.utils import extract_blocks 
    extracted_dialogue_blocks = extract_blocks(podcastDialogueResponse, ["txt", "text"])
    podcastDialogue = ""
    if extracted_dialogue_blocks:
        podcastDialogue = "\n".join(block.strip() for block in extracted_dialogue_blocks if block.strip()) if isinstance(extracted_dialogue_blocks, list) else str(extracted_dialogue_blocks).strip()
    else:
        print(colored("Could not extract dialogue from LLM response using ```txt blocks. Using raw response after attempting cleanup.", "yellow"))
        prompt_intro_for_cleanup = "Create a meditative expert discussion"
        actual_dialogue_text_start = podcastDialogueResponse[podcastDialogueResponse.find(prompt_intro_for_cleanup):] if prompt_intro_for_cleanup in podcastDialogueResponse else podcastDialogueResponse
        dialogue_start_markers = ["Liam:", "Chloe:", "[S1]", "[S2]"]
        first_speaker_occurrence = -1
        found_marker_pos = -1
        for marker in dialogue_start_markers:
            pos = actual_dialogue_text_start.find(marker)
            if pos != -1 and (first_speaker_occurrence == -1 or pos < first_speaker_occurrence): 
                first_speaker_occurrence = pos
                found_marker_pos = pos
        if found_marker_pos != -1: 
            podcastDialogue = actual_dialogue_text_start[found_marker_pos:].strip()
        elif analysisResponse in podcastDialogueResponse: 
            podcastDialogue = podcastDialogueResponse.split(analysisResponse, 1)[-1].strip()
        else: 
            podcastDialogue = podcastDialogueResponse.strip()

    if not podcastDialogue: 
        print(colored("Error: Podcast dialogue is empty after extraction/processing.", "red"))
        exit(1)
        
    return podcastDialogue, extracted_title

async def main():
    parser = argparse.ArgumentParser(description="Generate a podcast using AI.")
    parser.add_argument("-l", "--local", action="store_true", help="Use local Dia TTS model instead of Google TTS.")
    parser.add_argument("-o", "--output-dir", type=str, help="Custom directory to save podcast files. Defaults to 'podcast_generations' in script directory.")
    parser.add_argument("-i", "--input-file", type=str, help="Input text or PDF file to process into a podcast (alternative to clipboard).")
    parser.add_argument("-t", "--transcript-file", type=str, help="Input transcript file with pre-formatted dialogue (skips analysis and dialogue generation).")
    parser.add_argument("-a", "--auto", action="store_true", help="Automatically identify distinct subtopics and generate separate podcasts for each.")
    parser.add_argument("--llms", nargs='+', metavar=("LLM1", "LLM2"), help="Use specific LLMs for conversation. Provide 1 LLM (other speaker uses default) or 2 LLMs (e.g., --llms gemini-2.0-flash or --llms gemini-2.0-flash qwen3:4b).")
    parser.add_argument("--clipboard-content", type=str, help="Clipboard content to process (overrides built-in clipboard reading).")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive option selection and use command-line arguments only.")
    args = parser.parse_args()
    
    # Track multiple model mode separately from args.llms
    use_multiple_model = bool(args.llms)  # Default based on command line args
    use_multiple_podcasts = args.auto  # Default based on command line args
    
    # Show interactive options if not disabled and no specific modes are already set
    if not args.no_interactive and not any([args.local, args.transcript_file]):
        multiple_podcasts, single_model, multiple_model, use_local = await show_interactive_options()
        
        # Override args based on interactive selection
        if use_local:
            args.local = True
            
        if multiple_model:
            use_multiple_model = True
            if not args.llms:
                # Use automatic model selection with small models via LlmRouter
                args.llms = None  # Let LlmRouter automatically select models
                print(colored("Using automatic model selection for multiple model conversation (small models)", "cyan"))
        elif single_model:
            use_multiple_model = False
            print(colored("Using single model generation with analysis", "cyan"))
        
        if multiple_podcasts:
            use_multiple_podcasts = True
    
    if args.output_dir:
        global PODCAST_SAVE_LOCATION
        PODCAST_SAVE_LOCATION = os.path.abspath(args.output_dir)
        print(colored(f"Using custom output directory: {PODCAST_SAVE_LOCATION}", "cyan"))
    else:
        print(colored(f"Using default output directory: {PODCAST_SAVE_LOCATION}", "cyan"))
    os.makedirs(PODCAST_SAVE_LOCATION, exist_ok=True)
    
    g.local = args.local 

    if not args.local and not os.environ.get("GEMINI_API_KEY"):
        print(colored("Error: GEMINI_API_KEY environment variable not set. Needed for Google TTS.", "red"))
        print(colored("You can either set GEMINI_API_KEY or use the -l flag for local TTS (if configured).", "red"))
        exit(1)

    if not PYDUB_AVAILABLE:
        print(colored("CRITICAL: pydub library not found. MP3 files cannot be generated. This script will likely fail to save audio.", "red"))
        print(colored("Install pydub with: pip install pydub", "yellow"))
        print(colored("Note: You may also need ffmpeg (or libav) installed on your system for pydub to perform MP3 conversion.", "yellow"))

    start_time = time.time()
    
    # Step 1: Get input content and determine processing mode
    if args.transcript_file:
        # Use pre-formatted transcript file
        if not os.path.exists(args.transcript_file):
            print(colored(f"Error: Transcript file not found: {args.transcript_file}", "red"))
            exit(1)
        
        if use_multiple_podcasts:
            print(colored("Warning: Multiple podcasts mode is not compatible with transcript files (which are pre-formatted dialogues)", "yellow"))
            print(colored("Proceeding with single transcript processing", "yellow"))
        
        print(colored(f"Reading pre-formatted transcript from: {args.transcript_file}", "cyan"))
        with open(args.transcript_file, 'r', encoding='utf-8') as f:
            podcastDialogue = f.read().strip()
        
        if not podcastDialogue:
            print(colored("Error: Transcript file is empty", "red"))
            exit(1)
        
        # Extract title from filename
        base_filename = os.path.splitext(os.path.basename(args.transcript_file))[0]
        extracted_title = re.sub(r'[^\w\s-]', '', base_filename).strip()
        extracted_title = re.sub(r'\s+', '_', extracted_title)
        if not extracted_title:
            extracted_title = "Transcript_Podcast"
        
        print(colored(f"Using title from filename: {extracted_title}", "magenta"))
        
        # For transcript files, always use single podcast mode
        mp3_file_location = generate_podcast(podcastDialogue, extracted_title, use_local_dia=args.local)
        mp3_files = [mp3_file_location] if mp3_file_location else []
        
    elif args.input_file:
        # Use input file instead of clipboard
        if not os.path.exists(args.input_file):
            print(colored(f"Error: Input file not found: {args.input_file}", "red"))
            exit(1)
        
        # Determine file type and extract content
        file_extension = os.path.splitext(args.input_file)[1].lower()
        
        if file_extension == '.pdf':
            print(colored(f"Extracting text from PDF: {args.input_file}", "cyan"))
            try:
                content = extract_text_from_pdf(args.input_file)
            except ImportError as e:
                print(colored(str(e), "red"))
                print(colored("Install PyMuPDF with: pip install PyMuPDF", "yellow"))
                exit(1)
            except Exception as e:
                print(colored(f"Error processing PDF file: {e}", "red"))
                exit(1)
        else:
            # Assume it's a text file
            print(colored(f"Reading input text from: {args.input_file}", "cyan"))
            try:
                with open(args.input_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
            except UnicodeDecodeError:
                # Try with different encoding
                try:
                    with open(args.input_file, 'r', encoding='latin-1') as f:
                        content = f.read().strip()
                    print(colored("Used latin-1 encoding for text file", "yellow"))
                except Exception as e:
                    print(colored(f"Error reading text file: {e}", "red"))
                    exit(1)
            except Exception as e:
                print(colored(f"Error reading text file: {e}", "red"))
                exit(1)
        
        if not content:
            print(colored("Error: Input file appears to be empty or contains no extractable text", "red"))
            exit(1)
            
        print(colored(f"Extracted {len(content)} characters from input file", "green"))
            
        # Process content based on multiple podcasts mode
        if use_multiple_podcasts:
            print(colored("Multiple podcasts mode enabled - will analyze content for multiple subtopics", "magenta"))
            if args.local:
                print(colored("Using local Dia TTS. Ensure dialogue format matches Dia's expectations.", "yellow"))
            mp3_files = await generate_multiple_podcasts(content, args.llms, use_local_dia=args.local, use_multiple_model=use_multiple_model)
        else:
            # Single podcast mode or multiple model conversation
            if args.local: 
                print(colored("Using local Dia TTS. Ensure dialogue format matches Dia's expectations.", "yellow"))
            
            if use_multiple_model:
                # Use multiple model conversation
                podcastDialogue, extracted_title = await _generate_llm_conversation(content, args.llms, force_local=args.local)
            else:
                # Use single model with analysis
                podcastDialogue, extracted_title = await process_raw_content_to_dialogue(content, args.llms, force_local=args.local)
            
            mp3_file_location = generate_podcast(podcastDialogue, extracted_title, use_local_dia=args.local)
            mp3_files = [mp3_file_location] if mp3_file_location else []
        
    else:
        # Default behavior: use clipboard content or read from clipboard
        if args.clipboard_content:
            # Use provided clipboard content (from shell script)
            content = args.clipboard_content
            print(colored("Using clipboard content provided via argument", "cyan"))
        else:
            # Only show countdown if interactive mode was skipped
            if args.no_interactive or any([args.local, use_multiple_podcasts, args.transcript_file]):
                # Default behavior: read from clipboard with countdown
                for i in range(5, 0, -1): 
                    print(colored(f"Generating podcast via clipboard in {i} seconds...", "green"))
                    time.sleep(1)
            else:
                print(colored("Reading clipboard content...", "cyan"))
            
            content = pyperclip.paste()
        
        if not content.strip(): 
            print(colored("No text in clipboard", "red"))
            exit(1)
        
        # Process content based on multiple podcasts mode
        if use_multiple_podcasts:
            print(colored("Multiple podcasts mode enabled - will analyze content for multiple subtopics", "magenta"))
            if args.local:
                print(colored("Using local Dia TTS. Ensure dialogue format matches Dia's expectations.", "yellow"))
            mp3_files = await generate_multiple_podcasts(content, args.llms, use_local_dia=args.local, use_multiple_model=use_multiple_model)
        else:
            # Single podcast mode or multiple model conversation
            if args.local: 
                print(colored("Using local Dia TTS. Ensure dialogue format matches Dia's expectations.", "yellow"))
            
            if use_multiple_model:
                # Use multiple model conversation
                podcastDialogue, extracted_title = await _generate_llm_conversation(content, args.llms, force_local=args.local)
            else:
                # Use single model with analysis
                podcastDialogue, extracted_title = await process_raw_content_to_dialogue(content, args.llms, force_local=args.local)
            
            mp3_file_location = generate_podcast(podcastDialogue, extracted_title, use_local_dia=args.local)
            mp3_files = [mp3_file_location] if mp3_file_location else []

    # Step 2: Report results
    end_time = time.time()
    
    if mp3_files and any(mp3_files):
        successful_files = [f for f in mp3_files if f]
        if len(successful_files) == 1:
            print(colored(f"Podcast generation complete. Final MP3: {successful_files[0]}", "green"))
        else:
            print(colored(f"Podcast generation complete. Generated {len(successful_files)} MP3 files:", "green"))
            for i, file_path in enumerate(successful_files, 1):
                print(colored(f"  {i}. {file_path}", "cyan"))
        print(colored(f"Total time taken: {end_time - start_time:.2f} seconds", "green"))
    else:
        print(colored("Podcast generation failed or no MP3 files were produced.", "red"))
        exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(colored("\nPodcast generation was interrupted by user. Shutting down...", "yellow"))
        import sys
        sys.exit(0)
    except Exception as e:
        print(colored(f"\nPodcast generation encountered an error: {str(e)}", "red"))
        import sys
        sys.exit(1)