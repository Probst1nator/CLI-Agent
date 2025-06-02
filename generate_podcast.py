import argparse # Added
import base64
import mimetypes
import os
import re
import struct
import time
from dotenv import load_dotenv
# Google imports - only needed for Google TTS mode
try:
    from google import genai
    from google.genai import types
    from google.genai.types import SpeechConfig
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None
    types = None
from py_classes.cls_llm_router import LlmRouter
from py_classes.globals import g
import pyperclip
from termcolor import colored
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from enum import Enum
import sys

# Add pydub for MP3 conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

# Attempt to import Dia-related libraries conditionally later if needed
# import soundfile as sf
# from dia.model import Dia

# Add the parent directory to the path so we can import DiaHelper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from py_methods.dia_helper import get_dia_model

load_dotenv(g.CLIAGENT_ENV_FILE_PATH)

# Default podcast save location - can be overridden by command line argument
DEFAULT_PODCAST_SAVE_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "podcast_generations")
PODCAST_SAVE_LOCATION = DEFAULT_PODCAST_SAVE_LOCATION  # Will be updated by command line args
GOOGLE_TTS_MODEL_NAME = "gemini-2.5-flash-preview-tts" # User specified model

def save_binary_file(file_name, data):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wb") as f:
        f.write(data)

def convert_wav_to_mp3(wav_file_path: str) -> str:
    """
    Convert a WAV file to MP3 format using pydub.
    Returns the path to the MP3 file.
    """
    if not PYDUB_AVAILABLE:
        print(colored("Warning: pydub not available. Cannot convert to MP3. Install with 'pip install pydub'", "yellow"))
        return ""
    
    try:
        # Load WAV file
        audio = AudioSegment.from_wav(wav_file_path)
        
        # Generate MP3 filename
        mp3_file_path = wav_file_path.replace('.wav', '.mp3')
        
        # Export as MP3
        audio.export(mp3_file_path, format="mp3", bitrate="192k")
        print(colored(f"MP3 file saved to: {mp3_file_path}", "green"))
        return mp3_file_path
    except Exception as e:
        print(colored(f"Error converting WAV to MP3: {e}", "red"))
        return ""

def save_audio_files(audio_data, sample_rate: int, file_name_base: str, file_extension: str = ".wav") -> Tuple[str, str]:
    """
    Save audio data as both WAV and MP3 files.
    Returns tuple of (wav_file_path, mp3_file_path)
    """
    wav_file_path = os.path.join(PODCAST_SAVE_LOCATION, f"{file_name_base}.wav")
    mp3_file_path = ""
    
    # Save WAV file
    if isinstance(audio_data, np.ndarray):
        # For numpy array (Dia output)
        import soundfile as sf
        sf.write(wav_file_path, audio_data, sample_rate)
    else:
        # For binary data (Google TTS output)
        final_audio_to_save = create_wav_file(audio_data, sample_rate=sample_rate, bits_per_sample=16, num_channels=1)
        save_binary_file(wav_file_path, final_audio_to_save)
    
    print(colored(f"WAV file saved to: {wav_file_path} (Sample Rate: {sample_rate} Hz)", "green"))
    
    # Convert to MP3
    mp3_file_path = convert_wav_to_mp3(wav_file_path)
    
    return wav_file_path, mp3_file_path

# _is_speaker_line might still be useful for other purposes or if we need to validate format
def _is_speaker_line(line: str) -> bool:
    """Checks if a line indicates the start of a speaker's dialogue."""
    return bool(re.match(r"^\s*[\w\s.-]+\s*\([\w\s.-]+\):", line))

def _split_podcast_text_into_chunks(text: str, max_chars: int = 600, max_lines: int = 30, split_on_speakers: bool = True) -> list[str]:
    """
    Splits the podcast text into chunks based on character limits, line limits,
    and optionally speaker changes.
    """
    chunks = []
    current_chunk_lines = []
    
    input_lines = text.splitlines()
    if not input_lines:
        return []
        
    for i, line_to_add in enumerate(input_lines):
        temp_prospective_lines = current_chunk_lines + [line_to_add]
        prospective_char_count = len("\n".join(temp_prospective_lines))
        prospective_actual_line_count = len(temp_prospective_lines)

        must_split_before_adding_current_line = False
        
        if current_chunk_lines: # Only split if there's something in the current chunk
            if prospective_char_count > max_chars:
                must_split_before_adding_current_line = True
            elif prospective_actual_line_count > max_lines:
                must_split_before_adding_current_line = True
            elif split_on_speakers and _is_speaker_line(line_to_add): # Split when a new speaker line is encountered
                must_split_before_adding_current_line = True
        
        if must_split_before_adding_current_line:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = [] # Reset for the new chunk
        
        current_chunk_lines.append(line_to_add)

    # Add any remaining lines in current_chunk_lines as the last chunk
    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))
        
    return [chunk for chunk in chunks if chunk.strip()]


def generate_podcast(podcast_dialogue: str, title: str, use_local_dia: bool = False) -> str:
    safe_title_for_filename = re.sub(r'[^\w_-]+', '', title.replace(' ', '_')).strip('_')
    if not safe_title_for_filename:
        safe_title_for_filename = "podcast_episode"
    os.makedirs(PODCAST_SAVE_LOCATION, exist_ok=True)
    
    # Use different chunk sizes based on TTS method
    if use_local_dia:
        text_chunks = _split_podcast_text_into_chunks(podcast_dialogue, max_chars=600, max_lines=30, split_on_speakers=True)
        print(colored(f"[{title}] Successfully split into {len(text_chunks)} chunks for Dia.", "blue"))
    else:
        text_chunks = _split_podcast_text_into_chunks(podcast_dialogue, max_chars=4500, max_lines=100, split_on_speakers=False)
        print(colored(f"[{title}] Successfully split into {len(text_chunks)} chunks for Google TTS.", "blue"))

    if use_local_dia:
        print(colored(f"[{title}] Attempting to use local Dia TTS model.", "magenta"))

        try:
            import soundfile as sf # Moved import here
        except ImportError:
            print(colored("Error: 'soundfile' library not found. Please install it with 'pip install soundfile'", "red"))
            return ""

        print(colored(f"[{title}] NOTE: Dia TTS requires a compatible GPU and may take time to load the model.", "yellow"))
        dia_model = get_dia_model()
        if dia_model is None:
            print(colored("Failed to initialize Dia model. Falling back to Google TTS...", "red"))
            return generate_podcast(podcast_dialogue, title, use_local_dia=False) # Fallback
        print(colored(f"[{title}] Dia model loaded.", "blue"))

        audio_files = []
        total_chunks = len(text_chunks)
        for i, chunk in enumerate(text_chunks): # Use Dia specific chunks
            if not chunk.strip():
                continue
            print(colored(f"[{title}] Dia: Generating audio for chunk {i+1}/{total_chunks} ({len(chunk)} chars)...", "green"))
            try:
                output_waveform = dia_model.generate(chunk)
                dia_sample_rate = 44100 
                if output_waveform is None or output_waveform.size == 0:
                    print(colored(f"[{title}] Dia: No audio data for chunk {i+1}.", "yellow"))
                    continue
                temp_filename = f"temp_chunk_{i+1:03d}.wav"
                temp_filepath = os.path.join(PODCAST_SAVE_LOCATION, temp_filename)
                sf.write(temp_filepath, output_waveform, dia_sample_rate)
                audio_files.append(temp_filepath)
                print(colored(f"[{title}] Dia: Chunk {i+1} audio generated.", "cyan"))
            except Exception as e:
                print(colored(f"[{title}] Dia: Error for chunk {i+1}: {e}", "red"))
                import traceback
                traceback.print_exc()
                continue
        
        if not audio_files:
            print(colored(f"[{title}] Dia: No audio chunks generated.", "red"))
            return ""

        print(colored(f"[{title}] Dia: Merging {len(audio_files)} audio chunks...", "blue"))
        try:
            merged_audio_dia = []
            final_sample_rate_dia = None
            for af_path in audio_files:
                audio_data, sr = sf.read(af_path)
                if final_sample_rate_dia is None: final_sample_rate_dia = sr
                elif sr != final_sample_rate_dia:
                    print(colored(f"[{title}] Dia: Warning sample rate mismatch. Using {final_sample_rate_dia} Hz.", "yellow"))
                merged_audio_dia.append(audio_data)
            
            final_audio_dia = np.concatenate(merged_audio_dia)
            file_extension = ".wav"
            # ... (filename generation logic remains the same)
            highest_num = 0
            file_pattern = re.compile(rf"(\d+)_{re.escape(safe_title_for_filename)}\{re.escape(file_extension)}")
            for f_name in os.listdir(PODCAST_SAVE_LOCATION):
                match = file_pattern.match(f_name)
                if match:
                    num = int(match.group(1))
                    if num > highest_num:
                        highest_num = num
            episode_number = highest_num + 1
            file_name_base = f"{episode_number:03d}_{safe_title_for_filename}"
            file_location, mp3_file_location = save_audio_files(final_audio_dia, final_sample_rate_dia, file_name_base, file_extension)
            for af_path in audio_files: os.remove(af_path)
            print(colored(f"[{title}] Dia podcast saved to: {file_location} (SR: {final_sample_rate_dia} Hz)", "green"))
            if mp3_file_location:
                print(colored(f"[{title}] Dia podcast MP3 saved to: {mp3_file_location}", "green"))
            return file_location
        except Exception as e:
            print(colored(f"[{title}] Dia: Error merging audio: {e}", "red"))
            for af_path in audio_files: 
                try: os.remove(af_path)
                except OSError: pass
            return ""

    else: # Google TTS Implementation - Sequential chunk processing
        print(colored(f"[{title}] Using Google GenAI TTS model: {GOOGLE_TTS_MODEL_NAME} for sequential chunk processing.", "magenta"))

        if not GOOGLE_AVAILABLE:
            print(colored("Error: Google dependencies not available. Install with 'pip install google-genai'.", "red"))
            return ""

        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            print(colored("Error: GOOGLE_API_KEY environment variable not set.", "red"))
            return ""
        
        try:
            client = genai.Client(api_key=google_api_key)
        except Exception as e:
            print(colored(f"Failed to initialize Google GenAI Client: {e}", "red"))
            return ""

        generate_content_config = types.GenerateContentConfig(
            response_modalities=["audio"],
            speech_config=SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker="Chloe",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="Kore"
                                )
                            ),
                        ),
                        types.SpeakerVoiceConfig(
                            speaker="Liam",
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name="iapetus"
                                )
                            ),
                        ),
                    ]
                ),
            ),
        )

        all_audio_data = b""
        auto_retry_seconds = 30
        max_retries = 3
        total_chunks = len(text_chunks)

        # Process chunks sequentially
        for i, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue
                
            print(colored(f"[{title}] Google TTS: Generating audio for chunk {i+1}/{total_chunks} ({len(chunk)} chars)...", "green"))
            
            retry_count = 0
            chunk_audio_data = b""
            
            while retry_count <= max_retries:
                try:
                    # The 'contents' argument is where the textual guidance for narration
                    # (e.g., "(laughs)", "(stutters)") is passed to the TTS model.
                    # The model interprets these cues directly from the text.
                    response_stream = client.models.generate_content_stream(
                        model=f"models/{GOOGLE_TTS_MODEL_NAME}",
                        contents=chunk,
                        config=generate_content_config,
                    )

                    current_chunk_audio = b""
                    for response_part in response_stream:
                        if (
                            response_part.candidates is None
                            or not response_part.candidates[0].content
                            or not response_part.candidates[0].content.parts
                        ):
                            continue
                        
                        api_part_data = response_part.candidates[0].content.parts[0]
                        if hasattr(api_part_data, 'inline_data') and api_part_data.inline_data and hasattr(api_part_data.inline_data, 'data'):
                            current_chunk_audio += api_part_data.inline_data.data
                        elif hasattr(api_part_data, 'text') and api_part_data.text:
                            print(colored(f"ERROR: [{title}] API returned text instead of audio for chunk {i+1}: {api_part_data.text}", "red"))
                            current_chunk_audio = b""
                            break

                    if current_chunk_audio:
                        chunk_audio_data = current_chunk_audio
                        print(colored(f"[{title}] Google TTS: Chunk {i+1} audio generated ({len(chunk_audio_data)} bytes).", "cyan"))
                        break
                        
                except Exception as e:
                    # Handle any API errors including stop candidate exceptions
                    error_message = str(e)
                    if "stop" in error_message.lower() or "candidate" in error_message.lower():
                        print(colored(f"[{title}] Generation stopped for chunk {i+1}: {e}", "red"))
                        break
                    else:
                        print(colored(f"[{title}] Error generating audio for chunk {i+1}: {e}", "red"))
                
                retry_count += 1
                if retry_count <= max_retries:
                    print(colored(f"Retrying chunk {i+1} (attempt {retry_count}/{max_retries}) in {auto_retry_seconds} seconds...", "yellow"))
                    time.sleep(auto_retry_seconds)
                else:
                    print(colored(f"[{title}] Failed to process chunk {i+1} after {max_retries} attempts. Skipping.", "red"))
                    break
            
            if chunk_audio_data:
                all_audio_data += chunk_audio_data
            else:
                print(colored(f"[{title}] No audio data for chunk {i+1}. Continuing with remaining chunks.", "yellow"))

        if not all_audio_data:
            print(colored(f"[{title}] No audio data was generated for any chunks (Google TTS).", "red"))
            return ""

        print(colored(f"[{title}] Successfully generated audio from {total_chunks} chunks.", "green"))

        # Generate filename
        file_extension = ".wav"
        highest_num = 0
        file_pattern = re.compile(rf"(\d+)_{re.escape(safe_title_for_filename)}\{re.escape(file_extension)}")
        for f_name in os.listdir(PODCAST_SAVE_LOCATION):
            match = file_pattern.match(f_name)
            if match:
                num = int(match.group(1))
                if num > highest_num:
                    highest_num = num
        episode_number = highest_num + 1
        file_name_base = f"{episode_number:03d}_{safe_title_for_filename}"
        file_location, mp3_file_location = save_audio_files(all_audio_data, 24000, file_name_base, file_extension)
        return file_location

def create_wav_file(pcm_data: bytes, sample_rate: int = 24000, bits_per_sample: int = 16, num_channels: int = 1) -> bytes:
    """
    Create a proper WAV file from raw PCM data.
    (This function remains unchanged)
    """
    data_size = len(pcm_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    
    riff_chunk_id = b"RIFF"
    riff_chunk_size = 36 + data_size 
    wave_format = b"WAVE"
    
    fmt_chunk_id = b"fmt "
    fmt_chunk_size = 16  
    audio_format = 1  
    
    data_chunk_id = b"data"
    data_chunk_size = data_size
    
    wav_header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        riff_chunk_id, riff_chunk_size, wave_format,
        fmt_chunk_id, fmt_chunk_size, audio_format, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample,
        data_chunk_id, data_chunk_size
    )
    return wav_header + pcm_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a podcast using AI.")
    parser.add_argument("-l", "--local", action="store_true",
                        help="Use local Dia TTS model instead of Google TTS.")
    parser.add_argument("-o", "--output-dir", type=str, 
                        help="Custom directory to save podcast files. Defaults to 'podcast_generations' in script directory.")
    args = parser.parse_args()
    
    # Update PODCAST_SAVE_LOCATION if custom output directory is provided
    if args.output_dir:
        PODCAST_SAVE_LOCATION = os.path.abspath(args.output_dir)
        print(colored(f"Using custom output directory: {PODCAST_SAVE_LOCATION}", "cyan"))
    else:
        print(colored(f"Using default output directory: {PODCAST_SAVE_LOCATION}", "cyan"))
    
    g.local = args.local # Assuming g is a global settings object

    if not args.local and not os.environ.get("GOOGLE_API_KEY"):
        print(colored("Error: GOOGLE_API_KEY environment variable not set. Needed for Google TTS.", "red"))
        print(colored("You can either set GOOGLE_API_KEY or use the -l flag for local TTS (if configured).", "red"))
        exit(1)

    if not PYDUB_AVAILABLE:
        print(colored("Warning: pydub library not found. MP3 files will not be generated.", "yellow"))
        print(colored("Install pydub with: pip install pydub", "yellow"))
        print(colored("Note: You may also need ffmpeg installed on your system for MP3 conversion.", "yellow"))

    for i in range(5, 0, -1):
        print(colored(f"Generating podcast via clipboard in ", "green") + colored(f"{i}", "yellow") + colored(" seconds...", "green"))
        time.sleep(1)
    
    start_time = time.time()

    clipboard_content = pyperclip.paste()
    if not clipboard_content.strip():
        print(colored("No text in clipboard", "red"))
        exit(1)
    
    print(colored("Analyzing content for summary and title...", "blue"))
    try:
        analysisResponse = LlmRouter.generate_completion(
            "Hi, I have a text snippet which I would like you to analyze. Please reason about it and provide a summary of the context, fundamentals and key points. Include insightful analogies to other fields and concepts to help a reader remember the topic better.\nText:\n" + clipboard_content
        )
        titleResponseText = LlmRouter.generate_completion( # Renamed to avoid conflict
            "I am going to provide you with a text and you should generate a descriptive title for whatever content it's about. Ensure it is a short fit for a file name. Please think about the contents first and then like this:\nTitle: <title>\nThis is the text:\n" + analysisResponse
        )
    except Exception as e:
        print(colored(f"Error during LLM processing for analysis/title: {e}", "red"))
        import traceback
        traceback.print_exc()
        exit(1)

    # More robust title extraction
    title_match = re.search(r"Title:\s*(.*)", titleResponseText, re.IGNORECASE)
    extracted_title = ""
    if title_match:
        extracted_title = title_match.group(1).strip()
    
    if not extracted_title: # Fallback if "Title:" not found
        extracted_title = titleResponseText.splitlines()[0].strip() # Take first line as a guess

    MAX_TITLE_LEN = 50 
    if len(extracted_title) > MAX_TITLE_LEN:
        extracted_title = extracted_title[:MAX_TITLE_LEN]
    
    # Further sanitize title for filename
    extracted_title = re.sub(r'[^\w\s-]', '', extracted_title).strip() # Allow spaces and hyphens initially
    extracted_title = re.sub(r'\s+', '_', extracted_title) # Replace spaces with underscores
    if not extracted_title: extracted_title = "Untitled_Podcast"


    print(colored(f"Generated Title: {extracted_title}", "magenta"))
    print(colored("Generating podcast dialogue...", "blue"))

    # Modified podcastGenPrompt to instruct the LLM to include narration cues
    podcastGenPrompt = f"""Create a meditative and joyful expert discussion between an intelligent and very creative and educated student and an self-taught established expert.
The discussion should revolve around powerful insights and intuitions that enable easy understanding and retention of the topic.

To guide the emotional tone and style of the speech, include concise parenthetical cues like (laughs), (stutters), (happy), (sadly), (whispering), or (enthusiastically) directly within the dialogue.
**IMPORTANT: These cues are instructions for how the dialogue should be *spoken* by the text-to-speech system, and should NOT be spoken aloud themselves. They should only guide the voice's delivery.**

For example, if Chloe says something funny, write:
Chloe: That's absolutely hilarious! (laughs)

If Liam is hesitant:
Liam: I, uh, (stutters) I'm not entirely sure about that.

If Chloe is expressing joy:
Chloe: I'm so glad we covered this topic! (happy)

To provide the dialogue please use the following delimiters and this exact format:
```txt
Chloe: Welcome!
Liam: It's great to be here, lets attack the topic of {extracted_title}.
...
You don't need to use my example introduction, you can create a more fitting one for the topic.
The following information/topic(s) are provided to provide food for thought:
{analysisResponse}"""

    # Specific prompt adjustment for local Dia model if it requires different speaker tags
    if args.local:
        # Example adjustment: If Dia uses [S1], [S2]
        # This part needs to be conditional based on how Dia expects speaker tags
        # For now, we assume the Chloe: format is standard
        # podcastGenPrompt = podcastGenPrompt.replace("Chloe:", "[S1]").replace("Liam:", "[S2]")
        print(colored("Using local Dia TTS. Ensure dialogue format matches Dia's expectations if different from 'Speaker (Role):' format.", "yellow"))
        
    try:
        podcastDialogueResponse = LlmRouter.generate_completion(podcastGenPrompt)
    except Exception as e:
        print(colored(f"Error during LLM processing for podcast dialogue: {e}", "red"))
        exit(1)
    
    # Assuming extract_blocks is a valid utility you have
    from py_methods.utils import extract_blocks # Make sure this import is valid
    extracted_dialogue_blocks = extract_blocks(podcastDialogueResponse, ["txt", "text"])
    
    podcastDialogue = ""
    if extracted_dialogue_blocks:
        # extract_blocks returns a list, so we need to handle it properly
        if isinstance(extracted_dialogue_blocks, list):
            if len(extracted_dialogue_blocks) == 1:
                podcastDialogue = extracted_dialogue_blocks[0].strip()
            else:
                # If multiple blocks, join them with newlines
                podcastDialogue = "\n".join(block.strip() for block in extracted_dialogue_blocks if block.strip())
        else:
            # If it's not a list (unexpected), treat as string
            podcastDialogue = str(extracted_dialogue_blocks).strip()
    else:
        print(colored("Could not extract dialogue from LLM response. Using raw response.", "yellow"))
        # Fallback: use the whole response, but try to clean it up a bit
        # This might happen if ```txt ... ``` was not found
        # Basic cleanup: remove the prompt part if it's reflected in the response
        prompt_intro_for_cleanup = "Create a meditative expert discussion"
        intro_index = podcastDialogueResponse.find(prompt_intro_for_cleanup)
        if intro_index != -1:
            # Try to find the actual start of the dialogue after the prompt
            dialogue_start_markers = ["Chloe:", "Liam:", "[S1]", "[S2]"] # Add more if needed
            actual_dialogue_text = podcastDialogueResponse[intro_index:] # Search in the part after prompt
            first_speaker_occurrence = -1
            for marker in dialogue_start_markers:
                pos = actual_dialogue_text.find(marker)
                if pos != -1 and (first_speaker_occurrence == -1 or pos < first_speaker_occurrence):
                    first_speaker_occurrence = pos
            
            if first_speaker_occurrence != -1:
                podcastDialogue = actual_dialogue_text[first_speaker_occurrence:].strip()
            else: # If no speaker tags found after prompt, take a guess
                podcastDialogue = podcastDialogueResponse.split(analysisResponse)[-1].strip() if analysisResponse in podcastDialogueResponse else podcastDialogueResponse.strip()

        else: # If prompt intro not found, just use the raw response
            podcastDialogue = podcastDialogueResponse.strip()
            
    if not podcastDialogue:
        print(colored("Error: Podcast dialogue is empty after extraction/processing.", "red"))
        exit(1)

    # --- Call the updated generate_podcast function ---
    file_location = generate_podcast(podcastDialogue, extracted_title, use_local_dia=args.local)

    if file_location:
        end_time = time.time()
        print(colored(f"Total time taken: {end_time - start_time:.2f} seconds", "green"))
    else:
        print(colored("Podcast generation failed or produced no audio.", "red"))
        exit(1)