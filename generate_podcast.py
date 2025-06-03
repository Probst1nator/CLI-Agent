import argparse
import base64
import mimetypes
import os
import re
import struct
import time
import tempfile # Added for temporary file/directory management
from dotenv import load_dotenv

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

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from py_methods.dia_helper import get_dia_model

load_dotenv(g.CLIAGENT_ENV_FILE_PATH)

DEFAULT_PODCAST_SAVE_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "podcast_generations")
PODCAST_SAVE_LOCATION = DEFAULT_PODCAST_SAVE_LOCATION
GOOGLE_TTS_MODEL_NAME = "gemini-2.5-flash-preview-tts"

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

def _split_podcast_text_into_chunks(text: str, max_chars: int = 600, max_lines: int = 30, split_on_speakers: bool = True) -> list[str]:
    chunks = []
    current_chunk_lines = []
    input_lines = text.splitlines()
    if not input_lines: return []
    for i, line_to_add in enumerate(input_lines):
        temp_prospective_lines = current_chunk_lines + [line_to_add]
        prospective_char_count = len("\n".join(temp_prospective_lines))
        prospective_actual_line_count = len(temp_prospective_lines)
        must_split_before_adding_current_line = False
        if current_chunk_lines:
            if prospective_char_count > max_chars: must_split_before_adding_current_line = True
            elif prospective_actual_line_count > max_lines: must_split_before_adding_current_line = True
            elif split_on_speakers and _is_speaker_line(line_to_add): must_split_before_adding_current_line = True
        if must_split_before_adding_current_line:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
        current_chunk_lines.append(line_to_add)
    if current_chunk_lines: chunks.append("\n".join(current_chunk_lines))
    return [chunk for chunk in chunks if chunk.strip()]


def generate_podcast(podcast_dialogue: str, title: str, use_local_dia: bool = False) -> str:
    safe_title_for_filename = re.sub(r'[^\w_-]+', '', title.replace(' ', '_')).strip('_')
    if not safe_title_for_filename: safe_title_for_filename = "podcast_episode"
    os.makedirs(PODCAST_SAVE_LOCATION, exist_ok=True)
    
    if use_local_dia:
        text_chunks = _split_podcast_text_into_chunks(podcast_dialogue, max_chars=600, max_lines=30, split_on_speakers=True)
        print(colored(f"[{title}] Successfully split into {len(text_chunks)} chunks for Dia.", "blue"))
    else:
        text_chunks = _split_podcast_text_into_chunks(podcast_dialogue, max_chars=4500, max_lines=100, split_on_speakers=False)
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
            if GOOGLE_AVAILABLE and os.environ.get("GOOGLE_API_KEY"):
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
                file_pattern = re.compile(rf"(\d+)_{re.escape(safe_title_for_filename)}\{re.escape(file_extension_for_numbering)}")
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
        for i, chunk in enumerate(text_chunks):
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
                    else: print(colored(f"[{title}] Error generating audio for chunk {i+1}: {e}", "red"))
                retry_count += 1
                if retry_count <= max_retries:
                    print(colored(f"Retrying chunk {i+1} (attempt {retry_count}/{max_retries}) in {auto_retry_seconds} seconds...", "yellow")); time.sleep(auto_retry_seconds)
                else: print(colored(f"[{title}] Failed to process chunk {i+1} after {max_retries} attempts. Skipping.", "red")); break
            if chunk_audio_data: all_audio_data_bytes += chunk_audio_data
            else: print(colored(f"[{title}] No audio data for chunk {i+1}. Continuing.", "yellow"))

        if not all_audio_data_bytes:
            print(colored(f"[{title}] No audio data was generated for any chunks (Google TTS).", "red")); return ""
        print(colored(f"[{title}] Successfully gathered audio data from {total_chunks} chunks for Google TTS.", "green"))

        file_extension_for_numbering = ".mp3"
        highest_num = 0
        file_pattern = re.compile(rf"(\d+)_{re.escape(safe_title_for_filename)}\{re.escape(file_extension_for_numbering)}")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a podcast using AI.")
    parser.add_argument("-l", "--local", action="store_true", help="Use local Dia TTS model instead of Google TTS.")
    parser.add_argument("-o", "--output-dir", type=str, help="Custom directory to save podcast files. Defaults to 'podcast_generations' in script directory.")
    args = parser.parse_args()
    
    if args.output_dir:
        PODCAST_SAVE_LOCATION = os.path.abspath(args.output_dir)
        print(colored(f"Using custom output directory: {PODCAST_SAVE_LOCATION}", "cyan"))
    else:
        print(colored(f"Using default output directory: {PODCAST_SAVE_LOCATION}", "cyan"))
    os.makedirs(PODCAST_SAVE_LOCATION, exist_ok=True)
    
    g.local = args.local 

    if not args.local and not os.environ.get("GOOGLE_API_KEY"):
        print(colored("Error: GOOGLE_API_KEY environment variable not set. Needed for Google TTS.", "red"))
        print(colored("You can either set GOOGLE_API_KEY or use the -l flag for local TTS (if configured).", "red"))
        exit(1)

    if not PYDUB_AVAILABLE:
        print(colored("CRITICAL: pydub library not found. MP3 files cannot be generated. This script will likely fail to save audio.", "red"))
        print(colored("Install pydub with: pip install pydub", "yellow"))
        print(colored("Note: You may also need ffmpeg (or libav) installed on your system for pydub to perform MP3 conversion.", "yellow"))
        # Consider exiting if MP3 is strictly required.

    for i in range(5, 0, -1): print(colored(f"Generating podcast via clipboard in {i} seconds...", "green")); time.sleep(1)
    start_time = time.time()
    clipboard_content = pyperclip.paste()
    if not clipboard_content.strip(): print(colored("No text in clipboard", "red")); exit(1)
    
    print(colored("Analyzing content for summary and title...", "blue"))
    try:
        analysisResponse = LlmRouter.generate_completion(f"For the following text snippet, please highlight key insights and lay out the fundamentals of the topic of concern. Explore analogies to other fields and highlight related concepts to manifest a constructive framework for the topic.\nHere's the raw unfiltered text snippet:\n{clipboard_content}")
        titleResponseText = LlmRouter.generate_completion(f"I am going to provide you with a text and you should generate a descriptive title for whatever content it's about. Ensure it is a short fit for a file name. Please think about the contents first and then like this:\nTitle: <title>\nThis is the text:\n{analysisResponse}")
    except Exception as e:
        print(colored(f"Error during LLM processing for analysis/title: {e}", "red")); import traceback; traceback.print_exc(); exit(1)

    title_match = re.search(r"Title:\s*(.*)", titleResponseText, re.IGNORECASE)
    extracted_title = title_match.group(1).strip() if title_match else titleResponseText.splitlines()[0].strip()
    MAX_TITLE_LEN = 50 
    if len(extracted_title) > MAX_TITLE_LEN: extracted_title = extracted_title[:MAX_TITLE_LEN]
    extracted_title = re.sub(r'[^\w\s-]', '', extracted_title).strip()
    extracted_title = re.sub(r'\s+', '_', extracted_title)
    if not extracted_title: extracted_title = "Untitled_Podcast"

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
Liam: It's great to be here, lets attack the topic of {extracted_title}.
...
You don't need to use my example introduction, you can create a more fitting one for the topic.
The topic of {extracted_title} has been automatically reviewed and the following insights were distilled and are provided to enhance the clarity and depth of the exchange:
{analysisResponse}"""
    if args.local: print(colored("Using local Dia TTS. Ensure dialogue format matches Dia's expectations.", "yellow"))
        
    try:
        podcastGenChat = Chat(); podcastGenChat.add_message(Role.USER, podcastGenPrompt); podcastGenChat.add_message(Role.ASSISTANT, "<think>")
        podcastDialogueResponse = LlmRouter.generate_completion(podcastGenChat)
    except Exception as e: print(colored(f"Error during LLM processing for podcast dialogue: {e}", "red")); exit(1)
    
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
        first_speaker_occurrence = -1; found_marker_pos = -1
        for marker in dialogue_start_markers:
            pos = actual_dialogue_text_start.find(marker)
            if pos != -1 and (first_speaker_occurrence == -1 or pos < first_speaker_occurrence): first_speaker_occurrence = pos; found_marker_pos = pos
        if found_marker_pos != -1: podcastDialogue = actual_dialogue_text_start[found_marker_pos:].strip()
        elif analysisResponse in podcastDialogueResponse: podcastDialogue = podcastDialogueResponse.split(analysisResponse, 1)[-1].strip()
        else: podcastDialogue = podcastDialogueResponse.strip()

    if not podcastDialogue: print(colored("Error: Podcast dialogue is empty after extraction/processing.", "red")); exit(1)

    mp3_file_location = generate_podcast(podcastDialogue, extracted_title, use_local_dia=args.local)

    if mp3_file_location:
        end_time = time.time()
        print(colored(f"Podcast generation complete. Final MP3: {mp3_file_location}", "green"))
        print(colored(f"Total time taken: {end_time - start_time:.2f} seconds", "green"))
    else:
        print(colored("Podcast generation failed or no MP3 file was produced.", "red")); exit(1)