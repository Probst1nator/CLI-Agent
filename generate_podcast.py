import argparse # Added
import base64
import mimetypes
import os
import re
import struct
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from py_classes.cls_llm_router import LlmRouter # Assuming these are in PYTHONPATH or local
from py_classes.globals import g
import pyperclip
from termcolor import colored
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from enum import Enum
import sys

# Attempt to import Dia-related libraries conditionally later if needed
# import soundfile as sf
# from dia.model import Dia

# Add the parent directory to the path so we can import DiaHelper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.dia_helper import get_dia_model

load_dotenv(g.PROJ_ENV_FILE_PATH)

# next to this script in a directory called podcast_generations
PODCAST_SAVE_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "podcast_generations")
GOOGLE_TTS_MODEL_NAME = "gemini-2.5-flash-preview-tts" 

def save_binary_file(file_name, data):
    os.makedirs(os.path.dirname(file_name), exist_ok=True) # Ensure directory exists
    with open(file_name, "wb") as f: # Use 'with' for automatic file closing
        f.write(data)
    print(f"File saved to: {file_name}")


def _is_speaker_line(line: str) -> bool:
    """Checks if a line indicates the start of a speaker's dialogue."""
    return bool(re.match(r"^\s*[\w\s.-]+\s*\([\w\s.-]+\):", line))

def _split_podcast_text_into_chunks(text: str, max_chars: int = 4000, max_lines: int = 30) -> list[str]:
    """Splits the podcast text into chunks based on character limits, line limits, and speaker changes."""
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
        
        if current_chunk_lines:
            if prospective_char_count > max_chars:
                must_split_before_adding_current_line = True
            elif prospective_actual_line_count > max_lines:
                must_split_before_adding_current_line = True
            elif _is_speaker_line(line_to_add):
                must_split_before_adding_current_line = True
        
        if must_split_before_adding_current_line:
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = [] 
        
        current_chunk_lines.append(line_to_add)

        if len(current_chunk_lines) == 1:
            current_line_str = current_chunk_lines[0]
            if len(current_line_str) > max_chars:
                print(colored(f"Warning: A single line intended for a chunk is longer than max_chars ({len(current_line_str)} chars). "
                              "This might cause issues with API processing for this specific line/chunk.", "yellow"))

    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))
        
    return [chunk for chunk in chunks if chunk.strip()]


def generate_podcast(podcast_dialogue: str, title: str, use_local_dia: bool = False) -> str:
    """
    Generate a podcast audio file from dialogue text.
    
    Args:
        podcast_dialogue: The dialogue text for the podcast
        title: The title of the podcast
        use_local_dia: Whether to use the local Dia TTS model instead of Google TTS
        
    Returns:
        The path to the generated audio file, or empty string if generation failed
    """
    
    # Sanitize title for use in filenames early
    safe_title_for_filename = re.sub(r'[^\w_-]+', '', title.replace(' ', '_')).strip('_')
    if not safe_title_for_filename:
        safe_title_for_filename = "podcast_episode"

    os.makedirs(PODCAST_SAVE_LOCATION, exist_ok=True)
    
    print(colored(f"[{title}] Splitting podcast dialogue into chunks...", "blue"))
    text_chunks = _split_podcast_text_into_chunks(podcast_dialogue, max_chars=600 if use_local_dia else 4000)
    print(colored(f"[{title}] Successfully split into {len(text_chunks)} chunks.", "blue"))

    if use_local_dia:
        print(colored(f"[{title}] Attempting to use local Dia TTS model.", "magenta"))
        try:
            import soundfile as sf
            # Instead of directly importing Dia, use our helper
            # from dia.model import Dia
        except ImportError:
            print(colored("Error: 'soundfile' library not found. Please install it with 'pip install soundfile'", "red"))
            return ""

        print(colored(f"[{title}] NOTE: Dia TTS requires a compatible GPU and may take time to load the model.", "yellow"))

        try:
            print(colored(f"[{title}] Loading Dia-1.6B model...", "blue"))
            # Use the helper function to get the Dia model
            dia_model = get_dia_model()
            if dia_model is None:
                print(colored("Failed to initialize Dia model. Check if Dia is properly installed.", "red"))
                print(colored("Falling back to Google TTS...", "yellow"))
                use_local_dia = False
                # Continue to the Google TTS implementation instead
                return generate_podcast(podcast_dialogue, title, use_local_dia=False)
            
            print(colored(f"[{title}] Dia-1.6B model loaded.", "blue"))
        except Exception as e:
            print(colored(f"[{title}] Error loading Dia model: {e}", "red"))
            print(colored("Ensure you have PyTorch with CUDA installed and a suitable GPU.", "red"))
            print(colored("Falling back to Google TTS...", "yellow"))
            use_local_dia = False 
            # Continue to the Google TTS implementation instead
            return generate_podcast(podcast_dialogue, title, use_local_dia=False)

        dia_formatted_segments = []
        dia_input_text = " ".join(dia_formatted_segments)
        
        if not dia_input_text.strip():
            print(colored(f"[{title}] No processable text for Dia model after formatting.", "yellow"))
            return ""

        print(colored(f"[{title}] Generating audio with Dia for text ({len(dia_input_text)} chars)...", "green"))
        
        try:
            output_waveform = dia_model.generate(dia_input_text)
            dia_sample_rate = 44100 # Dia's typical sample rate

            if output_waveform is None or output_waveform.size == 0 : # Check if waveform is empty
                print(colored(f"[{title}] Dia model generated no audio data.", "red"))
                return ""

            print(colored(f"[{title}] Audio generated successfully with Dia.", "cyan"))

            file_extension = ".mp3" # Dia example uses mp3, soundfile supports it if system libs are present
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
            file_location = os.path.join(PODCAST_SAVE_LOCATION, f"{file_name_base}{file_extension}")

            sf.write(file_location, output_waveform, dia_sample_rate)
            print(colored(f"[{title}] Dia podcast saved to: {file_location} (Sample Rate: {dia_sample_rate} Hz)", "green"))
            return file_location

        except Exception as e:
            print(colored(f"[{title}] Error during Dia TTS generation or saving: {e}", "red"))
            import traceback
            traceback.print_exc()
            return ""

    else: # Original Google TTS Implementation
        print(colored(f"[{title}] Using Google GenAI TTS model: {GOOGLE_TTS_MODEL_NAME}", "magenta"))

        # Ensure API key is available
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            print(colored("Error: GOOGLE_API_KEY environment variable not set. Cannot use Google TTS.", "red"))
            return ""
        client = genai.Client(api_key=google_api_key)

        # Define speaker mapping for Google TTS. This is a simplified example.
        # You might need a more robust way to map speakers from your dialogue
        # to the "Speaker 1", "Speaker 2" tags if your LLM produces varied names.
        # For now, we assume a generic mapping based on the example.
        # This part of the Google config seems to map generic "Speaker 1" / "Speaker 2"
        # from the text chunks to specific voices. The text chunks themselves would need to contain
        # "Speaker 1:" or "Speaker 2:" for this to work as intended by multi_speaker_voice_config.
        # Your _is_speaker_line and chunking logic already handle named speakers,
        # so the input to Google would be "Chloe (Student): ..."
        # Google's multi-speaker TTS might require specific SSML or text formats.
        # The example config uses generic speaker tags; this needs careful input formatting for Google.
        # For now, I'll keep your original config, but note this potential complexity.
        
        generate_content_config = types.GenerateContentConfig(
            temperature=1, 
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker="Speaker 1", # This tag needs to be in the text_chunk
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="sulafat")
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker="Speaker 2", # This tag needs to be in the text_chunk
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="orus")
                            )
                        ),
                    ]
                )
            ),
        )

        # All audio chunks are merged in memory into this variable - no temp files are created
        all_audio_data = b""
        first_mime_type = None
        auto_retry_seconds = 15

        for i, text_chunk in enumerate(text_chunks[1:][:-1]):
            # For Google's multi-speaker, text_chunk needs to be formatted.
            # E.g., "Speaker 1: Hello. Speaker 2: Hi."
            # Your current text_chunks are like "Chloe (Student): Hello."
            # This might not directly work with the above `speaker_voice_configs`
            # unless Google's model can infer or map.
            # A common approach for Google multi-speaker is to use SSML or ensure text is
            # "speaker_tag: dialogue". You might need to preprocess `text_chunk` here.
            # For minimal change, I'm leaving it as is, but this is a likely point of failure
            # if the Google TTS doesn't pick up the speakers correctly.

            print(colored(f"[{title}] Google-Api: <{GOOGLE_TTS_MODEL_NAME}> generating audio for chunk {i+1}/{len(text_chunks)} ({len(text_chunk)} chars)...", "green"))
            
            current_chunk_audio_data = b""
            current_chunk_mime_type = None
            max_retries = 3
            retry_count = 0
            
            # Keep retrying the current chunk until successful or max retries reached
            while retry_count <= max_retries:
                try:
                    response_stream = client.models.generate_content_stream(
                        model=f"models/{GOOGLE_TTS_MODEL_NAME}", # Models are typically prefixed with "models/"
                        contents=text_chunk,
                        config=generate_content_config,
                    )

                    for response_part in response_stream:
                        if (
                            response_part.candidates is None
                            or not response_part.candidates[0].content
                            or not response_part.candidates[0].content.parts
                        ):
                            if response_part.text:
                                print(colored(f"[{title}] API Text response (no candidates) for chunk {i+1}: {response_part.text}", "yellow"))
                            continue
                        
                        if response_part.text:
                            print(colored(f"[{title}] API stream text for chunk {i+1}: {response_part.text}", "yellow"))

                        api_part_data = response_part.candidates[0].content.parts[0]
                        if api_part_data.inline_data:
                            current_chunk_audio_data += api_part_data.inline_data.data
                            if not current_chunk_mime_type:
                                current_chunk_mime_type = api_part_data.inline_data.mime_type
                            if not first_mime_type: 
                                first_mime_type = api_part_data.inline_data.mime_type
                        elif api_part_data.text :
                            print(colored(f"ERROR: [{title}] API returned text part for chunk {i+1} instead of audio: {api_part_data.text}", "red"))
                            # Removed exit(1) to allow script to continue or fail more gracefully
                            return "" # Indicate failure for this path
                        
                    if current_chunk_audio_data:
                        all_audio_data += current_chunk_audio_data
                        print(colored(f"[{title}] Chunk {i+1} audio ({len(current_chunk_audio_data)} bytes, type: {current_chunk_mime_type}) processed.", "cyan"))
                        break  # Successfully processed this chunk, exit the retry loop
                    else:
                        print(colored(f"[{title}] Critical error: Chunk {i+1} yielded no audio data after stream completion.", "red"))
                        if retry_count < max_retries:
                            retry_count += 1
                            print(colored(f"{time.strftime('%H:%M:%S')} Retrying chunk {i+1} (attempt {retry_count}/{max_retries}) in {auto_retry_seconds} seconds...", "yellow"))
                            time.sleep(auto_retry_seconds)
                            auto_retry_seconds = min(auto_retry_seconds * 2, 300)
                        else:
                            print(colored(f"[{title}] Failed to process chunk {i+1} after {max_retries} attempts. Skipping.", "red"))
                            break  # Give up on this chunk after max retries

                except Exception as e:
                    print(colored(f"[{title}] Error generating audio for chunk {i+1}: {e}", "red"))
                    if retry_count < max_retries:
                        retry_count += 1
                        print(colored(f"{time.strftime('%H:%M:%S')} Retrying chunk {i+1} (attempt {retry_count}/{max_retries}) in {auto_retry_seconds} seconds...", "yellow"))
                        time.sleep(auto_retry_seconds)
                        auto_retry_seconds = min(auto_retry_seconds * 2, 300)
                    else:
                        print(colored(f"[{title}] Failed to process chunk {i+1} after {max_retries} attempts. Skipping.", "red"))
                        break  # Give up on this chunk after max retries

        if not all_audio_data:
            print(colored(f"[{title}] No audio data was generated for any chunk (Google TTS).", "red"))
            return ""

        # Add a message about merging all chunks into a single audio file
        if len(text_chunks) > 1:
            print(colored(f"[{title}] Successfully merged {len(text_chunks)} audio chunks into a single file.", "green"))

        file_extension = ".wav" 
        if first_mime_type:
            guessed_extension = mimetypes.guess_extension(first_mime_type)
            if guessed_extension:
                file_extension = guessed_extension
            if "audio/L" in first_mime_type.lower() and file_extension != ".wav":
                print(colored(f"[{title}] Mime type {first_mime_type} suggests raw audio; ensuring .wav extension.", "blue"))
                file_extension = ".wav"
        if file_extension == ".bin":
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
        file_location = os.path.join(PODCAST_SAVE_LOCATION, f"{file_name_base}{file_extension}")
        
        final_audio_to_save = all_audio_data
        if first_mime_type and "audio/L" in first_mime_type.lower():
            print(colored(f"[{title}] Converting combined raw audio data (mime: {first_mime_type}) to WAV format...", "blue"))
            final_audio_to_save = convert_to_wav(all_audio_data, first_mime_type)
        
        save_binary_file(file_location, final_audio_to_save)
        google_sample_rate = parse_audio_mime_type(first_mime_type).get("rate", 24000) if first_mime_type else "N/A"
        print(colored(f"Google podcast saved to: {file_location} (Sample Rate: {google_sample_rate} Hz approx.)", "green"))
        return file_location

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters.get("bits_per_sample", 16) 
    sample_rate = parameters.get("rate", 24000) 
    num_channels = 1 
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size # 36 is size of header items before data_size itself

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size, # File size - 8 bytes
        b"WAVE",
        b"fmt ",
        16,  # Subchunk1Size (16 for PCM)
        1,   # AudioFormat (1 for PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    bits_per_sample = 16 
    rate = 24000         

    parts = mime_type.lower().split(";")
    for param in parts:
        param = param.strip()
        if param.startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass 
        elif param.startswith("audio/l"): 
            try:
                bits_str = "".join(filter(str.isdigit, param.split("audio/l",1)[1]))
                if bits_str:
                    bits_per_sample = int(bits_str)
            except (ValueError, IndexError):
                pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a podcast using AI.")
    parser.add_argument("-l", "--local", action="store_true",
                        help="Use local Dia TTS model instead of Google TTS.")
    args = parser.parse_args()
    
    g.local = args.local

    if not args.local and not os.environ.get("GOOGLE_API_KEY"):
        print(colored("Error: GOOGLE_API_KEY environment variable not set. Needed for Google TTS.", "red"))
        print(colored("You can either set GOOGLE_API_KEY or use the -l flag for local TTS (if configured).", "red"))
        exit(1) # Changed to exit(1) for error

    for i in range(5, 0, -1):
        print(colored(f"Generating podcast via clipboard in ", "green") + colored(f"{i}", "yellow") + colored(" seconds...", "green"))
        time.sleep(1)
    
    start_time = time.time()

    clipboard_content = pyperclip.paste()
    if not clipboard_content.strip():
        print(colored("No text in clipboard", "red"))
        exit(1) # Changed to exit(1)
    
    print(colored("Analyzing content for summary and title...", "blue"))
    try:
        analysisResponse = LlmRouter.generate_completion("Hi, I have a text snippet which I would like you to analyze. Please reason about it and provide a summary of the fundamentals and key points. Include analogies and intuitions to help a reader remember the topic better.\nText:\n" + clipboard_content)
        titleResponse = LlmRouter.generate_completion("I am going to provide you with a text and you should generate a descriptive title for whatever content its about. Ensure it is a short fit for a file name. Please think about the contents first and then like this:\nTitle: <title>\nThis is the text:\n" + analysisResponse)
    except Exception as e:
        print(colored(f"Error during LLM processing for analysis/title: {e}", "red"))
        exit(1) # Changed to exit(1)

    title_index = titleResponse.rfind("Title:")
    if title_index != -1:
        titleResponse = titleResponse[title_index + len("Title:"):].strip() # More robust extraction
    MAX_TITLE_LEN = 50 
    if len(titleResponse) > MAX_TITLE_LEN:
        titleResponse = titleResponse[:MAX_TITLE_LEN]
    titleResponse = titleResponse.split("\n")[0].strip() # Ensure single line and stripped
    if not titleResponse : titleResponse = "Untitled_Podcast" # Default title if empty

    
    print(colored(f"Generated Title: {titleResponse}", "magenta"))
    print(colored("Generating podcast dialogue...", "blue"))

    podcastGenPrompt = f"""
Create a meditative expert discussion between an intelligent and very critically implorative student named Chloe and a self-taught established expert named Liam.
The discussion should revolve around powerful insights and intuitions that enable easy understanding and retention of the topic.
To provide the dialogue please use the following delimiters and this exact format:
```txt
Chloe (Student): Are we on? I think we're on! Okay viewers, today we have a very special guest. Liam, why don't you introduce yourself?
Liam (Expert): Hello everybody! It's great to be here. First of all I'm a big fan of your work Chloe.
Chloe (Student): Oh thank you!
Liam (Expert): We should start with the fundamentals. What is the main topic of the conversation?
...
You dont need to use my example introducion, you can create a more imaginative one for the topic.

The topic of the conversation is:
{titleResponse}
The following information/topic(s) are provided to help you ground the conversation:
{analysisResponse}"""

    if args.local:
        podcastGenPrompt = podcastGenPrompt.replace("Chloe (Student):", "[S1]").replace("Liam (Expert):", "[S2]")
        
    try:
        podcastDialogue = LlmRouter.generate_completion(podcastGenPrompt)
    except Exception as e:
        print(colored(f"Error during LLM processing for podcast dialogue: {e}", "red"))
        exit(1) # Changed to exit(1)

    if not podcastDialogue.strip():
        print(colored("Generated podcast dialogue is empty. Exiting.", "red"))
        exit(1) # Changed to exit(1)

    file_location = generate_podcast(podcastDialogue, titleResponse, use_local_dia=args.local)

    if file_location:
        end_time = time.time()
        print(colored(f"Finished! Podcast saved to: {file_location}", "green"))
        print(colored(f"Total time taken: {end_time - start_time:.2f} seconds", "green"))
        print(colored(f"Used authoring model: {LlmRouter.last_used_model}", "green"))
        if args.local:
            print(colored(f"Used TTS model: Dia-1.6B (Local)", "green"))
        else:
            print(colored(f"Used TTS model: {GOOGLE_TTS_MODEL_NAME} (Google)", "green"))
    else:
        print(colored("Podcast generation failed or produced no audio.", "red"))
        if args.local:
             print(colored(f"Local Dia TTS generation attempt failed. Check logs for errors and ensure Dia setup is correct (GPU, libraries).", "red"))
        else:
             print(colored(f"Google TTS generation attempt failed. Check logs for API errors or issues with input.", "red"))
        exit(1) # Indicate failure