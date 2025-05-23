# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import mimetypes
import os
import re
import struct
from google import genai
from google.genai import types
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import Llm, LlmRouter
import pyperclip
from termcolor import colored
def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")


def generate_podcast(text: str):
    client = genai.Client(
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    model = "gemini-2.5-flash-preview-tts"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=[
            "audio",
        ],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 1",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Fenrir"
                            )
                        ),
                    ),
                    types.SpeakerVoiceConfig(
                        speaker="Speaker 2",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Gacrux"
                            )
                        ),
                    ),
                ]
            ),
        ),
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data:
            file_name = "ENTER_FILE_NAME"
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            if file_extension is None:
                file_extension = ".wav"
                data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)
            save_binary_file(f"{file_name}{file_extension}", data_buffer)
        else:
            print(chunk.text)

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file header.
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

    # http://soundfile.sapp.org/doc/WaveFormat/

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size (size of audio data)
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    bits_per_sample = 16
    rate = 24000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts: # Skip the main type part
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                # Handle cases like "rate=" with no value or non-integer value
                pass # Keep rate as default
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass # Keep bits_per_sample as default if conversion fails

    return {"bits_per_sample": bits_per_sample, "rate": rate}


if __name__ == "__main__":
    clipboard_content = pyperclip.paste()
    if clipboard_content == "":
        print(colored("No text in clipboard", "red"))
        exit()
    scientificExplorerChat = Chat("You are well walked scientist and explorer of the world. You are being asked to provide your critical and honest view on the presented topic. Work yourself throught the presented text coherently and allow yourself to be creative. Honesty is first and accuracy, clarity and focus are second.")
    scientificExplorerChat.add_message(Role.USER, "Hi, the following text was sent over by a viewer. Please elaborate on it and preferrably think aloud while doing so.\nText:\n" + clipboard_content)
    scientificExplorerDeepdive = LlmRouter.generate_completion(scientificExplorerChat)
    podcastTopicChat = Chat("You are an AI assistant that is tasked with asking another AI assistant to generate a podcast topic.")
    podcastTopicChat.add_message(Role.USER, "Can you please generate a question that asks to generate a podcast about the following text:\n" + scientificExplorerDeepdive)
    podcastTopic = LlmRouter.generate_completion(podcastTopicChat)
    
    
    podcastDialogueChat = Chat("You are a intelligent storyteller with deep education in science, the art of teaching and. You task is to create a dialogue in the style of a podcast between a intelligent and very critically implorative host and an expert guest, discussing a topic.")
    podcastGenPrompt = f"""
    Create a dialogue in the style of a podcast between an intelligent and very critically implorative host named Chloe and a self taught world class expert guest named Liam, who is trying to show off his true expertise on the subject.
    When you provide the dialogue please use the following delimiters and this exact format:
    ```txt
    Chloe (Host): Are we on? I think we're on! Okay viewers, today we have a very special guest. Liam, why don't you introduce yourself?
    Liam (Guest): Hello everybody! It's great to be here. First of all I'm a big fan of your work Chloe
    Chloe (Host): Oh thank you!
    Liam (Guest): and I'm very excited to share some of my thoughts with you and the audience
    ...
    ```
    The topic of the conversation is:
    {podcastTopic}
    The following information/topic(s) are provided to help you ground the conversation:
    {scientificExplorerDeepdive}
    """
    podcastDialogueChat.add_message(Role.USER, podcastGenPrompt)
    podcastDialogue = LlmRouter.generate_completion(podcastDialogueChat)

    generate_podcast(podcastDialogue)
