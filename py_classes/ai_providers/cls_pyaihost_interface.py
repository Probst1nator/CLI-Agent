import os
import re
import time
from dotenv import load_dotenv
from gtts import gTTS
import pygame
import whisper
from typing import Dict, Tuple, Optional
import pyttsx3
from py_classes.globals import g

class PyAiHost:
    """
    A class to interface with the Whisper model for speech recognition tasks.

    Attributes:
        whisper_model (Optional[whisper.Whisper]): The Whisper model instance used for transcription.

    Methods:
        initialize_whisper_model(whisper_model_key: str = 'medium') -> None:
            Initializes the Whisper model with a specified model key.
        
        transcribe_audio(audio_path: str, whisper_model_key: str = 'medium') -> Tuple[str, str]:
            Transcribes the audio file at the given path and returns the transcribed text and detected language.
        
        local_text_to_speech(text: str, lang_key: str = "en") -> None:
            Converts text to speech using pyttsx3 and plays it locally.
        
        online_text_to_speech(text: str, lang_key: str = "en") -> None:
            Converts text to speech using gTTS and plays it.
    """
    
    whisper_model: Optional[whisper.Whisper] = None  # Class variable to hold the model instance

    @classmethod
    def _initialize_whisper_model(cls, whisper_model_key: str = 'medium'):
        """
        Initialize the Whisper model if it hasn't been initialized yet.

        Args:
            whisper_model_key (str): The key of the Whisper model to use.
        """
        if cls.whisper_model is None:
            cls.whisper_model = whisper.load_model(whisper_model_key)

    @classmethod
    def transcribe_audio(cls, audio_path: str, whisper_model_key: str = 'small') -> Tuple[str, str]:
        """
        Transcribes the audio file located at the specified path.

        Args:
            audio_path (str): Path to the audio file to be transcribed.
            whisper_model_key (str): The key of the Whisper model to use.
            
            Available Whisper model-keys: tiny, base, small, medium, large

        Returns:
            Tuple[str, str]: A tuple containing the transcribed text and the detected language.
        """
        # Initialize the model if it hasn't been initialized yet
        if cls.whisper_model is None:
            cls._initialize_whisper_model(whisper_model_key)

        
        # Load the .env file
        load_dotenv(g.PROJ_ENV_FILE_PATH)
        
        # Get the initial_prompt from the .env file, defaulting to an empty string if not found
        voice_activation_whisper_prompt = os.getenv('VOICE_ACTIVATION_WHISPER_PROMPT', '')
        
        try:
            # Perform transcription using the Whisper model
            if voice_activation_whisper_prompt:
                result: Dict[str, any] = cls.whisper_model.transcribe(audio_path, initial_prompt=voice_activation_whisper_prompt)
            else:
                result: Dict[str, any] = cls.whisper_model.transcribe(audio_path, initial_prompt=voice_activation_whisper_prompt)
                
            
            # Extract the transcribed text and detected language
            transcribed_text: str = result.get('text', '')
            detected_language: str = result.get('language', '')

            return transcribed_text, detected_language

        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            return "", ""


    @staticmethod
    def text_to_speech(text: str, lang_key: str = "en"):
        """
        Convert the assistant's response to speech locally and play it using pyttsx3.
        
        Args:
        text (str): The text to convert to speech.
        lang_key (str, optional): The language of the text. Currently not used in pyttsx3.
        
        Returns:
        None
        """
        def remove_markdown(text: str):
            # Remove code blocks
            text = re.sub(r'```[\s\S]*?```', '', text)
            # Remove inline code
            text = re.sub(r'`[^`\n]+`', '', text)
            # Remove bold and italic
            text = re.sub(r'\*\*?(.*?)\*\*?', r'\1', text)
            return text.strip()

        # Clean the text
        cleaned_text = remove_markdown(text)
        
        # Initialize the TTS engine
        engine = pyttsx3.init()
        
        # Speak the text
        engine.say(cleaned_text)
        engine.runAndWait()
        