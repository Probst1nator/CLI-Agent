import os
import re
import time
from gtts import gTTS
import pygame
import whisper
from typing import Dict, Tuple, Optional
import pyttsx3

class PyAiHost:
    """
    A class to interface with the Whisper model for speech recognition tasks.

    Attributes:
        model (Optional): The Whisper model instance used for transcription.
        timeout (int): The maximum allowed transcription duration in seconds.

    Methods:
        initialize(model_name: str = 'base', timeout: int = 60) -> None:
            Initializes the PyAIHost class with a specified Whisper model and timeout value.
        
        transcribe_audio(audio_path: str) -> Tuple[str, str, bool]:
            Transcribes the audio file at the given path and returns the transcribed text, 
            detected language, and a timeout flag.
    """
    
    whisper_model: whisper.Whisper|None = None  # Class variable to hold the model instance

    @classmethod
    def initialize(cls, whisper_model_name: str = 'medium') -> None:
        """
        Initializes the PyAIHost class with a specified Whisper model and timeout value.
        Available Whisper models: tiny, base, small, medium, large

        Args:
            whisper_model_name (str): Name of the Whisper model to load. Default is 'medium'.
            timeout (int): Maximum allowed duration for transcription in seconds. Default is 60 seconds.
        """
        cls.whisper_model = whisper.load_model(whisper_model_name)

    @staticmethod
    def transcribe_audio(audio_path: str) -> Tuple[str, str]:
        """
        Transcribes the audio file located at the specified path.

        Args:
            audio_path (str): Path to the audio file to be transcribed.

        Returns:
            Tuple[str, str]: A tuple containing the transcribed text and the detected language.
        """
        if PyAiHost.whisper_model is None:
            raise ValueError("Model not initialized. Call PyAIHost.initialize() first.")

        try:
            
            # initial_prompt_en = "This text refers to FAU's STEM study offerings. It may mention the degree programs in Computational Mathematics, Data Science, Computer Science, and AI, as well as their contents and career prospects."
            # initial_prompt_de = "Die folgende Aufnahme enthält Informationen über Studiengänge an der Friedrich-Alexander-Universität (FAU) Erlangen-Nürnberg. Besonderer Fokus liegt auf Technomathematik, Data Science, Lehramt, Informatik, Künstliche Intelligenz (KI) und Materialtechnologie."
            initial_prompt_multilingual = "Pepper, Roboter, Friedrich-Alexander-Universität, FAU, University, Erlangen, Technomathematik, Data Science, Lehramt, Artificial Intelligence"
            # Perform transcription using the Whisper model
            result: Dict[str, any] = PyAiHost.whisper_model.transcribe(audio_path, initial_prompt=initial_prompt_multilingual)
            # Extract the transcribed text if available, else default to an empty string
            transcribed_text: str = result.get('text', '')
            # Extract the detected language if available, else default to an empty string
            detected_language: str = result.get('language', '')

            # Check if the transcription exceeded the specified timeout
            return transcribed_text, detected_language

        except Exception as e:
            # Handle any exception that occurs and print the error message
            print(f"An error occurred during transcription: {e}")
            # On error, return empty values and indicate a timeout occurred
            return "", ""
    
    def local_text_to_speech(text: str, lang_key: str = "en"):
        """
        Convert the assistant's response to speech and play it locally using pyttsx3.
        
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
        
        # Optional: Adjust the speech rate (default is 200)
        # engine.setProperty('rate', 150)
        
        # Optional: Set the voice (uncomment and modify if needed)
        # voices = engine.getProperty('voices')
        # engine.setProperty('voice', voices[1].id)  # 0 for male, 1 for female voice
        
        # Speak the text
        engine.say(cleaned_text)
        engine.runAndWait()

    # Convert assistant response to speech and play it
    def local_text_to_speech(text: str, lang_key: str = "en"):
        """
        THIS IS RUNNING ONLINE AND NOT LOCALLY, pls fix
        Convert the assistant's response to speech and play it.

        Args:
            text (str): The text to convert to speech.
            language (str, optional): The language of the text. Defaults to 'english'.

        Returns:
            None
        """
        def remove_single_asterisk(text: str):
            result = ""
            i = 0
            inside_single = False
            inside_double = False
            
            while i < len(text):
                if text[i:i+2] == "**":
                    result += "**"
                    inside_double = not inside_double
                    i += 2
                elif text[i] == "*" and not inside_double:
                    inside_single = not inside_single
                    i += 1
                elif not inside_single:
                    result += text[i]
                    i += 1
                else:
                    i += 1
            
            return result
        text = remove_single_asterisk(text)

        tts_file: str = "tts_response.mp3"

        tts = gTTS(text=text, lang=lang_key)
        tts.save(tts_file)
        pygame.mixer.init()
        pygame.mixer.music.load(tts_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        os.remove(tts_file)

