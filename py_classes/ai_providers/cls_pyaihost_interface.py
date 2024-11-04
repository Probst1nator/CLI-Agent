import os
import re
import time
from dotenv import load_dotenv
import whisper
from typing import Dict, Tuple, Optional
import torch
import soundfile as sf
import sounddevice as sd
import numpy as np
from bark import SAMPLE_RATE, generate_audio, preload_models
from py_classes.globals import g

class PyAiHost:
    """
    A class to interface with the Whisper model for speech recognition and Bark for speech synthesis.

    Attributes:
        whisper_model (Optional[whisper.Whisper]): The Whisper model instance used for transcription.
        bark_model_loaded (bool): Whether the Bark model has been loaded.
        device (str): The device to run the models on (cuda or cpu).
    """
    
    whisper_model: Optional[whisper.Whisper] = None
    bark_model_loaded: bool = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def _initialize_whisper_model(cls, whisper_model_key: str = 'medium'):
        """Initialize the Whisper model if it hasn't been initialized yet."""
        if cls.whisper_model is None:
            cls.whisper_model = whisper.load_model(whisper_model_key)

    @classmethod
    def _initialize_tts_model(cls):
        """Initialize the Bark models if they haven't been initialized yet."""
        if not cls.bark_model_loaded:
            preload_models(
                text_use_gpu=torch.cuda.is_available(),
                coarse_use_gpu=torch.cuda.is_available(),
                fine_use_gpu=torch.cuda.is_available(),
                codec_use_gpu=torch.cuda.is_available()
            )
            cls.bark_model_loaded = True

    @staticmethod
    def play_audio(audio_array: np.ndarray, sample_rate: int, blocking: bool = True):
        """
        Play audio using sounddevice.
        
        Args:
            audio_array (np.ndarray): The audio array to play.
            sample_rate (int): The sampling rate of the audio.
            blocking (bool): Whether to block until audio playback is complete.
        """
        try:
            # Ensure we have a clean audio stream
            sd.stop()
            
            # Ensure the audio array is float32 and normalized
            audio_array = audio_array.astype(np.float32)
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val * 0.9  # Leave some headroom
            
            # Play the audio
            sd.play(audio_array, sample_rate)
            
            if blocking:
                # Wait until the audio is done playing
                sd.wait()
                
        except Exception as e:
            print(f"An error occurred during audio playback: {e}")
            sd.stop()

    @classmethod
    def transcribe_audio(cls, audio_path: str, whisper_model_key: str = 'small') -> Tuple[str, str]:
        """
        Transcribes the audio file located at the specified path.

        Args:
            audio_path (str): Path to the audio file to be transcribed.
            whisper_model_key (str): The key of the Whisper model to use.
            
        Returns:
            Tuple[str, str]: A tuple containing the transcribed text and the detected language.
        """
        if cls.whisper_model is None:
            cls._initialize_whisper_model(whisper_model_key)
        
        load_dotenv(g.PROJ_ENV_FILE_PATH)
        voice_activation_whisper_prompt = os.getenv('VOICE_ACTIVATION_WHISPER_PROMPT', '')
        
        try:
            if voice_activation_whisper_prompt:
                result: Dict[str, any] = cls.whisper_model.transcribe(audio_path, initial_prompt=voice_activation_whisper_prompt)
            else:
                result: Dict[str, any] = cls.whisper_model.transcribe(audio_path)
            
            transcribed_text: str = result.get('text', '')
            detected_language: str = result.get('language', '')

            return transcribed_text, detected_language

        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            return "", ""

    @classmethod
    def text_to_speech(cls, text: str, lang_key: str = "en", output_path: str = "output.wav", play: bool = True):
        """
        Convert text to speech using Bark and optionally play it.
        
        Args:
            text (str): The text to convert to speech.
            lang_key (str): The language of the text (currently supports mainly English).
            output_path (str): Path where the output audio file will be saved.
            play (bool): Whether to play the audio after generation.
        
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

        # Initialize Bark if needed
        if not cls.bark_model_loaded:
            cls._initialize_tts_model()

        # Clean the text
        cleaned_text = remove_markdown(text)
        
        try:
            # Generate speech using Bark
            # Note: Bark will automatically use GPU if available due to our initialization
            audio_array = generate_audio(cleaned_text)
            
            # Save the audio if path provided
            if output_path:
                sf.write(output_path, audio_array, SAMPLE_RATE)
            
            # Play the audio if requested
            if play:
                cls.play_audio(audio_array, SAMPLE_RATE)
            
        except Exception as e:
            print(f"An error occurred during text-to-speech conversion: {e}")
            
    @classmethod
    def set_voice(cls, voice_preset: str = "v2/en_speaker_6"):
        """
        Set the voice preset for Bark TTS.
        Available presets can be found in Bark's documentation.
        
        Args:
            voice_preset (str): The voice preset to use.
        """
        os.environ["BARK_SPEAKER"] = voice_preset