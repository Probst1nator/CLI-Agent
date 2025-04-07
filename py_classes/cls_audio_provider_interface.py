from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import speech_recognition as sr

class AudioProviderInterface(ABC):
    """
    Abstract base class for audio provider interfaces.
    Defines the contract for speech recognition and synthesis services.
    """
    
    @abstractmethod
    def transcribe_audio(self, audio_data: Union[sr.AudioData, np.ndarray], 
                        language: str = "", 
                        model: str = "",
                        sample_rate: int = 44100) -> Tuple[str, str]:
        """
        Transcribes audio data to text.
        
        Args:
            audio_data: Audio data to transcribe
            language: Language hint for transcription
            model: Model to use for transcription
            sample_rate: Sample rate of the audio data if provided as numpy array
            
        Returns:
            Tuple[str, str]: (transcribed text, detected language)
        """
        pass
    
    @abstractmethod
    def record_and_transcribe(self,
                             language: str = "",
                             model: str = "",
                             sample_rate: int = 44100,
                             threshold: float = 0.05,
                             silence_duration: float = 2.0,
                             min_duration: float = 1.0,
                             max_duration: float = 30.0,
                             use_wake_word: bool = True) -> Tuple[str, str, np.ndarray]:
        """
        Records audio from microphone and transcribes it.
        
        Args:
            language: Language hint for transcription
            model: Model to use for transcription
            sample_rate: Sample rate for recording
            threshold: Volume threshold for speech detection
            silence_duration: Duration of silence to stop recording (seconds)
            min_duration: Minimum recording duration (seconds)
            max_duration: Maximum recording duration (seconds)
            use_wake_word: Whether to wait for wake word before recording
            
        Returns:
            Tuple[str, str, np.ndarray]: (transcribed text, detected language, audio data)
        """
        pass
    
    @abstractmethod
    def speak(self,
             text: str,
             voice: str = "",
             speed: float = 1.0,
             output_path: Optional[str] = None,
             play: bool = True) -> Union[List[str], None]:
        """
        Converts text to speech.
        
        Args:
            text: Text to convert to speech
            voice: Voice identifier
            speed: Speaking speed
            output_path: Optional path to save audio file
            play: Whether to play the audio
            
        Returns:
            Union[List[str], None]: List of generated audio file paths if output_path is provided, None otherwise
        """
        pass
    
    @abstractmethod
    def play_notification_sound(self) -> None:
        """
        Plays a notification sound.
        """
        pass
    
    @abstractmethod
    def play_audio_data(self, audio_array: np.ndarray, sample_rate: int = 44100, blocking: bool = True) -> None:
        """
        Plays audio data.
        
        Args:
            audio_array: Audio data to play
            sample_rate: Sample rate of the audio data
            blocking: Whether to block until audio finishes playing
        """
        pass 