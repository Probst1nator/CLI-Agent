import logging
import warnings
from typing import Tuple, Optional
import soundfile as sf
import numpy as np
from termcolor import colored

from py_classes.ai_providers.cls_whisper_interface import WhisperInterface
from py_methods import utils_audio

logger = logging.getLogger(__name__)

class PyAiHost:
    """
    DEPRECATED: This class is deprecated and will be removed in a future version.
    Please use the WhisperInterface class instead:
    
    from py_classes.ai_providers.cls_whisper_interface import WhisperInterface
    
    This class now serves as a thin wrapper around WhisperInterface for backward compatibility.
    """
    
    # Singleton instance of WhisperInterface
    _whisper_interface = None
    
    # Remote host client
    remote_host_client = None
    
    @classmethod
    def _get_whisper_interface(cls):
        """Get or create the WhisperInterface singleton instance."""
        if cls._whisper_interface is None:
            cls._whisper_interface = WhisperInterface()
        return cls._whisper_interface
    
    @classmethod
    def initialize_wake_word(cls):
        """
        DEPRECATED: This method is deprecated.
        
        Initialize wake word detector if not already initialized.
        """
        warnings.warn(
            "PyAiHost is deprecated. Use WhisperInterface directly instead.",
            DeprecationWarning, stacklevel=2
        )
        # No explicit initialization needed anymore - handled by utils
            
    @classmethod
    def initialize_remote_host_client(cls):
        """Initialize the remote host client."""
        warnings.warn(
            "PyAiHost is deprecated. Use WhisperInterface directly instead.",
            DeprecationWarning, stacklevel=2
        )
        if cls.remote_host_client is None:
            try:
                from py_classes.remote_host.cls_remote_host_client import RemoteHostClient
                cls.remote_host_client = RemoteHostClient()
                logger.info("Remote host client initialized")
            except ImportError as e:
                logger.error(f"Failed to import RemoteHostClient: {e}")
                print(f"Local: <{colored('PyAiHost', 'red')}> failed to import RemoteHostClient: {e}")
                cls.remote_host_client = None
            except Exception as e:
                logger.error(f"Failed to initialize remote host client: {e}")
                print(f"Local: <{colored('PyAiHost', 'red')}> failed to initialize remote host client: {e}")
                cls.remote_host_client = None

    @classmethod
    def wait_for_wake_word(cls, use_remote: bool = False) -> Optional[str]:
        """
        DEPRECATED: This method is deprecated.
        
        Listen for wake word and return either the detected wake word or None if failed.
        
        Args:
            use_remote: If True, use the remote host for wake word detection
                        instead of local detection
        
        Returns:
            Optional[str]: The detected wake word or None if detection failed
        """
        warnings.warn(
            "PyAiHost is deprecated. Use WhisperInterface directly instead.",
            DeprecationWarning, stacklevel=2
        )
        # Use remote host if requested
        if use_remote:
            # Initialize remote client if needed
            if cls.remote_host_client is None:
                cls.initialize_remote_host_client()
                
            # If we still don't have a remote client, fall back to local
            if cls.remote_host_client is None:
                logger.warning("Remote host requested but failed to initialize client. Falling back to local.")
                return cls.wait_for_wake_word(use_remote=False)
                
            # Check if wake word service is available
            if not cls.remote_host_client.is_service_available("wake_word"):
                logger.warning("Wake word service not available on remote host. Falling back to local.")
                return cls.wait_for_wake_word(use_remote=False)
                
            # Use remote client
            return cls.remote_host_client.wait_for_wake_word()
        
        # Import here to avoid circular imports
        from py_methods.utils_audio import wait_for_wake_word as utils_wait_for_wake_word
        return utils_wait_for_wake_word()

    @classmethod
    def _initialize_whisper_model(cls, whisper_model_key: str = ''):
        """
        DEPRECATED: This method is deprecated.
        
        Initialize the Whisper speech recognition model.
        """
        warnings.warn(
            "PyAiHost is deprecated. Use WhisperInterface directly instead.",
            DeprecationWarning, stacklevel=2
        )
        # No explicit initialization needed anymore - handled by utils/WhisperInterface

    @classmethod
    def play_notification(cls):
        """
        DEPRECATED: This method is deprecated.
        
        Play a gentle, pleasant notification sound to indicate when to start speaking.
        """
        warnings.warn(
            "PyAiHost is deprecated. Use WhisperInterface directly instead.",
            DeprecationWarning, stacklevel=2
        )
        cls._get_whisper_interface().play_notification_sound()

    @classmethod
    def record_audio(cls, sample_rate: int = 44100, threshold: float = 0.05,
                    silence_duration: float = 2.0, min_duration: float = 1.0, 
                    max_duration: float = 30.0, use_wake_word: bool = True,
                    use_remote: bool = False) -> Tuple[np.ndarray, int]:
        """
        DEPRECATED: This method is deprecated.
        
        Record audio with automatic speech detection and optional wake word detection.
        """
        warnings.warn(
            "PyAiHost is deprecated. Use WhisperInterface directly instead.",
            DeprecationWarning, stacklevel=2
        )
        # Check for wake word if enabled using potentially remote client
        if use_wake_word:
            if not cls.wait_for_wake_word(use_remote=use_remote):
                return np.array([]), sample_rate
        
        # Import here to avoid circular imports
        from py_methods.utils_audio import record_audio as utils_record_audio
        return utils_record_audio(
            sample_rate=sample_rate,
            threshold=threshold,
            silence_duration=silence_duration,
            min_duration=min_duration,
            max_duration=max_duration,
            use_wake_word=False  # Already handled wake word above
        )

    @classmethod
    def play_audio(cls, audio_array: np.ndarray, sample_rate: int, blocking: bool = True):
        """
        DEPRECATED: This method is deprecated.
        
        Play audio using sounddevice.
        """
        warnings.warn(
            "PyAiHost is deprecated. Use WhisperInterface directly instead.",
            DeprecationWarning, stacklevel=2
        )
        cls._get_whisper_interface().play_audio_data(audio_array, sample_rate, blocking)

    @classmethod
    def transcribe_audio(cls, audio_path: str, whisper_model_key: str = '') -> Tuple[str, str]:
        """
        DEPRECATED: This method is deprecated.
        
        Transcribes the audio file.
        """
        warnings.warn(
            "PyAiHost is deprecated. Use WhisperInterface directly instead.",
            DeprecationWarning, stacklevel=2
        )
        # Read the audio file
        audio_data, sample_rate = sf.read(audio_path)
        return cls._get_whisper_interface().transcribe_audio(
            audio_data=audio_data,
            sample_rate=sample_rate,
            model=whisper_model_key
        )

if __name__ == "__main__":
    print("\n=== Starting Voice Interaction Test with deprecated PyAiHost ===")
    print("You will hear an ascending tone. After the tone, please speak...")
    import time
    time.sleep(1)
    PyAiHost.play_notification()
    
    # Record audio
    audio_file = "recorded_speech.wav"
    audio_data, sample_rate = PyAiHost.record_audio()
    sf.write(audio_file, audio_data, sample_rate)
    
    # Transcribe the recorded audio
    transcribed_text, detected_lang = PyAiHost.transcribe_audio(
        audio_path=audio_file,
        whisper_model_key='small'
    )
    
    print(f"\nTranscribed Text: {transcribed_text}")
    print(f"Detected Language: {detected_lang}")

    if transcribed_text:
        print("\n=== Testing Speech Synthesis ===")
        
        print("\nPlaying English synthesis...")
        utils_audio.text_to_speech(
            text=transcribed_text,
            voice='af_heart',
            play=True,
            speed=1.0
        )
    else:
        print("\nNo text was transcribed, skipping speech synthesis.")