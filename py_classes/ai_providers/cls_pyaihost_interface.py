import logging
import os
import sys
import time
import warnings
from dotenv import load_dotenv
from termcolor import colored
import whisper
from typing import Dict, Tuple, Optional, List, Union
import torch
import soundfile as sf
import sounddevice as sd
import numpy as np
import outetts
from vosk import Model, KaldiRecognizer
import json
import queue
import sounddevice as sd
from kokoro import KPipeline
from py_classes.globals import g

logger = logging.getLogger(__name__)

class PyAiHost:
    whisper_model: Optional[whisper.Whisper] = None
    tts_interface: Optional[outetts.InterfaceHF] = None
    kokoro_pipeline: Optional[KPipeline] = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vosk_model: Optional[Model] = None
    
    @classmethod
    def initialize_wake_word(cls):
        """Initialize wake word detector if not already initialized."""
        if not cls.vosk_model:
            if not g.DEBUG_LOGGING:
                # Temporarily redirect stderr to suppress Vosk initialization logs
                stderr = sys.stderr
                with open(os.devnull, 'w') as devnull:
                    sys.stderr = devnull
                    try:
                        cls.vosk_model = Model(lang="en-us")  # Downloads small model automatically
                    finally:
                        sys.stderr = stderr
            else:
                # In debug mode, show all logs
                cls.vosk_model = Model(lang="en-us")  # Downloads small model automatically
            
    @classmethod
    def wait_for_wake_word(cls) -> Optional[str]:
        """Listen for wake word and return either the detected wake word or None if failed."""
        if not cls.vosk_model:
            cls.initialize_wake_word()
            
        q: queue.Queue = queue.Queue()
        
        def callback(indata: np.ndarray, frames: int, time: object, status: object) -> None:
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status, file=sys.stderr)
            q.put(bytes(indata))
            
        try:
            device_info = sd.query_devices(None, 'input')
            samplerate = int(device_info['default_samplerate'])
            
            rec = KaldiRecognizer(cls.vosk_model, samplerate)
            rec.SetWords(True)
            
            print(f"Local: <{colored('Vosk', 'green')}> is listening for wake word...")
            
            with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=None,
                                 dtype='int16', channels=1, callback=callback):
                
                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        json_result = json.loads(rec.Result())
                        result_str = json_result.get("text", "").lower()
                        print(f"Local: <{colored('Vosk', 'green')}> detected: {result_str}")
                        
                        wake_words: List[str] = [
                            # Single words
                            "ada", "ai", "assistant", "computer", "nova",
                            # Full phrases 
                            "a ai", "a assistant", "a computer", "a nova",
                            "hey ada", "hey ai", "hey assistant", "hey computer", "hey nova",
                            "ok ada", "ok ai", "ok assistant", "ok computer", "ok nova",
                            "Okay SmartHome", "Okay Zuhause",
                            "SmartHome", "Zuhause"
                        ]
                        
                        for wake_word in wake_words:
                            if wake_word in result_str:
                                return wake_word
                                
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            return None

    
    @classmethod
    def _initialize_whisper_model(cls, whisper_model_key: str = 'medium'):
        if cls.whisper_model is None:
            # Force CPU device if CUDA is not available
            device = "cpu"  # Override device selection to ensure CPU usage
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                cls.whisper_model = whisper.load_model(
                    whisper_model_key,
                    device=device
                )

    @classmethod
    def play_notification(cls):
        """Play a gentle, pleasant notification sound to indicate when to start speaking.
        
        The sound is designed to be:
        - Soft attack/decay to avoid startling
        - Uses musical intervals (perfect fifth) for pleasantness
        - Lower frequencies that feel less urgent
        - Short duration to be unobtrusive
        - Graduated fade in/out for smoothness
        """
        sample_rate = 44100
        duration = 0.5  # Slightly shorter duration
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Use G3 (196 Hz) and D4 (294 Hz) - a perfect fifth interval
        # Lower frequencies feel less urgent/alarming
        freq1, freq2 = 196, 294
        
        # Create two tones with the perfect fifth interval
        tone1 = 0.3 * np.sin(2 * np.pi * freq1 * t)
        tone2 = 0.2 * np.sin(2 * np.pi * freq2 * t)
        
        # Combine tones with slight delay on second tone for natural feel
        delay_samples = int(0.05 * sample_rate)
        signal = np.zeros_like(t)
        signal += tone1
        signal[delay_samples:] += tone2[:-delay_samples]

        # Add gentle amplitude envelope
        fade_duration = duration * 0.6  # Longer fade for softness
        fade_len = int(fade_duration * sample_rate)
        
        # Use cosine-based fades for smoother transitions
        fade_in = (1 - np.cos(np.linspace(0, np.pi, fade_len))) / 2
        fade_out = (1 + np.cos(np.linspace(0, np.pi, fade_len))) / 2
        
        # Apply envelope
        signal[:fade_len] *= fade_in
        signal[-fade_len:] *= fade_out
        
        # Add subtle reverb effect for warmth
        reverb_delay = int(0.5 * sample_rate)
        reverb = np.zeros_like(signal)
        reverb[reverb_delay:] = 0.2 * signal[:-reverb_delay]
        signal = signal + reverb
        
        # Normalize to prevent clipping
        signal = signal / np.max(np.abs(signal))
        
        cls.play_audio(signal, sample_rate)

    @classmethod
    def record_audio(cls, sample_rate: int = 44100, threshold: float = 0.05,
                    silence_duration: float = 2.0, min_duration: float = 1.0, 
                    max_duration: float = 30.0, use_wake_word: bool = True) -> Tuple[np.ndarray, int]:
        """
        Record audio with automatic speech detection and optional wake word detection.
        Record audio with automatic speech detection and optional wake word detection.
        
        Args:
            sample_rate (int): Sample rate for recording
            threshold (float): Volume threshold for speech detection
            silence_duration (float): Duration of silence to stop recording (seconds)
            min_duration (float): Minimum recording duration (seconds)
            max_duration (float): Maximum recording duration (seconds)
            use_wake_word (bool): Whether to wait for wake word before recording
                
            use_wake_word (bool): Whether to wait for wake word before recording
                
        Returns:
            Tuple[np.ndarray, int]: Recorded audio array and sample rate
        """
        # Check for wake word if enabled
        if use_wake_word and cls.porcupine_instance:
            if not cls.wait_for_wake_word():
                return np.array([]), sample_rate
            
        # Check for wake word if enabled
        if use_wake_word and cls.porcupine_instance:
            if not cls.wait_for_wake_word():
                return np.array([]), sample_rate
            
        chunk_duration = 0.1  # Process audio in 100ms chunks
        chunk_samples = int(chunk_duration * sample_rate)
        
        # Pre-allocate buffer for maximum duration
        max_chunks = int(max_duration / chunk_duration)
        recording_buffer = np.zeros((max_chunks * chunk_samples,), dtype=np.float32)
        
        # Recording state variables
        has_speech = False
        silent_chunks = 0
        needed_silent_chunks = int(silence_duration / chunk_duration)
        min_chunks = int(min_duration / chunk_duration)
        recorded_chunks = 0
        
        print("\nListening... (speak now, recording will stop after silence)")
        
        # Play notification sound after wake word detection
        cls.play_notification()
        
        # Play notification sound after wake word detection
        cls.play_notification()
        
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_samples
        )
        
        with stream:
            while recorded_chunks < max_chunks:
                audio_chunk, _ = stream.read(chunk_samples)
                chunk_max = np.max(np.abs(audio_chunk))
                
                # Detect if this chunk has speech
                is_speech = chunk_max > threshold
                
                if is_speech:
                    has_speech = True
                    silent_chunks = 0
                elif has_speech:
                    silent_chunks += 1
                
                # If we've detected speech at any point, record the chunk
                if has_speech:
                    chunk_start = recorded_chunks * chunk_samples
                    chunk_end = chunk_start + chunk_samples
                    recording_buffer[chunk_start:chunk_end] = audio_chunk.flatten()
                    recorded_chunks += 1
                    
                    # Print a simple indicator that we're recording
                    if is_speech:
                        print(".", end="", flush=True)
                    
                    # Only stop if we've met minimum duration and have enough silence
                    if recorded_chunks >= min_chunks and silent_chunks >= needed_silent_chunks:
                        print("\nSpeech complete!")
                        break
        
        # Trim the buffer to only the recorded chunks
        recording = recording_buffer[:recorded_chunks * chunk_samples]
        
        # If no speech was detected or recording is too short, return empty recording
        if not has_speech or recorded_chunks < min_chunks:
            print("\nNo valid speech detected or recording too short!")
            return np.array([]), sample_rate
        
        return recording, sample_rate

    @staticmethod
    def play_audio(audio_array: np.ndarray, sample_rate: int, blocking: bool = True):
        """Play audio using sounddevice."""
        try:
            sd.stop()
            audio_array = audio_array.astype(np.float32)
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array / max_val * 0.9
            
            sd.play(audio_array, sample_rate)
            
            if blocking:
                sd.wait()
                
        except Exception as e:
            print(f"An error occurred during audio playback: {e}")
            sd.stop()

    @classmethod
    def transcribe_audio(cls, audio_path: str, whisper_model_key: str = 'small') -> Tuple[str, str]:
        """Transcribes the audio file."""
        if cls.whisper_model is None:
            cls._initialize_whisper_model(whisper_model_key)
        
        load_dotenv(g.PROJ_ENV_FILE_PATH)
        voice_activation_whisper_prompt = os.getenv('VOICE_ACTIVATION_WHISPER_PROMPT', '')
        
        try:
            print(f"Local: <{colored('Whisper', 'green')}> is transcribing...")
            # Use CUDA if available but suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
    def _initialize_tts_model(cls, language: str = "en", use_flash_attention: bool = False):
        if cls.tts_interface is None:
            model_config = outetts.HFModelConfig_v1(
                model_path="OuteAI/OuteTTS-0.2-500M",
                language=language,
                dtype=torch.float32,
                device=cls.device,  # Add explicit device specification
            )
    

    @classmethod
    def text_to_speech(
        cls,
        text: str,
        voice: str = 'af_heart',
        speed: float = 1.0,
        output_path: Optional[str] = None,
        play: bool = True,
        split_pattern: str = r'\n+'
    ) -> Union[List[str], None]:
        """Convert text to speech using Kokoro TTS model.

        Args:
            text (str): The text to convert to speech
            language_code (str, optional): Language code. Defaults to 'a'.
                'a' or 'en' or 'en-us' => American English
                'b' or 'en-gb' => British English
                'e' or 'es' => Spanish
                'f' or 'fr' or 'fr-fr' => French
                'h' or 'hi' => Hindi
                'i' or 'it' => Italian
                'p' or 'pt' or 'pt-br' => Brazilian Portuguese
                'j' or 'ja' => Japanese (requires: pip install misaki[ja])
                'z' or 'zh' => Mandarin Chinese (requires: pip install misaki[zh])
            voice (str, optional): Voice ID to use. Defaults to 'af_heart'.
            speed (float, optional): Speech speed multiplier. Defaults to 1.0.
            output_path (Optional[str], optional): Base path for output files. Will append index if multiple segments.
            play (bool, optional): Whether to play the audio. Defaults to True.
            split_pattern (str, optional): Regex pattern for splitting text into segments. Defaults to r'\n+'.

        Returns:
            Union[List[str], None]: List of generated audio file paths if output_path is provided, None otherwise

        Raises:
            ImportError: If kokoro package is not installed
            ValueError: If invalid parameters are provided
            Exception: For other errors during synthesis
        """
        try:

            if cls.kokoro_pipeline is None:
                print(f"Local: <{colored('Kokoro', 'green')}> initializing pipeline...")
                cls.kokoro_pipeline = KPipeline(lang_code="b")

            if not 0.1 <= speed <= 3.0:
                raise ValueError("Speed must be between 0.1 and 3.0")

            print(f"Local: <{colored('Kokoro', 'green')}> is synthesizing speech...")
            
            output_files = []
            generator = cls.kokoro_pipeline(
                text,
                voice=voice,
                speed=speed,
                split_pattern=split_pattern
            )

            for i, (graphemes, phonemes, audio) in enumerate(generator):
                if output_path:
                    # Handle both with and without extension
                    base, ext = os.path.splitext(output_path)
                    if not ext:
                        ext = '.wav'
                    current_output = f"{base}_{i}{ext}" if i > 0 else f"{base}{ext}"
                    # Convert tensor to numpy array if needed
                    audio_np = audio.cpu().numpy() if torch.is_tensor(audio) else audio
                    sf.write(current_output, audio_np, 24000)
                    output_files.append(current_output)
                
                if play:
                    # Convert tensor to numpy array if needed
                    audio_np = audio.cpu().numpy() if torch.is_tensor(audio) else audio
                    cls.play_audio(audio_np, 24000)

            return output_files if output_path else None

        except ImportError:
            print("Kokoro is not installed. Please install it with: pip install kokoro>=0.3.4")
            print("For Japanese support: pip install misaki[ja]")
            print("For Chinese support: pip install misaki[zh]")
            return None
        except Exception as e:
            print(f"An error occurred during Kokoro text-to-speech conversion: {e}")
            return None


if __name__ == "__main__":
    print("\n=== Starting Voice Interaction Test ===")
    print("You will hear an ascending tone. After the tone, please speak...")
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
        PyAiHost.text_to_speech(
            text=transcribed_text,
            # speaker_path="speaker_en.wav",
            # speaker_transcript="Hey, can you see the raindrops falling down? I've seen some stuff but this is even more mesmerizing than expected.",
            language="en",
            play=True,
            temperature=0.4
        )
    else:
        print("\nNo text was transcribed, skipping speech synthesis.")