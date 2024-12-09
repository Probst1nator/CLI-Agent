import logging
import os
import re
import sys
import sys
import tempfile
import time
from dotenv import load_dotenv
from termcolor import colored
from termcolor import colored
import whisper
from typing import Dict, Tuple, Optional, List
import torch
import soundfile as sf
import sounddevice as sd
import numpy as np
import outetts
from vosk import Model, KaldiRecognizer
import json
import queue
import sounddevice as sd

from vosk import Model, KaldiRecognizer
import json
import queue
import sounddevice as sd


logger = logging.getLogger(__name__)

class GlobalConfig:
    PROJ_ENV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

g = GlobalConfig()

class PyAiHost:
    whisper_model: Optional[whisper.Whisper] = None
    tts_interface: Optional[outetts.InterfaceHF] = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vosk_model: Optional[Model] = None
    
    @classmethod
    def initialize_wake_word(cls):
        """Initialize wake word detector if not already initialized."""
        if not cls.vosk_model:
            cls.vosk_model = Model(lang="en-us")  # Downloads small model automatically
            
    @classmethod
    def wait_for_wake_word(cls) -> bool:
        """Listen for wake word before starting main recording."""
        if not cls.vosk_model:
            cls.initialize_wake_word()
            
        q = queue.Queue()
        
        def callback(indata, frames, time, status):
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
                        if json_result:
                            result_str = json_result["text"].lower()
                            # Check for wake word in the recognized text
                            wake_words: List[str] = ["hey computer", "hey nova", "hey assistant", "hey ai", " nova"]
                            if any(wake_word in result_str for wake_word in wake_words):
                                return True
                                
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            return False

    
    @classmethod
    def _initialize_whisper_model(cls, whisper_model_key: str = 'medium'):
        if cls.whisper_model is None:
            # Force CPU device if CUDA is not available
            device = "cpu"  # Override device selection to ensure CPU usage
            cls.whisper_model = whisper.load_model(
                whisper_model_key,
                device=device
            )
            
            # Force CPU device if CUDA is not available
            device = "cpu"  # Override device selection to ensure CPU usage
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
            print(f"Local: <{colored('Whisper', 'green')}> is transcribing...")
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
        language: str = "en",
        speaker_path: Optional[str] = None,
        speaker_transcript: Optional[str] = None,
        output_path: Optional[str] = "tts_output.wav",
        play: bool = True,
        temperature: float = 0.1
    ) -> Optional[str]:
        """Convert text to speech using OuteTTS.

        Args:
            text (str): The text to convert to speech
            language (str, optional): Language code. Defaults to "en"
            speaker_path (Optional[str], optional): Path to speaker profile. Defaults to None
            output_path (Optional[str], optional): Output file path. Defaults to "tts_output.wav"
            play (bool, optional): Whether to play the audio. Defaults to True
            temperature (float, optional): Generation temperature. Defaults to 0.1

        Returns:
            Optional[str]: Path to the output file if successful, None if failed

        Raises:
            ValueError: If temperature is not between 0 and 1
        """
        if not 0 <= temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")

        try:
            # Always reinitialize for a new language
            if cls.tts_interface is None or language != cls._current_language:
                cls.tts_interface = None
                cls._initialize_tts_model(language)
                cls._current_language = language

            speaker = None
            if speaker_path:
                try:
                    # For WAV files, create a speaker profile first
                    if speaker_path.lower().endswith('.wav'):
                        # Load and validate audio file
                        audio_data, sample_rate = sf.read(speaker_path)
                        if len(audio_data) > 0:
                            # Create a simple sample transcript
                            speaker = cls.tts_interface.create_speaker(
                                audio_path=speaker_path,
                                transcript=speaker_transcript
                            )
                        else:
                            raise ValueError("Empty audio file")
                    # For JSON files, load the existing speaker profile
                    else:
                        speaker = cls.tts_interface.load_speaker(speaker_path)
                except Exception as e:
                    logger.warning(f"Could not load speaker profile: {e}")
                    logger.info("Continuing with default voice...")
                    speaker = None

            if not speaker:
                speaker = cls.tts_interface.load_default_speaker(name="female_2")

            print(f"Local: <{colored('OuteTTS', 'green')}> is synthesizing speech...")

            print(f"Local: <{colored('OuteTTS', 'green')}> is synthesizing speech...")

            # Only proceed if we either have no speaker or a valid speaker
            output = cls.tts_interface.generate(
                text=text,
                temperature=temperature,
                repetition_penalty=1.1,
                max_length=4096,
                speaker=speaker
            )

            delete_output = False
            temp_file = None

            try:
                if not output_path:
                    delete_output = True
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    output_path = temp_file.name

                output.save(output_path)

                if play:
                    data, sample_rate = sf.read(output_path)
                    cls.play_audio(data, sample_rate)

                return output_path

            finally:
                if delete_output and temp_file is not None:
                    try:
                        temp_file.close()
                        os.remove(output_path)
                    except OSError as e:
                        logger.warning(f"Could not delete temporary file: {e}")

        except Exception as e:
            logger.error(f"An error occurred during text-to-speech conversion: {e}")
            if "FlashAttention" in str(e):
                logger.info("Retrying without flash attention...")
                cls.tts_interface = None
                cls._initialize_tts_model(language, use_flash_attention=False)
                return cls.text_to_speech(text, language, speaker_path, output_path, play, temperature)
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