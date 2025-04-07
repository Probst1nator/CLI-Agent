import logging
import os
import sys
import time
import warnings
from typing import Dict, Tuple, Optional, List, Union, Any

import json
import numpy as np
import outetts
import psutil
import queue
import sounddevice as sd
import soundfile as sf
import torch
import whisper
from scipy import signal
from termcolor import colored
from vosk import Model, KaldiRecognizer
from dotenv import load_dotenv

# Try to import kokoro if available
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global resources
_whisper_model: Optional[whisper.Whisper] = None
_whisper_model_key: Optional[str] = None
_tts_interface: Optional[outetts.InterfaceHF] = None
_kokoro_pipeline: Optional[Any] = None  # Using Any to avoid type issues if kokoro not available
_vosk_model: Optional[Model] = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_wake_word() -> None:
    """Initialize wake word detector if not already initialized."""
    global _vosk_model
    
    if not _vosk_model:
        if not logger.isEnabledFor(logging.DEBUG):
            # Temporarily redirect stderr to suppress Vosk initialization logs
            stderr = sys.stderr
            with open(os.devnull, 'w') as devnull:
                sys.stderr = devnull
                try:
                    _vosk_model = Model(lang="en-us")  # Downloads small model automatically
                finally:
                    sys.stderr = stderr
        else:
            # In debug mode, show all logs
            _vosk_model = Model(lang="en-us")  # Downloads small model automatically


def wait_for_wake_word() -> Optional[str]:
    """
    Listen for wake word and return either the detected wake word or None if failed.
    
    Returns:
        Optional[str]: The detected wake word or None if detection failed
    """
    global _vosk_model
    
    if not _vosk_model:
        initialize_wake_word()
        
    # Use a smaller queue to prevent memory buildup
    audio_queue = queue.Queue(maxsize=5)
    
    try:
        # Force the correct sample rate for Vosk
        target_sample_rate = 16000
        
        # Wake word list
        wake_words: List[str] = [
            # Single words
            "computer", "nova", "system",
            # Full phrases 
            "hey computer", "hey nova",
            "ok computer", "ok nova",
            "okay computer", "okay nova",
            "hey system", "okay system"
        ]
        
        # Create recognizer with correct sample rate
        rec = KaldiRecognizer(_vosk_model, target_sample_rate)
        rec.SetWords(True)
        
        print(f"<{colored('Vosk', 'green')}> is listening for wake word...")
        
        # Track last overflow message time to reduce spam
        last_overflow_time = time.time()
        overflow_count = 0
        
        def audio_callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            """Called for each audio block from the sound device."""
            nonlocal last_overflow_time, overflow_count
            
            # Handle status messages
            if status:
                now = time.time()
                if status.input_overflow:
                    overflow_count += 1
                    if now - last_overflow_time > 5.0:  # Only log every 5 seconds
                        print(f"Input overflow occurred {overflow_count} times")
                        last_overflow_time = now
                else:
                    print(f"Audio status: {status}")
            
            # Only add to queue if there's room (non-blocking)
            if not audio_queue.full():
                try:
                    audio_queue.put_nowait(bytes(indata))
                except queue.Full:
                    pass  # Queue is full, skip this frame
        
        # Device configuration options
        device_config = {
            'samplerate': target_sample_rate,
            'channels': 1,
            'dtype': 'int16',
            'latency': 'high',  # Use high latency for more stable processing
            'blocksize': 4000   # 250ms at 16kHz - large enough to reduce callbacks
        }
        
        # Start input stream
        with sd.RawInputStream(**device_config, callback=audio_callback):
            print("Listening for wake words...")
            
            while True:
                # Non-blocking queue get with timeout
                try:
                    audio_data = audio_queue.get(timeout=0.5)
                    
                    # Process audio data
                    if rec.AcceptWaveform(audio_data):
                        try:
                            result = json.loads(rec.Result())
                            text = result.get("text", "").lower().strip()
                            
                            if text:
                                print(f"Detected: '{text}'")
                                
                                # Check for wake words (simple contains check)
                                if text.count(" ") <= 4:  # Skip if too many words (likely not a wake word)
                                    for wake_word in wake_words:
                                        if wake_word in text:
                                            return wake_word
                        except json.JSONDecodeError:
                            pass  # Invalid JSON, just skip
                
                except queue.Empty:
                    # Timeout on queue - this is normal, just continue
                    continue
                
                except Exception as e:
                    # Log other errors but keep running
                    print(f"Error in speech recognition: {e}")
            
    except Exception as e:
        print(f"Error in wake word detection: {e}")
        import traceback
        traceback.print_exc()
        return None


def play_notification() -> None:
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
    signal_data = np.zeros_like(t)
    signal_data += tone1
    signal_data[delay_samples:] += tone2[:-delay_samples]

    # Add gentle amplitude envelope
    fade_duration = duration * 0.6  # Longer fade for softness
    fade_len = int(fade_duration * sample_rate)
    
    # Use cosine-based fades for smoother transitions
    fade_in = (1 - np.cos(np.linspace(0, np.pi, fade_len))) / 2
    fade_out = (1 + np.cos(np.linspace(0, np.pi, fade_len))) / 2
    
    # Apply envelope
    signal_data[:fade_len] *= fade_in
    signal_data[-fade_len:] *= fade_out
    
    # Add subtle reverb effect for warmth
    reverb_delay = int(0.5 * sample_rate)
    reverb = np.zeros_like(signal_data)
    reverb[reverb_delay:] = 0.2 * signal_data[:-reverb_delay]
    signal_data = signal_data + reverb
    
    # Normalize to prevent clipping
    signal_data = signal_data / np.max(np.abs(signal_data))
    
    play_audio(signal_data, sample_rate)


def play_audio(audio_array: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
    """Play audio using sounddevice.
    
    Args:
        audio_array: The audio data to play
        sample_rate: The sample rate of the audio data
        blocking: Whether to block until audio finishes playing
    """
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


def record_audio(
    sample_rate: int = 44100, 
    threshold: float = 0.05,
    silence_duration: float = 2.0, 
    min_duration: float = 1.0, 
    max_duration: float = 30.0, 
    use_wake_word: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Record audio with automatic speech detection and optional wake word detection.
    
    Args:
        sample_rate: Sample rate for recording
        threshold: Volume threshold for speech detection
        silence_duration: Duration of silence to stop recording (seconds)
        min_duration: Minimum recording duration (seconds)
        max_duration: Maximum recording duration (seconds)
        use_wake_word: Whether to wait for wake word before recording
                
    Returns:
        Tuple[np.ndarray, int]: Recorded audio array and sample rate
    """
    # Check for wake word if enabled
    if use_wake_word:
        if not wait_for_wake_word():
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
    play_notification()
    
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


def initialize_whisper_model(whisper_model_key: str = '') -> None:
    """
    Initialize the Whisper speech recognition model.
    
    Model Sizes:
    - tiny: 39M parameters
    - base: 74M parameters
    - small: 244M parameters
    - medium: 769M parameters
    - large: 1550M parameters
    - large-v2: 1550M parameters (improved version of large)
    
    Args:
        whisper_model_key: The key of the Whisper model to initialize. If empty, will select based on available resources.
    """
    global _whisper_model, _whisper_model_key, _device
    
    if _whisper_model is None:
        # If no key provided, decide using cuda availability and core count
        if whisper_model_key == '':
            if _device == "cuda":
                # Check free vram
                free_vram = torch.cuda.mem_get_info()[0]
                # If more than 2gb use large else medium
                if free_vram > 2 * 1024 * 1024 * 1024:  # 2 GB in bytes
                    whisper_model_key = 'large-v2'
                else:
                    whisper_model_key = 'medium'
            else:
                # Check core count
                core_count = psutil.cpu_count(logical=False)
                # If more than 8 use medium, if more than 4 use small, else tiny
                if core_count > 8:
                    whisper_model_key = 'medium'
                elif core_count > 4:
                    whisper_model_key = 'small'
                else:
                    whisper_model_key = 'tiny'
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            _whisper_model_key = whisper_model_key
            _whisper_model = whisper.load_model(
                whisper_model_key,
                device=_device
            )


def transcribe_audio(audio_path: str, whisper_model_key: str = '') -> Tuple[str, str]:
    """
    Transcribes an audio file using Whisper speech recognition.
    
    Args:
        audio_path: Path to the audio file to transcribe
        whisper_model_key: Whisper model to use (tiny, base, small, medium, large, large-v2)
                          If not specified, will use previously loaded model or choose based on system resources
                          
    Returns:
        Tuple[str, str]: Transcribed text and detected language
    """
    global _whisper_model, _whisper_model_key
    
    if _whisper_model is None:
        initialize_whisper_model(whisper_model_key)
    
    try:
        print(f"<{colored(f'Whisper - {_whisper_model_key}', 'green')}> is transcribing...")
        # Use CUDA if available but suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result: Dict[str, any] = _whisper_model.transcribe(audio_path)
        
        transcribed_text: str = result.get('text', '')
        detected_language: str = result.get('language', '')

        return transcribed_text, detected_language

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return "", ""


def initialize_kokoro_pipeline() -> bool:
    """
    Initialize the Kokoro TTS pipeline.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global _kokoro_pipeline
    
    if not KOKORO_AVAILABLE:
        print("Kokoro is not installed. Please install it with: pip install kokoro>=0.3.4")
        print("For Japanese support: pip install misaki[ja]")
        print("For Chinese support: pip install misaki[zh]")
        return False
    
    if _kokoro_pipeline is None:
        try:
            print(f"<{colored('Kokoro', 'green')}> initializing pipeline...")
            from kokoro import KPipeline
            _kokoro_pipeline = KPipeline(lang_code="b")
            return True
        except Exception as e:
            print(f"Error initializing Kokoro pipeline: {e}")
            return False
    
    return True


def text_to_speech(
    text: str,
    voice: str = 'af_heart',
    speed: float = 1.0,
    output_path: Optional[str] = None,
    play: bool = True,
    split_pattern: str = r'\n+'
) -> Union[List[str], None]:
    """
    Convert text to speech using Kokoro TTS model.

    Args:
        text: The text to convert to speech
        voice: Voice ID to use. Defaults to 'af_heart'.
        speed: Speech speed multiplier. Defaults to 1.0.
        output_path: Base path for output files. Will append index if multiple segments.
        play: Whether to play the audio. Defaults to True.
        split_pattern: Regex pattern for splitting text into segments.

    Returns:
        Union[List[str], None]: List of generated audio file paths if output_path is provided, None otherwise
    """
    global _kokoro_pipeline
    
    if not initialize_kokoro_pipeline():
        return None
    
    try:
        if not 0.1 <= speed <= 3.0:
            raise ValueError("Speed must be between 0.1 and 3.0")

        print(f"<{colored('Kokoro', 'green')}> is synthesizing speech...")
        
        output_files = []
        generator = _kokoro_pipeline(
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
                play_audio(audio_np, 24000)

        return output_files if output_path else None

    except Exception as e:
        print(f"An error occurred during Kokoro text-to-speech conversion: {e}")
        return None 