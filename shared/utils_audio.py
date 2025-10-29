# -*- coding: utf-8 -*-
"""
Audio processing utilities for voice interaction pipeline.

Provides wake word detection, audio recording, speech-to-text transcription,
and text-to-speech synthesis using local models with lazy loading for efficiency.
"""

import json
import logging
import queue
import struct
import subprocess
import time
import traceback
from typing import Any, Dict, Optional, Tuple, Union
import os
import tempfile
from datetime import datetime

import numpy as np
import psutil
import sounddevice as sd
import soundfile as sf
from termcolor import colored
from vosk import KaldiRecognizer, Model, SetLogLevel

# --- Centralized Logging Control ---
# Set to True to see detailed step-by-step logs for debugging.
# Set to False for a cleaner, user-focused output.
ENABLE_EXTENDED_LOGGING = False

# --- Global Resources ---
# We define our models globally so they are loaded only once.
_whisper_model: Optional[Any] = None
_whisper_model_key: Optional[str] = None
_faster_whisper_model: Optional[Any] = None
_faster_whisper_model_key: Optional[str] = None
_vosk_model: Optional[Model] = None
_tts_pipeline: Optional[Any] = None
_tts_model_id: Optional[str] = None
_device: Optional[str] = None

# Default parameters for robust transcription
DEFAULT_TRANSCRIPTION_PARAMS = {
    "temperature": [0.0, 0.2, 0.4],
    "no_speech_threshold": 0.01,
    "log_prob_threshold": -3.0,
    "compression_ratio_threshold": 10.0,
    "condition_on_previous_text": False,
    "beam_size": 5,
}

logger = logging.getLogger(__name__)

# audio.py
import fasttext
from pathlib import Path
import urllib.request

_FASTTEXT_MODEL_PATH = Path.home() / ".cache" / "piper-tts" / "lid.176.bin"
_FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
_lang_model = None


def _load_lang_model():
    """
    Downloads the fastText model if not present and loads it into memory.
    This is a robust, self-managing function.
    """
    global _lang_model
    if _lang_model:
        return _lang_model

    try:
        if not _FASTTEXT_MODEL_PATH.exists():
            print(f"<{colored('LangDetect', 'cyan')}> First-time setup: Downloading fastText model...")
            _FASTTEXT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_FASTTEXT_MODEL_URL, _FASTTEXT_MODEL_PATH)
            print(f"<{colored('LangDetect', 'green')}> Model downloaded to {_FASTTEXT_MODEL_PATH}")

        # Suppress the "Warning : `load_model` does not return WordVectorModel..." message
        fasttext.FastText.eprint = lambda x: None
        _lang_model = fasttext.load_model(str(_FASTTEXT_MODEL_PATH))
        return _lang_model
    except Exception as e:
        logger.error(f"Failed to load or download the language detection model: {e}")
        return None


def detect_language(text: str) -> str:
    """
    Robustly detects the language of a text using Meta's fastText model.

    Identifies German ('de') and defaults to English ('en') for all other
    languages or in case of an error, making it a safe replacement.

    Args:
        text: The input string to analyze.

    Returns:
        'de' for German, or 'en' for English/other.
    """
    model = _load_lang_model()
    if not model or not text or text.isspace():
        return 'en'  # Default to English if model fails or text is empty

    cleaned_text = text.replace('\n', ' ').strip()
    if not cleaned_text:
        return 'en'

    try:
        predictions = model.predict(cleaned_text, k=10)
        lang_code = predictions[0][0].replace('__label__', '')

        # We are only concerned with German vs. English for Piper/Coqui
        return 'de' if lang_code == 'de' else 'en'
    except Exception:
        return 'en' # Fallback on any error

def get_torch_device(force_reset: bool = False) -> str:
    """Determine the most appropriate device (CUDA or CPU) for PyTorch operations."""
    global _device

    if force_reset:
        _device = None

    if _device is None:
        env_device = os.getenv('CLIAGENT_FORCE_DEVICE')
        if env_device:
            _device = env_device.lower()
            if ENABLE_EXTENDED_LOGGING:
                print(f"<{colored('System', 'cyan')}> Using environment device override: {_device}")
            return _device

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                if ENABLE_EXTENDED_LOGGING:
                    print(f"<{colored('System', 'blue')}> Detected GPU: {gpu_name}")

                if "1080" in gpu_name:
                    _device = "cpu"
                    if ENABLE_EXTENDED_LOGGING:
                        print(f"<{colored('System', 'yellow')}> GTX 1080 Ti detected: Using CPU for stability")
                else:
                    _device = "cuda"
                    if ENABLE_EXTENDED_LOGGING:
                        print(f"<{colored('System', 'green')}> Using CUDA: {gpu_name}")
            else:
                _device = "cpu"
                if ENABLE_EXTENDED_LOGGING:
                    print(f"<{colored('System', 'blue')}> CUDA not available, using CPU")
        except ImportError:
            _device = "cpu"
            if ENABLE_EXTENDED_LOGGING:
                print(f"<{colored('System', 'yellow')}> PyTorch not available, using CPU")

    return _device


def initialize_faster_whisper_model(model_size: str = "small", compute_type: str = None, force_reset: bool = False) -> bool:
    """Lazily initialize the Faster-Whisper model, caching it for subsequent uses."""
    global _faster_whisper_model, _faster_whisper_model_key

    if force_reset:
        _faster_whisper_model, _faster_whisper_model_key = None, None
        if ENABLE_EXTENDED_LOGGING:
            print(f"<{colored('Faster-Whisper', 'cyan')}> Model cache reset requested")

    try:
        from faster_whisper import WhisperModel
        if ENABLE_EXTENDED_LOGGING:
            print(f"<{colored('Faster-Whisper', 'green')}> Faster-Whisper library available")
    except ImportError as e:
        print(f"<{colored('Error', 'red')}> Faster-Whisper not installed: {e}")
        print(f"<{colored('Info', 'yellow')}> Install with: pip install faster-whisper")
        return False

    device = get_torch_device(force_reset=force_reset)
    if compute_type is None:
        compute_type = "int8" if device == "cpu" else "float32"

    model_key = f"{model_size}_{compute_type}_{device}"

    if _faster_whisper_model and _faster_whisper_model_key == model_key and not force_reset:
        if ENABLE_EXTENDED_LOGGING:
            print(f"<{colored('Faster-Whisper', 'green')}> Model already loaded: {model_key}")
        return True

    print(f"<{colored('Faster-Whisper', 'magenta')}> Loading '{model_size}' model...")

    try:
        _faster_whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _faster_whisper_model_key = model_key
        print(f"<{colored('Faster-Whisper', 'green')}> Model loaded successfully! ({model_key})")
        return True

    except Exception as e:
        print(f"<{colored('Faster-Whisper', 'red')}> Failed to load model '{model_key}': {e}")
        if ENABLE_EXTENDED_LOGGING:
            print(f"<{colored('Debug', 'yellow')}> Device: {device}, Compute Type: {compute_type}")

        if compute_type != "int8":
            if ENABLE_EXTENDED_LOGGING:
                print(f"<{colored('Fallback', 'cyan')}> Trying fallback with int8 compute type...")
            try:
                _faster_whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
                _faster_whisper_model_key = f"{model_size}_int8_cpu_fallback"
                print(f"<{colored('Faster-Whisper', 'green')}> Fallback model loaded successfully!")
                return True
            except Exception as fallback_e:
                print(f"<{colored('Faster-Whisper', 'red')}> Fallback also failed: {fallback_e}")

        _faster_whisper_model = None
        return False


def _try_google_speech_recognition(audio_array: np.ndarray, sample_rate: int, language_code: str = "en-US") -> str:
    """Attempt speech recognition using Google's cloud service as a fallback."""
    try:
        import speech_recognition as sr
        boosted_audio = np.clip(audio_array * 10.0, -1.0, 1.0)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, boosted_audio, sample_rate)
            r = sr.Recognizer()
            with sr.AudioFile(tmp_file.name) as source:
                audio_data = r.record(source)
            try:
                return r.recognize_google(audio_data, language=language_code)
            except (sr.UnknownValueError, sr.RequestError):
                return ""
    except ImportError:
        if ENABLE_EXTENDED_LOGGING:
            print(f"<{colored('Fallback', 'yellow')}> speech_recognition library not available")
        return ""
    except Exception as e:
        print(f"<{colored('Fallback', 'red')}> Google Speech Recognition failed: {e}")
        return ""


def _transcribe_with_faster_whisper(
    audio_array: np.ndarray,
    sample_rate: int = 16000,
    language: Optional[str] = None,
    fallback_to_google: bool = True
) -> Tuple[str, str]:
    """Transcribe audio with Faster-Whisper, with optional Google fallback."""
    if ENABLE_EXTENDED_LOGGING:
        print(f"<{colored('Faster-Whisper', 'cyan')}> Starting local transcription...")

    if not _faster_whisper_model:
        print(f"<{colored('Faster-Whisper', 'red')}> Local transcription unavailable - model not initialized")
        if fallback_to_google:
            if ENABLE_EXTENDED_LOGGING:
                print(f"<{colored('Fallback', 'cyan')}> Will use Google Speech Recognition instead...")
            google_result = _try_google_speech_recognition(audio_array, sample_rate)
            if google_result:
                return google_result, "unknown"
        return "Error: Faster-Whisper model not initialized.", "unknown"

    transcription_params = DEFAULT_TRANSCRIPTION_PARAMS.copy()
    if language:
        transcription_params["language"] = language
        if ENABLE_EXTENDED_LOGGING:
            print(f"<{colored('Faster-Whisper', 'magenta')}> Transcribing with specified language: {language}...")
    elif ENABLE_EXTENDED_LOGGING:
        print(f"<{colored('Faster-Whisper', 'magenta')}> Transcribing with auto-detected language...")

    if ENABLE_EXTENDED_LOGGING:
        save_debug_audio(audio_array, sample_rate, "faster_whisper_input")

    audio_float32 = audio_array.astype(np.float32)
    rms_original = np.sqrt(np.mean(audio_float32**2))

    if rms_original > 0 and rms_original < 0.05:
        boost_factor = min(0.15 / rms_original, 15.0)
        audio_float32 *= boost_factor
        if ENABLE_EXTENDED_LOGGING:
            print(f"<{colored('Audio-Boost', 'yellow')}> Quiet audio detected (RMS={rms_original:.4f}), boosting by {boost_factor:.1f}x")

    max_val = np.max(np.abs(audio_float32))
    if max_val > 1.0:
        audio_float32 = audio_float32 / max_val

    if ENABLE_EXTENDED_LOGGING:
        duration = len(audio_float32) / sample_rate
        rms = np.sqrt(np.mean(audio_float32**2))
        print(f"<{colored('Faster-Whisper', 'blue')}> Audio: {duration:.1f}s, RMS={rms:.3f}")
        print(f"<{colored('Faster-Whisper', 'blue')}> Parameters: {transcription_params}")

    try:
        segments, info = _faster_whisper_model.transcribe(
            audio_float32, vad_filter=True, **transcription_params
        )

        full_text = " ".join([seg.text.strip() for seg in segments]).strip()
        detected_language = getattr(info, 'language', 'unknown')

        if ENABLE_EXTENDED_LOGGING and hasattr(info, 'language_probability'):
            print(f"<{colored('Faster-Whisper', 'blue')}> Detected language: '{detected_language}' with probability {info.language_probability:.2f}")

        print(f"<{colored('Whisper', 'green')}> Transcription: '{full_text}'")

        if fallback_to_google and (not full_text or full_text in ["...", ". . .", "... ..."]):
            if ENABLE_EXTENDED_LOGGING:
                print(f"<{colored('Fallback', 'cyan')}> Trying Google Speech Recognition...")
            google_lang_map = {"de": "de-DE", "en": "en-US", "es": "es-ES", "fr": "fr-FR"}
            google_lang_code = google_lang_map.get(detected_language, "en-US")
            google_result = _try_google_speech_recognition(audio_array, sample_rate, language_code=google_lang_code)
            if google_result and len(google_result.strip()) > 3:
                print(f"<{colored('Fallback', 'green')}> Using Google result: '{google_result}'")
                return google_result, detected_language

        return full_text, detected_language

    except Exception as e:
        print(f"<{colored('Faster-Whisper', 'red')}> Transcription failed: {e}")
        if fallback_to_google:
            print(f"<{colored('Fallback', 'cyan')}> Trying Google Speech Recognition as last resort...")
            google_result = _try_google_speech_recognition(audio_array, sample_rate)
            if google_result:
                return google_result, "unknown"
        return "", "error"


def save_debug_audio(audio_array: np.ndarray, sample_rate: int = 16000, prefix: str = "debug_audio") -> str:
    """Save audio array to a temporary file for manual inspection."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, f"{prefix}_{timestamp}.wav")

    audio_float32 = audio_array.astype(np.float32)
    max_val = np.max(np.abs(audio_float32))
    if max_val > 1.0:
        audio_float32 /= max_val

    sf.write(filepath, audio_float32, sample_rate)
    if ENABLE_EXTENDED_LOGGING:
        print(f"<{colored('Debug', 'cyan')}> Audio saved to: {filepath}")
    return filepath


def initialize_wake_word() -> None:
    """Lazily initialize the Vosk wake word detector model."""
    global _vosk_model
    if not _vosk_model:
        if ENABLE_EXTENDED_LOGGING:
            print(f"<{colored('Vosk', 'green')}> Initializing wake word model...")
        SetLogLevel(-1)  # Suppress verbose Vosk logging
        _vosk_model = Model(lang="en-us")


def wait_for_wake_word() -> Optional[str]:
    """Listen continuously for a wake word using Vosk."""
    initialize_wake_word()
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=10)
    wake_words = ["computer", "nova", "system", "hey computer", "hey nova", "ok computer", "ok nova"]
    rec = KaldiRecognizer(_vosk_model, 16000, json.dumps(wake_words))

    def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status: logger.warning(f"Audio stream status: {status}")
        try: audio_queue.put_nowait(bytes(indata))
        except queue.Full: pass

    try:
        with sd.RawInputStream(samplerate=16000, channels=1, dtype='int16', blocksize=4000, callback=audio_callback):
            print(f"<{colored('Vosk', 'green')}> Listening for a wake word... ({', '.join(wake_words)})")
            while True:
                audio_data = audio_queue.get()
                if rec.AcceptWaveform(audio_data):
                    result_json = json.loads(rec.Result())
                    text = result_json.get("text", "").strip()
                    if text:
                        print(f"<{colored('Vosk', 'green')}> Detected wake word: '{text}'")
                        return text
    except Exception as e:
        logger.error(f"Error in wake word detection: {e}\n{traceback.format_exc()}")
        return None


def play_notification() -> None:
    """Play a short, pleasant two-tone sound to signal that recording has started."""
    sample_rate = 44100
    duration, freq1, freq2 = 0.4, 196.00, 293.66
    t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
    tone1 = 0.3 * np.sin(2 * np.pi * freq1 * t)
    tone2 = 0.2 * np.sin(2 * np.pi * freq2 * t)
    signal = tone1 + tone2
    fade_len = int(0.05 * sample_rate)
    fade_in, fade_out = np.linspace(0., 1., fade_len), np.linspace(1., 0., fade_len)
    signal[:fade_len] *= fade_in
    signal[-fade_len:] *= fade_out
    play_audio(signal, sample_rate, blocking=False)


def play_audio(audio_array: np.ndarray, sample_rate: int, blocking: bool = True, volume: float = 1.0) -> None:
    """Play a NumPy audio array through the default sound device."""
    try:
        audio_float32 = audio_array.astype(np.float32)
        max_val = np.max(np.abs(audio_float32))
        if max_val > 0:
            audio_float32 /= max_val

        # Apply volume control
        audio_float32 *= volume

        # Ensure mono audio is properly formatted
        if len(audio_float32.shape) == 1:
            # Mono audio - reshape to (samples, 1) for sounddevice
            audio_float32 = audio_float32.reshape(-1, 1)

        # Play with explicit parameters to avoid channel conflicts
        sd.play(audio_float32, samplerate=sample_rate, blocking=blocking)
        if blocking:
            # blocking=True already waits, no need for sd.wait()
            pass
    except Exception:
        # Since Piper interface is working perfectly, just silently skip problematic direct audio
        # The TTS engines will handle their own audio playback correctly
        pass


def listen_microphone(
    max_listening_duration: Optional[int] = 60,
    private_remote_wake_detection: bool = False,
    remote_voice_enabled: bool = False,
    language: Optional[str] = None
) -> Tuple[str, str, Union[bool, str]]:
    """High-level function to record audio and transcribe it using Faster-Whisper."""
    first_iteration = True
    while True:
        audio_data = record_audio(max_duration=float(max_listening_duration or 60), notify_user=first_iteration)
        first_iteration = False
        
        if audio_data is None:
            return "", "unknown", False

        if ENABLE_EXTENDED_LOGGING:
            print(f"<{colored('Debug', 'yellow')}> Original audio array shape: {audio_data.shape}")

        audio_data = audio_data.flatten()

        if ENABLE_EXTENDED_LOGGING:
            print(f"<{colored('Debug', 'yellow')}> Flattened audio array shape: {audio_data.shape}")

        initialize_faster_whisper_model("small")

        transcription, detected_language = _transcribe_with_faster_whisper(
            audio_data, 16000, language=language, fallback_to_google=remote_voice_enabled
        )

        transcription = transcription.strip()

        if (transcription == ""):
            continue
        else:
            break
    play_notification()
    return transcription, detected_language, False
            


def record_audio(
    sample_rate: int = 16000,
    threshold: float = None,
    silence_duration: float = None,
    max_duration: float = 20.0,
    notify_user: bool = True
) -> Optional[np.ndarray]:
    """Record audio from the microphone, stopping after a period of silence."""
    if threshold is None:
        threshold = float(os.environ.get('CLIAGENT_AUDIO_THRESHOLD', '0.02'))
    if silence_duration is None:
        silence_duration = float(os.environ.get('CLIAGENT_SILENCE_DURATION', '2.0'))

    print(f"<{colored('Recording', 'red')}> Listening...")
    if notify_user:
        play_notification()
        time.sleep(1)
    if ENABLE_EXTENDED_LOGGING:
        print(f" (threshold: {threshold:.3f}, silence: {silence_duration}s)")

    q: queue.Queue[np.ndarray] = queue.Queue()
    recording, silent_chunks = False, 0
    num_silent_chunks_to_stop = int(silence_duration * 10)

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any):
        if status: logger.warning(status)
        q.put(indata.copy())

    recorded_frames = []
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=int(sample_rate / 10), callback=callback):
        start_time = time.time()
        while True:
            frame = q.get()
            is_loud = np.abs(frame).mean() > threshold

            if time.time() - start_time > max_duration:
                if ENABLE_EXTENDED_LOGGING: print("\n<Recording> Reached maximum recording time.")
                break

            if recording:
                recorded_frames.append(frame)
                silent_chunks = 0 if is_loud else silent_chunks + 1
                if silent_chunks > num_silent_chunks_to_stop:
                    if ENABLE_EXTENDED_LOGGING: print("\n<Recording> Detected end of speech.")
                    break
            elif is_loud:
                if ENABLE_EXTENDED_LOGGING: print("<Recording> Speech detected, starting...", end="", flush=True)
                recording = True
                recorded_frames.append(frame)

    if not recorded_frames:
        if ENABLE_EXTENDED_LOGGING: print("\n<Recording> No speech detected.")
        return None

    data = np.concatenate(recorded_frames, axis=0)
    return data


def initialize_whisper_model(model_key: str = None) -> None:
    """Lazily initialize the standard OpenAI Whisper model."""
    global _whisper_model, _whisper_model_key
    try:
        import whisper
    except ImportError:
        logger.error("Whisper library not found. Please install it with 'pip install openai-whisper'")
        return

    if model_key is None:
        model_key = get_optimal_whisper_model()

    if _whisper_model and _whisper_model_key == model_key:
        return

    available_mem = psutil.virtual_memory().available
    mem_reqs = {"large": 10e9, "medium": 5e9, "small": 2e9, "base": 1e9}
    if get_torch_device() == 'cpu' and available_mem < mem_reqs.get(model_key, 0):
        if ENABLE_EXTENDED_LOGGING:
            logger.warning(f"Not enough RAM for Whisper '{model_key}' on CPU. Falling back.")
        for key, req in sorted(mem_reqs.items(), key=lambda item: item[1], reverse=True):
            if available_mem > req:
                model_key = key
                break
        else:
            model_key = 'tiny'

    print(f"<{colored('Whisper', 'magenta')}> Loading '{model_key}' model...")
    _whisper_model = whisper.load_model(model_key, device=get_torch_device())
    _whisper_model_key = model_key
    print(f"<{colored('Whisper', 'magenta')}> Model loaded successfully.")


def get_optimal_whisper_model() -> str:
    """Select an optimal Whisper model size based on available GPU memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            if ENABLE_EXTENDED_LOGGING: print(f"<{colored('Whisper', 'yellow')}> No CUDA, using 'base' model")
            return 'base'

        torch.cuda.empty_cache()
        free_memory_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('System', 'blue')}> Free GPU Memory: {free_memory_gb:.1f}GB")

        model = 'large' if free_memory_gb > 4.0 else 'medium' if free_memory_gb > 2.0 else 'small' if free_memory_gb > 1.0 else 'base'
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('Whisper', 'green')}> Selected '{model}' model based on memory")
        return model
    except Exception as e:
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('Whisper', 'yellow')}> Could not detect GPU memory: {e}, using 'medium'")
        return 'medium'


def transcribe_audio(
    audio_array: np.ndarray,
    sample_rate: int = 16000,
    whisper_model_key: str = None,
    language: Optional[str] = None,
    use_faster_whisper: bool = True
) -> Tuple[str, str]:
    """High-level function to transcribe audio, preferring Faster-Whisper."""
    if use_faster_whisper:
        try:
            if ENABLE_EXTENDED_LOGGING: print(f"<{colored('System', 'green')}> Using Faster-Whisper for transcription")
            return _transcribe_with_faster_whisper(audio_array, sample_rate, language=language)
        except Exception as e:
            print(f"<{colored('System', 'yellow')}> Faster-Whisper failed: {e}. Falling back to OpenAI Whisper...")

    initialize_whisper_model(whisper_model_key)
    if not _whisper_model:
        return "Error: Whisper model not initialized.", "unknown"

    if ENABLE_EXTENDED_LOGGING:
        print(f"<{colored('Whisper', 'magenta')}> Transcribing audio with OpenAI Whisper...")
        save_debug_audio(audio_array, sample_rate, "whisper_input")

    audio_float32 = audio_array.astype(np.float32)
    max_val = np.max(np.abs(audio_float32))
    if max_val > 0:
        audio_float32 /= max_val

    text, detected_language = "", "unknown"
    try:
        result = _whisper_model.transcribe(
            audio_float32, fp16=(get_torch_device()=='cuda'),
            language=language, temperature=0.0
        )
        text = result.get("text", "").strip()
        detected_language = result.get("language", "unknown")
    except Exception as e:
        print(f"<{colored('Whisper', 'red')}> OpenAI Whisper transcription failed: {e}")

    print(f"<{colored('Whisper', 'magenta')}> Final transcription: '{text}'")
    return text, detected_language


def initialize_tts_pipeline(model_id: str) -> bool:
    """Lazily initialize a Hugging Face Text-to-Speech pipeline."""
    global _tts_pipeline, _tts_model_id
    if _tts_pipeline and _tts_model_id == model_id:
        return True

    try:
        from transformers import pipeline
        print(f"<{colored('TTS', 'cyan')}> Initializing HF TTS pipeline with model: {model_id}...")
        _tts_pipeline = pipeline("text-to-speech", model=model_id, device=get_torch_device())
        _tts_model_id = model_id
        print(f"<{colored('TTS', 'cyan')}> Pipeline ready!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize TTS pipeline '{model_id}': {e}\n{traceback.format_exc()}")
        _tts_pipeline, _tts_model_id = None, None
        return False


def text_to_speech(
    text: str, voice: str = "default", speed: float = 1.0, volume: float = 0.8,
    output_path: Optional[str] = None, play: bool = True, use_piper: bool = True,
    language: str = "auto", gender: str = "female", quality: str = "medium",
    **generation_kwargs: Any
) -> bool:
    """
    Convert text to speech using multiple TTS engines.

    Tries engines in order: Piper (local) > Coqui > HuggingFace > OpenAI > ElevenLabs > System.

    Args:
        text: Text to convert to speech
        voice: Voice model ("default", "en_GB-alba-medium", "alloy", etc.)
        speed: Speech speed multiplier (0.5-2.0)
        volume: Audio volume (0.0-1.0+)
        output_path: File path to save audio (None = don't save)
        play: Whether to play audio immediately
        use_piper: Include Piper TTS in engine priority
        language: Language code ("auto", "en", "de")
        gender: Voice gender preference ("female", "male")
        quality: Quality level ("low", "medium", "high")

    Returns:
        Success: bool
    """
    if ENABLE_EXTENDED_LOGGING:
        print(f"<{colored('TTS', 'cyan')}> Converting text to speech: '{text[:50]}...'")

    # Resolve language if set to 'auto' using our robust detector
    resolved_language = language
    if language == "auto":
        resolved_language = detect_language(text)
        if ENABLE_EXTENDED_LOGGING:
            print(f"<{colored('LangDetect', 'blue')}> Auto-detected language as: {resolved_language}")

    # Try engines in priority order
    try:
        from core.globals import g
        engine_priority = g.TTS_ENGINE_PRIORITY
    except:
        engine_priority = ["piper", "coqui", "huggingface", "system", "openai", "elevenlabs"]

    # 1. Try Piper TTS first (working, natural, offline)
    if use_piper:
        try:
            from core.providers.cls_piper_interface import get_piper_interface
            piper = get_piper_interface()
            if piper.is_available():
                if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS Piper', 'magenta')}> Using Piper TTS")
                result = piper.speak(
                    text=text, voice=voice if voice != "default" else "default", speed=speed,
                    output_path=output_path, play=play, language=resolved_language, gender=gender, quality=quality,
                    volume=volume
                )
                if result:
                    if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS Piper', 'magenta')}> Piper TTS synthesis successful")
                    return True
            elif ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS Piper', 'yellow')}> Piper TTS not available, falling back")
        except (ImportError, Exception) as e:
            if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS Piper', 'yellow')}> Piper TTS error: {e}, falling back")

    # 2. Try Coqui TTS (XTTS model for natural speech)
    # Pass the already resolved language to avoid re-detecting
    if _try_coqui_tts(text, voice, speed, output_path, play, language=resolved_language, volume=volume, **generation_kwargs):
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS Coqui', 'green')}> Coqui TTS synthesis successful")
        return True

    # 3. Try improved Hugging Face models
    if _try_modern_hf_tts(text, voice, speed, output_path, play, volume=volume, **generation_kwargs):
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS HF Modern', 'green')}> Modern HF TTS synthesis successful")
        return True

    # 4. Try remote APIs if allowed
    if _try_openai_tts(text, voice, speed, output_path, play, volume=volume, **generation_kwargs):
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS OpenAI', 'green')}> OpenAI TTS synthesis successful")
        return True

    if _try_elevenlabs_tts(text, voice, speed, output_path, play, volume=volume, **generation_kwargs):
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS ElevenLabs', 'green')}> ElevenLabs TTS synthesis successful")
        return True

    # 5. Last resort: System TTS
    if _try_improved_system_tts(text, volume=volume):
        return True

    print(f"<{colored('TTS Fallback', 'red')}> All TTS methods failed.")
    return False


def filter_latex_for_tts(text: str) -> str:
    r"""
    Remove LaTeX mathematical formulas and markdown formatting from text for TTS.

    Filters out:
    - Display math: $$...$$ and \[...\]
    - Inline math: $...$ and \(...\)
    - Equation environments: \begin{equation}...\end{equation}, etc.
    - Markdown code blocks: ```...```
    - Markdown tables
    - Markdown formatting: # * _ `

    Args:
        text: Input text potentially containing LaTeX and markdown

    Returns:
        Cleaned text suitable for TTS
    """
    import re

    # --- Remove Markdown Code Blocks ---
    # Remove fenced code blocks (```...```)
    text = re.sub(r'```[^`]*```', '[code block]', text, flags=re.DOTALL)

    # Remove inline code (`...`)
    text = re.sub(r'`[^`]+`', '', text)

    # --- Remove Markdown Tables ---
    # Tables are lines with pipes | and dashes ---
    # Remove table separator lines (|---|---|)
    text = re.sub(r'\|[\s\-:]+\|[\s\-:|]*', '', text, flags=re.MULTILINE)

    # Remove table rows (| cell | cell |)
    # Only if multiple pipes on same line
    text = re.sub(r'^.*?\|.*?\|.*?$', '', text, flags=re.MULTILINE)

    # --- Remove LaTeX Math ---
    # Display math with $$ delimiters
    text = re.sub(r'\$\$[^$]+\$\$', '[mathematical formula]', text, flags=re.DOTALL)

    # Display math with \[ \] delimiters
    text = re.sub(r'\\\[[^\]]+\\\]', '[mathematical formula]', text, flags=re.DOTALL)

    # Equation environments (equation, align, gather, multline, split)
    text = re.sub(r'\\begin\{(?:equation|align|gather|multline|split)\*?\}.*?\\end\{(?:equation|align|gather|multline|split)\*?\}',
                  '[mathematical formula]', text, flags=re.DOTALL)

    # Inline math with $ delimiters (careful with currency)
    # Only match if content contains letters or LaTeX commands (backslash)
    # This avoids matching pure numbers like $5 or $10
    text = re.sub(r'\$([^$\s]*[a-zA-Z\\^_{}][^$]*?)\$', '[math]', text)

    # Inline math with \( \) delimiters
    text = re.sub(r'\\\([^\)]+\\\)', '[math]', text)

    # --- Remove Markdown Formatting Characters ---
    # Remove headers (# ## ### etc.)
    text = re.sub(r'#+\s*', '', text)

    # Remove bold/italic markers (* ** ___)
    text = re.sub(r'[*_]{1,3}', '', text)

    # Remove remaining backticks
    text = re.sub(r'`', '', text)

    # Clean up multiple spaces and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    text = re.sub(r' {2,}', ' ', text)  # Max 1 space

    return text.strip()


def _try_coqui_tts(text: str, voice: str = "default", speed: float = 1.0,
                  output_path: Optional[str] = None, play: bool = True,
                  language: str = "auto", volume: float = 1.0, **kwargs) -> bool:
    """Try Coqui TTS XTTS model for natural multi-lingual speech."""
    try:
        from TTS.api import TTS
        import tempfile

        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS Coqui', 'magenta')}> Using Coqui XTTS")

        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        tts = TTS(model_name, progress_bar=False, gpu=get_torch_device() == "cuda")

        # Resolve language if it's still 'auto' (though it should be pre-resolved)
        resolved_language = language
        if language == "auto":
            resolved_language = detect_language(text)

        temp_output = None
        effective_output_path = output_path
        if not effective_output_path:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                effective_output_path = tmp_file.name
            temp_output = effective_output_path

        tts.tts_to_file(
            text=text,
            file_path=effective_output_path,
            language=resolved_language,
            speed=speed
        )

        if play:
            _play_audio_file_simple(effective_output_path, volume)

        if temp_output:
            try:
                os.unlink(temp_output)
            except:
                pass

        return True

    except ImportError:
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS Coqui', 'yellow')}> Coqui TTS library not available")
        return False
    except Exception as e:
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS Coqui', 'yellow')}> Coqui TTS failed: {e}")
        return False


def _try_openai_tts(text: str, voice: str = "default", speed: float = 1.0,
                   output_path: Optional[str] = None, play: bool = True, volume: float = 1.0, **kwargs) -> bool:
    """Try OpenAI TTS API for highest quality speech synthesis."""
    try:
        import openai
        import tempfile

        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS OpenAI', 'yellow')}> No API key found")
            return False

        client = openai.OpenAI(api_key=api_key)

        # Map voice names to OpenAI voices
        voice_map = {
            "default": "nova",
            "female": "nova",
            "male": "onyx",
            "alloy": "alloy",
            "echo": "echo",
            "fable": "fable",
            "nova": "nova",
            "onyx": "onyx",
            "shimmer": "shimmer"
        }

        openai_voice = voice_map.get(voice, "nova")

        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS OpenAI', 'magenta')}> Using OpenAI TTS with voice: {openai_voice}")

        # Create TTS request
        response = client.audio.speech.create(
            model="tts-1-hd",  # Use HD model for better quality
            voice=openai_voice,
            input=text,
            speed=max(0.25, min(4.0, speed))  # OpenAI accepts 0.25 to 4.0
        )

        # Handle output
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(response.content)

        if play:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name

            # Play the audio with volume control
            _play_audio_file_simple(tmp_path, volume)

            # Clean up if no permanent file
            if not output_path:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        return True

    except ImportError:
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS OpenAI', 'yellow')}> OpenAI library not available")
        return False
    except Exception as e:
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS OpenAI', 'yellow')}> OpenAI TTS failed: {e}")
        return False


def _try_elevenlabs_tts(text: str, voice: str = "default", speed: float = 1.0,
                       output_path: Optional[str] = None, play: bool = True, volume: float = 1.0, **kwargs) -> bool:
    """Try ElevenLabs TTS API for high-quality natural voices."""
    try:
        import requests
        import tempfile

        # Check for API key
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS ElevenLabs', 'yellow')}> No API key found")
            return False

        # Map voice names to ElevenLabs voice IDs (popular voices)
        voice_map = {
            "default": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "female": "21m00Tcm4TlvDq8ikWAM",   # Rachel
            "male": "29vD33N1CtxCmqQRPOHJ",      # Drew
            "rachel": "21m00Tcm4TlvDq8ikWAM",
            "drew": "29vD33N1CtxCmqQRPOHJ",
            "clyde": "2EiwWnXFnvU5JabPnv8n",
            "dave": "CYw3kZ02Hs0563khs1Fj"
        }

        voice_id = voice_map.get(voice, voice_map["default"])

        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS ElevenLabs', 'magenta')}> Using ElevenLabs TTS")

        # Create TTS request
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }

        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "speed": max(0.5, min(2.0, speed))  # ElevenLabs accepts 0.5 to 2.0
            }
        }

        response = requests.post(url, json=data, headers=headers, timeout=30)

        if response.status_code == 200:
            # Handle output
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(response.content)

            if play:
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name

                # Play the audio with volume control
                _play_audio_file_simple(tmp_path, volume)

                # Clean up if no permanent file
                if not output_path:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

            return True
        else:
            if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS ElevenLabs', 'yellow')}> API error: {response.status_code}")
            return False

    except ImportError:
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS ElevenLabs', 'yellow')}> Requests library not available")
        return False
    except Exception as e:
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS ElevenLabs', 'yellow')}> ElevenLabs TTS failed: {e}")
        return False


def _try_coqui_tts(text: str, voice: str = "default", speed: float = 1.0,
                  output_path: Optional[str] = None, play: bool = True,
                  language: str = "auto", volume: float = 1.0, **kwargs) -> bool:
    """Try Coqui TTS XTTS model for natural multi-lingual speech."""
    try:
        from TTS.api import TTS
        import tempfile

        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS Coqui', 'magenta')}> Using Coqui XTTS")

        # Initialize XTTS model (supports multiple languages and voice cloning)
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        tts = TTS(model_name, progress_bar=False, gpu=get_torch_device() == "cuda")

        # Auto-detect language
        if language == "auto":
            german_indicators = ["der", "die", "das", "und", "ist", "können", "wasserstoff"]
            text_lower = text.lower()
            language = "de" if any(word in text_lower for word in german_indicators) else "en"

        # Create temporary output if needed
        temp_output = None
        if not output_path:
            temp_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_output.name
            temp_output.close()

        # Generate speech
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            language=language,
            speed=speed
        )

        if play:
            _play_audio_file_simple(output_path, volume)

        # Clean up temporary file if created
        if temp_output and not play:
            try:
                os.unlink(output_path)
            except:
                pass

        return True

    except ImportError:
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS Coqui', 'yellow')}> Coqui TTS library not available")
        return False
    except Exception as e:
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS Coqui', 'yellow')}> Coqui TTS failed: {e}")
        return False


def _try_modern_hf_tts(text: str, voice: str = "default", speed: float = 1.0,
                      output_path: Optional[str] = None, play: bool = True, volume: float = 1.0, **kwargs) -> bool:
    """Try modern Hugging Face TTS models (better than speecht5)."""
    try:
        from transformers import pipeline
        import tempfile

        # Use better HF models
        modern_models = [
            "microsoft/speecht5_tts",  # Keep as fallback but use better voices
            "facebook/mms-tts-eng",    # Meta's multilingual model
            "espnet/kan-bayashi_ljspeech_vits",  # VITS model
        ]

        for model_name in modern_models:
            try:
                if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS HF Modern', 'magenta')}> Trying {model_name}")

                synthesizer = pipeline("text-to-speech", model=model_name, device=get_torch_device())

                # Generate speech
                output = synthesizer(text)
                audio_array, sample_rate = output["audio"], output["sampling_rate"]

                # Handle speed adjustment
                if speed != 1.0:
                    try:
                        import scipy.signal
                        new_length = int(len(audio_array) / speed)
                        audio_array = scipy.signal.resample(audio_array, new_length)
                    except ImportError:
                        pass  # Skip speed adjustment if scipy not available

                # Save to file if requested
                if output_path:
                    sf.write(output_path, audio_array, sample_rate)

                if play:
                    play_audio(audio_array, sample_rate, volume=volume)

                return True

            except Exception as model_error:
                if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS HF Modern', 'yellow')}> Model {model_name} failed: {model_error}")
                continue

        return False

    except ImportError:
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS HF Modern', 'yellow')}> Transformers library not available")
        return False
    except Exception as e:
        if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS HF Modern', 'yellow')}> Modern HF TTS failed: {e}")
        return False


def _play_audio_file_simple(audio_path: str, volume: float = 1.0) -> None:
    """Simple cross-platform audio file playback with volume control."""
    try:
        # Try sounddevice first (better volume control)
        try:
            import sounddevice as sd
            import soundfile as sf

            audio_data, sample_rate = sf.read(audio_path)
            # Apply volume control
            audio_data = audio_data * volume
            sd.play(audio_data, sample_rate)
            sd.wait()
            return
        except ImportError:
            pass

        # Fallback to system audio players (limited volume control)
        import platform
        import subprocess

        system = platform.system().lower()

        # Note: System players don't have easy volume control, but at least they work
        if system == "linux":
            # Try multiple Linux audio players
            players = ['aplay', 'paplay', 'sox', 'ffplay']
            for player in players:
                try:
                    subprocess.run([player, audio_path], check=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
        elif system == "darwin":  # macOS
            subprocess.run(['afplay', audio_path], check=True, timeout=30)
        elif system == "windows":
            subprocess.run(['start', audio_path], shell=True, check=True, timeout=30)
        else:
            logger.warning(f"No audio playback method for {system}")

    except Exception as e:
        logger.error(f"Failed to play audio file {audio_path}: {e}")


def _try_improved_system_tts(text: str, volume: float = 1.0) -> bool:
    """Try improved system TTS engines, avoiding robotic espeak when possible."""
    # Prioritize better system TTS engines over espeak
    tts_commands = [
        ['say', text],           # macOS (natural sounding)
        ['spd-say', text],       # Speech Dispatcher (Linux, often better than espeak)
        ['festival', '--tts'],   # Festival (better than espeak)
        ['flite', '-t', text],   # Flite (lighter, sometimes better than espeak)
        ['espeak-ng', text],     # espeak-ng (improved espeak)
        ['espeak', text]         # espeak (last resort - robotic)
    ]

    for cmd in tts_commands:
        try:
            process_args = {'check': True, 'timeout': 10, 'capture_output': True}
            if cmd[0] == 'festival':
                subprocess.run(cmd, input=text, text=True, **process_args)
            else:
                subprocess.run(cmd, **process_args)
            if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS System', 'green')}> Speech successful using {cmd[0]}")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue

    if ENABLE_EXTENDED_LOGGING: print(f"<{colored('TTS System', 'red')}> No working system TTS found")
    return False




def save_binary_file(file_name: str, data: bytes) -> str:
    """Save binary data to a file."""
    try:
        with open(file_name, "wb") as f:
            f.write(data)
        logger.info(f"File saved to: {file_name}")
        return file_name
    except Exception as e:
        logger.error(f"Error saving file {file_name}: {e}")
        raise


def parse_audio_mime_type(mime_type: str) -> Dict[str, int]:
    """Parse audio parameters from a MIME type string."""
    bits_per_sample, rate = 16, 24000
    for param in mime_type.split(";"):
        param = param.strip()
        if param.lower().startswith("rate="):
            try: rate = int(param.split("=", 1)[1])
            except (ValueError, IndexError): pass
        elif param.startswith("audio/L"):
            try: bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError): pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Prepend a WAV header to raw audio data."""
    params = parse_audio_mime_type(mime_type)
    bits_per_sample, sample_rate, num_channels = params["bits_per_sample"], params["rate"], 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size
    header = struct.pack("<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample, b"data", data_size)
    return header + audio_data