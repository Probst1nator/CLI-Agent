"""
Piper TTS Provider Interface

High-quality, multi-lingual text-to-speech using Piper TTS.
Supports German, English, and many other languages with offline processing.
"""

import os
import tempfile
import logging
from typing import Optional, List, Union, Tuple, Dict, Any
from pathlib import Path
import urllib.request
import numpy as np
from piper import PiperVoice, SynthesisConfig
import speech_recognition as sr

from py_classes.unified_interfaces import AudioProviderInterface

logger = logging.getLogger(__name__)

class PiperInterface(AudioProviderInterface):
    """
    Piper TTS provider interface for high-quality multi-lingual speech synthesis.

    Supports:
    - German voices: thorsten-high, thorsten-medium, mls-medium
    - English voices: amy-medium, lessac-medium
    - Offline processing with auto-download of voice models
    """

    # Voice model definitions - updated with correct Hugging Face URLs
    VOICES = {
        # German voices
        "de_DE-thorsten-high": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json",
            "language": "de",
            "gender": "male",
            "quality": "high"
        },
        "de_DE-thorsten-medium": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json",
            "language": "de",
            "gender": "male",
            "quality": "medium"
        },
        "de_DE-mls-medium-f2": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/mls/medium/de_DE-mls-medium.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/mls/medium/de_DE-mls-medium.onnx.json",
            "language": "de",
            "gender": "female",
            "quality": "medium",
            "voice_id": 14
        },
        "de_DE-mls-medium-f1": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/mls/medium/de_DE-mls-medium.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/mls/medium/de_DE-mls-medium.onnx.json",
            "language": "de",
            "gender": "female",
            "quality": "high",
            "voice_id": 14
        },
        "de_DE-mls-medium-m1": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/mls/medium/de_DE-mls-medium.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/mls/medium/de_DE-mls-medium.onnx.json",
            "language": "de",
            "gender": "male",
            "quality": "medium",
            "voice_id": 150
        },
        # English voices
        "en_GB-alba-medium": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alba/medium/en_GB-alba-medium.onnx.json",
            "language": "en",
            "gender": "female",
            "quality": "medium"
        },
        "en_GB-alan-medium": {
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
            "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
            "language": "en",
            "gender": "male",
            "quality": "medium"
        }
    }

    def __init__(self):
        self._voices_cache = {}  # Cache loaded voices
        self._voices_dir = Path.home() / ".cache" / "piper-tts" / "voices"
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        self._available = True

        # Check if piper is available
        try:
            from piper import PiperVoice
            logger.debug("Piper TTS is available")
        except ImportError as e:
            logger.error(f"Piper TTS not available: {e}")
            self._available = False

    def is_available(self) -> bool:
        """Check if Piper TTS is available for use."""
        return self._available

    def _download_voice(self, voice_name: str) -> bool:
        """Download voice model if not present."""
        if voice_name not in self.VOICES:
            logger.error(f"Unknown voice: {voice_name}")
            return False

        voice_info = self.VOICES[voice_name]
        model_path = self._voices_dir / f"{voice_name}.onnx"
        config_path = self._voices_dir / f"{voice_name}.onnx.json"

        # Check if files already exist
        if model_path.exists() and config_path.exists():
            return True

        logger.info(f"Downloading voice model: {voice_name}")

        try:
            # Download model file
            if not model_path.exists():
                logger.info(f"Downloading {voice_info['url']}")
                urllib.request.urlretrieve(voice_info["url"], model_path)

            # Download config file
            if not config_path.exists():
                logger.info(f"Downloading {voice_info['config_url']}")
                urllib.request.urlretrieve(voice_info["config_url"], config_path)

            logger.info(f"Voice {voice_name} downloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to download voice {voice_name}: {e}")
            # Clean up partial downloads
            if model_path.exists():
                model_path.unlink()
            if config_path.exists():
                config_path.unlink()
            return False

    def _load_voice(self, voice_name: str) -> Optional[PiperVoice]:
        """Load a Piper voice model."""
        if voice_name in self._voices_cache:
            return self._voices_cache[voice_name]

        # Download if needed
        if not self._download_voice(voice_name):
            return None

        try:
            from piper import PiperVoice

            model_path = self._voices_dir / f"{voice_name}.onnx"
            config_path = self._voices_dir / f"{voice_name}.onnx.json"

            # Load the voice
            voice = PiperVoice.load(str(model_path), str(config_path))
            self._voices_cache[voice_name] = voice

            logger.debug(f"Loaded voice: {voice_name}")
            return voice

        except Exception as e:
            logger.error(f"Failed to load voice {voice_name}: {e}")
            return None

    def _select_voice(self, language: str = "en", gender: str = "female", quality: str = "medium", speaker_id: Union[None, str] = "") -> str:
        """Select appropriate voice based on parameters."""
        # Default voice mappings
        defaults = {
            ("de", "male", "medium"): "de_DE-mls-medium-m1",
            ("de", "male", "high"): "de_DE-thorsten-high",
            ("de", "female", "medium"): "de_DE-mls-medium-f1",
            ("en", "female", "medium"): "en_GB-alba-medium",
            ("en", "male", "medium"): "en_GB-alan-medium"
        }

        # Try exact match first
        key = (language, gender, quality)
        if key in defaults:
            return defaults[key]

        # Fallback logic
        if language == "de":
            return "de_DE-thorsten-medium"  # Good quality German default
        else:
            return "en_GB-alba-medium"  # Good quality English default

    def speak(self,
              text: str,
              speaker_name: str = "default",
              voice: str = "default",
              speed: float = 1.0,
              volume: float = 1.0,
              output_path: Optional[str] = None,
              play: bool = True,
              language: str = "auto",
              gender: str = "female",
              quality: str = "medium") -> bool:
        """
        Convert text to speech using Piper TTS.

        Args:
            text: Text to convert to speech
            speaker_name: Ignored (for compatibility)
            voice: Specific voice name or "default"
            speed: Speech speed (not directly supported by Piper)
            output_path: Optional path to save audio file
            play: Whether to play the audio
            language: Language code ("de", "en", "auto")
            gender: Voice gender ("male", "female")
            quality: Voice quality ("low", "medium", "high")

        Returns:
            List of generated audio file paths if output_path provided
        """
        if not self._available:
            logger.error("Piper TTS not available")
            return False

        # Text preprocessing
        text = text.replace("*", "")

        # if (len(text)>1000):
        #     speed *= 1.2

        # # Auto-detect language if needed
        # if language == "auto":
        #     # Simple language detection based on common German words
        #     german_indicators = ["der", "die", "das", "und", "ist", "mit", "zu", "auf", "für", "von",
        #                        "wasserstoff", "elektronenkonfiguration", "thermodynamik", "können", "erklären"]
        #     text_lower = text.lower()
        #     german_score = sum(1 for word in german_indicators if word in text_lower)
        #     language = "de" if german_score > 0 else "en"

        # Select voice
        if voice == "default":
            voice_name = self._select_voice(language, gender, quality)
        else:
            voice_name = voice

        # Load voice
        piper_voice = self._load_voice(voice_name)
        if not piper_voice:
            logger.error(f"Failed to load voice: {voice_name}")
            return False

            # Piper TTS synthesis configuration parameters  
        
        length_scale: float = 1  
        """Speech rate multiplier (0.75=faster, 1.0=normal, 1.25=slower)"""  
        
        noise_scale: float = 0.2
        """Overall prosodic variability (0.333=consistent, 0.667-1.4=more expressive)"""  
        
        noise_w_scale: float = 0.1 
        """Stochastic duration of phonemes/words (controls timing variability)"""  
        
        normalize_audio: bool = True  
        """Whether to normalize output audio levels for consistent volume"""  
        
        volume: float = 1.0  
        """Output volume multiplier (1.0=normal, >1.0=louder, <1.0=quieter)"""

        if "voice_id" in self.VOICES[voice_name]:  
            speaker_id = self.VOICES[voice_name]["voice_id"]  
            synthConf = SynthesisConfig(speaker_id, length_scale, noise_scale, noise_w_scale, normalize_audio, volume)
        else:
            synthConf = SynthesisConfig()

        try:
            logger.info(f"Generating speech with {voice_name}: '{text[:50]}...'")
            audio_chunks = list(piper_voice.synthesize(text, synthConf))  

            # Extract audio data from chunks
            if not audio_chunks:
                logger.error("No audio chunks generated")
                return False

            # Combine all audio chunks
            audio_arrays = []
            for chunk in audio_chunks:
                audio_arrays.append(chunk.audio_float_array)

            # Concatenate all chunks
            audio_data = np.concatenate(audio_arrays) if audio_arrays else np.array([])

            # Handle speed adjustment (simple resampling approximation)
            if speed != 1.0 and speed > 0:
                try:
                    import scipy.signal
                    original_length = len(audio_data)
                    new_length = int(original_length / speed)
                    audio_data = scipy.signal.resample(audio_data, new_length)
                    # Keep as float for now - conversion to int16 handled later if needed
                except ImportError:
                    logger.warning("scipy not available for speed adjustment")

            # Ensure audio data is in proper float format [-1.0, 1.0]
            if audio_data.dtype != np.float32:
                # If it's int16, convert back to float
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32767.0
                else:
                    audio_data = audio_data.astype(np.float32)

            # Validate audio content (detect silent audio)
            rms_level = np.sqrt(np.mean(audio_data**2))
            non_zero_ratio = np.count_nonzero(audio_data) / len(audio_data) if len(audio_data) > 0 else 0

            if rms_level < 1e-6 or non_zero_ratio < 0.01:
                logger.warning(f"Generated audio appears to be silent (RMS: {rms_level:.2e}, non-zero: {non_zero_ratio:.1%})")
            else:
                logger.info(f"Audio generated successfully (RMS: {rms_level:.3f}, non-zero: {non_zero_ratio:.1%})")

            # Save to file if requested
            saved_files = []
            temp_path = None

            if output_path or play:
                import soundfile as sf

                if output_path:
                    if not output_path.lower().endswith('.wav'):
                        output_path += ".wav"
                    # Convert float audio [-1.0, 1.0] to int16 PCM [-32768, 32767]
                    audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
                    sf.write(output_path, audio_int16, 22050)
                    saved_files.append(output_path)
                    logger.info(f"Piper audio saved to: {output_path}")

                if play:
                    # Create temporary file for playback
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        temp_path = tmp_file.name
                        # Convert float audio to int16 for playback
                        audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
                        sf.write(temp_path, audio_int16, 22050)

                    # Play audio with volume control
                    self._play_audio_file(temp_path, volume)

                    # Clean up if no permanent file was saved
                    if not output_path:
                        try:
                            os.unlink(temp_path)
                        except:
                            pass
            print("finished playback")
            return True

        except Exception as e:
            logger.error(f"Piper speech generation failed: {e}")
            return False

    def _play_audio_file(self, audio_path: str, volume: float = 1.0) -> None:
        """Play audio file using available system methods."""
        try:
            # Try sounddevice first (if available)
            try:
                import sounddevice as sd
                import soundfile as sf

                audio_data, sample_rate = sf.read(audio_path)
                # Apply volume control
                audio_data = audio_data * volume

                # Ensure proper audio format for sounddevice
                if len(audio_data.shape) == 1:
                    # Mono audio - reshape to (samples, 1)
                    audio_data = audio_data.reshape(-1, 1)

                # Play with explicit parameters
                sd.play(audio_data, samplerate=sample_rate, blocking=True)
                return
            except ImportError:
                pass
            except Exception as sd_error:
                # If sounddevice fails (ALSA issues, etc.), fall back to system players
                logger.warning(f"Sounddevice playback failed: {sd_error}")
                pass

            # Fallback to system commands
            import subprocess
            import platform

            system = platform.system().lower()

            if system == "linux":
                subprocess.run(['aplay', audio_path], check=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif system == "darwin":  # macOS
                subprocess.run(['afplay', audio_path], check=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif system == "windows":
                subprocess.run(['start', audio_path], shell=True, check=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                logger.warning(f"No audio playback method available for {system}")

        except Exception as e:
            logger.error(f"Failed to play audio: {e}")

    def get_available_voices(self) -> List[str]:
        """Get list of available voices."""
        return list(self.VOICES.keys())

    def get_voice_info(self, voice_name: str) -> Dict[str, Any]:
        """Get information about a specific voice."""
        if voice_name in self.VOICES:
            return self.VOICES[voice_name].copy()
        return {}

    def get_model_info(self) -> dict:
        """Get information about the Piper TTS setup."""
        return {
            "provider": "Piper TTS",
            "version": "1.3.0",
            "voices_available": len(self.VOICES),
            "voices_cached": len(self._voices_cache),
            "voices_dir": str(self._voices_dir),
            "available": self._available,
            "supports_languages": ["de", "en"],
            "german_voices": [v for v in self.VOICES.keys() if v.startswith("de_")],
            "english_voices": [v for v in self.VOICES.keys() if v.startswith("en_")]
        }

    # Abstract method implementations for AudioProviderInterface

    def transcribe_audio(self, audio_data: Union[sr.AudioData, np.ndarray],
                        language: str = "",
                        model: str = "",
                        sample_rate: int = 44100) -> Tuple[str, str]:
        """
        Piper is a TTS model, not STT.
        This method is not implemented - use WhisperInterface instead.
        """
        logger.warning("Piper TTS does not support speech-to-text. Use WhisperInterface instead.")
        return "", "unknown"

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
        Piper is a TTS model, not STT.
        This method is not implemented - use WhisperInterface instead.
        """
        logger.warning("Piper TTS does not support speech-to-text. Use WhisperInterface instead.")
        return "", "unknown", np.array([])

    def play_notification_sound(self) -> None:
        """Play a simple notification sound."""
        try:
            # Generate a simple notification tone
            sample_rate = 22050
            duration = 0.3
            t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)

            # Two-tone notification (G3 and D4)
            freq1, freq2 = 196.00, 293.66
            tone1 = 0.3 * np.sin(2 * np.pi * freq1 * t)
            tone2 = 0.2 * np.sin(2 * np.pi * freq2 * t)
            signal = tone1 + tone2

            # Fade in/out
            fade_len = int(0.05 * sample_rate)
            fade_in = np.linspace(0., 1., fade_len)
            fade_out = np.linspace(1., 0., fade_len)
            signal[:fade_len] *= fade_in
            signal[-fade_len:] *= fade_out

            self.play_audio_data(signal, sample_rate, blocking=False)
        except Exception as e:
            logger.warning(f"Could not play notification sound: {e}")

    def play_audio_data(self, audio_array: np.ndarray, sample_rate: int = 22050, blocking: bool = True) -> None:
        """
        Play audio data through the default sound device.

        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate in Hz
            blocking: Whether to wait for playback to complete
        """
        try:
            # Try sounddevice first (if available)
            try:
                import sounddevice as sd

                audio_array = audio_array.astype(np.float32)
                max_val = np.max(np.abs(audio_array))
                if max_val > 0:
                    audio_array /= max_val

                # Ensure proper audio format for sounddevice
                if len(audio_array.shape) == 1:
                    # Mono audio - reshape to (samples, 1)
                    audio_array = audio_array.reshape(-1, 1)

                # Play with explicit parameters
                sd.play(audio_array, samplerate=sample_rate, blocking=blocking)
                return
            except ImportError:
                pass

            # Fallback to file-based playback
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
                import soundfile as sf
                sf.write(temp_path, audio_array, sample_rate)

            self._play_audio_file(temp_path)

            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass

        except Exception as e:
            logger.error(f"Could not play audio: {e}")


# Global instance for reuse
_piper_instance = None

def get_piper_interface() -> PiperInterface:
    """Get or create global Piper interface instance."""
    global _piper_instance
    if _piper_instance is None:
        _piper_instance = PiperInterface()
    return _piper_instance