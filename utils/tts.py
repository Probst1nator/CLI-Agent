import json
import logging
import subprocess
import traceback
from typing import Any, Optional, Dict

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from termcolor import colored

from py_classes.cls_util_base import UtilBase

logger = logging.getLogger(__name__)

# Global resources to be loaded lazily
_tts_pipeline: Optional[Any] = None
_tts_model_id: Optional[str] = None
_device: Optional[str] = None


def get_torch_device() -> str:
    """
    Determines the optimal torch device and returns it as a string.
    Caches the result in a global variable.
    """
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"<{colored('System', 'blue')}> Determined torch device: {_device}")
    return _device


def play_audio(audio_array: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
    """
    Plays a NumPy array of audio data through the default sound device.
    """
    try:
        audio_array = audio_array.astype(np.float32)
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array /= max_val
        
        sd.play(audio_array, sample_rate)
        if blocking:
            sd.wait()
    except Exception as e:
        logger.error(f"Could not play audio: {e}")
        sd.stop()


def initialize_tts_pipeline(model_id: str) -> bool:
    """
    Initializes the Hugging Face TTS pipeline lazily.
    """
    global _tts_pipeline, _tts_model_id
    if _tts_pipeline and _tts_model_id == model_id:
        return True

    try:
        from transformers import pipeline
        print(f"<{colored('TTS', 'cyan')}> Initializing pipeline with model: {model_id}...")
        device = get_torch_device()
        _tts_pipeline = pipeline("text-to-speech", model=model_id, device=device)
        _tts_model_id = model_id
        print(f"<{colored('TTS', 'cyan')}> Pipeline ready!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize TTS pipeline '{model_id}': {e}\n{traceback.format_exc()}")
        _tts_pipeline = None
        _tts_model_id = None
        return False


def get_speaker_embeddings(model_id: str):
    """
    Gets speaker embeddings for models that require them (like SpeechT5).
    """
    if "speecht5" in model_id.lower():
        try:
            from datasets import load_dataset
            print(f"<{colored('TTS', 'cyan')}> Loading speaker embeddings...")
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
            return speaker_embeddings
        except Exception as e:
            logger.warning(f"Failed to load speaker embeddings: {e}")
            return None
    return None


class TtsUtil(UtilBase):
    """
    A utility for converting text to speech using various models.
    This tool provides a voice for the agent, allowing it to speak its responses.
    """

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["speak", "say", "read aloud", "text to speech", "voice", "audio", "narrate"],
            "use_cases": [
                "Read the summary of the search results out loud.",
                "Say 'Hello, how can I help you today?'.",
                "Convert the article text to an audio file."
            ],
            "arguments": {
                "text": "The text to be converted into speech.",
                "model_id": "The Hugging Face model ID for the TTS engine.",
                "output_path": "Optional file path to save the generated audio as a .wav file.",
                "play": "Whether to play the audio through the system's speakers."
            }
        }
    
    @staticmethod
    def _run_logic(
        text: str,
        model_id: str = "microsoft/speecht5_tts",
        output_path: Optional[str] = None,
        play: bool = True,
        **generation_kwargs: Any
    ) -> str:
        """
        Converts text to speech using a Hugging Face model or a system fallback.

        Args:
            text (str): The text to be spoken.
            model_id (str): The Hugging Face model ID for TTS.
            output_path (Optional[str]): If provided, saves the audio to this file path.
            play (bool): If True, plays the audio directly.
            **generation_kwargs: Additional arguments for the TTS model.

        Returns:
            str: A JSON string indicating success or failure.
        """
        if not text or not text.strip():
            return json.dumps({"status": "error", "message": "No text provided for speech synthesis."})

        print(f"<{colored('TTS', 'cyan')}> Converting text to speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")

        # Try Hugging Face TTS first
        if not initialize_tts_pipeline(model_id):
            print(f"<{colored('TTS Fallback', 'yellow')}> Hugging Face model failed. Trying system TTS...")
            try:
                subprocess.run(['espeak', text], check=True, capture_output=True)
                return json.dumps({"status": "success", "method": "system_tts", "message": "Speech synthesized using system TTS."})
            except (FileNotFoundError, subprocess.CalledProcessError):
                logger.error("No system TTS found. Cannot speak.")
                return json.dumps({"status": "error", "message": "All TTS options failed."})

        print(f"<{colored('TTS', 'cyan')}> Synthesizing speech...")
        try:
            # Check if this model needs speaker embeddings
            forward_params = generation_kwargs.copy()
            if "speecht5" in model_id.lower():
                speaker_embeddings = get_speaker_embeddings(model_id)
                if speaker_embeddings is not None:
                    forward_params["speaker_embeddings"] = speaker_embeddings
                else:
                    raise Exception("SpeechT5 model requires speaker embeddings but none could be loaded")
            
            output = _tts_pipeline(text, forward_params=forward_params)
            audio_array = output["audio"]
            sample_rate = output["sampling_rate"]

            result_data = {
                "status": "success",
                "method": "huggingface_tts",
                "model_id": model_id,
                "sample_rate": sample_rate,
                "message": "Speech synthesized successfully."
            }

            if play:
                play_audio(audio_array, sample_rate)
                result_data["played"] = True

            if output_path:
                if not output_path.lower().endswith('.wav'):
                    output_path += ".wav"
                sf.write(output_path, audio_array, sample_rate)
                logger.info(f"Audio saved to: {output_path}")
                result_data["output_path"] = output_path
                result_data["saved"] = True
            
            return json.dumps(result_data)

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}\n{traceback.format_exc()}")
            return json.dumps({"status": "error", "message": f"TTS synthesis failed: {e}"})


# Module-level run function for CLI-Agent compatibility
def run(text: str, model_id: str = "microsoft/speecht5_tts", output_path: Optional[str] = None, speaker_id: Optional[int] = None, temperature: float = 1.0, use_cache: bool = True) -> str:
    """
    Module-level wrapper for TTSUtil._run_logic() to maintain compatibility with CLI-Agent.
    
    Args:
        text (str): The text to convert to speech
        model_id (str): The TTS model to use
        output_path (Optional[str]): Path to save the audio file
        speaker_id (Optional[int]): Speaker ID for multi-speaker models
        temperature (float): Temperature for synthesis
        use_cache (bool): Whether to use cached results
        
    Returns:
        str: JSON string with result or error
    """
    return TTSUtil._run_logic(text=text, model_id=model_id, output_path=output_path, speaker_id=speaker_id, temperature=temperature, use_cache=use_cache) 