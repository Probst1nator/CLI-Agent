import tempfile
import os
import threading
import logging
from typing import Any, Optional, Tuple
from openai import OpenAI
from termcolor import colored
from core.chat import Chat
from py_classes.unified_interfaces import AIProviderInterface
import speech_recognition as sr
from core.globals import g

logger = logging.getLogger(__name__)

class OpenAIAPI(AIProviderInterface):
    """
    Implementation of the AIProviderInterface for the OpenAI API.
    """

    # Class-level cancellation support
    _cancellation_event = None
    _current_task = None
    _cancellation_lock = threading.Lock()

    @classmethod
    def cancel_current_generation(cls):
        """
        Cancel any currently running OpenAI generation.
        This sets a cancellation event that can be checked by streaming operations.
        """
        try:
            with cls._cancellation_lock:
                if cls._cancellation_event:
                    cls._cancellation_event.set()
                    logger.debug("OpenAI stream cancellation event set")

                # Note: OpenAI SDK doesn't support direct HTTP request cancellation
                # The cancellation will be handled at the stream iteration level

        except Exception as e:
            logger.warning(f"Error cancelling OpenAI generation: {e}")

    @staticmethod
    def generate_response(chat: Chat, model_key: str, temperature: float = 0.7, silent_reason: str = "", thinking_budget: Optional[int] = None) -> Any:
        """
        Generates a response using the OpenAI API.

        Args:
            chat (Chat): The chat object containing messages or a string prompt.
            model_key (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent_reason (str): Reason for suppressing print statements.

        Returns:
            Any: A stream object that yields response chunks.

        Raises:
            Exception: If there's an error generating the response, to be handled by the router.
        """

        # Configure the client (let any error here bubble up to the router)
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Informational logging (not error handling)
        if silent_reason:
            temp_str = "" if temperature == 0 else f" at temperature {temperature}"
            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
            g.debug_log(f"OpenAI-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is {colored('silently', 'green')} generating response{temp_str}...", force_print=True, prefix=prefix)
        else:
            temp_str = "" if temperature == 0 else f" at temperature {temperature}"
            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
            g.debug_log(f"OpenAI-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is generating response{temp_str}...", "green", force_print=True, prefix=prefix)

        # Set up cancellation event for this generation
        OpenAIAPI._cancellation_event = threading.Event()

        # Let any errors here bubble up to the router for centralized handling
        raw_stream = client.chat.completions.create(
            model=model_key,
            messages=chat.to_openai(),
            temperature=temperature,
            stream=True
        )

        # Return a cancellable stream wrapper
        return OpenAIAPI._create_cancellable_stream(raw_stream, model_key)

    @staticmethod
    def transcribe_audio(audio_data: sr.AudioData, language: str = "", model: str = "whisper-1", chat: Optional[Chat] = None) -> Tuple[str,str]:
        """
        Transcribes an audio file using the OpenAI Whisper API.

        Args:
            audio_data (sr.AudioData): The audio data object.
            model (str): The model identifier for Whisper.
            language (str): The language of the audio.
            chat (Optional[Chat]): The chat object for debug printing.

        Returns:
            Tuple[str,str]: (transcribed text from the audio file, language)

        Raises:
            Exception: If an error occurs during transcription.
        """
        temp_audio_file_path = None
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"OpenAI-Api: Transcribing audio using {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')}...", force_print=True, prefix=prefix)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=g.CLIAGENT_PERSISTENT_STORAGE_PATH) as temp_audio_file:
                temp_audio_file.write(audio_data.get_wav_data())
                temp_audio_file_path = temp_audio_file.name

            with open(temp_audio_file_path, 'rb') as audio_file:
                response = client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    response_format="verbose_json",
                    language=language
                )

            no_speech_prob = response.segments[0]['no_speech_prob']
            if (no_speech_prob > 0.7):
                if chat:
                    prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                    g.debug_log("No speech detected", force_print=True, prefix=prefix)
                return "", "english"

            language = response.language

            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Transcription complete. Detected language: {language}", "green", force_print=True, prefix=prefix)

            return response.text, language

        except Exception as e:
            error_msg = f"OpenAI Whisper API error: {e}"
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
            raise Exception(error_msg)

        finally:
            if temp_audio_file_path and os.path.exists(temp_audio_file_path):
                os.remove(temp_audio_file_path)

    @classmethod
    def _create_cancellable_stream(cls, raw_stream, model_key: str):
        """
        Create a cancellable stream wrapper that monitors the cancellation event.
        This allows for immediate interruption of OpenAI streaming responses.
        """
        class CancellableOpenAIStream:
            def __init__(self, stream, cancellation_event, model_key):
                self.stream = stream
                self.cancellation_event = cancellation_event
                self.model_key = model_key

            def __iter__(self):
                try:
                    for chunk in self.stream:
                        # Check for cancellation before yielding each chunk
                        if self.cancellation_event and self.cancellation_event.is_set():
                            logger.debug(f"OpenAI stream for {self.model_key} cancelled via interruption")
                            break
                        yield chunk
                finally:
                    # Cleanup: close the stream if possible
                    try:
                        if hasattr(self.stream, 'close'):
                            self.stream.close()
                    except Exception as e:
                        logger.debug(f"Error closing OpenAI stream: {e}")

        return CancellableOpenAIStream(raw_stream, cls._cancellation_event, model_key)