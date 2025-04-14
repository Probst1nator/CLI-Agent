import tempfile
import os
import time
from typing import List, Optional, Any, Dict, Union
from collections.abc import Callable
from openai import OpenAI
from termcolor import colored
import logging
from py_classes.cls_chat import Chat
from py_classes.unified_interfaces import AIProviderInterface
from py_classes.cls_custom_coloring import CustomColoring
import speech_recognition as sr
from py_classes.globals import g

class OpenAIAPI(AIProviderInterface):
    """
    Implementation of the AIProviderInterface for the OpenAI API.
    """

    @staticmethod
    def generate_response(chat: Union[Chat, str], model_key: str, temperature: float, silent_reason: str = "") -> Any:
        """
        Generates a response using the OpenAI API.
        
        Args:
            chat (Union[Chat, str]): The chat object containing messages or a string prompt.
            model_key (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent_reason (str): Whether to suppress print statements.
            
        Returns:
            Any: A stream object that yields response chunks.
        """
        # Convert string to Chat object if needed
        if isinstance(chat, str):
            from py_classes.cls_chat import Chat, Role
            chat_obj = Chat()
            chat_obj.add_message(Role.USER, chat)
            chat = chat_obj
            
        debug_print = AIProviderInterface.create_debug_printer(chat)
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            if silent_reason:
                debug_print(f"OpenAI-Api: <{colored(model_key, 'green')}> is {colored('silently', 'green')} generating response...", force_print=True)
            else:
                debug_print(f"OpenAI-Api: <{colored(model_key, 'green')}> is generating response...", "green", force_print=True)

            return client.chat.completions.create(
                model=model_key,
                messages=chat.to_openai(),
                temperature=temperature,
                stream=True
            )

        except Exception as e:
            error_msg = f"OpenAI API error: {e}"
            debug_print(error_msg, "red", is_error=True)
            raise Exception(error_msg)

    @staticmethod
    def transcribe_audio(audio_data: sr.AudioData, language: str = "", model: str = "whisper-1", chat: Optional[Chat] = None) -> tuple[str,str]:
        """
        Transcribes an audio file using the OpenAI Whisper API.

        Args:
            audio_data (sr.AudioData): The audio data object.
            model (str): The model identifier for Whisper.
            language (str): The language of the audio.
            chat (Optional[Chat]): The chat object for debug printing.

        Returns:
            tuple[str,str]: (transcribed text from the audio file, language)
        """
        debug_print = None
        if chat:
            debug_print = AIProviderInterface.create_debug_printer(chat)
            
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            if debug_print:
                debug_print(f"OpenAI-Api: Transcribing audio using <{colored(model, 'green')}>...", force_print=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=g.PROJ_PERSISTENT_STORAGE_PATH) as temp_audio_file:
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
                if debug_print:
                    debug_print("No speech detected", force_print=True)
                return "", "english"
            
            language = response.language
            
            if debug_print:
                debug_print(f"Transcription complete. Detected language: {language}", "green", force_print=True)
                
            return response.text, language

        except Exception as e:
            error_msg = f"OpenAI Whisper API error: {e}"
            if debug_print:
                debug_print(error_msg, "red", is_error=True)
            else:
                logger.error(error_msg)
            raise Exception(error_msg)

        finally:
            if 'temp_audio_file_path' in locals() and os.path.exists(temp_audio_file_path):
                os.remove(temp_audio_file_path)