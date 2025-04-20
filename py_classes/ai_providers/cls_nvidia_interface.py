import os
from typing import List, Optional, Dict, Any, Union
from collections.abc import Callable
import logging
from openai import OpenAI
from termcolor import colored
from py_classes.cls_chat import Chat
from py_classes.unified_interfaces import AIProviderInterface
from py_classes.cls_custom_coloring import CustomColoring

class NvidiaAPI(AIProviderInterface):
    """
    Implementation of the ChatClientInterface for the NVIDIA NeMo API.
    """

    def __init__(self):
        """
        Initialize the NvidiaAPI with the NVIDIA-specific OpenAI client.
        """
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv('NVIDIA_API_KEY')
        )

    def generate_response(self, chat: Union[Chat, str], model_key: str = "nvidia/llama-3.1-nemotron-70b-instruct", temperature: float = 0.7, silent_reason: str = "", base64_images: List[str] = []) -> Any:
        """
        Generates a response using the NVIDIA NeMo API.
        Args:
        chat (Union[Chat, str]): The chat object containing messages or a string prompt.
        model_key (str): The model identifier.
        temperature (float): The temperature setting for the model.
        silent_reason (str): Whether to suppress print statements.
        base64_images (List[str]): List of base64 encoded images (not used in this implementation).
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
            if silent_reason:
                temp_str = "" if temperature == 0 else f" at temperature {temperature}"
                debug_print(f"NVIDIA-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is {colored('silently', 'green')} generating response{temp_str}...", force_print=True)
            else:
                temp_str = "" if temperature == 0 else f" at temperature {temperature}"
                debug_print(f"NVIDIA-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is generating response{temp_str}...", "green", force_print=True)

            return self.client.chat.completions.create(
                model=model_key,
                messages=chat.to_openai(),
                temperature=temperature,
                top_p=1,
                max_tokens=1024,
                stream=True
            )

        except Exception as e:
            error_msg = f"NVIDIA NeMo API error: {e}"
            debug_print(error_msg, "red", is_error=True)
            raise Exception(error_msg)

    @staticmethod
    def transcribe_audio(audio_data, language: str = "", model: str = ""):
        """
        Transcribes an audio file using the NVIDIA NeMo API.
        This method is not implemented for the NVIDIA NeMo API in this example.
        """
        raise NotImplementedError("Audio transcription is not implemented for the NVIDIA NeMo API in this example.")