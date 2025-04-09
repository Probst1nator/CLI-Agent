import os
from typing import List, Optional, Dict, Any
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

    def generate_response(self, chat: Chat, model: str = "nvidia/llama-3.1-nemotron-70b-instruct", temperature: float = 0.7, silent_reason: str = False, base64_images: List[str] = []) -> Optional[str]:
        """
        Generates a response using the NVIDIA NeMo API.
        Args:
        chat (Chat): The chat object containing messages.
        model (str): The model identifier.
        temperature (float): The temperature setting for the model.
        silent_reason (str): Whether to suppress print statements.
        base64_images (List[str]): List of base64 encoded images (not used in this implementation).
        Returns:
        Optional[str]: The generated response, or None if an error occurs.
        """
        debug_print = AIProviderInterface.create_debug_printer(chat)
        try:
            if silent_reason:
                debug_print(f"NVIDIA-Api: <{colored(model, 'green')}> is {colored('silently', 'green')} generating response...", force_print=True)
            else:
                debug_print(f"NVIDIA-Api: <{colored(model, 'green')}> is generating response...", "green", force_print=True)

            stream = self.client.chat.completions.create(
                model=model,
                messages=chat.to_openai(),
                temperature=temperature,
                top_p=1,
                max_tokens=1024,
                stream=True
            )

            full_response = ""
            token_keeper = CustomColoring()

            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    if not silent_reason:
                        debug_print(token_keeper.apply_color(token), end="", with_title=False)
                    full_response += token

            if not silent_reason:
                debug_print("", with_title=False)

            return full_response

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