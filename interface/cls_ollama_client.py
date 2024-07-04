import logging
from typing import List, Optional
import ollama
from termcolor import colored
from interface.cls_chat import Chat
from cls_custom_coloring import CustomColoring
from PIL import Image
import base64
import io
import os
import tiktoken

from interface.cls_chat_client_interface import ChatClientInterface

class OllamaClient(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the Ollama API.
    """

    @staticmethod
    def generate_response(chat: Chat, model: str = "phi3", temperature: float = 0.8, silent: bool = False, base64_images: List[str] = [], **kwargs) -> Optional[str]:
        """
        Generates a response using the Ollama API.

        Args:
            chat (Chat): The chat object containing messages.
            model (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent (bool): Whether to suppress print statements.
            base64_images (List[str]): List of base64-encoded images.
            **kwargs: Additional keyword arguments.

        Returns:
            Optional[str]: The generated response, or None if an error occurs.
        """
        tooling = CustomColoring()
        if not silent:
            print(f"Ollama-Api: <{colored(model, 'green')}> is generating response...")

        str_temperature = str(temperature)
        try:
            data = {
                "model": model,
                "messages": chat._to_dict(),
                "temperature": str_temperature,
                **kwargs,
            }

            if base64_images:
                for image_base64 in base64_images:
                    image_bytes = base64.b64decode(image_base64)
                    image = Image.open(io.BytesIO(image_bytes))
                    print(f"Image Resolution: {image.size} (Width x Height)")
                data["images"] = base64_images

            response_stream = None
            for env_var in os.environ:
                if env_var.startswith("OLLAMA_HOST_"):
                    host = os.getenv(env_var)
                    try:
                        client = ollama.Client(host=f'http://{host}:11434')
                        response_stream = client.chat(model, data["messages"], True)
                        if response_stream:
                            break
                    except Exception as e:
                        logging.error(f"Failed to connect to {host}: {e}")

            full_response = ""
            for line in response_stream:
                next_string = line["message"]["content"]
                full_response += next_string
                if not silent:
                    print(tooling.apply_color(next_string), end="")
            if not silent:
                print()

            return full_response

        except Exception as e:
            logging.error(f"Ollama API error: {e}")
            return None

    @staticmethod
    def count_tokens(text: str, model: str = "phi3") -> int:
        """
        Counts the number of tokens in the given text for the specified model.

        Args:
            text (str): The input text.
            model (str): The model identifier.

        Returns:
            int: The number of tokens in the input text.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logging.error(f"Token counting error: {e}")
            return 0
