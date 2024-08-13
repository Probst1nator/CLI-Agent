import logging
from typing import List, Optional
import ollama
from termcolor import colored
from classes.cls_chat import Chat
from classes.cls_custom_coloring import CustomColoring
from PIL import Image
import base64
import io
import os
import tiktoken

from classes.cls_ai_provider_interface import ChatClientInterface

class OllamaClient(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the Ollama API.
    """

    @staticmethod
    def generate_response(chat: Chat, model: str = "phi3", temperature: float = 0.8, silent: bool = False, **kwargs) -> Optional[str]:
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
        
        if silent:
            print(f"Ollama-Api: <{colored(model, 'green')}> is {colored('silently', 'green')} generating response...")
        else:
            print(f"Ollama-Api: <{colored(model, 'green')}> is generating response...")

        try:
            response_stream = None
            for env_var in os.environ:
                if env_var.startswith("OLLAMA_HOST_"):
                    host = os.getenv(env_var)
                    try:
                        client = ollama.Client(host=f'http://{host}:11434')
                        response_stream = client.chat(model, chat.to_ollama(), True, keep_alive=1800)
                        if response_stream:
                            print(f"Ollama-Api: <{colored(host, 'green')}> is generating response...")
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
            raise Exception(f"Ollama API error: {e}")


