import logging
from typing import Dict, List, Union
import ollama
from termcolor import colored
from interface.cls_chat import Chat
from cls_custom_coloring import CustomColoring
from PIL import Image
import base64
import io
import os

class OllamaClient:
    @staticmethod
    def generate_completion(prompt: Chat, model: str = "phi3", temperature: float = 0.8, silent: bool = False, base64_images: List[str] = [], **kwargs) -> str:
        tooling = CustomColoring()
        if not silent:
            print(f"Ollama-Api: <{colored(model, 'green')}> is generating response...")

        str_temperature = str(temperature)
        try:
            data = {
                "model": model,
                "messages": prompt._to_dict(),
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
            return ""
