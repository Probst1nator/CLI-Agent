import json
from typing import List, Optional
import ollama
from termcolor import colored
from classes.cls_chat import Chat
from classes.cls_custom_coloring import CustomColoring
import os
from logger import logger
from classes.cls_ai_provider_interface import ChatClientInterface
import socket

class OllamaClient(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the Ollama API.
    """
    validated_hosts: List[str] = []
    failed_hosts: List[str] = []

    @staticmethod
    def validate_host(host: str) -> bool:
        """
        Validates if a host is reachable using a socket connection.
        
        Args:
            host (str): The hostname to validate.
        
        Returns:
            bool: True if the host is reachable, False otherwise.
        """
        try:
            # Extract hostname and port (default to 11434 if not specified)
            if ':' in host:
                hostname, port_str = host.split(':')
                port = int(port_str)
            else:
                hostname = host
                port = 11434

            # Attempt to create a socket connection
            with socket.create_connection((hostname, port), timeout=3) as sock:
                return True
        except (socket.timeout, socket.error):
            return False

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

        Returns:
            Optional[str]: The generated response, or None if an error occurs.
        """
        tooling = CustomColoring()
        logger.debug(json.dumps({"last_message":chat.messages[-1][1]}, indent=2))
        # logger.debug(json.dumps({"last_message":chat.messages[0], "model":model, "temperature":temperature, "silent":silent}, indent=2))
        while True:
            try:
                response_stream = None
                for env_var in os.environ:
                    if env_var.startswith("OLLAMA_HOST_"):
                        host = os.getenv(env_var)
                        if not host in OllamaClient.validated_hosts and not host in OllamaClient.failed_hosts and not host+model in OllamaClient.failed_hosts:
                            if OllamaClient.validate_host(host):
                                OllamaClient.validated_hosts.append(host)
                            else:
                                OllamaClient.failed_hosts.append(host)
                        if host in OllamaClient.validated_hosts and not host in OllamaClient.failed_hosts and not host+model in OllamaClient.failed_hosts:
                            try:
                                if silent:
                                    print(f"Ollama-Api: <{colored(model, 'green')}> is {colored('silently', 'green')} generating response using <{colored(host, 'green')}>...")
                                else:
                                    print(f"Ollama-Api: <{colored(model, 'green')}> is generating response using <{colored(host, 'green')}>...")
                                # Check if the host is reachable
                                client = ollama.Client(host=f'http://{host}:11434')
                                response_stream = client.chat(model, chat.to_ollama(), True, keep_alive=1800, options=ollama.Options())
                            except Exception as e:
                                print(f"Ollama-Api: Failed to generate response using <{colored(host, 'red')}> with model <{colored(model, 'red')}>: {e}")
                                OllamaClient.failed_hosts.append(host+model)
                                if ("try pulling it first" in str(e).lower()):
                                    continue
                                logger.error(f"Ollama-Api: Failed to generate response using <{host}> with model <{model}>: {e}")

                refused = False
                full_response = ""
                for line in response_stream:
                    next_string = line["message"]["content"]
                    full_response += next_string
                    if not silent:
                        print(tooling.apply_color(next_string), end="")
                    # refusal check
                    lower_response = full_response.lower()
                    if model != "WizardLM-2-7B-abliterated-Q4_K_M.gguf" and any(item in lower_response for item in ["tut mir leid", "i am sorry", "i'm sorry", "entschuldigung", "i apologize", "i apologise", "ich kann keine"]):
                        print(f"Ollama-Api: <{colored(model, 'red')}> refused...")
                        model = "WizardLM-2-7B-abliterated-Q4_K_M.gguf"
                        refused = True
                        break
                if refused:
                    continue
                if not silent:
                    print()

                logger.debug(json.dumps({"full_response":full_response}, indent=2))
                return full_response

            except Exception as e:
                logger.error(json.dumps({"error":{e}}, indent=2))
                raise Exception(f"Ollama API error: {e}")


    @staticmethod
    def generate_embedding( text: str, model: str = "bge-m3") -> List[float]:
        """
        Generates an embedding for the given text using the specified Ollama model.
        
        Args:
        text (str): The input text to generate an embedding for.
        model (str): The embedding model to use.
        
        Returns:
        List[float]: The generated embedding as a list of floats.
        
        Raises:
        Exception: If there's an error in generating the embedding.
        """
        response:List[float] = []
        for env_var in os.environ:
            if env_var.startswith("OLLAMA_HOST_"):
                host = os.getenv(env_var)
                if not host in OllamaClient.validated_hosts and not host in OllamaClient.failed_hosts and not host+model in OllamaClient.failed_hosts:
                    if OllamaClient.validate_host(host):
                        OllamaClient.validated_hosts.append(host)
                    else:
                        OllamaClient.failed_hosts.append(host)
                if host in OllamaClient.validated_hosts and not host in OllamaClient.failed_hosts and not host+model in OllamaClient.failed_hosts:
                    try:
                        print(f"Ollama-Api: <{colored(model, 'green')}> is generating embedding using <{colored(host, 'green')}>... ")
                        client = ollama.Client(host=f'http://{host}:11434')
                        response = client.embeddings(model=model, prompt=text)["embedding"]
                        break
                    except Exception as e:
                        print(f"Ollama-Api: Failed to generate embedding using <{colored(host, 'red')}> with model <{colored(model, 'red')}>: {e}")
                        OllamaClient.failed_hosts.append(host+model)
                        if ("try pulling it first" in str(e).lower()):
                            continue
                        logger.error(f"Ollama-Api: Failed to generate embedding using <{host}> with model <{model}>: {e}")
        return response