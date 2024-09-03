import json
import sys
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import ollama
from termcolor import colored
from classes.cls_chat import Chat
from classes.cls_custom_coloring import CustomColoring
import os
from logger import logger
from classes.cls_ai_provider_interface import ChatClientInterface
import socket
from dataclasses import dataclass, field
from typing import List, Optional
import json
from datetime import datetime

@dataclass
class OllamaDetails:
    format: str
    family: str
    parameter_size: str
    quantization_level: str
    parent_model: str
    families: Optional[List[str]] = None

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "family": self.family,
            "families": self.families,
            "parent_model": self.parent_model,
            "parameter_size": self.parameter_size,
            "quantization_level": self.quantization_level
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'OllamaDetails':
        return cls(**data)

@dataclass
class OllamaModel:
    name: str
    modified_at: str
    size: int
    digest: str
    details: OllamaDetails

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "modified_at": self.modified_at,
            "size": self.size,
            "digest": self.digest,
            "details": self.details.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'OllamaModel':
        details = OllamaDetails.from_dict(data['details'])
        return cls(
            name=data['name'],
            modified_at=data['modified_at'],
            size=data['size'],
            digest=data['digest'],
            details=details
        )

@dataclass
class OllamaModelList:
    models: List[OllamaModel] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"models": [model.to_dict() for model in self.models]}, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'OllamaModelList':
        data = json.loads(json_str)
        return cls(models=[OllamaModel.from_dict(model_data) for model_data in data['models']])



class OllamaClient(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the Ollama API.
    """
    reached_hosts: List[str] = []
    unreachable_hosts: List[str] = []

    @staticmethod
    def check_host_reachability(host: str) -> bool:
        """
        Validates if a host is reachable using a socket connection.
        
        Args:
            host (str): The hostname to validate.
        
        Returns:
            bool: True if the host is reachable, False otherwise.
        """
        try:
            hostname, port = host.split(':') if ':' in host else (host, 11434)
            with socket.create_connection((hostname, int(port)), timeout=3):
                return True
        except (socket.timeout, socket.error):
            return False


    @staticmethod
    def get_valid_client(model_key: str) -> Optional[ollama.Client]:
        """
        Returns a valid client for the given model, pulling the model if necessary on auto-download hosts.
        
        Args:
            model_key (str): The model to find a valid client for.
        
        Returns:
            Optional[ollama.Client]: A valid client if found, None otherwise.
        """
        ollama_hosts = [os.getenv(env_var) for env_var in os.environ if env_var.startswith("OLLAMA_HOST_") and env_var.count('_') == 2]
        auto_download_hosts = set(os.getenv(env_var) for env_var in os.environ if env_var.startswith("OLLAMA_HOST_AUTO_DOWNLOAD_MODELS_"))
        
        for host in ollama_hosts:
            if host not in OllamaClient.reached_hosts and host not in OllamaClient.unreachable_hosts:
                if OllamaClient.check_host_reachability(host):
                    OllamaClient.reached_hosts.append(host)
                else:
                    OllamaClient.unreachable_hosts.append(host)
            
            if host in OllamaClient.reached_hosts and host not in OllamaClient.unreachable_hosts:
                client = ollama.Client(host=f'http://{host}:11434')
                try:
                    model_list = OllamaModelList.from_json(json.dumps(client.list()))
                    if any(model_key in model.name for model in model_list.models):
                        return client
                    elif host in auto_download_hosts:
                        print(colored(f"{host} is pulling {model_key}...", "yellow"))
                        try:
                            
                            def bytes_to_mb(bytes_value):
                                return bytes_value / (1024 * 1024)

                            for response in client.pull(model_key, stream=True):
                                if "status" in response:
                                    if response["status"] == "pulling manifest":
                                        status = colored("Pulling manifest...", "yellow")
                                    elif response["status"].startswith("pulling"):
                                        digest = response.get("digest", "")
                                        total = bytes_to_mb(response.get("total", 0))
                                        completed = bytes_to_mb(response.get("completed", 0))
                                        status = colored(f"Pulling {digest}: {completed:.2f}/{total:.2f} MB", "yellow")
                                    else:
                                        continue
                                    
                                    sys.stdout.write('\r' + status)
                                    sys.stdout.flush()
                            return client
                        except Exception as e:
                            print(f"Error pulling model {model_key} on host {host}: {e}")
                except Exception as e:
                    print(f"Error checking models on host {host}: {e}")
                    OllamaClient.unreachable_hosts.append(host)
        
        print(f"No valid client found for model {model_key}")
        return None


    @staticmethod
    def generate_response(chat: Chat, model: str = "phi3.5:3.8b", temperature: Optional[float] = 0.75, silent: bool = False, tools: Optional[List[Dict[str, Any]]] = None) -> Optional[str | List[Dict[str, Any]]]:
        """
        Generates a response using the Ollama API, with support for tool calling.

        Args:
            chat (Chat): The chat object containing messages.
            model (str): The model identifier (e.g., "phi3.5:3.8b", "llama3.1").
            temperature (float): The temperature setting for the model.
            silent (bool): Whether to suppress print statements.
            tools (List[Dict[str, Any]], optional): A list of tool definitions for the model to use.

        Returns:
            Optional[str]: The generated response, or None if an error occurs.
        """
        options = ollama.Options()
        if "hermes" in model.lower():
            options.update(stop=["<|end_of_text|>"])
        if temperature:
            options.update(temperature=temperature)
        
        tooling = CustomColoring()
        logger.debug(json.dumps({"last_message": chat.messages[-1][1]}, indent=2))

        client: ollama.Client | None = OllamaClient.get_valid_client(model)
        if not client:
            logger.error(f"No valid host found for model {model}")
            return None
        assert client is not None
        host: str = client._client.base_url.host

        try:
            if silent:
                print(f"Ollama-Api: <{colored(model, 'green')}> is {colored('silently', 'green')} generating response using <{colored(host, 'green')}>...")
            else:
                print(f"Ollama-Api: <{colored(model, 'green')}> is generating response using <{colored(host, 'green')}>...")
            
            if tools:
                response = client.chat(model=model, messages=chat.to_ollama(), stream=False, options=options, keep_alive=1800, tools=tools)
                return response["tool_calls"]
            else:
                response_stream = client.chat(model=model, messages=chat.to_ollama(), stream=True, options=options, keep_alive=1800)
                full_response = ""
                for line in response_stream:
                    next_string = line["message"]["content"]
                    full_response += next_string
                    if not silent:
                        print(tooling.apply_color(next_string), end="")
                if not silent:
                    print()
                logger.debug(json.dumps({"full_response": full_response}, indent=2))
                return full_response

        except Exception as e:
            print(f"Ollama-Api: Failed to generate response using <{colored(host, 'red')}> with model <{colored(model, 'red')}>: {e}")
            OllamaClient.unreachable_hosts.append(f"{host}{model}")
            logger.error(f"Ollama-Api: Failed to generate response using <{host}> with model <{model}>: {e}")
            return None

    @staticmethod
    def generate_response_raw(prompt: str, model: str = "nuextract", host: str = None) -> Optional[Dict[str, Any]]:
        """
        Generates a raw response from the Ollama API.

        Args:
            prompt (str): The input prompt.
            model (str): The model to use.
            host (str, optional): The specific host to use. If None, uses the first validated host.

        Returns:
            Optional[Dict[str, Any]]: The raw response from the API, or None if an error occurs.
        """
        if not host:
            host = OllamaClient.get_valid_host(model)
            if not host:
                raise ValueError("No validated Ollama hosts available")

        try:
            client = ollama.Client(host=f'http://{host}:11434')
            response = client.generate(model=model, prompt=prompt, stream=False, keep_alive=1800)
            return response
        except Exception as e:
            logger.error(f"Ollama-Api: Failed to generate raw response using <{host}> with model <{model}>: {e}")
            return None

    @staticmethod
    def generate_embedding(text: str, model: str = "bge-m3") -> Optional[List[float]]:
        """
        Generates an embedding for the given text using the specified Ollama model.
        
        Args:
            text (str): The input text to generate an embedding for.
            model (str): The embedding model to use.
        
        Returns:
            Optional[List[float]]: The generated embedding as a list of floats, or None if an error occurs.
        """
        client: ollama.Client | None = OllamaClient.get_valid_client(model)
        if not client:
            logger.error(f"No valid host found for model {model}")
            return None
        assert client is not None
        host: str = client._client.base_url.host

        try:
            print(f"Ollama-Api: <{colored(model, 'green')}> is generating embedding using <{colored(host, 'green')}>...")
            client = ollama.Client(host=f'http://{host}:11434')
            response = client.embeddings(model=model, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Ollama-Api: Failed to generate embedding using <{colored(host, 'red')}> with model <{colored(model, 'red')}>: {e}")
            OllamaClient.unreachable_hosts.append(f"{host}{model}")
            logger.error(f"Ollama-Api: Failed to generate embedding using <{host}> with model <{model}>: {e}")
            return None