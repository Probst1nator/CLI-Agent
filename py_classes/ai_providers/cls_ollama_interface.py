import ast
from enum import Enum
import json
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import chromadb
import ollama
from termcolor import colored
from py_classes.cls_chat import Chat
from py_classes.cls_custom_coloring import CustomColoring
from py_classes.cls_ai_provider_interface import ChatClientInterface
import os
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime
from py_classes.globals import g
import logging

# Configure logger with proper settings to prevent INFO level messages from being displayed
logger = logging.getLogger(__name__)

# Remove any existing handlers and set up console handler to only show ERROR or higher
for handler in logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        logger.removeHandler(handler)

# Add a console handler that only shows ERROR level and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
logger.addHandler(console_handler)

class OllamaClient(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the Ollama API.
    """
    reached_hosts: List[str] = []
    unreachable_hosts: List[str] = []

    @classmethod
    def check_host_reachability(cls, host: str, chat: Optional[Chat] = None) -> bool:
        """
        Validates if a host is reachable using a socket connection.
        
        Args:
            host (str): The hostname to validate.
            chat (Optional[Chat]): Chat object for debug printing with title.
        
        Returns:
            bool: True if the host is reachable, False otherwise.
        """
        try:
            debug_print = None
            if chat:
                debug_print = ChatClientInterface.create_debug_printer(chat)
            
            hostname, port = host.split(':') if ':' in host else (host, 11434)
            if debug_print:
                debug_print(f"Checking host {host}...", "green", force_print=True)
            else:
                print(colored(f"Checking host {host}...", "green"))
                
            with socket.create_connection((hostname, int(port)), timeout=3):
                return True
        except (socket.timeout, socket.error):
            if debug_print:
                debug_print(f"Host {host} is unreachable", "red", is_error=True)
            else:
                print(colored(f"Host {host} is unreachable", "red"))
            return False

    @staticmethod
    def get_valid_client(model_key: str, chat: Optional[Chat] = None) -> Tuple[ollama.Client|None, str]:
        """
        Returns a valid client for the given model, pulling the model if necessary on auto-download hosts.
        
        Args:
            model_key (str): The model to find a valid client for.
            chat (Optional[Chat]): Chat object for debug printing with title.
        
        Returns:
            Tuple[Optional[ollama.Client], str]: [A valid client or None, found model_key].
        """
        # Get hosts from comma-separated environment variables
        ollama_hosts = os.getenv("OLLAMA_HOST", "").split(",")
        auto_download_hosts = set(os.getenv("OLLAMA_HOST_AUTO_DOWNLOAD_MODELS", "").split(","))
        
        debug_print = None
        if chat:
            debug_print = ChatClientInterface.create_debug_printer(chat)
        
        for host in ollama_hosts:
            host = host.strip()
            if not host:
                continue
                
            if host not in OllamaClient.reached_hosts and host not in OllamaClient.unreachable_hosts:
                if OllamaClient.check_host_reachability(host, chat):
                    OllamaClient.reached_hosts.append(host)
                else:
                    OllamaClient.unreachable_hosts.append(host)
            
            if host in OllamaClient.reached_hosts and host not in OllamaClient.unreachable_hosts:
                client = ollama.Client(host=f'http://{host}:11434')
                try:
                    response = client.list()
                    logger.debug("=== START MODEL PROCESSING ===")
                    logger.debug(f"Raw response type: {type(response)}")
                    logger.debug(f"Raw response dir: {dir(response)}")
                    logger.debug(f"Models list type: {type(response.models)}")
                    logger.debug(f"Number of models: {len(response.models)}")
                    
                    # Examine first model's structure if available
                    if response.models:
                        first_model = response.models[0]
                        logger.debug(f"First model type: {type(first_model)}")
                        logger.debug(f"First model dir: {dir(first_model)}")
                        logger.debug(f"First model attributes: {[(attr, getattr(first_model, attr, None)) for attr in dir(first_model) if not attr.startswith('_')]}")
                    
                    # Convert ListResponse to dict before JSON serialization
                    response_dict = {"models": []}
                    for idx, model in enumerate(response.models):
                        logger.debug(f"\nProcessing model {idx + 1}:")
                        try:
                            # Get raw model attributes
                            model_attrs = {
                                "model": getattr(model, "model", None),
                                "name": getattr(model, "name", None),
                                "modified_at": getattr(model, "modified_at", None),
                                "size": getattr(model, "size", None),
                                "digest": getattr(model, "digest", None),
                            }
                            logger.debug(f"Raw model attributes: {model_attrs}")
                            
                            # Clean and validate attributes
                            model_dict = {
                                "model": model_attrs["model"] or model_attrs["name"] or "",
                                "modified_at": (model_attrs["modified_at"].isoformat() 
                                              if hasattr(model_attrs["modified_at"], 'isoformat') 
                                              else model_attrs["modified_at"] or datetime.now().isoformat()),
                                "size": model_attrs["size"] or 0,
                                "digest": model_attrs["digest"] or "",
                            }
                            logger.debug(f"Processed model dict: {model_dict}")
                            
                            # Verify JSON serialization
                            json.dumps(model_dict)  # Test serialization
                            response_dict["models"].append(model_dict)
                            logger.debug(f"Successfully added model to response_dict")
                        except Exception as e:
                            logger.error(f"Error processing model {idx + 1}: {str(e)}")
                            continue
                    
                    logger.debug(f"\nFinal response dict: {response_dict}")
                    logger.debug("Testing full JSON serialization...")
                    serialized = json.dumps(response_dict)
                    logger.debug("JSON serialization successful")
                    
                    logger.debug("Creating OllamaModelList...")
                    model_list = OllamaModelList.from_json(serialized)
                    logger.debug(f"Created model list with {len(model_list.models)} models")
                    
                    # Look for model name in all available models
                    logger.debug(f"\nSearching for model key: {model_key}")
                    found_model_key = next((
                        model.model 
                        for model in model_list.models 
                        if model_key.lower() in model.model.lower()
                    ), None)
                    logger.debug(f"Found model key: {found_model_key}")
                    logger.debug("=== END MODEL PROCESSING ===\n")
                    if found_model_key:
                        return client, found_model_key 
                    elif host in auto_download_hosts:
                        if debug_print:
                            debug_print(f"{host} is pulling {model_key}...", "yellow", force_print=True)
                        else:
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
                            print()
                            return client, model_key
                        except Exception as e:
                            if debug_print:
                                debug_print(f"Error pulling model {model_key} on host {host}: {e}", "red", is_error=True)
                            else:
                                print(f"Error pulling model {model_key} on host {host}: {e}")
                except Exception as e:
                    if debug_print:
                        debug_print(f"Error checking models on host {host}: {e}", "red", is_error=True)
                    else:
                        print(f"Error checking models on host {host}: {e}")
                    OllamaClient.unreachable_hosts.append(host)
        
        if debug_print:
            debug_print(f"No valid client found for model {model_key}", "red", is_error=True)
        else:
            print(f"No valid client found for model {model_key}")
        return None, None

    @staticmethod
    def generate_response(
        chat: Chat,
        model_key: str = "phi3.5:3.8b",
        temperature: Optional[float] = 0.75,
        silent_reason: str = ""
    ) -> Optional[str]:
        """
        Generates a response using the Ollama API, with support for tool calling.

        Args:
            chat (Chat): The chat object containing messages.
            model (str): The model identifier (e.g., "phi3.5:3.8b", "llama3.1").
            temperature (float): The temperature setting for the model.
            silent (bool): Whether to suppress print statements.
            tools (List[FunctionTool], optional): A list of tool definitions for the model to use.

        Returns:
            Optional[Union[str, List[ToolCall]]]: The generated response, or None if an error occurs.
        """
        # options = ollama.Options()
        # if "hermes" in model_key.lower():
        #     options.update(stop=["<|end_of_text|>"])
        # if temperature:
        #     options.update(temperature=temperature)
        # options.update(max_tokens=1500)
        
        # ! Hack to try lower ram usage
        if "mistral-small" in model_key:
            model_key = "mistral-small:22b-instruct-2409-q3_K_M"
        
        debug_print = ChatClientInterface.create_debug_printer(chat)
        tooling = CustomColoring()
        logger.debug(json.dumps({"last_message": chat.messages[-1][1]}, indent=2))

        client: ollama.Client | None
        client, model_key = OllamaClient.get_valid_client(model_key, chat)
        if not client:
            error_msg = f"No valid host found for model {model_key}"
            debug_print(error_msg, "red", is_error=True)
            return None
        assert client is not None
        host: str = client._client.base_url.host

        # Check if host is in OLLAMA_HOST_FORCE_FAST_MODELS list
        force_fast_hosts = os.getenv("OLLAMA_HOST_FORCE_FAST_MODELS", "").split(",")
        force_fast_hosts = [h.strip() for h in force_fast_hosts if h.strip()]
        if host in force_fast_hosts:
            debug_print(f"Host {host} is in OLLAMA_HOST_FORCE_FAST_MODELS list. Using optimized settings.", force_print=True)
            # Add your fast model optimizations here if needed

        try:
            if silent_reason:
                debug_print(f"Ollama-Api: <{colored(model_key, 'green')}> is using <{colored(host, 'green')}> to perform the action: <{colored(silent_reason, 'yellow')}>", force_print=True)
            else:
                debug_print(f"Ollama-Api: <{colored(model_key, 'green')}> is using <{colored(host, 'green')}> to generate a response...", "green", force_print=True)
            
            response_stream = client.chat(model=model_key, messages=chat.to_ollama(), stream=True, keep_alive=1800, options={"num_predict": 32768})
            full_response = ""
            for line in response_stream:
                next_string = line["message"]["content"]
                full_response += next_string
                if not silent_reason:
                    debug_print(tooling.apply_color(next_string), end="", with_title=False)
            if not silent_reason:
                debug_print("", with_title=False)
            logger.debug(json.dumps({"full_response": full_response}, indent=2))
            if "instruction" in full_response.lower():
                full_response = full_response.split("instruction")[0]
            return full_response

        except Exception as e:
            error_msg = f"Ollama-Api: Failed to generate response using <{colored(host, 'red')}> with model <{colored(model_key, 'red')}>: {e}"
            debug_print(error_msg, "red", is_error=True)
            OllamaClient.unreachable_hosts.append(f"{host}{model_key}")
            return None

    @staticmethod
    def generate_response_raw(prompt: str, model: str = "nuextract", host: str = None, chat: Optional[Chat] = None) -> Optional[Dict[str, Any]]:
        """
        Generates a raw response from the Ollama API.

        Args:
            prompt (str): The input prompt.
            model (str): The model to use.
            host (str, optional): The specific host to use. If None, uses the first validated host.
            chat (Optional[Chat]): The chat object for debug printing.

        Returns:
            Optional[Dict[str, Any]]: The raw response from the API, or None if an error occurs.
        """
        debug_print = None
        if chat:
            debug_print = ChatClientInterface.create_debug_printer(chat)
            
        if not host:
            client, model_key = OllamaClient.get_valid_client(model, chat)
            if not client:
                error_msg = "No validated Ollama hosts available"
                if debug_print:
                    debug_print(error_msg, "red", is_error=True)
                else:
                    logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            client = ollama.Client(host=f'http://{host}:11434')

        try:
            if debug_print:
                debug_print(f"Ollama-Api: Generating raw response using model <{colored(model, 'green')}>...", force_print=True)
                
            response = client.generate(model=model, prompt=prompt, stream=False, keep_alive=1800)
            return response
        except Exception as e:
            error_msg = f"Ollama-Api: Failed to generate raw response using <{host}> with model <{model}>: {e}"
            if debug_print:
                debug_print(error_msg, "red", is_error=True)
            else:
                logger.error(error_msg)
            return None

    @staticmethod
    def generate_embedding(text: Union[str, List[str]], model: str = "bge-m3", chat: Optional[Chat] = None) -> Optional[Union[List[float], List[List[float]]]]:
        """
        Generates embeddings for the given text(s) using the specified Ollama model.
        
        Args:
            text (Union[str, List[str]]): The input text or list of texts to generate embeddings for.
            model (str): The embedding model to use.
            chat (Optional[Chat]): The chat object for debug printing.
        
        Returns:
            Optional[Union[List[float], List[List[float]]]]: The generated embedding(s) as a list of floats or list of list of floats,
                                                            or None if an error occurs.
        """
        # Handle empty input cases
        if isinstance(text, str) and len(text) < 3:
            return None
        if isinstance(text, list) and (len(text) == 0 or all(len(t) < 3 for t in text)):
            return None

        debug_print = None
        if chat:
            debug_print = ChatClientInterface.create_debug_printer(chat)

        client, model_key = OllamaClient.get_valid_client(model, chat)
        if not client:
            error_msg = f"No valid host found for model {model}"
            if debug_print:
                debug_print(error_msg, "red", is_error=True)
            else:
                logger.error(error_msg)
            return None
            
        assert client is not None
        host: str = client._client.base_url.host

        try:
            if isinstance(text, str):
                # Single text case
                if debug_print:
                    debug_print(f"Ollama-Api: <{colored(model, 'green')}> is generating embedding using <{colored(host, 'green')}>...", force_print=True)
                else:
                    print(f"Ollama-Api: <{colored(model, 'green')}> is generating embedding using <{colored(host, 'green')}>...")
                
                response = client.embeddings(model=model, prompt=text, keep_alive=7200)
                return response["embedding"]
            else:
                # List of texts case
                if debug_print:
                    debug_print(f"Ollama-Api: <{colored(model, 'green')}> is generating {len(text)} embeddings using <{colored(host, 'green')}>", force_print=True)
                else:
                    print(f"Ollama-Api: <{colored(model, 'green')}> is generating {len(text)} embeddings using <{colored(host, 'green')}>")
                
                embeddings = []
                
                for i, t in enumerate(text, 1):
                    if len(t) < 3:
                        embeddings.append(None)
                        continue
                    
                    response = client.embeddings(model=model, prompt=t, keep_alive=7200)
                    embeddings.append(response["embedding"])
                    
                    # Print a dot for each successful embedding generation
                    sys.stdout.write(".")
                    sys.stdout.flush()
                
                print()  # New line after all dots
                return embeddings

        except Exception as e:
            error_msg = f"Ollama-Api: Failed to generate embedding(s) using <{colored(host, 'red')}> with model <{colored(model, 'red')}>: {e}"
            if debug_print:
                debug_print(error_msg, "red", is_error=True)
            else:
                print(colored(error_msg, "red"))
                logger.error(f"Ollama-Api: Failed to generate embedding(s) using <{host}> with model <{model}>: {e}")
            
            OllamaClient.unreachable_hosts.append(f"{host}{model}")
            return None

@dataclass
class OllamaModel:
    """
    Represents an Ollama model with its metadata.
    
    Attributes:
        model (str): The name/identifier of the model.
        modified_at (str): The last modification timestamp of the model.
        size (int): The size of the model in bytes.
        digest (str): The digest (hash) of the model.
    """
    model: str
    modified_at: str
    size: int
    digest: str
    
    def to_dict(self) -> dict:
        """
        Converts OllamaModel to a dictionary.
        
        Returns:
            dict: A dictionary representation of the OllamaModel instance.
        """
        return {
            "model": self.model,
            "modified_at": self.modified_at,
            "size": self.size,
            "digest": self.digest,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'OllamaModel':
        """
        Creates an OllamaModel instance from a dictionary.
        
        Args:
            data (dict): A dictionary containing OllamaModel attributes.
        
        Returns:
            OllamaModel: An instance of OllamaModel.
        """
        return cls(**data)

@dataclass
class OllamaModelList:
    """
    Represents a list of Ollama models.
    
    Attributes:
        models (List[OllamaModel]): A list of OllamaModel instances.
    """
    models: List[OllamaModel] = field(default_factory=list)
    
    def to_json(self) -> str:
        """
        Converts OllamaModelList to a JSON string.
        
        Returns:
            str: A JSON string representation of the OllamaModelList.
        """
        return json.dumps({"models": [model.to_dict() for model in self.models]}, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'OllamaModelList':
        """
        Creates an OllamaModelList instance from a JSON string.
        
        Args:
            json_str (str): A JSON string containing a list of models.
        
        Returns:
            OllamaModelList: An instance of OllamaModelList.
        """
        data = json.loads(json_str)
        return cls(models=[OllamaModel.from_dict(model_data) for model_data in data['models']])