import ast
from enum import Enum
import json
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from collections.abc import Callable
import chromadb
import ollama
from termcolor import colored
from py_classes.cls_chat import Chat, Role
from py_classes.unified_interfaces import AIProviderInterface
from py_classes.cls_text_stream_painter import TextStreamPainter
from py_classes.cls_rate_limit_tracker import rate_limit_tracker
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

class OllamaClient(AIProviderInterface):
    """
    Implementation of the ChatClientInterface for the Ollama API.
    """
    reached_hosts: List[str] = []
    unreachable_hosts: List[str] = []
    
    @classmethod
    def reset_host_cache(cls):
        """Reset the host reachability cache to allow retrying all hosts."""
        cls.reached_hosts.clear()
        cls.unreachable_hosts.clear()
        # Note: unreachable_hosts now also contains host:model combinations

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
            hostname, port = host.split(':') if ':' in host else (host, 11434)
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Ollama-Api: Checking host " + colored("<" + host + ">", "green") + "...", "green", force_print=True, prefix=prefix)
            else:
                print(colored(f"Ollama-Api: Checking host " + colored("<" + host + ">", "green") + "...", "green"))
                
            with socket.create_connection((hostname, int(port)), timeout=3):
                return True
        except (socket.timeout, socket.error):
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Ollama-Api: Host " + colored("<" + host + ">", "red") + " is unreachable", "red", is_error=True, prefix=prefix)
            else:
                print(f"Ollama-Api: Host " + colored("<" + host + ">", "red") + " is unreachable", "red")
            return False

    @staticmethod
    def get_valid_client(model_key: str, chat: Optional[Chat] = None, auto_download: bool = True, is_small_model: bool = False) -> Tuple[ollama.Client|None, str]:
        """
        Returns a valid client for the given model, pulling the model if necessary on auto-download hosts.
        
        Args:
            model_key (str): The model to find a valid client for.
            chat (Optional[Chat]): Chat object for debug printing with title.
            auto_download (bool): Whether to automatically download models if not found.
            is_small_model (bool): Whether this is a small/fast model.
        
        Returns:
            Tuple[Optional[ollama.Client], str]: [A valid client or None, found model_key].
        """
        # Get hosts from comma-separated environment variables
        ollama_hosts = os.getenv("OLLAMA_HOST", "").split(",")
        
        # Remove the localhost from the list
        force_local_remote_host = os.getenv("FORCE_REMOTE_HOST_FOR_HOSTNAME", "")
        if socket.gethostname() in force_local_remote_host:
            try:
                ollama_hosts.remove("localhost")
                ollama_hosts.remove(socket.gethostbyname(socket.gethostname()))
            except Exception as e:
                pass
        
        auto_download_hosts = set(os.getenv("OLLAMA_HOST_AUTO_DOWNLOAD_MODELS", "").split(","))
        small_only_hosts = set(os.getenv("OLLAMA_HOST_ONLY_SMALL_MODELS", "").split(","))
        
        # If no auto_download hosts are set, default to all hosts for local development
        if not auto_download_hosts or (len(auto_download_hosts) == 1 and "" in auto_download_hosts):
            auto_download_hosts = set(ollama_hosts)
        
        # Remove empty strings from fast_only_hosts
        small_only_hosts.discard("")
        
        # Track failed hosts for this specific attempt to reduce noise
        failed_hosts_this_attempt = []
        
        for host in ollama_hosts:
            host = host.strip()
            if not host:
                continue
            
            # Skip hosts that are restricted to fast models if this isn't a small model
            if host in small_only_hosts and not is_small_model:
                continue
            
            # Skip host+model combinations that have recently failed with connection issues
            problematic_identifier = f"{host}:{model_key}"
            if problematic_identifier in OllamaClient.unreachable_hosts:
                continue
                
            if host not in OllamaClient.reached_hosts and host not in OllamaClient.unreachable_hosts:
                if OllamaClient.check_host_reachability(host, chat):
                    OllamaClient.reached_hosts.append(host)
                else:
                    OllamaClient.unreachable_hosts.append(host)
                    failed_hosts_this_attempt.append(host)
            
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
                    elif auto_download and host in auto_download_hosts:
                        if chat:
                            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                            g.debug_log(f"{host} is pulling {model_key}...", "yellow", force_print=True, prefix=prefix)
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
                            if chat:
                                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                                g.debug_log(f"Error pulling model {model_key} on host {host}: {e}", "red", is_error=True, prefix=prefix)
                            else:
                                print(f"Error pulling model {model_key} on host {host}: {e}")
                except Exception as e:
                    if chat:
                        prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                        g.debug_log(f"Error checking models on host {host}: {e}", "red", is_error=True, prefix=prefix)
                    else:
                        print(f"Error checking models on host {host}: {e}")
                    # Only mark host as unreachable if it's a connection issue, not a model issue
                    error_str = str(e).lower()
                    if any(conn_error in error_str for conn_error in ['connection', 'timeout', 'refused', 'unreachable', 'network']):
                        OllamaClient.unreachable_hosts.append(host)
                        failed_hosts_this_attempt.append(host)
                    # If it's just a model not found or other non-connection error, continue to next host
                    # but don't mark this host as completely unreachable
        
        # Only show summary error if no hosts were reachable and we actually tried some
        if failed_hosts_this_attempt and len(failed_hosts_this_attempt) == len([h.strip() for h in ollama_hosts if h.strip() and (not small_only_hosts or h.strip() not in small_only_hosts or is_small_model)]):
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"No reachable Ollama hosts found for model {model_key}", "yellow", prefix=prefix)
        
        return None, None

    @staticmethod
    def generate_response(
        chat: Chat | str,
        model_key: str = "phi3.5:3.8b",
        temperature: Optional[float] = None,
        silent_reason: str = ""
    ) -> Any:
        """
        Generates a response using the Ollama API, with support for tool calling.

        Args:
            chat (Chat | str): The chat object containing messages or a string prompt.
            model_key (str): The model identifier (e.g., "phi3.5:3.8b", "llama3.1").
            temperature (Optional[float]): The temperature setting for the model.
            silent_reason (str): If provided, suppresses output and shows this reason.

        Returns:
            Any: A stream object that yields response chunks.
            
        Raises:
            Exception: If there's an error generating the response, to be handled by the router.
        """
        # Convert string to Chat object if needed
        if isinstance(chat, str):
            chat_obj = Chat()
            chat_obj.add_message(Role.USER, chat)
            chat_inner = chat_obj
        else:
            chat_inner = chat
            
        # Store original model_key for error messages
        original_model_key = model_key
        
        # Determine if this is a small model by checking the LlmRouter's model definitions
        is_small_model = False
        try:
            # Import here to avoid circular imports
            from py_classes.cls_llm_router import Llm
            from py_classes.enum_ai_strengths import AIStrengths
            
            # Find the model in the available LLMs list
            available_llms = Llm.get_available_llms()
            matching_llm = next((llm for llm in available_llms if model_key in llm.model_key), None)
            
            if matching_llm:
                is_small_model = any(s == AIStrengths.SMALL for s in matching_llm.strengths)
            else:
                # Fallback to name-based detection if model not found in definitions
                is_small_model = any(size in model_key.lower() for size in ['1b', '2b', '3b', '4b', '7b', '8b']) or 'small' in model_key.lower()
        except ImportError:
            # Fallback to name-based detection if import fails
            is_small_model = any(size in model_key.lower() for size in ['1b', '2b', '3b', '4b', '7b', '8b']) or 'small' in model_key.lower()
        
        client: ollama.Client | None
        client, found_model_key = OllamaClient.get_valid_client(model_key, chat_inner, is_small_model=is_small_model)
        if not client:
            # Raise exception to be handled by the router instead of returning None
            error_msg = f"No valid host found for model {original_model_key}"
            raise Exception(error_msg)
        
        # Use the found model key (which might be slightly different than requested)
        model_key = found_model_key or original_model_key
            
        assert client is not None
        host: str = client._client.base_url.host

        # Create options dictionary with temperature
        options = {}
        if temperature is not None and temperature != 0:
            options["temperature"] = temperature

        # Log the start of generation (this is informational, not error handling)
        if silent_reason:
            temp_str = "" if temperature == 0 or temperature is None else f" at temperature {temperature}"
            prefix = chat_inner.get_debug_title_prefix() if hasattr(chat_inner, 'get_debug_title_prefix') else ""
            g.debug_log(f"Ollama-Api: {colored('<' + model_key + '>', 'green')} is using {colored('<' + host + '>', 'green')} for: {colored('<' + silent_reason + '>', 'yellow')}{temp_str}", force_print=True, with_title=False, prefix=prefix)
        else:
            temp_str = "" if temperature == 0 or temperature is None else f" at temperature {temperature}"
            prefix = chat_inner.get_debug_title_prefix() if hasattr(chat_inner, 'get_debug_title_prefix') else ""
            g.debug_log(f"Ollama-Api: {colored('<' + model_key + '>', 'green')} is using {colored('<' + host + '>', 'green')}{temp_str} to generate a response...", force_print=True, prefix=prefix)
        
        # Let any errors here bubble up to the router for centralized handling
        is_chat = isinstance(chat_inner, Chat)
        
        # Add timeout to prevent hanging on connection issues
        import signal
        import time
        
        def ollama_timeout_handler(signum, frame):
            raise Exception(f"Ollama client timeout after 30 seconds for model {model_key} on host {host}")
        
        # Set a 30-second timeout for the ollama client call
        signal.signal(signal.SIGALRM, ollama_timeout_handler)
        signal.alarm(30)
        
        try:
            if is_chat:
                result = client.chat(model=model_key, messages=chat_inner.to_ollama(), stream=True, keep_alive=1800, options=options)
            else:
                result = client.generate(model=model_key, prompt=chat_inner, stream=True, keep_alive=1800, options=options)
            signal.alarm(0)  # Clear the alarm if successful
            return result
        except Exception as e:
            signal.alarm(0)  # Clear the alarm on error
            # Check if this is an EOF or connection error that should mark the host as problematic
            error_str = str(e).lower()
            if any(issue in error_str for issue in ['eof', 'connection', 'timeout', 'refused', 'unreachable']):
                # Add this host+model combo to unreachable list temporarily
                problematic_identifier = f"{host}:{model_key}"
                if problematic_identifier not in OllamaClient.unreachable_hosts:
                    OllamaClient.unreachable_hosts.append(problematic_identifier)
            raise

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

        # Embedding models are typically small, so mark as small model
        client, model_key = OllamaClient.get_valid_client(model, chat, is_small_model=True)
        if not client:
            error_msg = f"No valid host found for model {model}"
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
            else:
                logger.error(error_msg)
            return None
            
        assert client is not None
        host: str = client._client.base_url.host

        try:
            if isinstance(text, str):
                # Single text case
                if chat:
                    prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                    g.debug_log(f"Ollama-Api: {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')} is generating embedding using {colored('<', 'green')}{colored(host, 'green')}{colored('>', 'green')}...", force_print=True, prefix=prefix)
                else:
                    print(f"Ollama-Api: {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')} is generating embedding using {colored('<', 'green')}{colored(host, 'green')}{colored('>', 'green')}...")
                
                response = client.embeddings(model=model, prompt=text, keep_alive=7200)
                return response["embedding"]
            else:
                # List of texts case
                if chat:
                    prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                    g.debug_log(f"Ollama-Api: {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')} is generating {len(text)} embeddings using {colored('<', 'green')}{colored(host, 'green')}{colored('>', 'green')}", force_print=True, prefix=prefix, end="")
                else:
                    print(f"Ollama-Api: {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')} is generating {len(text)} embeddings using {colored('<', 'green')}{colored(host, 'green')}{colored('>', 'green')}", end="")
                
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
            error_msg = f"Ollama-Api: Failed to generate embedding(s) using {colored('<', 'red')}{colored(host, 'red')}{colored('>', 'red')} with model {colored('<', 'red')}{colored(model, 'red')}{colored('>', 'red')}: {e}"
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
            else:
                print(colored(error_msg, "red"))
                logger.error(f"Ollama-Api: Failed to generate embedding(s) using <{host}> with model <{model}>: {e}")
            
            # Don't mark host as unreachable for embedding failures - could be model-specific
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