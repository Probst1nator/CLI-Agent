import json
import sys
from typing import Any, Dict, List, Optional, Tuple
import ollama
from termcolor import colored
from py_classes.cls_chat import Chat, Role
from py_classes.unified_interfaces import AIProviderInterface
import os
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime
from py_classes.globals import g
import logging
import time

logger = logging.getLogger(__name__)

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
            dict: A dictionary representation of the OllamaModel.
        """
        return asdict(self)

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
        Creates OllamaModelList from a JSON string.
        
        Args:
            json_str (str): A JSON string to parse.
        
        Returns:
            OllamaModelList: The parsed OllamaModelList instance.
        """
        data = json.loads(json_str)
        models = [
            OllamaModel(
                model=model_data["model"],
                modified_at=model_data["modified_at"],
                size=model_data["size"],
                digest=model_data["digest"]
            )
            for model_data in data["models"]
        ]
        return cls(models=models)

class OllamaClient(AIProviderInterface):
    """
    Implementation of the ChatClientInterface for the Ollama API.
    """
    reached_hosts: List[str] = []
    unreachable_hosts: List[str] = []
    current_host: Optional[str] = None
    # Host discovery cache with timestamps (host -> timestamp)
    _host_cache: Dict[str, Dict[str, float]] = {"reachable": {}, "unreachable": {}}
    _cache_ttl: float = 30.0  # Cache for 30 seconds
    
    @classmethod
    def reset_host_cache(cls):
        """Reset the host reachability cache to allow retrying all hosts."""
        cls.reached_hosts.clear()
        cls.unreachable_hosts.clear()
        cls._host_cache["reachable"].clear()
        cls._host_cache["unreachable"].clear()
        # Note: unreachable_hosts now also contains host:model combinations

    @classmethod
    def check_host_reachability(cls, host: str, chat: Optional[Chat] = None) -> bool:
        """
        Validates if a host is reachable using a socket connection with caching.
        
        Args:
            host (str): The hostname to validate.
            chat (Optional[Chat]): Chat object for debug printing with title.
        
        Returns:
            bool: True if the host is reachable, False otherwise.
        """
        current_time = time.time()
        
        # Check cache first
        if host in cls._host_cache["reachable"]:
            if current_time - cls._host_cache["reachable"][host] < cls._cache_ttl:
                return True
            else:
                # Cache expired, remove from cache
                del cls._host_cache["reachable"][host]
        
        if host in cls._host_cache["unreachable"]:
            if current_time - cls._host_cache["unreachable"][host] < cls._cache_ttl:
                return False
            else:
                # Cache expired, remove from cache
                del cls._host_cache["unreachable"][host]
        
        try:
            hostname, port_str = host.split(':') if ':' in host else (host, '11434')
            port = int(port_str)
            
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Ollama-Api: Checking host <{host}>...", "green", force_print=True, prefix=prefix)
            else:
                logging.debug(f"Ollama-Api: Checking host <{host}>...")
                
            with socket.create_connection((hostname, port), timeout=1): # 1-second timeout
                # Cache successful result
                cls._host_cache["reachable"][host] = current_time
                return True
        except (socket.timeout, socket.error, ValueError):
            # Cache failure result
            cls._host_cache["unreachable"][host] = current_time
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Ollama-Api: Host <{host}> is unreachable", "red", is_error=True, prefix=prefix)
            else:
                logging.debug(f"Ollama-Api: Host <{host}> is unreachable")
            return False

    @staticmethod
    def get_valid_client(model_key: str, chat: Optional[Chat] = None, auto_download: bool = True, is_small_model: bool = False) -> Tuple[ollama.Client|None, str, str]:
        """
        Returns a valid client for the given model, pulling the model if necessary on auto-download hosts.
        
        Args:
            model_key (str): The model to find a valid client for.
            chat (Optional[Chat]): Chat object for debug printing with title.
            auto_download (bool): Whether to automatically download models if not found.
            is_small_model (bool): Whether this is a small/fast model.
        
        Returns:
            Tuple[Optional[ollama.Client], str, str]: [A valid client or None, found model_key, host].
        """
        # Get hosts from comma-separated environment variables
        ollama_host_env = os.getenv("OLLAMA_HOST", "")
        if ollama_host_env:
            ollama_hosts = ollama_host_env.split(",")
        else:
            # Default to localhost if no OLLAMA_HOST is set
            ollama_hosts = ["localhost"]
        
        # Remove the localhost from the list if explicitly configured to force remote
        force_local_remote_host = os.getenv("FORCE_REMOTE_HOST_FOR_HOSTNAME", "")
        if socket.gethostname() in force_local_remote_host:
            try:
                ollama_hosts.remove("localhost")
                ollama_hosts.remove(socket.gethostbyname(socket.gethostname()))
            except Exception:
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
                    
                    # Handle both dict and ollama._types.ListResponse
                    if hasattr(response, 'models'):
                        models_data = response.models
                    elif isinstance(response, dict) and 'models' in response:
                        models_data = response.get('models', [])
                    else:
                        logger.warning(f"Ollama host {host} returned unexpected response format: {type(response)}")
                        continue # Skip to the next host

                    logger.debug("=== START MODEL PROCESSING ===")

                    # Convert to dict for JSON serialization, compatible with OllamaModelList
                    response_dict = {"models": []}
                    for model_data in models_data:
                        model_name = model_data.get('model') or model_data.get('name') or ''
                        modified_at = model_data.get('modified_at')
                        
                        if hasattr(modified_at, 'isoformat'):
                            modified_at_str = modified_at.isoformat()
                        else:
                            modified_at_str = modified_at or datetime.now().isoformat()

                        response_dict["models"].append({
                            "model": model_name,
                            "modified_at": modified_at_str,
                            "size": model_data.get('size') or 0,
                            "digest": model_data.get('digest') or ""
                        })
                    
                    serialized = json.dumps(response_dict)
                    model_list = OllamaModelList.from_json(serialized)
                    
                    # Sort models by modification date (newest first)
                    model_list.models.sort(key=lambda m: m.modified_at, reverse=True)
                    
                    # Look for model name in all available models
                    logger.debug(f"\nSearching for model key: {model_key}")
                    found_model_key = next((
                        model.model 
                        for model in model_list.models 
                        if model.model and model_key.lower() in model.model.lower()
                    ), None)
                    logger.debug(f"Found model key: {found_model_key}")
                    logger.debug("=== END MODEL PROCESSING ===\n")
                    
                    if found_model_key:
                        # Set current_host for logging consistency
                        OllamaClient.current_host = host
                        return client, found_model_key, host
                    elif auto_download and host in auto_download_hosts:
                        if chat:
                            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                            g.debug_log(f"{host} is pulling {model_key}...", "yellow", force_print=True, prefix=prefix)
                        else:
                            logging.info(colored(f"{host} is pulling {model_key}...", "yellow"))
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
                            # Set current_host for logging consistency
                            OllamaClient.current_host = host
                            return client, model_key, host
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
        
        return None, None, None

    @staticmethod
    def generate_response(
        chat: Chat | str,
        model_key: str = "phi3.5:3.8b",
        temperature: Optional[float] = None,
        silent_reason: str = "",
        thinking_budget: Optional[int] = None,
        auto_download: bool = True
    ) -> Any:
        """
        Generates a response using the Ollama API, with support for tool calling.

        Args:
            chat (Chat | str): The chat object containing messages or a string prompt.
            model_key (str): The model identifier (e.g., "phi3.5:3.8b", "llama3.1").
            temperature (Optional[float]): The temperature setting for the model.
            silent_reason (str): If provided, suppresses output and shows this reason.
            thinking_budget (Optional[int]): Not used in Ollama, kept for compatibility.
            auto_download (bool): Whether to automatically download models if not found.

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
        
        # Determine if this is a small model for routing
        is_small_model = any(size in model_key.lower() for size in ['1b', '2b', '3b', '4b', 'small'])
        
        # Get a valid client for this model
        client, found_model_key, host = OllamaClient.get_valid_client(model_key, chat_inner, auto_download, is_small_model)
        if not client:
            raise Exception("No valid host found for Ollama models")
        
        # Store the host information on the class for logging
        OllamaClient.current_host = host
        
        # Use the found model key (which might be different from requested if we found a partial match)
        model_key = found_model_key
        
        # Default temperature if not specified
        if temperature is None:
            temperature = 0.0
            
        # Convert chat messages to Ollama format - get the messages first
        openai_messages = chat_inner.to_openai()
        
        messages = []
        for message_dict in openai_messages:
            role = message_dict["role"]
            content = message_dict["content"]
            if role == "system":
                messages.append({"role": "system", "content": content})
            elif role == "user":
                # Handle images in user messages
                if hasattr(chat_inner, 'base64_images') and chat_inner.base64_images:
                    message = {"role": "user", "content": content, "images": chat_inner.base64_images}
                    chat_inner.base64_images = []  # Clear after use
                else:
                    message = {"role": "user", "content": content}
                messages.append(message)
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
        
        try:
            
            # Make the streaming API call
            stream = client.chat(
                model=model_key,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                }
            )
            
            # Verify stream is valid before returning
            if stream is None:
                host = "unknown host"
                if hasattr(client, '_client') and hasattr(client._client, 'base_url'):
                    host = client._client.base_url.host
                raise Exception(f"Ollama API at {host} returned None stream for model {model_key}")
                
            return stream
        except ollama.ResponseError as e:
            # This handles specific API errors from Ollama, like model not found.
            # We create a more descriptive error message to be handled by the router.
            host = "unknown host"
            if hasattr(client, '_client') and hasattr(client._client, 'base_url'):
                host = client._client.base_url.host
            error_message = f"Ollama API error from host {host}: {e.error}"
            raise Exception(error_message) from e
        except Exception as e:
            # For other errors (like connection errors), let the router classify them.
            # Add host info for better debugging
            host = "unknown host"
            try:
                if hasattr(client, '_client') and hasattr(client._client, 'base_url'):
                    host = client._client.base_url.host
            except:
                pass
            
            # Re-raise with host context
            if "host" not in str(e).lower():
                raise Exception(f"Ollama error from host {host}: {str(e)}") from e
            else:
                raise e

    @staticmethod
    def get_downloaded_models(host: str = "localhost") -> List[Dict[str, Any]]:
        """
        Get list of downloaded models from specified Ollama host.
        
        Args:
            host (str): The Ollama host to query
            
        Returns:
            List[Dict[str, Any]]: List of model information dictionaries
        """
        try:
            client = ollama.Client(host=f'http://{host}:11434')
            response = client.list()
            
            # Handle new dictionary-based response from ollama-python >= 0.2.0
            if isinstance(response, dict) and 'models' in response:
                models = []
                for model_data in response['models']:
                    model_info = {
                        'name': model_data.get('name', ''),
                        'size': model_data.get('size', 0),
                        'modified_at': model_data.get('modified_at', None)
                    }
                    models.append(model_info)
                # Sort by modification date (newest first), handling None
                models.sort(key=lambda m: m.get('modified_at') or datetime.min, reverse=True)
                return models
            
            # Fallback for older object-based response
            elif hasattr(response, 'models'):
                models = []
                for model in response.models:
                    model_info = {
                        'name': getattr(model, 'name', '') or getattr(model, 'model', ''),
                        'size': getattr(model, 'size', 0),
                        'modified_at': getattr(model, 'modified_at', None)
                    }
                    models.append(model_info)
                # Sort by modification date (newest first), handling None
                models.sort(key=lambda m: m.get('modified_at') or datetime.min, reverse=True)
                return models
            
            else:
                return []
            
        except Exception:
            return []

    @staticmethod
    def get_comprehensive_model_status(hosts: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive model status across multiple hosts.
        
        Args:
            hosts (List[str]): List of hosts to check
            
        Returns:
            Dict[str, Dict[str, Any]]: Model status information
        """
        model_status = {}
        
        for host in hosts:
            try:
                downloaded_models = OllamaClient.get_downloaded_models(host)
                for model_info in downloaded_models:
                    model_name = model_info['name']
                    base_name = model_name.split(':')[0]  # Remove tag
                    
                    if base_name not in model_status:
                        model_status[base_name] = {
                            'downloaded': False,
                            'hosts': []
                        }
                    
                    model_status[base_name]['downloaded'] = True
                    model_status[base_name]['hosts'].append({
                        'host': host,
                        'full_name': model_name,
                        'size': model_info.get('size', 0)
                    })
            
            except Exception:
                continue
                
        return model_status

    @staticmethod
    def get_model_context_length(model_key: str, host: str = "localhost") -> Optional[int]:
        """
        Get context length for a specific model.
        
        Args:
            model_key (str): The model key to check
            host (str): The host to query
            
        Returns:
            Optional[int]: Context length if available, None otherwise
        """
        try:
            client = ollama.Client(host=f'http://{host}:11434')
            
            # Try to get model info - this might not work for all Ollama versions
            try:
                model_info = client.show(model_key)
                # Context length might be in different places depending on model
                # This is a best-effort attempt
                if hasattr(model_info, 'parameters') and model_info.parameters:
                    params = model_info.parameters
                    if 'num_ctx' in params:
                        return int(params['num_ctx'])
            except Exception:
                pass
            
            # Fallback to reasonable defaults based on model name
            model_lower = model_key.lower()
            if any(x in model_lower for x in ['large', '70b', '72b']):
                return 8192
            elif any(x in model_lower for x in ['medium', '13b', '14b', '32b', '34b']):
                return 32768
            else:
                return 128000  # Most modern models support this
                
        except Exception:
            return None

    @staticmethod
    def generate_embedding(text: str, model: str = "bge-m3", **kwargs) -> Optional[List[float]]:
        """
        Generate embeddings for text using Ollama embedding models.
        
        Args:
            text (str): The text to generate embeddings for
            model (str): The embedding model to use
            
        Returns:
            Optional[List[float]]: The embedding vector or None if failed
        """
        try:
            # Get a valid client for this model
            client, found_model_key, _ = OllamaClient.get_valid_client(model, None, True, False)
            if not client:
                return None
                
            # Use the found model key
            model_key = found_model_key
            
            # Generate embedding using the newer embed method
            response = client.embed(model=model_key, input=text)
            
            # Extract embedding from response
            if hasattr(response, 'embeddings') and response.embeddings:
                # Return the first embedding (since we're only passing one input)
                return response.embeddings[0] if response.embeddings else None
            elif isinstance(response, dict) and 'embeddings' in response:
                # Return the first embedding from the list
                embeddings = response['embeddings']
                return embeddings[0] if embeddings else None
            else:
                return None
                
        except Exception as e:
            # Fail silently to allow fallback to other embedding methods
            logger.debug(f"Failed to generate embedding with Ollama model {model}: {e}")
            return None


def test_ollama_functionality():
    """
    Test function to manually verify Ollama functionality.
    Run this script directly to test the Ollama interface.
    """
    print(colored("=== Testing Ollama Interface ===", "cyan", attrs=["bold"]))
    
    # Test 1: Check host reachability
    print(colored("\n1. Testing host reachability...", "yellow"))
    from py_classes.globals import g
    test_hosts = g.DEFAULT_OLLAMA_HOSTS
    for host in test_hosts:
        reachable = OllamaClient.check_host_reachability(host)
        status = colored("✓ Reachable", "green") if reachable else colored("✗ Unreachable", "red")
        print(f"   {host}: {status}")
    
    # Test 2: List available models
    print(colored("\n2. Testing model listing...", "yellow"))
    try:
        test_models = ["qwen3-coder:latest", "bge-m3", "phi3.5:3.8b"]
        for model in test_models:
            client, found_model, _ = OllamaClient.get_valid_client(model, None, False, False)
            if client and found_model:
                print(f"   {model}: {colored('✓ Available as ' + found_model, 'green')}")
            else:
                print(f"   {model}: {colored('✗ Not available', 'red')}")
    except Exception as e:
        print(f"   {colored('Error listing models: ' + str(e), 'red')}")
    
    # Test 3: Test embedding generation
    print(colored("\n3. Testing embedding generation...", "yellow"))
    test_text = "Hello world, this is a test"
    try:
        embedding = OllamaClient.generate_embedding(test_text, "bge-m3")
        if embedding:
            print(f"   bge-m3: {colored('✓ Generated embedding of length ' + str(len(embedding)), 'green')}")
            print(f"   Sample values: {embedding[:5]}...")
        else:
            print(f"   bge-m3: {colored('✗ Failed to generate embedding', 'red')}")
    except Exception as e:
        print(f"   bge-m3: {colored('✗ Error: ' + str(e), 'red')}")
    
    # Test 4: Test response generation
    print(colored("\n4. Testing response generation...", "yellow"))
    test_prompt = "What is 2+2? Please respond briefly."
    test_models = ["qwen3-coder:latest", "phi3.5:3.8b"]
    
    for model in test_models:
        try:
            print(f"   Testing {model}...")
            stream = OllamaClient.generate_response(test_prompt, model, temperature=0.0)
            if stream:
                print(f"   {model}: {colored('✓ Stream created successfully', 'green')}")
                
                # Try to read a few chunks from the stream
                response_text = ""
                chunk_count = 0
                try:
                    for chunk in stream:
                        if hasattr(chunk, 'message') and chunk.message.content:
                            response_text += chunk.message.content
                            chunk_count += 1
                            if chunk_count > 10:  # Limit to prevent too much output
                                break
                    
                    if response_text:
                        print(f"     Response preview: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
                    else:
                        print(f"     {colored('⚠ Stream created but no content received', 'yellow')}")
                        
                except Exception as stream_e:
                    print(f"     {colored('⚠ Error reading stream: ' + str(stream_e), 'yellow')}")
            else:
                print(f"   {model}: {colored('✗ Failed to create stream', 'red')}")
                
        except Exception as e:
            print(f"   {model}: {colored('✗ Error: ' + str(e), 'red')}")
    
    # Test 5: Test model info
    print(colored("\n5. Testing model information...", "yellow"))
    try:
        status = OllamaClient.get_comprehensive_model_status(["localhost"])
        if status:
            print(f"   Found {len(status)} model types:")
            for model_name, info in list(status.items())[:3]:  # Show first 3
                downloaded = colored("✓ Downloaded", "green") if info['downloaded'] else colored("✗ Not downloaded", "red")
                print(f"     {model_name}: {downloaded}")
        else:
            print(f"   {colored('No model status available', 'yellow')}")
    except Exception as e:
        print(f"   {colored('Error getting model status: ' + str(e), 'red')}")
    
    print(colored("\n=== Test Complete ===", "cyan", attrs=["bold"]))


if __name__ == "__main__":
    test_ollama_functionality()