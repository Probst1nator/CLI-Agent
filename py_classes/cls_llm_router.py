from collections.abc import Callable
import hashlib
import json
import os
from random import shuffle
import shutil
import time
from typing import Dict, List, Optional, Set, Any, Union, Iterator
from termcolor import colored
from py_classes.ai_providers.cls_human_as_interface import HumanAPI
from py_classes.ai_providers.cls_nvidia_interface import NvidiaAPI
from py_classes.cls_custom_coloring import CustomColoring
from py_classes.cls_chat import Chat, Role
from enum import Enum
from py_classes.ai_providers.cls_anthropic_interface import AnthropicAPI
from py_classes.unified_interfaces import AIProviderInterface
from py_classes.ai_providers.cls_groq_interface import GroqAPI, TimeoutException, RateLimitException
from py_classes.ai_providers.cls_ollama_interface import OllamaClient
from py_classes.ai_providers.cls_openai_interface import OpenAIAPI
from py_classes.globals import g
from py_classes.cls_debug_utils import get_debug_title_prefix, DEBUG_TITLE_FORMAT
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

# Custom exception for user interruption
class UserInterruptedException(Exception):
    """Exception raised when the user interrupts model generation (e.g., with Ctrl+C)."""
    pass

class AIStrengths(Enum):
    """Enum class to represent AI model strengths."""
    UNCENSORED = 6
    REASONING = 5
    CODE = 4
    GUARD = 3
    GENERAL = 2
    FAST = 1
    LOCAL = 8
    VISION = 9

class Llm:
    """
    Class representing a Language Model (LLM) with its properties and capabilities.
    """

    def __init__(
        self, 
        provider: AIProviderInterface, 
        model_key: str, 
        pricing_in_dollar_per_1M_tokens: Optional[float], 
        context_window: int, 
        strength: List[AIStrengths] = [], 
    ):
        """
        Initialize an LLM instance.

        Args:
            provider (ChatClientInterface): The chat client interface for the LLM.
            model_key (str): Unique identifier for the model.
            pricing_in_dollar_per_1M_tokens (Optional[float]): Pricing information.
            context_window (int): The context window size of the model.
            strength (AIStrengths): The strength category of the model.
        """
        self.provider = provider
        self.model_key = model_key
        self.pricing_in_dollar_per_1M_tokens = pricing_in_dollar_per_1M_tokens
        self.context_window = context_window
        self.strengths = strength
    
    def __str__(self) -> str:
        """
        Returns a string representation of the LLM.
        
        Returns:
            str: A formatted string with the LLM's attributes
        """
        provider_name = self.provider.__class__.__name__
        pricing = f"${self.pricing_in_dollar_per_1M_tokens}/1M tokens" if self.pricing_in_dollar_per_1M_tokens else "Free"
        strengths = ", ".join(s.name for s in self.strengths) if self.strengths else "None"
        
        return f"LLM(provider={provider_name}, model={self.model_key}, pricing={pricing}, " \
               f"context_window={self.context_window}, strengths=[{strengths}])"
    
    @property
    def local(self) -> bool:
        """Returns whether the model is available locally."""
        return any(s == AIStrengths.LOCAL for s in self.strengths)
    
    @property
    def has_vision(self) -> bool:
        """Returns whether the model has vision capabilities."""
        return any(s == AIStrengths.VISION for s in self.strengths)
    
    @classmethod
    def get_available_llms(cls, exclude_guards: bool = False) -> List["Llm"]:
        """
        Get the list of available LLMs.
        
        Returns:
            List[Llm]: A list of Llm instances representing the available models.
        """
        # Define and return a list of available LLM instances
        llms = [
            # Llm(HumanAPI(), "human", None, 131072, [AIStrengths.STRONG, AIStrengths.LOCAL, AIStrengths.VISION]), # For testing
            # Llm(GroqAPI(), "llama-3.3-70b-specdec", None, 8192, [AIStrengths.GENERAL, AIStrengths.CODE]),
            Llm(GroqAPI(), "llama-3.3-70b-versatile", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE]),
            
            Llm(GroqAPI(), "qwen-2.5-32b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE]),
            Llm(GroqAPI(), "qwen-2.5-coder-32b", None, 128000, [AIStrengths.CODE]),
            
            Llm(GroqAPI(), "deepseek-r1-distill-llama-70b", None, 128000, [AIStrengths.GENERAL, AIStrengths.REASONING]),
            Llm(GroqAPI(), "deepseek-r1-distill-qwen-32b", None, 128000, [AIStrengths.GENERAL, AIStrengths.REASONING]),
            Llm(GroqAPI(), "qwen-qwq-32b", None, 128000, [AIStrengths.GENERAL, AIStrengths.REASONING]),
            
            Llm(GroqAPI(), "llama-3.2-90b-vision-preview", None, 32768, [AIStrengths.GENERAL, AIStrengths.VISION]),
            Llm(GroqAPI(), "llama-3.1-8b-instant", None, 128000, [AIStrengths.FAST, AIStrengths.CODE]),
            
            Llm(AnthropicAPI(), "claude-3-7-sonnet-20250219", 3, 200000, [AIStrengths.GENERAL, AIStrengths.CODE]),
            Llm(AnthropicAPI(), "claude-3-5-haiku-20241022", 0.8, 200000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.FAST]),
            
            Llm(OllamaClient(), "mistral-nemo:12b", None, 128000, [AIStrengths.GENERAL, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "cogito:32b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "cogito:14b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "cogito:8b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "cogito:3b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.FAST]),
            Llm(OllamaClient(), "gemma3:27b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.VISION]),
            Llm(OllamaClient(), "gemma3:12b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.VISION]),
            Llm(OllamaClient(), "gemma3:4b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.FAST, AIStrengths.VISION]),
            
            Llm(OllamaClient(), "deepcoder:14b", None, 131072, [AIStrengths.CODE, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "deepcoder:1.5b", None, 128000, [AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.FAST]),
            Llm(OllamaClient(), "Captain-Eris_Violet-GRPO-v0.420.i1-Q4_K_M:latest", None, 128000, [AIStrengths.GENERAL, AIStrengths.UNCENSORED, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "L3-8B-Stheno-v3.2-Q4_K_M-imat:latest", None, 128000, [AIStrengths.GENERAL, AIStrengths.UNCENSORED, AIStrengths.LOCAL]),
            
            # Guard models
            Llm(GroqAPI(), "llama-guard-3-8b", None, 8192, [AIStrengths.GUARD]),
            Llm(OllamaClient(), "llama-guard3:8b", None, 4096, [AIStrengths.GUARD, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "shieldgemma:2b", None, 4096, [AIStrengths.GUARD, AIStrengths.LOCAL, AIStrengths.FAST]),
        ]
        if exclude_guards:
            llms = [llm for llm in llms if not any(s == AIStrengths.GUARD for s in llm.strengths)]
        return llms
        
        


class LlmRouter:
    """
    Singleton class for routing and managing LLM requests.
    """

    _instance: Optional["LlmRouter"] = None
    call_counter: int = 0
    last_used_model: str = ""
    _model_limits: Dict[str, int] = {}
    _model_limits_loaded: bool = False
    
    def __new__(cls, *args, **kwargs) -> "LlmRouter":
        """
        Create a new instance of LlmRouter if it doesn't exist, otherwise return the existing instance.
        """
        if cls._instance is None:
            cls._instance = super(LlmRouter, cls).__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self) -> None:
        """
        Initialize the LlmRouter instance.
        """
        # Set up cache directory and file path
        self.cache_file_path = f"{g.PROJ_PERSISTENT_STORAGE_PATH}/llm_cache.json"
        
        # Load cache and initialize retry models and failed models set
        self.cache = self._load_cache()
        self.retry_models = Llm.get_available_llms()
        self._load_dynamic_model_limits()
        self.failed_models: Set[str] = set()
    
    def _load_dynamic_model_limits(self) -> None:
        """Load model limits from disk if not already loaded."""
        if not self._model_limits_loaded:
            try:
                if os.path.exists(g.MODEL_TOKEN_LIMITS_PATH):
                    with open(g.MODEL_TOKEN_LIMITS_PATH, 'r') as f:
                        self._model_limits = json.load(f)
                self._model_limits_loaded = True
            except Exception as e:
                logger.error(f"Failed to load model token limits: {e}")
                self._model_limits = {}
                self._model_limits_loaded = True

    def _generate_hash(self, model_key: str, temperature: str, prompt: str, images: List[str]) -> str:
        """
        Generate a hash for caching based on model, temperature, prompt, and images.
        
        Args:
            model_key (str): Model identifier.
            temperature (str): Temperature setting for the model.
            prompt (str): The prompt string.
            images (List[str]): List of image encodings.

        Returns:
            str: The generated hash string.
        """
        # Combine inputs and generate SHA256 hash
        hash_input = f"{model_key}:{temperature}:{prompt}{':'.join(images)}".encode()
        return hashlib.sha256(hash_input).hexdigest()

    def _load_cache(self) -> Dict[str, str]:
        if not os.path.exists(self.cache_file_path):
            return {}
        try:
            with open(self.cache_file_path, "r") as json_file:
                return json.load(json_file)
        except json.JSONDecodeError:
            print(colored("Failed to load cache file: Invalid JSON format", "red"))
            print("Creating a new cache file...")
            return {}
        except Exception as e:
            print(colored(f"Unexpected error loading cache: {e}", "red"))
            return {}

    def _get_cached_completion(self, model_key: str, temperature: str, key: str, images: List[str]) -> Optional[str]:
        """
        Retrieve a cached completion if available.
        
        Args:
            model_key (str): Model identifier.
            temperature (str): Temperature setting for the model.
            chat (Chat): The chat prompt.
            images (List[str]): List of image encodings.

        Returns:
            Optional[str]: The cached completion string if available, otherwise None.
        """
        # Generate cache key and return cached completion if it exists
        cache_key = self._generate_hash(model_key, temperature, key, images)
        return self.cache.get(cache_key)

    def _update_cache(self, model_key: str, temperature: str, key: str, images: List[str], completion: str) -> None:
        """
        Update the cache with a new completion.
        """
        # Generate cache key
        cache_key = self._generate_hash(model_key, temperature, key, images)
        
        # Update the in-memory cache
        self.cache[cache_key] = completion
        
        try:
            # Read existing cache from file
            if os.path.exists(self.cache_file_path):
                with open(self.cache_file_path, "r") as json_file:
                    existing_cache = json.load(json_file)
            else:
                existing_cache = {}
            
            # Update the existing cache with the new entry
            existing_cache.update({cache_key: completion})
            
            # Write the updated cache back to the file
            with open(self.cache_file_path, "w") as json_file:
                json.dump(existing_cache, json_file, indent=4, ensure_ascii=False)
        except json.JSONDecodeError as je:
            print(colored(f"Failed to parse existing cache: {je}", "red"))
            print("Creating a new cache file...")
            with open(self.cache_file_path, "w") as json_file:
                json.dump({cache_key: completion}, json_file, indent=4, ensure_ascii=False)
        except Exception as e:
            print(colored(f"Failed to update cache: {e}", "red"))
            print("Continuing without updating cache file...")

    def model_capable_check(self, model: Llm, chat: Chat, strength: List[AIStrengths], local: bool, force_free: bool = False, has_vision: bool = False, allow_general: bool = True) -> bool:
        """
        Check if a model is capable of handling the given constraints.
        
        Args:
            model (Llm): The model to check.
            chat (Chat): The chat to process.
            strength (List[AIStrengths]): The required strengths.
            local (bool): Whether the model should be local.
            force_free (bool): Whether to force free models only.
            has_vision (bool): Whether vision capability is required.
            allow_general (bool): Whether to allow GENERAL strength as a fallback.

        Returns:
            bool: True if the model is capable, False otherwise.
        """
        if force_free and model.pricing_in_dollar_per_1M_tokens is not None:
            return False
        if has_vision and not model.has_vision:
            return False
            
        if model.model_key in self._model_limits:
            token_limit = self._model_limits[model.model_key]
            if len(chat) >= token_limit:
                return False
        
        if model.context_window < len(chat):
            return False
        if strength and model.strengths:
            # Check if ALL of the required strengths are included in the model's strengths
            if not all(s.value in [ms.value for ms in model.strengths] for s in strength):
                # Only check for GENERAL if allowed
                if allow_general:
                    return any(s.value == AIStrengths.GENERAL.value for s in model.strengths)
                return False
        if local != model.local:
            return False
        return True

    @classmethod
    def get_models(cls, preferred_models: List[str] = [], strength: List[AIStrengths] = [], chat: Chat = Chat(), force_local: bool = False, force_free: bool = False, has_vision: bool = False) -> List[Llm]:
        """
        Get a list of available models based on the given constraints.
        
        Args:
            preferred_models (List[str]): List of preferred model keys.
            strength (List[AIStrengths]): The required strengths of the model.
            chat (Chat): The chat which the model will be processing.
            force_local (bool): Whether to force local models only.
            force_free (bool): Whether to force free models only.
            has_vision (bool): Whether to require models with vision capability.

        Returns:
            List[Llm]: A list of available Llm instances that meet the specified criteria.
        """
        instance = cls()
        available_models: List[Llm] = []

        # First try to find models with exact capability matches
        for model_key in preferred_models:
            if model_key and model_key not in instance.failed_models:
                model = next((model for model in instance.retry_models if model_key in model.model_key), None)
                if model and instance.model_capable_check(model, chat, strength, model.local, force_free, has_vision, allow_general=False):
                    available_models.append(model)
        
        # If no preferred models with exact capabilities, check all models
        if not available_models:
            for model in instance.retry_models:
                if model.model_key not in instance.failed_models and not model.model_key in [model.model_key for model in available_models]:
                    if (not force_local or model.local) and instance.model_capable_check(model, chat, strength, model.local, force_free, has_vision, allow_general=False):
                        available_models.append(model)
        
        # If still no models found, try again allowing GENERAL capability
        if not available_models and strength:
            # First check preferred models
            for model_key in preferred_models:
                if model_key and model_key not in instance.failed_models:
                    model = next((model for model in instance.retry_models if model_key in model.model_key), None)
                    if model and instance.model_capable_check(model, chat, strength, model.local, force_free, has_vision, allow_general=True):
                        available_models.append(model)
            
            # Then check all models
            if not available_models:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and not model.model_key in [model.model_key for model in available_models]:
                        if (not force_local or model.local) and instance.model_capable_check(model, chat, strength, model.local, force_free, has_vision, allow_general=True):
                            available_models.append(model)

        return available_models

    @classmethod
    def get_model(cls, preferred_models: List[str] = [], strength: List[AIStrengths] = [], chat: Chat = Chat(), force_local: bool = False, force_free: bool = False, has_vision: bool = False, force_preferred_model: bool = False) -> Optional[Llm]:
        """
        Route to the next available model based on the given constraints.
        
        Args:
            preferred_models (List[str]): List of preferred model keys.
            strength (List[AIStrengths]): The required strengths of the model.
            chat (Chat): The chat which the model will be processing.
            force_local (bool): Whether to force local models only.
            force_free (bool): Whether to force free models only.
            has_vision (bool): Whether to require models with vision capability.
            force_preferred_model (bool): Whether to only consider preferred models.

        Returns:
            Optional[Llm]: The next available Llm instance if available, otherwise None.
        """
        instance = cls()
        if (force_local):
            force_fast_hosts = os.getenv("OLLAMA_HOST_FORCE_FAST_MODELS", "").split(",")
            force_fast_hosts = [h.strip() for h in force_fast_hosts if h.strip()]
        else:
            force_fast_hosts = []
        
        # Debug print for large token counts
        if (len(chat) > 4000 and not force_free and not force_local):
            print(colored("DEBUG: len(chat) returned: " + str(len(chat)), "yellow"))
        
        # Try models in order of preference
        candidates = []
        
        # First try to find preferred model with exact capabilities
        for model_key in preferred_models:
            if (model_key not in instance.failed_models) and model_key:
                model = next((model for model in instance.retry_models if model_key in model.model_key and (force_local == False or force_local == model.local) and (has_vision == False or has_vision == model.has_vision)), None)
                if model and instance.model_capable_check(model, chat, strength, model.local, force_free, has_vision, allow_general=False):
                    candidates.append(model)

        # If no preferred candidates and force_preferred_model is True
        if not candidates and force_preferred_model:
            if force_local:
                # return a dummy model to force Ollama to download it
                return Llm(OllamaClient(), preferred_models[0], 0, 8192, [AIStrengths.GENERAL, AIStrengths.LOCAL])
            print(colored(f"Could not find preferred model {preferred_models[0]}", "red"))
            return None
        
        # Continue gathering candidates from other models if needed
        if not candidates and not force_preferred_model:
            # Search online models by exact capability next
            if not force_local:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and not model.local:
                        if instance.model_capable_check(model, chat, strength, local=False, force_free=force_free, has_vision=has_vision, allow_general=False):
                            candidates.append(model)
                
                # Add online models with GENERAL capability
                if not candidates and strength:
                    for model in instance.retry_models:
                        if model.model_key not in instance.failed_models and not model.local:
                            if instance.model_capable_check(model, chat, strength, local=False, force_free=force_free, has_vision=has_vision, allow_general=True):
                                candidates.append(model)

            # Add local models by exact capability
            if not candidates or force_local:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and model.local:
                        if instance.model_capable_check(model, chat, strength, local=True, force_free=force_free, has_vision=has_vision, allow_general=False):
                            candidates.append(model)
            
            # Add local models with GENERAL capability
            if not candidates and strength:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and model.local:
                        if instance.model_capable_check(model, chat, strength, local=True, force_free=force_free, has_vision=has_vision, allow_general=True):
                            candidates.append(model)
            
            # Last resort: try with empty chat to ignore context length
            if not candidates:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and model.local:
                        if instance.model_capable_check(model, Chat(), strength, local=True, force_free=force_free, has_vision=has_vision, allow_general=True):
                            candidates.append(model)
        
        # Now, check Ollama candidates against force_fast_hosts
        if force_fast_hosts:
            valid_candidates = []
            
            for model in candidates:
                # Skip non-Ollama models
                if not isinstance(model.provider, OllamaClient):
                    valid_candidates.append(model)
                    continue
                    
                # Skip fast models (they're always valid)
                if any(s == AIStrengths.FAST for s in model.strength):
                    valid_candidates.append(model)
                    continue
                    
                # For non-fast Ollama models, check which host would be used
                try:
                    client, _ = model.provider.get_valid_client(model.model_key, None, False)
                    if client:
                        host = client._client.base_url.host
                        # If host is not in force_fast_hosts, it's valid
                        if host not in force_fast_hosts:
                            valid_candidates.append(model)
                        else:
                            print(colored(f"Skipping model {model.model_key} because host {host} requires FAST models", "yellow"))
                except Exception as e:
                    # If we can't determine the host, be conservative and skip
                    print(colored(f"Error checking host for {model.model_key}: {e}", "red"))
                    
            # Replace candidates with valid ones
            candidates = valid_candidates
        
        # Return the first valid candidate
        return candidates[0] if candidates else None
    
    @classmethod
    def _process_stream(
        cls,
        stream: Union[Iterator[Dict[str, Any]], Iterator[str], Any],
        debug_print: Callable,
        token_keeper: CustomColoring,
        hidden_reason: str,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Process a stream of tokens from any provider.
        
        Args:
            stream (Union[Iterator[Dict[str, Any]], Iterator[str], Any]): The stream object from the provider
            debug_print (Callable): Function to print debug messages
            token_keeper (CustomColoring): Token coloring utility
            hidden_reason (str): Reason for hidden mode
            callback (Optional[Callable]): Callback function for each token
            
        Returns:
            str: The full response string
        """
        full_response = ""
        
        # Handle different stream types
        if hasattr(stream, 'text_stream'):  # Anthropic
            for token in stream.text_stream:
                if token:
                    full_response += token
                    if not hidden_reason:
                        debug_print(token_keeper.apply_color(token), end="", with_title=False)
                    if callback is not None:
                        if callback(token):
                            return full_response
        elif hasattr(stream, 'choices'):  # OpenAI/NVIDIA
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, 'content'):
                    token = chunk.choices[0].delta.content
                    if token:
                        full_response += token
                        if not hidden_reason:
                            debug_print(token_keeper.apply_color(token), end="", with_title=False)
                        if callback is not None:
                            if callback(token):
                                return full_response
        else:  # Ollama/Groq
            for chunk in stream:
                if hasattr(chunk, 'choices'):  # Groq ChatCompletionChunk
                    if hasattr(chunk.choices[0].delta, 'content'):
                        token = chunk.choices[0].delta.content
                    else:
                        continue
                elif isinstance(chunk, dict):  # Ollama dictionary chunks
                    token = chunk.get('message', {}).get('content', '') or chunk.get('response', '')
                elif hasattr(chunk, 'message'):  # Ollama response object
                    if hasattr(chunk.message, 'content'):
                        token = chunk.message.content
                    else:
                        continue
                else:
                    token = str(chunk)
                if token:
                    full_response += token
                    if not hidden_reason:
                        debug_print(token_keeper.apply_color(token), end="", with_title=False)
                    if callback is not None:
                        if callback(token):
                            return full_response
        
        if not full_response.endswith("\n"):
            print()
            
        return full_response

    @classmethod
    def _process_cached_response(
        cls,
        cached_completion: str,
        model: Llm,
        debug_print: Callable,
        tooling: CustomColoring,
        hidden_reason: str,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Process a cached response.
        
        Args:
            cached_completion (str): The cached completion string
            model (Llm): The model that generated the response
            debug_print (Callable): Function to print debug messages
            tooling (CustomColoring): Token coloring utility
            hidden_reason (str): Reason for hidden mode
            callback (Optional[Callable]): Callback function for each token
            
        Returns:
            str: The processed response string
        """
        if not hidden_reason:
            debug_print(f"{colored('Cache - ' + model.provider.__module__.split('.')[-1], 'green')} <{colored(model.model_key, 'green')}>", "blue", force_print=True)
            for char in cached_completion:
                debug_print(tooling.apply_color(char), end="", with_title=False)
                if callback:
                    callback(char)
                time.sleep(0)  # better observable for the user
            debug_print("", with_title=False)
        return cached_completion

    @classmethod
    def _handle_model_error(
        cls,
        e: Exception,
        model: Optional[Llm],
        instance: "LlmRouter",
        chat: Chat,
        log_print: Callable
    ) -> None:
        """
        Handle errors that occur during model generation.
        
        Args:
            e (Exception): The error that occurred
            model (Optional[Llm]): The model that failed
            instance (LlmRouter): The router instance
            chat (Chat): The chat being processed
            log_print (Callable): Function to print log messages
        """
        if "too large" in str(e):
            # Save the model's maximum token limit
            print(colored(f"Too large request for {model.model_key}, saving token limit {len(chat)}", "yellow"))
            instance._save_dynamic_token_limit_for_model(model, len(chat))
        
        if model is not None:
            if model.model_key in instance.failed_models:
                return
            instance.failed_models.add(model.model_key)
            instance.retry_models.remove(model)
        
        # Special handling for timeout exceptions and rate limit errors
        if (isinstance(e, TimeoutException) or 
            isinstance(e, RateLimitException) or
            "request timed out" in str(e).lower() or 
            "timeout" in str(e).lower() or 
            "timed out" in str(e).lower() or
            "connection" in str(e).lower() or
            ("Groq" in str(e) and "rate_limit_exceeded" in str(e))):
            # Silently handle timeout errors and rate limits
            if model is not None:
                logger.info(f"Network/timeout/rate-limit issue with model {model.model_key}: {e}")
            return
        
        # Display other errors
        if model is not None:
            log_print(f"\ngenerate_completion error with model {model.model_key}: {e}", "red", is_error=True)
        else:
            log_print(f"generate_completion error: {e}", "red", is_error=True)

    @classmethod
    def generate_completion(
        cls,
        chat: Chat|str,
        preferred_models: List[str] | List[Llm] = [],
        strengths: List[AIStrengths] | AIStrengths = [],
        temperature: float = 0.7,
        base64_images: List[str] = [],
        force_local: bool = False,
        force_free: bool = True,
        force_preferred_model: bool = False,
        hidden_reason: str = "",
        exclude_reasoning_tokens: bool = False,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Generate a completion response using the appropriate LLM.
        
        Args:
            chat (Chat|str): The chat prompt or string.
            preferred_models (List[str]): List of preferred model keys.
            strength (List[AIStrengths] | AIStrengths): The required strengths of the model.
            temperature (float): Temperature setting for the model.
            base64_images (List[str]): List of base64-encoded images.
            force_local (Optional[bool]): Whether to force local models only.
            force_free (bool): Whether to force free models only.
            force_preferred_model (bool): Whether to force using only preferred models.
            hidden_reason (str): Reason for hidden mode.
            exclude_reasoning_tokens (bool): Whether to exclude reasoning tokens.
            callback (Optional[Callable]): A function to call with each chunk of streaming data.

        Returns:
            str: The generated completion string.
        """
        instance = cls()
        instance.failed_models.clear()
        tooling = CustomColoring()
        cls.call_counter += 1
        
        if g.FORCE_LOCAL:
            force_local = True
        
        def exclude_reasoning(response: str) -> str:
            if exclude_reasoning_tokens and "</think>" in response:
                return response.split("</think>")[1]
            return response
            
        # Custom print function that prepends the chat debug title
        def log_print(message: str, color: str = None, end: str = '\n', with_title: bool = True, is_error: bool = False, force_print: bool = False) -> None:
            """
            Print log information with chat title prefix and logging.
            
            Args:
                message (str): The message to print
                color (str, optional): Color for the message
                end (str): End character
                with_title (bool): Whether to include the chat title
                is_error (bool): Whether this is an error message
                force_print (bool): Force printing to console even for info messages
            """
            if with_title:
                prefix = LlmRouter.get_debug_title_prefix(chat)
                log_message = f"{prefix}{message}"
                
                # Log to appropriate logger level (ignoring color)
                if is_error:
                    logger.error(log_message)
                    # For errors, always print to console
                    if color:
                        print(colored(log_message, color), end=end)
                    else:
                        print(log_message, end=end)
                else:
                    # For info level, log to logger
                    logger.info(log_message)
                    # Only print to console if forced
                    if force_print:
                        if color:
                            print(colored(log_message, color), end=end)
                        else:
                            print(log_message, end=end)
            else:
                # For character-by-character printing, don't log to the logger
                # But still print to console
                if color:
                    print(colored(message, color), end=end)
                else:
                    print(message, end=end)
        
        if base64_images:
            chat.base64_images = base64_images
        
        if not preferred_models or preferred_models == [""] or preferred_models == [None]:
            preferred_models = []
            
        # FIX FOR BREAKING CHANGE: Ensure strength is a list
        if not isinstance(strengths, list):
            strengths = [strengths] if strengths else []
        
        # Find llm and generate response, excepts on user interruption, or total failure
        while True:
            try:
                if not preferred_models or (preferred_models and isinstance(preferred_models[0], str)):
                    # Get an appropriate model
                    model = instance.get_model(strength=strengths, preferred_models=preferred_models, chat=chat, force_local=force_local, force_free=force_free, has_vision=bool(base64_images), force_preferred_model=force_preferred_model)
                else:
                    for preferred_model in preferred_models:
                        if preferred_model.model_key not in instance.failed_models:
                            model = preferred_model
                            break
                
                # If no model is available, clear failed models and retry
                if not model:
                    log_print("# # # Could not find valid model # # # RETRYING... # # #", "red", is_error=True)
                    instance.failed_models.clear()
                    if preferred_models and isinstance(preferred_models[0], str):
                        model = instance.get_model(strength=strengths, preferred_models=preferred_models, chat=chat, force_local=force_local, force_free=force_free, has_vision=bool(base64_images), force_preferred_model=force_preferred_model)

                cached_completion = instance._get_cached_completion(model.model_key, str(temperature), str(chat), base64_images)
                if cached_completion:
                    return exclude_reasoning(cls._process_cached_response(
                        cached_completion, model, log_print, tooling, hidden_reason, callback
                    ))

                try:
                    # Get the stream from the provider
                    stream = model.provider.generate_response(chat, model.model_key, temperature, hidden_reason)
                    instance.last_used_model = model.model_key
                    
                    # Process the stream
                    full_response = cls._process_stream(stream, log_print, CustomColoring(), hidden_reason, callback)
                    
                    # Cache the response
                    instance._update_cache(model.model_key, str(temperature), str(chat), base64_images, full_response)
                    
                    # Save the chat completion pair if requested
                    if not force_local:
                        instance._save_chat_completion_pair(chat.to_openai(), full_response, model.model_key)
                    
                    return exclude_reasoning(full_response)

                except KeyboardInterrupt:
                    # Explicitly catch Ctrl+C during model generation
                    log_print("User interrupted model generation (Ctrl+C).", "yellow", is_error=True, force_print=True)
                    raise UserInterruptedException("Model generation interrupted by user (Ctrl+C).")
                
            except UserInterruptedException:
                # Re-raise the specific user interruption exception
                raise
            except Exception as e:
                cls._handle_model_error(e, model, instance, chat, log_print)

    def _save_dynamic_token_limit_for_model(self, model: Llm, token_count: int) -> None:
        """
        Save or update the token limit for a model that encountered a 'too large' error.
        
        Args:
            model (Llm): The model that encountered the error
            token_count (int): The token count that caused the error
        """
        try:
            # Ensure limits are loaded
            self._load_dynamic_model_limits()
            
            # Update the cached limits
            self._model_limits[model.model_key] = min(
                token_count,
                self._model_limits.get(model.model_key, float('inf'))
            )
            
            # Save updated limits to disk
            with open(g.MODEL_TOKEN_LIMITS_PATH, 'w') as f:
                json.dump(self._model_limits, f, indent=4)
            
            # logger.info(f"Updated token limit for {model.model_key}: {token_count} tokens")
        except Exception as limit_error:
            # Create a simple local version of log_print for this method
            def error_log(message: str):
                logger.error(message)
                print(colored(message, "red"))
                
            error_log(f"Failed to save model token limit: {limit_error}")

    @classmethod
    def _save_chat_completion_pair(cls, chat_str: str, response: str, model_key: str) -> None:
        """
        Save a chat completion pair for finetuning.
        
        Args:
            chat (Chat): The input chat context
            response (str): The model's response
            model_key (str): The key of the model that generated the response
        """
        try:
            # Create the finetuning data directory if it doesn't exist
            os.makedirs(g.UNCONFIRMED_FINETUNING_PATH, exist_ok=True)
            
            # Create a filename with timestamp to avoid collisions
            timestamp = int(time.time())
            filename = os.path.join(g.UNCONFIRMED_FINETUNING_PATH, f'{timestamp}_completion_pair.jsonl')
            
            # Create the training example
            training_example = {
                "input": chat_str,
                "output": response,
                "metadata": {
                    "model": model_key,
                    "timestamp": timestamp
                }
            }
            
            # Save to JSONL file
            with open(filename, 'a') as f:
                f.write(json.dumps(training_example) + '\n')
                
            logger.info(f"Saved chat completion pair to {filename}")
        except Exception as e:
            logger.error(f"Failed to save chat completion pair: {e}")
            print(colored(f"Failed to save chat completion pair: {e}", "red"))
    
    @classmethod
    def has_unconfirmed_data(cls) -> bool:
        """Check if there are any unconfirmed finetuning data files."""
        try:
            if not os.path.exists(g.UNCONFIRMED_FINETUNING_PATH):
                return False
            return len(os.listdir(g.UNCONFIRMED_FINETUNING_PATH)) > 0
        except Exception:
            return False

    @classmethod
    def confirm_finetuning_data(cls) -> None:
        """Move unconfirmed finetuning data to confirmed directory."""
        os.makedirs(g.CONFIRMED_FINETUNING_PATH, exist_ok=True)
        if not os.path.exists(g.UNCONFIRMED_FINETUNING_PATH):
            return
        
        # move all files from unconfirmed_dir to confirmed_dir
        for file in os.listdir(g.UNCONFIRMED_FINETUNING_PATH):
            shutil.move(
                os.path.join(g.UNCONFIRMED_FINETUNING_PATH, file), 
                os.path.join(g.CONFIRMED_FINETUNING_PATH, file)
            )
    
    @classmethod
    def clear_unconfirmed_finetuning_data(cls) -> None:
        """Delete all unconfirmed finetuning data."""
        if not os.path.exists(g.UNCONFIRMED_FINETUNING_PATH):
            return
        for file in os.listdir(g.UNCONFIRMED_FINETUNING_PATH):
            os.remove(os.path.join(g.UNCONFIRMED_FINETUNING_PATH, file))
        
    @staticmethod
    def get_debug_title_prefix(chat: Chat) -> str:
        """
        Get a formatted prefix string for debug messages that includes the chat's debug title if available.
        
        Args:
            chat (Chat): The chat whose debug_title should be included
            
        Returns:
            str: The formatted prefix string
        """
        return get_debug_title_prefix(chat)

