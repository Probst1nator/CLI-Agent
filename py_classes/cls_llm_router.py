from collections.abc import Callable
import hashlib
import json
import os
from random import shuffle
import shutil
import socket
import time
from typing import Dict, List, Optional, Set, Any, Union, Iterator
from termcolor import colored
from py_classes.ai_providers.cls_human_as_interface import HumanAPI
from py_classes.ai_providers.cls_nvidia_interface import NvidiaAPI
from py_classes.cls_text_stream_painter import TextStreamPainter
from py_classes.cls_chat import Chat, Role
from py_classes.ai_providers.cls_anthropic_interface import AnthropicAPI
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.unified_interfaces import AIProviderInterface
from py_classes.ai_providers.cls_groq_interface import GroqAPI, TimeoutException, RateLimitException
from py_classes.ai_providers.cls_ollama_interface import OllamaClient
from py_classes.ai_providers.cls_openai_interface import OpenAIAPI
from py_classes.ai_providers.cls_google_interface import GoogleAPI
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

# Custom exception for user interruption
class UserInterruptedException(Exception):
    """Exception raised when the user interrupts model generation (e.g., with Ctrl+C)."""
    pass

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
        strengths: List[AIStrengths] = [], 
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
        self.strengths = strengths
    
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
    
    @property
    def is_small_model(self) -> bool:
        """Returns whether this is a small/fast model."""
        return any(s == AIStrengths.SMALL for s in self.strengths)
    
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
            
            Llm(GoogleAPI(), "gemini-2.5-flash-preview-04-17", None, 1000000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.VISION, AIStrengths.ONLINE]),
            Llm(GoogleAPI(), "gemini-2.5-flash-preview-05-20", None, 1000000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.VISION, AIStrengths.ONLINE, AIStrengths.STRONG]),
            Llm(GoogleAPI(), "gemini-2.5-pro-exp-03-25", None, 1000000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.VISION, AIStrengths.ONLINE, AIStrengths.STRONG]),
            Llm(GoogleAPI(), "gemini-2.0-flash", None, 1000000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.VISION, AIStrengths.ONLINE, AIStrengths.SMALL]),
            
            Llm(GroqAPI(), "llama-3.3-70b-versatile", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.ONLINE, AIStrengths.SMALL]),
            Llm(GroqAPI(), "deepseek-r1-distill-llama-70b", None, 128000, [AIStrengths.GENERAL, AIStrengths.REASONING, AIStrengths.ONLINE, AIStrengths.SMALL]),
            Llm(GroqAPI(), "qwen-qwq-32b", None, 128000, [AIStrengths.GENERAL, AIStrengths.REASONING, AIStrengths.ONLINE, AIStrengths.SMALL]),
            Llm(GroqAPI(), "llama-3.1-8b-instant", None, 128000, [AIStrengths.SMALL, AIStrengths.CODE, AIStrengths.ONLINE, AIStrengths.SMALL]),
            
            # Llm(AnthropicAPI(), "claude-3-7-sonnet-20250219", 3, 200000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.ONLINE]),
            
            # Llm(OllamaClient(), "gemma3n:4b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.SMALL]),
            Llm(OllamaClient(), "qwen3:30b-a3b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.STRONG]),
            # Llm(OllamaClient(), "qwen2.5vl:3b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.SMALL, AIStrengths.VISION]),
            Llm(OllamaClient(), "qwen3:4b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.SMALL]),
            Llm(OllamaClient(), "gemma3:4b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.SMALL, AIStrengths.VISION]),
            # Llm(OllamaClient(), "cogito:3b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.SMALL]),
            
            
            Llm(OllamaClient(), "cogito:8b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "gemma3:12b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.VISION]),
            Llm(OllamaClient(), "cogito:14b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "mistral-nemo:12b", None, 128000, [AIStrengths.GENERAL, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "cogito:32b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.STRONG]),
            Llm(OllamaClient(), "gemma3:27b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.VISION, AIStrengths.STRONG]),
            Llm(OllamaClient(), "devstral:24b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.STRONG]),
            Llm(OllamaClient(), "Captain-Eris_Violet-GRPO-v0.420.i1-Q4_K_M:latest", None, 128000, [AIStrengths.GENERAL, AIStrengths.UNCENSORED, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "L3-8B-Stheno-v3.2-Q4_K_M-imat:latest", None, 128000, [AIStrengths.GENERAL, AIStrengths.UNCENSORED, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "DeepHermes-Egregore-v2-RLAIF-8b-Atropos-Q4:latest", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "gemma3:1b", None, 128000, [AIStrengths.GENERAL, AIStrengths.CODE, AIStrengths.LOCAL, AIStrengths.SMALL, AIStrengths.VISION]),
            
            # Guard models
            Llm(GroqAPI(), "llama-guard-4-12b", None, 128000, [AIStrengths.GUARD]),
            Llm(OllamaClient(), "llama-guard3:8b", None, 8192, [AIStrengths.GUARD, AIStrengths.LOCAL]),
            Llm(OllamaClient(), "shieldgemma:2b", None, 8192, [AIStrengths.GUARD, AIStrengths.LOCAL, AIStrengths.SMALL]),
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
        self.cache_file_path = f"{g.CLIAGENT_PERSISTENT_STORAGE_PATH}/llm_cache.json"
        
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

    def _generate_hash(self, model_key: str, prompt: str, images: List[str]) -> str:
        """
        Generate a hash for caching based on model, prompt, and images.
        
        Args:
            model_key (str): Model identifier.
            prompt (str): The prompt string.
            images (List[str]): List of image encodings.

        Returns:
            str: The generated hash string.
        """
        # Combine inputs and generate SHA256 hash
        hash_input = f"{model_key}:{prompt}{':'.join(images)}".encode()
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

    def _get_cached_completion(self, model_key: str, key: str, images: List[str]) -> Optional[str]:
        """
        Retrieve a cached completion if available.
        
        Args:
            model_key (str): Model identifier.
            chat (Chat): The chat prompt.
            images (List[str]): List of image encodings.

        Returns:
            Optional[str]: The cached completion string if available, otherwise None.
        """
        # Generate cache key and return cached completion if it exists
        cache_key = self._generate_hash(model_key,  key, images)
        return self.cache.get(cache_key)

    def _update_cache(self, model_key: str, key: str, images: List[str], completion: str) -> None:
        """
        Update the cache with a new completion.
        """
        # Generate cache key
        cache_key = self._generate_hash(model_key, key, images)
        
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
    def get_model(cls, preferred_models: List[str] = [], strengths: List[AIStrengths] = [], chat: Chat = Chat(), force_local: bool = False, force_free: bool = False, has_vision: bool = False, force_preferred_model: bool = False) -> Optional[Llm]:
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

        # Debug print for large token counts
        if (len(chat) > 4000 and not force_free and not force_local):
            print(colored("DEBUG: len(chat) returned: " + str(len(chat)), "yellow"))
        
        # Try models in order of preference
        candidate_models = []
        
        # First try to find preferred model with exact capabilities
        for model_key in preferred_models:
            if (model_key not in instance.failed_models) and model_key:
                model = next((model for model in instance.retry_models if model_key in model.model_key and (has_vision == False or has_vision == model.has_vision)), None)
                if model and instance.model_capable_check(model, chat, strengths, model.local, False, has_vision, allow_general=False):
                    candidate_models.append(model)

        # If no preferred candidates and force_preferred_model is True
        if not candidate_models and force_preferred_model:
            # Check if the preferred model looks like a local model (contains ':' or common model naming patterns)
            model_name = preferred_models[0].lower()
            is_likely_local = (':' in model_name or 
                             any(pattern in model_name for pattern in ['llama', 'qwen', 'gemma', 'phi', 'mistral', 'cogito', 'devstral']))
            
            if force_local or is_likely_local:
                # return a dummy model to force Ollama to try downloading it
                return Llm(OllamaClient(), preferred_models[0], 0, 8192, [AIStrengths.GENERAL, AIStrengths.LOCAL])
            
            print(colored(f"Could not find preferred model {preferred_models[0]}", "red"))
            return None
        
        # Continue gathering candidates from other models if needed
        if not candidate_models and not force_preferred_model:
            # Search online models by exact capability next
            if not force_local:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and not model.local:
                        if instance.model_capable_check(model, chat, strengths, local=False, force_free=force_free, has_vision=has_vision, allow_general=False):
                            candidate_models.append(model)
                
                # Add online models with GENERAL capability
                if not candidate_models and strengths:
                    for model in instance.retry_models:
                        if model.model_key not in instance.failed_models and not model.local:
                            if instance.model_capable_check(model, chat, strengths, local=False, force_free=force_free, has_vision=has_vision, allow_general=True):
                                candidate_models.append(model)

            # Add local models by exact capability
            if not candidate_models or force_local:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and model.local:
                        if instance.model_capable_check(model, chat, strengths, local=True, force_free=force_free, has_vision=has_vision, allow_general=False):
                            candidate_models.append(model)
            
            # Add local models with GENERAL capability
            if not candidate_models and strengths:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and model.local:
                        if instance.model_capable_check(model, chat, strengths, local=True, force_free=force_free, has_vision=has_vision, allow_general=True):
                            candidate_models.append(model)
            
            # Last resort: try with empty chat to ignore context length
            if not candidate_models:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and model.local:
                        if instance.model_capable_check(model, Chat(), strengths, local=True, force_free=force_free, has_vision=has_vision, allow_general=True):
                            candidate_models.append(model)
        
        # Return the first valid candidate
        return candidate_models[0] if candidate_models else None
    
    @classmethod
    def _process_stream(
        cls,
        stream: Union[Iterator[Dict[str, Any]], Iterator[str], Any],
        provider: AIProviderInterface,
        hidden_reason: str,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Process a stream of tokens from any provider.
        
        Args:
            stream (Union[Iterator[Dict[str, Any]], Iterator[str], Any]): The stream object from the provider
            provider (AIProviderInterface): The provider interface
            hidden_reason (str): Reason for hidden mode
            callback (Optional[Callable]): Callback function for each token
            
        Returns:
            str: The full response string
        """
        full_response = ""
        finished_response = ""
        token_stream_painter = TextStreamPainter()
        
        # Handle different stream types
        # ! Anthropic
        if isinstance(provider, AnthropicAPI):
            if hasattr(stream, 'text_stream'):  
                for token in stream.text_stream:
                    if token:
                        full_response += token
                        if callback is not None:
                            finished_response = callback(token, hidden_reason)
                            if finished_response and isinstance(finished_response, str):
                                break
                        elif not hidden_reason:
                            g.print_token(token_stream_painter.apply_color(token))
        # ! OpenAI/NVIDIA
        elif isinstance(provider, OpenAIAPI) or isinstance(provider, NvidiaAPI):
            if hasattr(stream, 'choices'):  
                for chunk in stream:
                    # Safely access delta content
                    token = None
                    if chunk.choices and hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        token = chunk.choices[0].delta.content
                    
                    if token is not None:  # Ensure token is not None (can be empty string)
                        full_response += token
                        if callback is not None:
                            finished_response = callback(token)
                            if finished_response and isinstance(finished_response, str):
                                break
                        elif not hidden_reason:
                            g.print_token(token_stream_painter.apply_color(token))
        # ! Google Gemini - IMPROVED HANDLING
        elif isinstance(provider, GoogleAPI):
            try:
                first_chunk_processed = False
                for chunk in stream:  # chunk is a GenerateContentResponse
                    if not first_chunk_processed:
                        first_chunk_processed = True
                        # Minimal check for immediate prompt blocking on the first chunk
                        if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback and \
                           hasattr(chunk.prompt_feedback, 'block_reason') and chunk.prompt_feedback.block_reason:
                            return ""  # Prompt was blocked, no content will follow.

                    token_from_this_chunk = ""
                    try:
                        # Safely attempt to extract text.
                        # The .parts property on GenerateContentResponse is a shortcut for candidates[0].content.parts
                        if hasattr(chunk, 'parts') and chunk.parts:
                            for part in chunk.parts:
                                if hasattr(part, 'text') and part.text is not None:
                                    token_from_this_chunk += part.text
                        # If chunk.parts is empty or not present, token_from_this_chunk remains ""
                        # This avoids the error from directly accessing chunk.text if parts are missing.
                    except AttributeError:
                        # This catches if `chunk.parts` itself or `part.text` is missing when expected.
                        # Silently treat as no token for this chunk to prevent crash.
                        pass 
                    
                    if token_from_this_chunk:
                        full_response += token_from_this_chunk
                        if callback is not None:
                            # Callback can return the final response string to terminate early
                            result_from_callback = callback(token_from_this_chunk)
                            if result_from_callback and isinstance(result_from_callback, str):
                                finished_response = result_from_callback
                                break 
                        elif not hidden_reason:
                            g.print_token(token_stream_painter.apply_color(token_from_this_chunk))
                
                # If callback signaled to finish early
                if finished_response and isinstance(finished_response, str):
                    return finished_response

            except StopIteration:
                pass  # Normal end of stream
        # ! Ollama/Groq
        elif isinstance(provider, OllamaClient) or isinstance(provider, GroqAPI):
            for chunk in stream:
                token = None
                if isinstance(provider, GroqAPI):
                    if hasattr(chunk, 'choices') and chunk.choices and \
                       hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        token = chunk.choices[0].delta.content
                elif isinstance(provider, OllamaClient):
                    if isinstance(chunk, dict):  # Ollama dictionary chunks
                        token = chunk.get('message', {}).get('content', '') or chunk.get('response', '')
                    elif hasattr(chunk, 'message'):  # Ollama response object
                        if hasattr(chunk.message, 'content'):
                            token = chunk.message.content
                
                if token is not None:  # Ensure token is not None
                    full_response += token
                    if callback is not None:
                        finished_response = callback(token)
                        if finished_response and isinstance(finished_response, str):
                            break
                    elif not hidden_reason:
                        g.print_token(token_stream_painter.apply_color(token))
        
        # Fallback for other unknown stream types (original logic)
        else:  
            for chunk_item in stream:  # Renamed to avoid conflict
                token = str(chunk_item)  # Basic conversion
                if token:
                    full_response += token
                    if callback is not None:
                        finished_response = callback(token)
                        if finished_response and isinstance(finished_response, str):
                            break
                    elif not hidden_reason:
                        g.print_token(token_stream_painter.apply_color(token))
            
        # If callback returned a final response string at any point (and broke the loop)
        if finished_response and isinstance(finished_response, str):
            return finished_response
        
        return full_response

    @classmethod
    def _process_cached_response(
        cls,
        cached_completion: str,
        model: Llm,
        text_stream_painter: TextStreamPainter,
        hidden_reason: str,
        callback: Optional[Callable] = None
    ) -> str:
        """
        Process a cached response.
        
        Args:
            cached_completion (str): The cached completion string
            model (Llm): The model that generated the response
            text_stream_painter (TextStreamPainter): Token coloring utility
            hidden_reason (str): Reason for hidden mode
            callback (Optional[Callable]): Callback function for each token
            
        Returns:
            str: The processed response string
        """
        if not hidden_reason:
            g.debug_log(f"{colored('Cache - ' + model.provider.__module__.split('.')[-1], 'green')} <{colored(model.model_key, 'green')}>", "blue", force_print=True)
            for char in cached_completion:
                if callback:
                    finished_response = callback(char, not hidden_reason)
                    if finished_response and isinstance(finished_response, str):
                        return finished_response
                elif not hidden_reason:
                    g.print_token(text_stream_painter.apply_color(char))
                time.sleep(0)  # better observable for the user
            print()  # Add newline at the end
        return cached_completion

    @classmethod
    def _get_descriptive_error(cls, error_msg: str, model_key: str) -> str:
        """
        Convert generic error messages into more descriptive ones.
        
        Args:
            error_msg (str): The original error message
            model_key (str): The model that failed
            
        Returns:
            str: A more descriptive error message
        """
        if "timeout" in error_msg.lower():
            return f"Model {model_key} timed out during generation (possibly stuck in loop)"
        elif "empty response" in error_msg.lower():
            return f"Model {model_key} returned empty response (generation failure)"
        elif "connection" in error_msg.lower():
            return f"Connection failed to model {model_key}: {error_msg}"
        elif "rate" in error_msg.lower() and "limit" in error_msg.lower():
            return f"Rate limit exceeded for model {model_key}"
        elif "model not found" in error_msg.lower():
            return f"Model {model_key} not found (trying next model)"
        else:
            return f"Failed to generate response with model {model_key}: {error_msg}"

    @classmethod
    def _handle_model_error(
        cls,
        e: Exception,
        model: Optional[Llm],
        instance: "LlmRouter",
        chat: Chat
    ) -> None:
        """
        Handle errors that occur during model generation.
        
        Args:
            e (Exception): The error that occurred
            model (Optional[Llm]): The model that failed
            instance (LlmRouter): The router instance
            chat (Chat): The chat being processed
        """
        # Check if the exception has already been logged by a provider
        if hasattr(e, 'already_logged') and getattr(e, 'already_logged'):
            # This error was already logged, don't log it again
            # We still need to update the failed models list though
            if model is not None and model.model_key not in instance.failed_models:
                instance.failed_models.add(model.model_key)
                instance.retry_models.remove(model)
            return
            
        error_msg = str(e)
        prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
        provider_name = model.provider.__class__.__name__ if model else "Unknown"
        model_key = model.model_key if model else "unknown"
        
        # Handle token limit error
        if "too large" in error_msg:
            # Save the model's maximum token limit
            print(colored(f"Too large request for {model_key}, saving token limit {len(chat)}", "yellow"))
            instance._save_dynamic_token_limit_for_model(model, len(chat))
        
        # Update failed models list
        if model is not None:
            if model.model_key in instance.failed_models:
                return
            instance.failed_models.add(model.model_key)
            instance.retry_models.remove(model)
        
        # Special handling for network and rate limit issues
        if (isinstance(e, TimeoutException) or 
            isinstance(e, RateLimitException) or
            "request timed out" in error_msg.lower() or 
            "timeout" in error_msg.lower() or 
            "timed out" in error_msg.lower() or
            "connection" in error_msg.lower() or
            "rate_limit" in error_msg.lower()):
            
            # Log silently to file but don't show to user
            if model is not None:
                logger.info(f"Network/timeout/rate-limit issue with model {model_key}: {e}")
            return
        
        # Check if this error has already been logged by the Google API provider
        if "Google API" in error_msg and "error" in error_msg.lower():
            # This error was already logged by the Google API provider, don't log it again
            return
        
        # Provider-specific error handling
        if "OllamaClient" in provider_name:
            # Provide more detailed error messages for common Ollama issues
            if "timeout" in error_msg.lower():
                g.debug_log(f"❌ Ollama-Api: Model {model_key} timed out during generation (stream may be stuck)", "red", is_error=True, prefix=prefix)
            elif "empty response" in error_msg.lower():
                g.debug_log(f"❌ Ollama-Api: Model {model_key} returned empty response (generation failure)", "red", is_error=True, prefix=prefix)
            elif "No valid host found" in error_msg:
                g.debug_log(f"⚠️  Ollama-Api: {error_msg} (trying next model)", "yellow", prefix=prefix)
            elif "model not found" in error_msg.lower():
                g.debug_log(f"⚠️  Ollama-Api: Model {model_key} not found (trying next model)", "yellow", prefix=prefix)
            else:
                g.debug_log(f"❌ Ollama-Api: Failed to generate response with model {model_key}: {e}", "red", is_error=True, prefix=prefix)
            # Add to unreachable hosts if applicable
            if model and hasattr(model.provider, "unreachable_hosts") and hasattr(model.provider, "_client"):
                try:
                    host = model.provider._client.base_url.host
                    model.provider.unreachable_hosts.append(f"{host}{model_key}")
                except Exception:
                    pass
        elif "GroqAPI" in provider_name:
            descriptive_error = cls._get_descriptive_error(error_msg, model_key)
            g.debug_log(f"❌ Groq-Api: {descriptive_error}", "red", is_error=True, prefix=prefix)
        elif "GoogleAPI" in provider_name:
            descriptive_error = cls._get_descriptive_error(error_msg, model_key)
            g.debug_log(f"❌ Google-Api: {descriptive_error}", "red", is_error=True, prefix=prefix)
        elif "OpenAIAPI" in provider_name:
            descriptive_error = cls._get_descriptive_error(error_msg, model_key)
            g.debug_log(f"❌ OpenAI-Api: {descriptive_error}", "red", is_error=True, prefix=prefix)
        elif "AnthropicAPI" in provider_name:
            descriptive_error = cls._get_descriptive_error(error_msg, model_key)
            g.debug_log(f"❌ Anthropic-Api: {descriptive_error}", "red", is_error=True, prefix=prefix)
        elif "NvidiaAPI" in provider_name:
            descriptive_error = cls._get_descriptive_error(error_msg, model_key)
            g.debug_log(f"❌ NVIDIA-Api: {descriptive_error}", "red", is_error=True, prefix=prefix)
        elif "HumanAPI" in provider_name:
            g.debug_log(f"❌ Human-Api: Failed to generate response: {e}", "red", is_error=True, prefix=prefix)
        else:
            # Generic error handling for unknown providers or when model is None
            if model is not None:
                descriptive_error = cls._get_descriptive_error(error_msg, model_key)
                g.debug_log(f"❌ Generation error with model {model_key}: {descriptive_error}", "red", is_error=True, prefix=prefix)
            else:
                g.debug_log(f"❌ Generation error: {e}", "red", is_error=True, prefix=prefix)
            time.sleep(1)

    @classmethod
    def generate_completion(
        cls,
        chat: Chat|str,
        preferred_models: List[str] | List[Llm] = [],
        strengths: List[AIStrengths] = [],
        temperature: float = 0,
        base64_images: List[str] = [],
        force_local: bool = False,
        force_free: bool = True,
        force_preferred_model: bool = False,
        hidden_reason: str = "",
        exclude_reasoning_tokens: bool = True,
        generation_stream_callback: Optional[Callable] = None,
        follows_condition_callback: Optional[Callable] = None
    ) -> str:
        """
        Generate a completion response using the appropriate LLM.
        
        Args:
            chat (Chat|str): The chat prompt or string.
            preferred_models (List[str]): List of preferred model keys.
            strength (List[AIStrengths] | AIStrengths): The required strengths of the model.
            temperature (float): Temperature setting for the model.
            base64_images (List[str]): List of base64-encoded images.
            force_local (bool): Whether to force local models only.
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
        cls.call_counter += 1
        
        # Check global force flags
        if g.LLM:
            preferred_models = [g.LLM]
        if g.FORCE_FAST:
            strengths.append(AIStrengths.SMALL)
        if g.LLM_STRENGTHS:
            strengths.extend(g.LLM_STRENGTHS)
        if g.FORCE_ONLINE:
            strengths = [s for s in strengths if s != AIStrengths.LOCAL]
            strengths.append(AIStrengths.ONLINE)
            # ! force_free = False
        
        # Local has higher priority than online
        if g.FORCE_LOCAL:
            force_local = True
            strengths = [s for s in strengths if s != AIStrengths.ONLINE]
            strengths.append(AIStrengths.LOCAL)
        
        def exclude_reasoning(response: str) -> str:
            if exclude_reasoning_tokens and ("</think>" in response or "</thinking>" in response):
                if "</think>" in response:
                    return response.split("</think>")[1]
                elif "</thinking>" in response:
                    return response.split("</thinking>")[1]
                elif "</" in response: # Weird fallback, helps for small models
                    return response.split("</")[1].split(">")[1]
            return response
        
        if base64_images:
            chat.base64_images = base64_images
        
        # FIX FOR BREAKING CHANGE: Ensure strength is a list
        if not isinstance(strengths, list):
            strengths = [strengths] if strengths else []
        
        if isinstance(chat, str):
            prompt = chat
            chat = Chat()
            chat.add_message(Role.USER, prompt)
        
        # Find llm and generate response, excepts on user interruption, or total failure
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                model = None  # Initialize model variable to avoid UnboundLocalError
                if not preferred_models or (preferred_models and isinstance(preferred_models[0], str)):
                    # Get an appropriate model
                    model = instance.get_model(strengths=strengths, preferred_models=preferred_models, chat=chat, force_local=force_local, force_free=force_free, has_vision=bool(base64_images), force_preferred_model=force_preferred_model)
                else:
                    for preferred_model in preferred_models:
                        if preferred_model.model_key not in instance.failed_models:
                            model = preferred_model
                            break
                
                # If no model is available, clear failed models and retry
                if not model:
                    retry_count += 1
                    if retry_count >= max_retries:
                        break  # Exit the retry loop
                    
                    prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                    g.debug_log(f"# # # Could not find valid model # # # RETRYING... ({retry_count}/{max_retries}) # # #", "red", is_error=True, prefix=prefix)
                    instance.failed_models.clear()
                    
                    # Also reset OllamaClient host cache to allow trying different hosts
                    try:
                        from py_classes.ai_providers.cls_ollama_interface import OllamaClient
                        OllamaClient.reset_host_cache()
                    except ImportError:
                        pass  # OllamaClient not available
                    
                    if preferred_models and isinstance(preferred_models[0], str):
                        model = instance.get_model(strengths=strengths, preferred_models=preferred_models, chat=chat, force_local=force_local, force_free=force_free, has_vision=bool(base64_images), force_preferred_model=force_preferred_model)
                    continue  # Continue the retry loop

                # Check if model is still None after retry
                if not model:
                    prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                    g.debug_log("No valid model found after retry, exiting", "red", is_error=True, prefix=prefix)
                    raise Exception("No valid model available")
                
                enable_caching = False
                instance.last_used_model = model.model_key
                
                if temperature == 0:
                    enable_caching = True
                    cached_completion = instance._get_cached_completion(model.model_key, str(chat), base64_images)
                    if cached_completion:
                        return exclude_reasoning(cls._process_cached_response(
                            cached_completion, model, TextStreamPainter(), hidden_reason, generation_stream_callback
                        ))

                try:
                    # Get the stream from the provider
                    stream = model.provider.generate_response(chat, model.model_key, temperature, hidden_reason)
                    
                    # Check if stream is None before processing
                    if stream is None:
                        raise Exception(f"Model {model.model_key} returned None stream")
                    
                    # MODIFIED LINE: Pass model.provider to _process_stream with timeout
                    import signal
                    import time
                    
                    def timeout_handler(signum, frame):
                        raise Exception(f"Stream processing timeout after 60 seconds for model {model.model_key}")
                    
                    # Set timeout for stream processing
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(60)  # 60 second timeout
                    
                    try:
                        full_response = cls._process_stream(stream, model.provider, hidden_reason, generation_stream_callback)
                        
                        # Check for empty responses
                        if not full_response or not full_response.strip():
                            raise Exception(f"Model {model.model_key} returned empty response - possible generation failure")
                            
                    finally:
                        signal.alarm(0)  # Clear the alarm
                    
                    if (not full_response.endswith("\n") and not hidden_reason):
                        print()
                    
                    if enable_caching:
                        # Cache the response
                        instance._update_cache(model.model_key, str(chat), base64_images, full_response)
                    
                    # Save the chat completion pair if requested
                    if not force_local:
                        instance._save_chat_completion_pair(chat.to_openai(), full_response, model.model_key)
                    
                    return exclude_reasoning(full_response)

                except KeyboardInterrupt:
                    # Explicitly catch Ctrl+C during model generation
                    prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                    g.debug_log("User interrupted model generation (Ctrl+C).", "yellow", is_error=True, force_print=True, prefix=prefix)
                    raise UserInterruptedException("Model generation interrupted by user (Ctrl+C).")
                
            except UserInterruptedException:
                # Re-raise the specific user interruption exception
                raise
            except Exception as e:
                cls._handle_model_error(e, model, instance, chat)
        
        # If we've exhausted all retries, raise an exception
        raise Exception(f"Failed to find a valid model after {max_retries} retries")

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
            logger.error(f"Failed to save model token limit: {limit_error}")
            print(colored(f"Failed to save model token limit: {limit_error}", "red"))

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
        return chat.get_debug_title_prefix()

