import hashlib
import json
import os
from random import shuffle
import shutil
import time
from typing import Dict, List, Optional, Set
from termcolor import colored
from py_classes.ai_providers.cls_human_as_interface import HumanAPI
from py_classes.ai_providers.cls_nvidia_interface import NvidiaAPI
from py_classes.cls_custom_coloring import CustomColoring
from py_classes.cls_chat import Chat, Role
from enum import Enum
from py_classes.ai_providers.cls_anthropic_interface import AnthropicAPI
from py_classes.cls_ai_provider_interface import ChatClientInterface
from py_classes.ai_providers.cls_groq_interface import GroqAPI
from py_classes.ai_providers.cls_ollama_interface import OllamaClient
from py_classes.ai_providers.cls_openai_interface import OpenAIAPI
from py_classes.globals import g
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIStrengths(Enum):
    """Enum class to represent AI model strengths."""
    TOOLUSE = 8
    OMNIMODAL = 7
    UNCENSORED = 6
    REASONING = 5
    CODE = 4
    GUARD = 3
    GENERAL = 2
    FAST = 1

class Llm:
    """
    Class representing a Language Model (LLM) with its properties and capabilities.
    """

    def __init__(
        self, 
        provider: ChatClientInterface, 
        model_key: str, 
        pricing_in_dollar_per_1M_tokens: Optional[float], 
        has_tool_use: bool, 
        available_local: bool, 
        has_vision: bool, 
        context_window: int, 
        strength: List[AIStrengths] = [], 
    ):
        """
        Initialize an LLM instance.

        Args:
            provider (ChatClientInterface): The chat client interface for the LLM.
            model_key (str): Unique identifier for the model.
            pricing_in_dollar_per_1M_tokens (Optional[float]): Pricing information.
            available_local (bool): Whether the model is available locally.
            has_vision (bool): Whether the model has vision capabilities.
            context_window (int): The context window size of the model.
            max_output (Optional[int]): Maximum output tokens.
            strength (AIStrengths): The strength category of the model.
        """
        self.provider = provider
        self.model_key = model_key
        self.pricing_in_dollar_per_1M_tokens = pricing_in_dollar_per_1M_tokens
        self.has_tool_use = has_tool_use
        self.local = available_local
        self.has_vision = has_vision
        self.context_window = context_window
        self.strength = strength
    
    @classmethod
    def get_available_llms(cls) -> List["Llm"]:
        """
        Get the list of available LLMs.
        
        Returns:
            List[Llm]: A list of Llm instances representing the available models.
        """
        # Define and return a list of available LLM instances
        return [
            # Llm(HumanAPI(), "human", None, True, True, True, 131072, 30000, [AIStrengths.STRONG]), # For testing
            Llm(GroqAPI(), "llama-3.3-70b-specdec", None, False, False, False, 8192, [AIStrengths.GENERAL, AIStrengths.FAST, AIStrengths.TOOLUSE]),
            Llm(GroqAPI(), "llama-3.3-70b-versatile", None, False, False, False, 128000, [AIStrengths.GENERAL, AIStrengths.TOOLUSE]),
            Llm(GroqAPI(), "llama-3.1-8b-instant", None, False, False, False, 128000, [AIStrengths.FAST, AIStrengths.TOOLUSE]),
            Llm(GroqAPI(), "llama3-70b-8192", None, False, False, False, 8192, [AIStrengths.GENERAL, AIStrengths.TOOLUSE]),
            Llm(GroqAPI(), "qwen-2.5-coder-32b", None, False, False, False, 128000, [AIStrengths.CODE]),
            Llm(GroqAPI(), "qwen-2.5-32b", None, False, False, False, 8192, [AIStrengths.FAST, AIStrengths.GENERAL, AIStrengths.TOOLUSE, AIStrengths.CODE]),
            Llm(GroqAPI(), "llama3-8b-8192", None, False, False, False, 8192, [AIStrengths.FAST]),
            Llm(GroqAPI(), "gemma2-9b-it", None, False, False, False, 8192, [AIStrengths.FAST]),
            
            Llm(GroqAPI(), "deepseek-r1-distill-llama-70b", None, False, False, False, 128000, [AIStrengths.GENERAL, AIStrengths.REASONING]),
            Llm(GroqAPI(), "llama-3.2-90b-vision-preview", None, False, False, True, 32768, [AIStrengths.GENERAL, AIStrengths.OMNIMODAL]),
            Llm(GroqAPI(), "deepseek-r1-distill-qwen-32b", None, False, False, False, 128000, [AIStrengths.GENERAL, AIStrengths.FAST, AIStrengths.REASONING]),
            
            
            Llm(AnthropicAPI(), "claude-3-5-sonnet-latest", 9, False, False, False, 200000, [AIStrengths.GENERAL, AIStrengths.OMNIMODAL, AIStrengths.CODE, AIStrengths.TOOLUSE]),
            # Llm(AnthropicAPI(), "claude-3-haiku-20240307", 1, False, False, False, 200000, [AIStrengths.FAST]),
            
            # Llm(OpenAIAPI(), "gpt-4o", 10, False, False, True, 128000, [AIStrengths.STRONG]),
            # Llm(OpenAIAPI(), "gpt-4o-mini", 0.4, False, False, True, 128000, [AIStrengths.FAST]),
            Llm(OllamaClient(), "gemma3:27b", None, True, True, False, 128000, [AIStrengths.GENERAL, AIStrengths.TOOLUSE]),
            Llm(OllamaClient(), "gemma3:12b", None, True, True, False, 128000, [AIStrengths.GENERAL, AIStrengths.TOOLUSE]),
            Llm(OllamaClient(), "gemma3:4b", None, True, True, False, 128000, [AIStrengths.GENERAL, AIStrengths.FAST, AIStrengths.TOOLUSE]),
            # Llm(OllamaClient(), "deepseek-r1:14b", None, True, True, False, 128000, [AIStrengths.REASONING]),
            Llm(OllamaClient(), "deepseek-r1:8b", None, True, True, False, 128000, [AIStrengths.REASONING, AIStrengths.FAST]),
            Llm(OllamaClient(), "command-r7b:latest", None, True, True, False, 128000, [AIStrengths.TOOLUSE]),
            Llm(OllamaClient(), "qwen2.5-coder:7b-instruct", None, False, True, False, 131072, [AIStrengths.GENERAL, AIStrengths.CODE]),
            Llm(OllamaClient(), "mistral-nemo:12b", None, False, True, False, 128000, [AIStrengths.GENERAL]),
            Llm(OllamaClient(), "mistral-small:22b", None, False, True, False, 128000, [AIStrengths.GENERAL]),
            Llm(OllamaClient(), 'llama3.1:8b', None, True, True, False, 8192, [AIStrengths.GENERAL]),
            Llm(OllamaClient(), "MN-12B-Mag-Mell-Q4_K_M.gguf:latest", None, True, True, False, 128000, [AIStrengths.UNCENSORED]),
            Llm(OllamaClient(), "llama3.2:3b", None, False, True, False, 4096, [AIStrengths.FAST]),
            
            Llm(OllamaClient(), "minicpm-v:8b", None, False, True, True, 32768, []),
            
            # Specialised models below
            
            # RAG models
            # Llm(NvidiaAPI(), "nvidia/llama-3.1-nemotron-70b-instruct", None, False, False, False, 128000, [AIStrengths.STRONG]),
            
            # Guard models
            Llm(GroqAPI(), "llama-guard-3-8b", None, False, False, False, 8192, [AIStrengths.GUARD]),
            Llm(OllamaClient(), "llama-guard3:8b", None, False, True, False, 4096, [AIStrengths.GUARD]),
            Llm(OllamaClient(), "llama-guard3:1b", None, False, True, False, 4096, [AIStrengths.GUARD]),
        ]


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
                if os.path.exists(g.DYNAMIC_MODEL_LIMITS_PATH):
                    with open(g.DYNAMIC_MODEL_LIMITS_PATH, 'r') as f:
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

    def _get_cached_completion(self, model_key: str, temperature: str, chat: Chat, images: List[str]) -> Optional[str]:
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
        cache_key = self._generate_hash(model_key, temperature, chat.to_json(), images)
        return self.cache.get(cache_key)

    def _update_cache(self, model_key: str, temperature: str, chat: Chat, images: List[str], completion: str) -> None:
        """
        Update the cache with a new completion.
        """
        # Generate cache key
        cache_key = self._generate_hash(model_key, temperature, chat.to_json(), images)
        
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
            if chat.count_tokens() >= token_limit:
                return False
        
        if model.context_window < chat.count_tokens():
            return False
        if strength and model.strength:
            # Check if ALL of the required strengths are included in the model's strengths
            if not all(s.value in [ms.value for ms in model.strength] for s in strength):
                # Only check for GENERAL if allowed
                if allow_general:
                    return any(s.value == AIStrengths.GENERAL.value for s in model.strength)
                return False
        if model.local != local:
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

        Returns:
            Optional[Llm]: The next available Llm instance if available, otherwise None.
        """
        try:
            instance = cls()
            
            # Debug print for large token counts
            if (chat.count_tokens() > 4000 and not force_free and not force_local):
                print(colored("DEBUG: chat.count_tokens() returned: " + str(chat.count_tokens()), "yellow"))
            
            # First try to find preferred model with exact capabilities
            for model_key in preferred_models:
                if (model_key not in instance.failed_models) and model_key:
                    model = next((model for model in instance.retry_models if model_key in model.model_key and (force_local == False or force_local == model.local) and (has_vision == False or has_vision == model.has_vision)), None)
                    if model and instance.model_capable_check(model, chat, strength, model.local, force_free, has_vision, allow_general=False):
                        return model

            if force_preferred_model:
                if force_local:
                    # return a dummy model to force Ollama to download it
                    return Llm(OllamaClient(), preferred_models[0], 0, True, True, True, 8192, [AIStrengths.GENERAL])
                print(colored(f"Could not find preferred model {preferred_models[0]}", "red"))
                return None
            
            # Search online models by exact capability next
            if not force_local:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and not model.local:
                        if instance.model_capable_check(model, chat, strength, local=False, force_free=force_free, has_vision=has_vision, allow_general=False):
                            return model
                
                # Try online models with GENERAL capability
                if strength:
                    for model in instance.retry_models:
                        if model.model_key not in instance.failed_models and not model.local:
                            if instance.model_capable_check(model, chat, strength, local=False, force_free=force_free, has_vision=has_vision, allow_general=True):
                                return model

            # Search local models by exact capability
            for model in instance.retry_models:
                if model.model_key not in instance.failed_models and model.local:
                    if instance.model_capable_check(model, chat, strength, local=True, force_free=force_free, has_vision=has_vision, allow_general=False):
                        return model
            
            # Try local models with GENERAL capability
            if strength:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and model.local:
                        if instance.model_capable_check(model, chat, strength, local=True, force_free=force_free, has_vision=has_vision, allow_general=True):
                            return model
            
            # Last resort: try with empty chat to ignore context length
            for model in instance.retry_models:
                if model.model_key not in instance.failed_models and model.local:
                    if instance.model_capable_check(model, Chat(), strength, local=True, force_free=force_free, has_vision=has_vision, allow_general=True):
                        return model
        except Exception as e:
            print(colored(f"Error in get_model: {e}", "red"))
            logger.error(f"Error in get_model: {e}")
        return None
    
    @classmethod
    def generate_completion(
        cls,
        chat: Chat|str,
        preferred_models: List[str] | List[Llm] = [],
        strength: List[AIStrengths] | AIStrengths = [],
        start_response_with: str = "",
        instruction: str = "",
        temperature: float = 0.7,
        base64_images: List[str] = [],
        include_start_response_str: bool = True,
        use_cache: bool = True,
        force_local: bool = False,
        force_free: bool = True,
        force_preferred_model: bool = False,
        silent_reason: str = "",
        re_print_prompt: bool = False,
        exclude_reasoning_tokens: bool = False
    ) -> str:
        """
        Generate a completion response using the appropriate LLM.
        
        Args:
            chat (Chat|str): The chat prompt or string.
            preferred_models (List[str]): List of preferred model keys.
            strength (List[AIStrengths] | AIStrengths): The required strengths of the model.
            start_response_with (str): Initial string to start the response with.
            instruction (str): Instruction for the chat.
            temperature (float): Temperature setting for the model.
            base64_images (List[str]): List of base64-encoded images.
            include_start_response_str (bool): Whether to include the start response string.
            use_cache (bool): Whether to use the cache.
            force_local (Optional[bool]): Whether to force local models only.
            force_free (bool): Whether to force free models only.
            force_preferred_model (bool): Whether to force using only preferred models.
            silent_reason (str): Reason for silent mode.
            re_print_prompt (bool): Whether to reprint the prompt.

        Returns:
            str: The generated completion string.
        """
        instance = cls()
        instance.failed_models.clear()
        tooling = CustomColoring()
        cls.call_counter += 1
        
        if g.FORCE_LOCAL:
            force_local = True
        
        def preprocess_response(response: str) -> str:
            if exclude_reasoning_tokens and "</think>" in response:
                return response.split("</think>")[1]
            return response
        
        # Convert string input to Chat object if necessary
        if isinstance(chat, str):
            chat = Chat(instruction, debug_title="generate_completion() call").add_message(Role.USER, chat)
        
        if start_response_with:
            chat.add_message(Role.ASSISTANT, start_response_with)
        
        if base64_images:
            chat.base64_images = base64_images
        
        if not preferred_models or preferred_models == [""] or preferred_models == [None]:
            preferred_models = []
            
        # FIX FOR BREAKING CHANGE: Ensure strength is a list
        if not isinstance(strength, list):
            strength = [strength] if strength else []
        
        while True:
            try:
                if not preferred_models or (preferred_models and isinstance(preferred_models[0], str)):
                    # Get an appropriate model
                    model = instance.get_model(strength=strength, preferred_models=preferred_models, chat=chat, force_local=force_local, force_free=force_free, has_vision=bool(base64_images), force_preferred_model=force_preferred_model)
                else:
                    for preferred_model in preferred_models:
                        if preferred_model.model_key not in instance.failed_models:
                            model = preferred_model
                            break
                
                # If no model is available, clear failed models and retry
                if not model:
                    print(colored(f"# # # Could not find valid model # # # RETRYING... # # #", "red"))
                    instance.failed_models.clear()
                    if preferred_models and isinstance(preferred_models[0], str):
                        model = instance.get_model(strength=strength, preferred_models=preferred_models, chat=chat, force_local=force_local, force_free=force_free, has_vision=bool(base64_images), force_preferred_model=force_preferred_model)

                if re_print_prompt:
                    print(colored(f"\n\nPROMPT: {chat.messages[-1][1]}", "blue"))
                if use_cache:
                    cached_completion = instance._get_cached_completion(model.model_key, str(temperature), chat, base64_images)
                    if cached_completion:
                        if not silent_reason:
                            print(colored(f"Successfully fetched from cache instead of <{colored(model.provider.__module__, 'green')}> <{colored(model.model_key, 'green')}>","blue"))
                            for char in cached_completion:
                                print(tooling.apply_color(char), end="")
                                time.sleep(0) # better observable for the user
                            print()
                        return preprocess_response(cached_completion)

                response = model.provider.generate_response(chat, model.model_key, temperature, silent_reason)
                instance.last_used_model = model.model_key
                instance._update_cache(model.model_key, str(temperature), chat, base64_images, response)
                
                # Save the chat completion pair if requested
                if not force_local:
                    instance._save_chat_completion_pair(chat, response, model.model_key)
                
                response = preprocess_response(response)
                return start_response_with + response if include_start_response_str else response

            except Exception as e:
                if "too large" in str(e):
                    # Save the model's maximum token limit
                    instance._save_dynamic_token_limit_for_model(model, chat.count_tokens())
                
                if 'model' in locals() and model is not None:
                    if model.model_key in instance.failed_models:
                        return None
                    instance.failed_models.add(model.model_key)
                    instance.retry_models.remove(model)
                # Do not print Groq rate limit errors
                if (not ("Groq" in str(e) and "rate_limit_exceeded" in str(e))):
                    if 'model' in locals() and model is not None:
                        print(colored(f"generate_completion error with model {model.model_key}: {e}", "red"))
                        logger.error(f"generate_completion error with model {model.model_key}: {e}")
                    else:
                        print(colored(f"generate_completion error: {e}", "red"))
                        logger.error(f"generate_completion error: {e}")

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
            with open(g.DYNAMIC_MODEL_LIMITS_PATH, 'w') as f:
                json.dump(self._model_limits, f, indent=4)
            
            # logger.info(f"Updated token limit for {model.model_key}: {token_count} tokens")
        except Exception as limit_error:
            logger.error(f"Failed to save model token limit: {limit_error}")

    @classmethod
    def _save_chat_completion_pair(cls, chat: Chat, response: str, model_key: str) -> None:
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
            cleaned_debug_title = chat.debug_title.replace(" ", "_")
            filename = os.path.join(g.UNCONFIRMED_FINETUNING_PATH, f'completion_pair_{cleaned_debug_title}_{timestamp}.jsonl')
            
            # Create the training example
            training_example = {
                "input": chat.to_openai(),  # Convert chat to OpenAI format
                "output": response,
                "metadata": {
                    "model": model_key,
                    "timestamp": timestamp,
                    "debug_title": cleaned_debug_title
                }
            }
            
            # Save to JSONL file
            with open(filename, 'a') as f:
                f.write(json.dumps(training_example) + '\n')
                
            logger.info(f"Saved chat completion pair to {filename}")
        except Exception as e:
            logger.error(f"Failed to save chat completion pair: {e}")
    
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
        
