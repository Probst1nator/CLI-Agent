import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Set
from termcolor import colored
from py_classes.ai_providers.cls_human_as_interface import HumanAPI
from py_classes.cls_custom_coloring import CustomColoring
from py_classes.cls_chat import Chat, Role
from enum import Enum
from py_classes.ai_providers.cls_anthropic_interface import AnthropicAPI
from py_classes.cls_ai_provider_interface import ChatClientInterface
from py_classes.ai_providers.cls_groq_interface import GroqAPI
from py_classes.ai_providers.cls_ollama_interface import OllamaClient
from py_classes.ai_providers.cls_openai_interface import OpenAIAPI
from py_methods.logger import logger
from py_classes.globals import g

class AIStrengths(Enum):
    """Enum class to represent AI model strengths."""
    STRONG = 2
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
        max_output: Optional[int],
        strength: AIStrengths, 
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
        self.max_output = max_output
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
            # Llm(GroqChat(), "llama-3.1-405b-reasoning", None, False, False, 131072, None, AIStrengths.STRONG),
            Llm(HumanAPI(), "human", None, True, True, True, 131072, 30000, AIStrengths.STRONG),
            Llm(GroqAPI(), "llama-3.1-70b-versatile", None, False, False, False, 131072, 30000, AIStrengths.STRONG),
            Llm(GroqAPI(), "llama-3.1-8b-instant", None, False, False, False, 131072, 30000, AIStrengths.FAST),
            Llm(GroqAPI(), "llama3-70b-8192", None, False, False, False, 8192, 6000, AIStrengths.STRONG),
            Llm(GroqAPI(), "llama3-8b-8192", None, False, False, False, 8192, 30000, AIStrengths.FAST),
            # Llm(GroqChat(), "llama3-groq-70b-8192-tool-use-preview", None, False, False, False, 8192, 30000, AIStrengths.STRONG),
            # Llm(GroqChat(), "llama3-groq-8b-8192-tool-use-preview", None, False, False, False, 8192, 30000, AIStrengths.FAST),
            Llm(GroqAPI(), "gemma2-9b-it", None, False, False, False, 8192, 15000, AIStrengths.FAST),
            
            Llm(AnthropicAPI(), "claude-3-5-sonnet", 9, False, False, False, 200000, 4096, AIStrengths.STRONG),
            Llm(AnthropicAPI(), "claude-3-haiku-20240307", 1, False, False, False, 200000, 4096, AIStrengths.FAST),
            Llm(OpenAIAPI(), "gpt-4o", 10, False, False, True, 128000, None, AIStrengths.STRONG),
            Llm(OpenAIAPI(), "gpt-4o-mini", 0.4, False, False, True, 128000, None, AIStrengths.FAST),
            
            # Llm(OllamaClient(), "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf:latest", None, False, True, False, 4096, None, AIStrengths.STRONG),
            Llm(OllamaClient(), "phi3.5:3.8b", None, False, True, False, 4096, None, AIStrengths.STRONG),
            Llm(OllamaClient(), 'llama3.1:8b', None, True, True, False, 4096, None, AIStrengths.STRONG),
            Llm(OllamaClient(), "llava-llama3:8b", None, False, True, True, 4096, None, AIStrengths.STRONG),
            Llm(OllamaClient(), "llava-phi3:3.8b", None, False, True, True, 4096, None, AIStrengths.FAST),
            Llm(OllamaClient(), "mistral-nemo:12b", None, False, True, True, 128000, None, AIStrengths.STRONG),
        ]


class LlmRouter:
    """
    Singleton class for routing and managing LLM requests.
    """

    _instance: Optional["LlmRouter"] = None
    call_counter: int = 0
    last_used_model: str = ""
    
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
        self.cache_file_path = f"{g.PROJ_VSCODE_DIR_PATH}/llm_cache.json"
        
        # Load cache and initialize retry models and failed models set
        self.cache = self._load_cache()
        self.retry_models = Llm.get_available_llms()
        self.failed_models: Set[str] = set()

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
        """
        Load the cache from a JSON file.
        
        Returns:
            Dict[str, str]: The cache dictionary.
        """
        # Load cache from file or return empty dict if file doesn't exist or is invalid
        if not os.path.exists(self.cache_file_path):
            return {}
        with open(self.cache_file_path, "r") as json_file:
            try:
                return json.load(json_file)
            except json.JSONDecodeError:
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
        
        Args:
            model_key (str): Model identifier.
            temperature (str): Temperature setting for the model.
            chat (Chat): The chat prompt.
            images (List[str]): List of image encodings.
            completion (str): The generated completion string.
        """
        # Generate cache key, update cache, and save to file
        cache_key = self._generate_hash(model_key, temperature, chat.to_json(), images)
        self.cache[cache_key] = completion
        try:
            with open(self.cache_file_path, "w") as json_file:
                json.dump(self.cache, json_file, indent=4)
        except Exception as e:
            logger.error(f"Failed to update cache: {e}")

    @classmethod
    def get_models(cls, preferred_model_keys: List[str] = [], strength: Optional[AIStrengths] = None, chat: Chat = Chat(), force_local: bool = False, force_free: bool = False, has_vision: bool = False) -> List[Llm]:
        """
        Get a list of available models based on the given constraints.
        
        Args:
            preferred_model_keys (List[str]): List of preferred model keys.
            strength (AIStrengths): The required strength of the model.
            chat (Chat): The chat which the model will be processing.
            force_local (bool): Whether to force local models only.
            force_free (bool): Whether to force free models only.
            has_vision (bool): Whether to require models with vision capability.

        Returns:
            List[Llm]: A list of available Llm instances that meet the specified criteria.
        """
        instance = cls()
        available_models: List[Llm] = []

        # Check for exact matches in preferred model keys
        for model_key in preferred_model_keys:
            if model_key and model_key not in instance.failed_models:
                model = next((model for model in instance.retry_models if model_key in model.model_key), None)
                if model and instance.model_capable_check(model, chat, strength, model.local, force_free, has_vision):
                    available_models.append(model)

        # Check all models based on capabilities
        for model in instance.retry_models:
            if model.model_key not in instance.failed_models and not model.model_key in [model.model_key for model in available_models]:
                if (not force_local or model.local) and instance.model_capable_check(model, chat, strength, model.local, force_free, has_vision):
                    available_models.append(model)

        return available_models

    def get_model(self, preferred_model_keys: List[str] = [], strength: Optional[AIStrengths] = None, chat: Chat = Chat(), force_local: bool = False, force_free: bool = False, has_vision: bool = False, force_preferred_model: bool = False) -> Optional[Llm]:
        """
        Route to the next available model based on the given constraints.
        
        Args:
            preferred_model_keys (List[str]): List of preferred model keys.
            strength (AIStrengths): The required strength of the model.
            chat (Chat): The chat which the model will be processing.
            force_local (bool): Whether to force local models only.
            force_free (bool): Whether to force free models only.
            has_vision (bool): Whether to require models with vision capability.

        Returns:
            Optional[Llm]: The next available Llm instance if available, otherwise None.
        """
        # Debug print for large token counts
        if (chat.count_tokens() > 4000 and not force_free and not force_local):
            print(colored("DEBUG: chat.count_tokens() returned: " + str(chat.count_tokens()), "yellow"))
        
        # Search for model key match first
        for model_key in preferred_model_keys:
            if (model_key not in self.failed_models or force_preferred_model) and model_key:
                model = next((model for model in self.retry_models if model_key in model.model_key and (force_local == False or force_local == model.local)), None)
                if model:
                    return model

        if force_preferred_model:
            # return a dummy model to force Ollama to download it
            return Llm(OllamaClient(), preferred_model_keys[0], 0, True, True, True, 8192, 8192, AIStrengths.STRONG)
        
        # Search online models by capability next
        if not force_local:
            for model in self.retry_models:
                if model.model_key not in self.failed_models and not model.local:
                    if self.model_capable_check(model, chat, strength, local=False, force_free=force_free, has_vision=has_vision):
                        return model
            for model in self.retry_models:
                if model.model_key not in self.failed_models and not model.local:
                    if self.model_capable_check(model, chat, None, local=False, force_free=force_free, has_vision=has_vision):
                        return model

        # search local models last
        # first by capability
        for model in self.retry_models:
            if model.model_key not in self.failed_models and model.local:
                if self.model_capable_check(model, chat, strength, local=True, force_free=force_free, has_vision=has_vision):
                    return model
        # ignore strength if no model is found
        for model in self.retry_models:
            if model.model_key not in self.failed_models and model.local:
                if self.model_capable_check(model, chat, None, local=True, force_free=force_free, has_vision=has_vision):
                    return model
        # ignore context_length and strength if no model is found
        for model in self.retry_models:
            if model.model_key not in self.failed_models:
                if (model.local):
                    if self.model_capable_check(model, Chat(), None, local=True, force_free=force_free, has_vision=has_vision):
                        return model

        return None
    
    def model_capable_check(self, model: Llm, chat: Chat, strength: Optional[AIStrengths], local: bool, force_free: bool = False, has_vision: bool = False) -> bool:
        """
        Check if a model is capable of handling the given constraints.
        
        Args:
            model (Llm): The model to check.
            chat (Chat): The chat to process.
            strength (AIStrengths): The required strength.
            local (bool): Whether the model should be local.
            force_free (bool): Whether to force free models only.
            has_vision (bool): Whether vision capability is required.

        Returns:
            bool: True if the model is capable, False otherwise.
        """
        if force_free and model.pricing_in_dollar_per_1M_tokens is not None:
            return False
        if has_vision and not model.has_vision:
            return False
        if model.context_window < chat.count_tokens():
            return False
        if strength:
            if model.strength.value != strength.value:
                return False
        if model.local != local:
            return False
        return True

    @classmethod
    def generate_completion(
        cls,
        chat: Chat|str,
        preferred_model_keys: List[str] = [],
        strength: AIStrengths = AIStrengths.STRONG,
        start_response_with: str = "",
        instruction: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        base64_images: List[str] = [],
        include_start_response_str: bool = True,
        use_cache: bool = True,
        force_local: bool = False,
        force_free: bool = False,
        force_preferred_model: bool = False,
        silent: bool = False,
        re_print_prompt: bool = False,
    ) -> str:
        """
        Generate a completion response using the appropriate LLM.
        
        Args:
            chat (Chat|str): The chat prompt or string.
            preferred_model_keys (List[str]): List of preferred model keys.
            strength (Optional[AIStrengths]): The required strength of the model.
            start_response_with (str): Initial string to start the response with.
            instruction (str): Instruction for the chat.
            temperature (float): Temperature setting for the model.
            base64_images (List[str]): List of base64-encoded images.
            include_start_response_str (bool): Whether to include the start response string.
            use_cache (bool): Whether to use the cache.
            force_local (Optional[bool]): Whether to force local models only.
            force_free (bool): Whether to force free models only.
            silent (bool): Whether to suppress output.

        Returns:
            str: The generated completion string.
        """
        instance = cls()
        tooling = CustomColoring()
        cls.call_counter += 1
        
        # Convert string input to Chat object if necessary
        if isinstance(chat, str):
            chat = Chat(instruction).add_message(Role.USER, chat)
        if start_response_with:
            chat.add_message(Role.ASSISTANT, start_response_with)
        
        if base64_images:
            chat.base64_images = base64_images
        
        if not preferred_model_keys or preferred_model_keys == [""]:
            preferred_model_keys = []
        
        while True:
            try:
                # Get an appropriate model
                model = instance.get_model(strength=strength, preferred_model_keys=preferred_model_keys, chat=chat, force_local=force_local, force_free=force_free, has_vision=bool(base64_images), force_preferred_model=force_preferred_model)
                
                # If no model is available, clear failed models and retry
                if not model:
                    print(colored(f"# # # All models failed # # # RETRYING... # # #", "red"))
                    instance.failed_models.clear()
                    model = instance.get_model(strength=strength, preferred_model_keys=preferred_model_keys, chat=chat, force_local=force_local, force_free=force_free, has_vision=bool(base64_images), force_preferred_model=force_preferred_model)

                if re_print_prompt:
                    print(colored(f"\n\nPROMPT: {chat.messages[-1][1]}", "blue"))
                if use_cache:
                    cached_completion = instance._get_cached_completion(model.model_key, str(temperature), chat, base64_images)
                    if cached_completion:
                        if not silent:
                            print(colored(f"Successfully fetched from cache instead of <{colored(model.provider.__module__, 'green')}>","blue"))
                            for char in cached_completion:
                                print(tooling.apply_color(char), end="")
                                time.sleep(0) # better observable for the user
                            print()
                        return cached_completion

                response = model.provider.generate_response(chat, model.model_key, temperature, silent)
                instance.last_used_model = model.model_key
                instance._update_cache(model.model_key, str(temperature), chat, base64_images, response)
                return start_response_with + response if include_start_response_str else response

            except Exception as e:
                print(colored(f"Error with model {model.model_key}: {e}", "red"))
                logger.error(f"Error with model {model.model_key}: {e}")
                instance.failed_models.add(model.model_key)

    @classmethod
    def generate_completion_raw(
        cls,
        chat: Chat|str,
        model: Llm,
        start_response_with: str = "",
        instruction: str = "You are a helpful assistant.",
        temperature: float = 0.75,
        base64_images: List[str] = [],
        include_start_response_str: bool = True,
        use_cache: bool = True,
        silent: bool = False
    ) -> Optional[str]:
        """
        Generate a completion response using the specified LLM.
        
        Args:
            chat (Chat|str): The chat prompt or string.
            model (Llm): The specific model to use.
            strength (AIStrengths): The required strength of the model.
            start_response_with (str): Initial string to start the response with.
            instruction (str): Instruction for the chat.
            temperature (float): Temperature setting for the model.
            base64_images (List[str]): List of base64-encoded images.
            include_start_response_str (bool): Whether to include the start response string.
            use_cache (bool): Whether to use the cache.
            force_local (Optional[bool]): Whether to force local models only.
            force_free (bool): Whether to force free models only.
            silent (bool): Whether to suppress output.

        Returns:
            Optional[str]: The generated completion string, or None if the model fails to generate a completion.
        """
        instance = cls()
        tooling = CustomColoring()
        cls.call_counter += 1
        
        # Convert string input to Chat object if necessary
        if isinstance(chat, str):
            chat = Chat(instruction).add_message(Role.USER, chat)
        if start_response_with:
            chat.add_message(Role.ASSISTANT, start_response_with)
        
        if base64_images:
            chat.base64_images = base64_images
        
        try:
            if use_cache:
                cached_completion = instance._get_cached_completion(model.model_key, str(temperature), chat, base64_images)
                if cached_completion:
                    if not silent:
                        print(colored(f"Successfully fetched from cache instead of <{colored(model.provider.__module__, 'green')}>","blue"))
                        for char in cached_completion:
                            print(tooling.apply_color(char), end="")
                            time.sleep(0.001) # better observable for the user
                        print()
                    return cached_completion

            response = model.provider.generate_response(chat, model.model_key, temperature, silent)
            instance.last_used_model = model.model_key
            instance._update_cache(model.model_key, str(temperature), chat, base64_images, response)
            return start_response_with + response if include_start_response_str else response

        except Exception as e:
            logger.error(f"Error with model {model.model_key}: {e}")
            return None
