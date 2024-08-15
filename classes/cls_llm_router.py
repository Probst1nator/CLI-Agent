import hashlib
import json
import os
from typing import Dict, List, Optional, Set

from termcolor import colored

from classes.cls_custom_coloring import CustomColoring
from classes.cls_chat import Chat, Role
from enum import Enum
from classes.ai_providers.cls_anthropic_interface import AnthropicChat
from classes.cls_ai_provider_interface import ChatClientInterface
from classes.ai_providers.cls_groq_interface import GroqChat
from classes.ai_providers.cls_ollama_interface import OllamaClient
from classes.ai_providers.cls_openai_interface import OpenAIChat
from logger import logger

class AIStrengths(Enum):
    STRONG = 2
    FAST = 1

class Llm:
    def __init__(
        self, 
        provider: ChatClientInterface, 
        model_key: str, 
        pricing_in_dollar_per_1M_tokens: Optional[float], 
        available_local: bool, 
        has_vision: bool, 
        context_window: int, 
        max_output: Optional[int],
        strength: AIStrengths, 
    ):
        self.provider = provider
        self.model_key = model_key
        self.pricing_in_dollar_per_1M_tokens = pricing_in_dollar_per_1M_tokens
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
        return [
            Llm(GroqChat(), "llama-3.1-405b-reasoning", None, False, True, 131072, None, AIStrengths.STRONG),
            Llm(GroqChat(), "llama-3.1-70b-versatile", None, False, False, 131072, 30000, AIStrengths.STRONG),
            Llm(GroqChat(), "llama-3.1-8b-instant", None, False, False, 131072, 30000, AIStrengths.FAST),
            Llm(GroqChat(), "llama3-70b-8192", None, False, False, 8192, 6000, AIStrengths.STRONG),
            Llm(GroqChat(), "llama3-8b-8192", None, False, False, 8192, 30000, AIStrengths.FAST),
            Llm(GroqChat(), "llama3-groq-70b-8192-tool-use-preview", None, False, False, 8192, 30000, AIStrengths.STRONG),
            Llm(GroqChat(), "llama3-groq-8b-8192-tool-use-preview", None, False, False, 8192, 30000, AIStrengths.FAST),
            Llm(GroqChat(), "gemma2-9b-it", None, False, False, 8192, 15000, AIStrengths.FAST),
            
            Llm(AnthropicChat(), "claude-3-5-sonnet", 9, False, False, 200000, 4096, AIStrengths.STRONG),
            Llm(AnthropicChat(), "claude-3-haiku-20240307", 1, False, False, 200000, 4096, AIStrengths.FAST),
            Llm(OpenAIChat(), "gpt-4o", 10, False, True, 128000, None, AIStrengths.STRONG),
            Llm(OpenAIChat(), "gpt-4o-mini", 0.4, False, True, 128000, None, AIStrengths.FAST),
            
            Llm(OllamaClient(), 'llama3.1', None, True, False, 4096, None, AIStrengths.STRONG),
            Llm(OllamaClient(), "phi3:3.8b", None, True, False, 4096, None, AIStrengths.FAST),
            Llm(OllamaClient(), "llava-llama3", None, True, True, 4096, None, AIStrengths.STRONG),
            Llm(OllamaClient(), "llava-phi3", None, True, True, 4096, None, AIStrengths.FAST),
            Llm(OllamaClient(), "WizardLM-2-7B-abliterated-Q4_K_M.gguf", None, True, True, 4096, None, AIStrengths.STRONG),
            # Llm(OllamaClient(), "nuextract", None, True, True, 4096, None, AIStrengths.STRONG),
            # Llm(OllamaClient(), "mistral-nemo", None, True, True, 128000, None, AIStrengths.STRONG),
            Llm(OllamaClient(), "deepseek-v2", None, True, True, 4096, None, AIStrengths.STRONG),
            Llm(OllamaClient(), "phi3:medium-128k", None, True, True, 4096, None, AIStrengths.STRONG),
        ]


class LlmRouter:
    _instance: Optional["LlmRouter"] = None
    call_counter: int = 0
    
    def __new__(cls, *args, **kwargs) -> "LlmRouter":
        if cls._instance is None:
            cls._instance = super(LlmRouter, cls).__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self) -> None:
        user_cli_agent_dir = os.path.expanduser('~/.local/share') + "/cli-agent"
        os.makedirs(user_cli_agent_dir, exist_ok=True)
        self.cache_file_path = f"{user_cli_agent_dir}/llm_cache.json"
        self.cache = self._load_cache()
        self.retry_models = Llm.get_available_llms()
        self.failed_models: Set[str] = set()

    def _generate_hash(self, model_key: str, temperature: str, prompt: str, images: List[str]) -> str:
        """
        Generate a hash for caching based on model, temperature, prompt, and images.
        
        Args:
            model (str): Model identifier.
            temperature (str): Temperature setting for the model.
            prompt (str): The prompt string.
            images (List[str]): List of image encodings.

        Returns:
            str: The generated hash string.
        """
        hash_input = f"{model_key}:{temperature}:{prompt}{':'.join(images)}".encode()
        return hashlib.sha256(hash_input).hexdigest()

    def _load_cache(self) -> Dict[str, str]:
        """
        Load the cache from a JSON file.
        
        Returns:
            Dict[str, str]: The cache dictionary.
        """
        # with open(self.cache_file_path, "w") as json_file:
        #     json.dump(self.cache, json_file, indent=4)
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
            prompt (Chat): The chat prompt.
            images (List[str]): List of image encodings.

        Returns:
            Optional[str]: The cached completion string if available, otherwise None.
        """
        cache_key = self._generate_hash(model_key, temperature, chat.to_json(), images)
        return self.cache.get(cache_key)

    def _update_cache(self, model_key: str, temperature: str, chat: Chat, images: List[str], completion: str) -> None:
        """
        Update the cache with a new completion.
        
        Args:
            model_key (str): Model identifier.
            temperature (str): Temperature setting for the model.
            prompt (Chat): The chat prompt.
            images (List[str]): List of image encodings.
            completion (str): The generated completion string.
        """
        cache_key = self._generate_hash(model_key, temperature, chat.to_json(), images)
        self.cache[cache_key] = completion
        try:
            with open(self.cache_file_path, "w") as json_file:
                json.dump(self.cache, json_file, indent=4)
        except Exception as e:
            logger.error(f"Failed to update cache: {e}")


    def get_model(self, preferred_model_keys: List[str], strength: AIStrengths, chat: Chat, force_local: bool = False, force_free: bool = False, has_vision: bool = False) -> Optional[Llm]:
        """
        Route to the next available model based on the given constraints.
        
        Args:
            current_model (str): The current model identifier.
            chat (chat): The chat which the model will be processing.
            force_local (bool): Whether to force local models only.
            force_free (bool): Whether to force free models only.
            has_vision (bool): Whether to require models with vision capability.

        Returns:
            Optional[str]: The next model identifier if available, otherwise None.
        """
        if (chat.count_tokens()>4000 and not force_free and not force_local):
            print(colored("DEBUG: chat.count_tokens() returned: " + str(chat.count_tokens()), "yellow"))
        
        # search for exact model key match first
        for model_key in preferred_model_keys:
            if model_key not in self.failed_models and model_key:
                model = next((model for model in self.retry_models if model_key in model.model_key), None)
                if model:
                    return model
        # search online models by capability next
        if not force_local:
            for model in self.retry_models:
                if model.model_key not in self.failed_models:
                    if (not model.local):
                        if self.model_capable_check(model, chat, strength, local=False, force_free=force_free, has_vision=has_vision):
                            return model
            for model in self.retry_models:
                if model.model_key not in self.failed_models:
                    if (not model.local):
                        if self.model_capable_check(model, chat, None, local=False, force_free=force_free, has_vision=has_vision):
                            return model
        # search local models last
        # first by capability
        for model in self.retry_models:
            if model.model_key not in self.failed_models:
                if (model.local):
                    if self.model_capable_check(model, chat, strength, local=True, force_free=force_free, has_vision=has_vision):
                        return model
        # ignore strength if no model is found
        for model in self.retry_models:
            if model.model_key not in self.failed_models:
                if (model.local):
                    if self.model_capable_check(model, chat, None, local=True, force_free=force_free, has_vision=has_vision):
                        return model
        # ignore context_length and strength if no model is found
        for model in self.retry_models:
            if model.model_key not in self.failed_models:
                if (model.local):
                    if self.model_capable_check(model, Chat(), None, local=True, force_free=force_free, has_vision=has_vision):
                        return model
        return None
    
    def model_capable_check(self, model: Llm, chat: Chat, strength: AIStrengths, local: bool, force_free: bool = False, has_vision: bool = False) -> bool:
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
        instruction: str = "You are a helpful assistant",
        temperature: float = 0.75,
        base64_images: List[str] = [],
        include_start_response_str: bool = True,
        use_cache: bool = True,
        force_local: Optional[bool] = None,
        force_free: bool = False,
        silent: bool = False
    ) -> Optional[str]:
        """
        Generate a completion response using the appropriate LLM.
        
        Args:
            prompt (Chat): The chat prompt.
            model_key (str): Model identifier.
            start_response_with (str): Initial string to start the response with.
            instruction (str): Instruction for the chat.
            temperature (float): Temperature setting for the model.
            base64_images (List[str]): List of base64-encoded images.
            include_start_response_str (bool): Whether to include the start response string.
            ignore_cache (bool): Whether to ignore the cache.
            force_local (Optional[bool]): Whether to force local models only.
            force_free (bool): Whether to force free models only.
            silent (bool): Whether to suppress output.

        Returns:
            Optional[str]: The generated completion string if successful, otherwise None.
        """
        instance = cls()
        tooling = CustomColoring()
        cls.call_counter += 1
        
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
                model = instance.get_model(strength=strength, preferred_model_keys=preferred_model_keys, chat=chat, force_local=force_local, force_free=force_free, has_vision=bool(base64_images))
                if not model:
                    print(colored(f"# # # All models failed # # # RETRYING... # # #", "red"))
                    instance.failed_models.clear()
                    model = instance.get_model(strength=strength, preferred_model_keys=preferred_model_keys, chat=chat, force_local=force_local, force_free=force_free, has_vision=bool(base64_images))

                if use_cache:
                    cached_completion = instance._get_cached_completion(model.model_key, str(temperature), chat, base64_images)
                    if cached_completion:
                        if not silent:
                            print(colored(f"Successfully fetched from cache instead of <{colored(model.provider.__module__, 'green')}>","blue"))
                            for char in cached_completion:
                                print(tooling.apply_color(char), end="")
                            print()
                        return cached_completion

                response = model.provider.generate_response(chat, model.model_key, temperature, silent)
                instance._update_cache(model.model_key, str(temperature), chat, base64_images, response)
                return start_response_with + response if include_start_response_str else response

            except Exception as e:
                logger.error(f"Error with model {model.model_key}: {e}")
                instance.failed_models.add(model.model_key)
