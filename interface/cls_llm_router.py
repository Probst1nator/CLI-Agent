import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Set

from termcolor import colored

from cls_custom_coloring import CustomColoring
from interface.cls_chat import Chat, Role
from enum import Enum
from interface.cls_anthropic_interface import AnthropicChat
from interface.cls_chat_client_interface import ChatClientInterface
from interface.cls_groq_interface import GroqChat
from interface.cls_ollama_client import OllamaClient
from interface.cls_openai_interface import OpenAIChat

# class LlmProviders(Enum):
#     AnthropicChat = AnthropicChat
#     GroqChat = GroqChat
#     OllamaClient = OllamaClient
#     OpenAIChat = OpenAIChat

class AIStrengths(Enum):
    STRONG = 2
    MEDIUM = 1
    WEAK = 0

class Llm:
    def __init__(
        self, 
        provider: ChatClientInterface, 
        model_key: str, 
        pricing_in_dollar_per_1M_tokens: Optional[int], 
        available_local: bool, 
        has_vision: bool, 
        context_window: int, 
        max_output: Optional[int],
        strength: AIStrengths, 
    ):
        self.provider = provider
        self.model_key = model_key
        self.pricing_in_dollar_per_1M_tokens = pricing_in_dollar_per_1M_tokens
        self.available_local = available_local
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
            Llm(GroqChat(), "llama3-70b-8192", None, False, False, 8192, 6000, AIStrengths.MEDIUM),
            Llm(GroqChat(), "gemma2-9b-it", None, False, False, 8192, 15000, AIStrengths.MEDIUM),
            Llm(GroqChat(), "mixtral-8x7b-32768", None, False, False, 32768, 5000, AIStrengths.MEDIUM),
            Llm(GroqChat(), "llama3-8b-8192", None, False, True, 8192, 30000, AIStrengths.WEAK),
            Llm(AnthropicChat(), "claude-3-5-sonnet", 9, False, False, 200000, 4096, AIStrengths.STRONG),
            Llm(AnthropicChat(), "claude-3-haiku-20240307", 1, False, False, 200000, 4096, AIStrengths.WEAK),
            Llm(OpenAIChat(), "gpt-4o", 10, False, True, 128000, None, AIStrengths.STRONG),
            Llm(OllamaClient(), "phi3", None, True, False, 4096, None, AIStrengths.WEAK),
            Llm(OllamaClient(), "llava-phi3", None, True, True, 4096, None, AIStrengths.WEAK),
        ]

class LlmRouter:
    _instance: Optional["LlmRouter"] = None
    
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

    def _generate_hash(self, model: str, temperature: str, prompt: str, images: List[str]) -> str:
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
        hash_input = f"{model}:{temperature}:{prompt}{':'.join(images)}".encode()
        return hashlib.sha256(hash_input).hexdigest()

    def _load_cache(self) -> Dict[str, str]:
        """
        Load the cache from a JSON file.
        
        Returns:
            Dict[str, str]: The cache dictionary.
        """
        if not os.path.exists(self.cache_file_path):
            return {}
        with open(self.cache_file_path, "r") as json_file:
            try:
                return json.load(json_file)
            except json.JSONDecodeError:
                return {}

    def _get_cached_completion(self, model: str, temperature: str, prompt: Chat, images: List[str]) -> Optional[str]:
        """
        Retrieve a cached completion if available.
        
        Args:
            model (str): Model identifier.
            temperature (str): Temperature setting for the model.
            prompt (Chat): The chat prompt.
            images (List[str]): List of image encodings.

        Returns:
            Optional[str]: The cached completion string if available, otherwise None.
        """
        cache_key = self._generate_hash(model, temperature, prompt.to_json(), images)
        return self.cache.get(cache_key)

    def _update_cache(self, model: str, temperature: str, prompt: Chat, images: List[str], completion: str) -> None:
        """
        Update the cache with a new completion.
        
        Args:
            model (str): Model identifier.
            temperature (str): Temperature setting for the model.
            prompt (Chat): The chat prompt.
            images (List[str]): List of image encodings.
            completion (str): The generated completion string.
        """
        cache_key = self._generate_hash(model, temperature, prompt.to_json(), images)
        self.cache[cache_key] = completion
        try:
            with open(self.cache_file_path, "w") as json_file:
                json.dump(self.cache, json_file, indent=4)
        except Exception as e:
            logging.error(f"Failed to update cache: {e}")


    def get_model(self, model_key: str, min_strength: AIStrengths, chat: Chat, force_local: bool = False, force_free: bool = False, has_vision: bool = False) -> Optional[Llm]:
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
        print(colored("DEBUG: chat.count_tokens() returned: " + str(chat.count_tokens()), "yellow"))
        if model_key not in self.failed_models and model_key:
            model = next((model for model in self.retry_models if model.model_key == model_key and model.context_window > chat.count_tokens()), None)
            if model:
                return model
        for model in self.retry_models:
            if model.model_key not in self.failed_models:
                if self.model_capable_check(model, chat, min_strength, force_local, force_free, has_vision):
                    return model
        return None
    
    def model_capable_check(self, model: Llm, chat: Chat, min_strength: AIStrengths = AIStrengths.WEAK, force_local: bool = False, force_free: bool = False, has_vision: bool = False) -> bool:
        if force_local and not model.available_local:
            return False
        if force_free and model.pricing_in_dollar_per_1M_tokens is not None:
            return False
        if has_vision and not model.has_vision:
            return False
        if model.context_window < chat.count_tokens():
            return False
        if min_strength:
            if model.strength.value < min_strength.value:
                return False
        
        return True

    @classmethod
    def generate_completion(
        cls,
        chat: Chat,
        model_key: str = "",
        min_strength: AIStrengths = AIStrengths.WEAK,
        start_response_with: str = "",
        instruction: str = "The highly advanced AI assistant provides thorough responses to the user. It displays a deep understanding and offers expert service in whatever domain the user requires.",
        temperature: float = 0.8,
        base64_images: List[str] = [],
        include_start_response_str: bool = True,
        ignore_cache: bool = False,
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
            **kwargs: Additional keyword arguments.

        Returns:
            Optional[str]: The generated completion string if successful, otherwise None.
        """
        instance = cls()
        tooling = CustomColoring()

        if isinstance(chat, str):
            chat = Chat(instruction).add_message(Role.USER, chat)
        if start_response_with:
            chat.add_message(Role.ASSISTANT, start_response_with)
        
        while True:
            try:
                model = instance.get_model(min_strength=min_strength, model_key=model_key, chat=chat, force_local=force_local, force_free=force_free, has_vision=bool(base64_images))
                if not model:
                    print(colored(f"All models failed.", "red"))
                    return None

                if not ignore_cache:
                    cached_completion = instance._get_cached_completion(model.model_key, str(temperature), chat, base64_images)
                    if cached_completion:
                        if not silent:
                            for char in cached_completion:
                                print(tooling.apply_color(char), end="")
                            print()
                        return cached_completion

                if base64_images:
                    response = OllamaClient.generate_response(chat, model.model_key, temperature, silent, base64_images)
                    instance._update_cache(model.model_key, str(temperature), chat, base64_images, response)
                    return start_response_with + response if include_start_response_str else response
                else:
                    response = model.provider.generate_response(chat, model.model_key, temperature, silent)
                    instance._update_cache(model.model_key, str(temperature), chat, [], response)
                    return start_response_with + response if include_start_response_str else response

            except Exception as e:
                logging.error(f"Error with model {model.model_key}: {e}")
                instance.failed_models.add(model.model_key)
