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
from interface.cls_groq_interface import GroqChat
from interface.cls_ollama_client import OllamaClient
from interface.cls_openai_interface import OpenAIChat

class LlmProviders(Enum):
    AnthropicChat = AnthropicChat
    GroqChat = GroqChat
    OllamaClient = OllamaClient
    OpenAIChat = OpenAIChat

class Llm:
    def __init__(
        self, 
        provider: LlmProviders, 
        model_key: str, 
        pricing_in_dollar_per_1M_tokens: Optional[int], 
        available_local: bool, 
        has_vision: bool, 
        context_window: int, 
        max_output: Optional[int]
    ):
        self.provider = provider
        self.model_key = model_key
        self.pricing_in_dollar_per_1M_tokens = pricing_in_dollar_per_1M_tokens
        self.available_local = available_local
        self.has_vision = has_vision
        self.context_window = context_window
        self.max_output = max_output
    
    @classmethod
    def get_available_llms(cls) -> List["Llm"]:
        """
        Get the list of available LLMs.
        
        Returns:
            List[Llm]: A list of Llm instances representing the available models.
        """
        return [
            Llm(LlmProviders.GroqChat, "llama3-70b-8192", None, False, False, 8192, 6000),
            Llm(LlmProviders.GroqChat, "llama3-8b-8192", None, False, True, 8192, 30000),
            Llm(LlmProviders.AnthropicChat, "claude-3-5-sonnet", 9, False, False, 200000, 4096),
            Llm(LlmProviders.OpenAIChat, "gpt4-o", 10, False, True, 128000, None),
            Llm(LlmProviders.GroqChat, "mixtral-8x7b-32768", None, False, False, 32768, 5000),
            Llm(LlmProviders.GroqChat, "gemma-7b-it", None, False, False, 8192, 15000),
            Llm(LlmProviders.OllamaClient, "phi3", None, False, False, 4096, None),
            Llm(LlmProviders.OllamaClient, "llava-phi3", None, False, True, 4096, None),
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

    def route_to_next_model(self, current_model: str, force_local: bool = False, force_free: bool = False, has_vision: bool = False) -> Optional[str]:
        """
        Route to the next available model based on the given constraints.
        
        Args:
            current_model (str): The current model identifier.
            force_local (bool): Whether to force local models only.
            force_free (bool): Whether to force free models only.
            has_vision (bool): Whether to require models with vision capability.

        Returns:
            Optional[str]: The next model identifier if available, otherwise None.
        """
        self.failed_models.add(current_model)
        for model in self.retry_models:
            if model.model_key not in self.failed_models:
                if force_local and not model.available_local:
                    continue
                if force_free and model.pricing_in_dollar_per_1M_tokens is not None:
                    continue
                if has_vision and not model.has_vision:
                    continue
                return model.model_key
        return None

    @classmethod
    def generate_completion(
        cls,
        prompt: Chat,
        model: str = "",
        start_response_with: str = "",
        instruction: str = "The highly advanced AI assistant provides thorough responses to the user. It displays a deep understanding and offers expert service in whatever domain the user requires.",
        temperature: float = 0.8,
        base64_images: List[str] = [],
        include_start_response_str: bool = True,
        ignore_cache: bool = False,
        force_local: Optional[bool] = None,
        force_free: bool = False,
        silent: bool = False,
        **kwargs,
    ) -> Optional[str]:
        """
        Generate a completion response using the appropriate LLM.
        
        Args:
            prompt (Chat): The chat prompt.
            model (str): Model identifier.
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

        if isinstance(prompt, str):
            prompt = Chat(instruction).add_message(Role.USER, prompt)
        if start_response_with:
            prompt.add_message(Role.ASSISTANT, start_response_with)
            
        if not model:
            model = instance.route_to_next_model(model)

        if not ignore_cache:
            cached_completion = instance._get_cached_completion(model, str(temperature), prompt, base64_images)
            if cached_completion:
                if not silent:
                    for char in cached_completion:
                        print(tooling.apply_color(char), end="")
                    print()
                return cached_completion

        while True:
            try:
                if not force_local and (not model or "llama3" in model or "mixtral" in model or "70b" in model) and "dolphin" not in model:
                    response = GroqChat.generate_response(prompt, model, temperature, silent)
                    instance._update_cache(model, str(temperature), prompt, [], response)
                    return start_response_with + response if include_start_response_str else response

                if not force_local and not force_free and ("claude" in model or len(model) == 0):
                    response = AnthropicChat.generate_response(prompt, model, temperature, silent)
                    instance._update_cache(model, str(temperature), prompt, [], response)
                    return start_response_with + response if include_start_response_str else response

                if not force_local and not force_free and "gpt" in model:
                    response = OpenAIChat.generate_response(prompt, model, temperature, silent)
                    instance._update_cache(model, str(temperature), prompt, [], response)
                    return start_response_with + response if include_start_response_str else response

                response = OllamaClient.generate_completion(prompt, model, temperature, silent, base64_images, **kwargs)
                instance._update_cache(model, str(temperature), prompt, base64_images, response)
                return start_response_with + response if include_start_response_str else response

            except Exception as e:
                logging.error(f"Error with model {model}: {e}")
                next_model = instance.route_to_next_model(model, force_local, force_free, bool(base64_images))
                if not next_model:
                    print(colored(f"All models failed.", "red"))
                    return None
                print(colored(f"{model} failed, retrying with: {next_model}", "red"))
                model = next_model
