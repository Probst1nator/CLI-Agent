from collections.abc import Callable
import hashlib
import json
import os
import shutil
import time
import threading
from typing import Dict, List, Optional, Set, Any, Union, Iterator
from termcolor import colored
import math # <-- Added for token calculation

from core.providers.cls_nvidia_interface import NvidiaAPI
from agent.text_painter.stream_painter import TextStreamPainter
from core.chat import Chat, Role
from core.providers.cls_anthropic_interface import AnthropicAPI
from core.ai_strengths import AIStrengths
from py_classes.unified_interfaces import AIProviderInterface
from core.providers.cls_groq_interface import GroqAPI, TimeoutException, RateLimitException
from core.providers.cls_openai_interface import OpenAIAPI
from core.providers.cls_google_interface import GoogleAPI
from core.llm import Llm
from core.globals import g
from core.cache.llm_cache_manager import LLMCacheManager
import logging

# Get a logger for this module. It will inherit its configuration from the root logger set up in main.py.
logger = logging.getLogger(__name__)

# Custom exception for user interruption
class UserInterruptedException(Exception):
    """Exception raised when the user interrupts model generation (e.g., with Ctrl+C)."""
    pass

# PASTE THE EXCEPTION DEFINITION HERE
class StreamInterruptedException(Exception):
    """
    Control flow exception (not an error) raised by stream callback to signal that a complete
    tool block has been detected and should be executed immediately rather than waiting for
    the full response. This is normal, intended behavior for interactive tool execution.
    """
    def __init__(self, response):
        self.response = response
        super().__init__("Stream completion signal: complete tool block detected")

class ModelGenerationFailedException(Exception):
    """Exception raised when model generation fails after all retry attempts."""
    pass

class LlmRouter:
    """
    Singleton class for routing and managing LLM requests.
    """

    _instance: Optional["LlmRouter"] = None
    call_counter: int = 0
    last_used_model: str = ""
    _model_limits: Dict[str, int] = {}
    _model_limits_loaded: bool = False

    @property
    def discovered_models(self) -> List[Llm]:
        """Get list of discovered models."""
        return self.retry_models

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
        # Set up SQLite cache manager
        cache_db_path = os.path.join(g.CLIAGENT_CACHE_DIR, "llm_cache.db")
        self.cache_manager = LLMCacheManager(cache_db_path)

        # Legacy cache file path for JSON migration (handled by cache manager)
        self.cache_file_path = os.path.join(g.CLIAGENT_CACHE_DIR, "llm_cache.json")
        # Start with static models only to avoid startup delays - dynamic discovery happens when needed
        self.retry_models = Llm.get_available_llms(include_dynamic=False)
        self._dynamic_discovery_done = False  # Track if we've done dynamic discovery
        self._load_dynamic_model_limits()
        self.failed_models: Set[str] = set()
        # Track cache keys used in current runtime to avoid duplicate fetches for non-zero temperature
        self.runtime_used_cache_keys: Set[str] = set()
        # Lock for thread-safe cache access in MCT mode
        self._cache_lock = threading.Lock()
        # Store detailed failure reason for better error messages
        self._last_failure_reason: str = "Unknown reason"
        # Store original error from single model attempts for better reporting
        self._original_single_model_error: str = ""
        self._failed_model_key: str = ""

    def _ensure_dynamic_models_loaded(self) -> None:
        """Load dynamic models if not already loaded."""
        if not self._dynamic_discovery_done:
            # Replace static models with full discovery
            self.retry_models = Llm.get_available_llms(include_dynamic=True)
            self._dynamic_discovery_done = True

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

    def _get_cached_completion(self, model_key: str, key: str, images: List[str], temperature: float = 0) -> Optional[str]:
        """
        Retrieve a cached completion if available.

        Args:
            model_key (str): Model identifier.
            key (str): The chat prompt key.
            images (List[str]): List of image encodings.
            temperature (float): Temperature setting - used to determine cache behavior.

        Returns:
            Optional[str]: The cached completion string if available, otherwise None.
        """
        # Use the SQLite cache manager for retrieval
        cached_result = self.cache_manager.get_cached_completion(
            model_key=model_key,
            prompt=key,
            images=images,
            temperature=temperature
        )

        # Use lock to prevent race conditions in MCT mode with non-zero temperature
        with self._cache_lock:
            # For non-zero temperature, check if we've already used this prompt in current runtime
            runtime_key = f"{model_key}:{key}:{temperature}"
            if temperature != 0 and runtime_key in self.runtime_used_cache_keys:
                return None  # Skip cache to allow variety in Monte Carlo scenarios

            if cached_result and temperature != 0:
                # Track usage for non-zero temperature to ensure variety
                self.runtime_used_cache_keys.add(runtime_key)

            return cached_result

    def _update_cache(self, model_key: str, key: str, images: List[str], completion: str, temperature: float = 0) -> None:
        """
        Update the cache with a new completion.
        """
        # Use the SQLite cache manager for storage
        self.cache_manager.cache_completion(
            model_key=model_key,
            prompt=key,
            response=completion,
            images=images,
            temperature=temperature
        )

    def model_capable_check(self, model: Llm, chat: Chat, strengths: List[AIStrengths], local: bool, force_free: bool = False, has_vision: bool = False, debug_model_key: Optional[str] = None) -> tuple[bool, str | None]:
        """
        Check if a model is capable of handling the given constraints.

        Args:
            model (Llm): The model to check.
            chat (Chat): The chat to process.
            strengths (List[AIStrengths]): The required strengths.
            local (bool): Whether the model should be local.
            force_free (bool): Whether to force free models only.
            has_vision (bool): Whether vision capability is required.
            debug_model_key (str): Optional model key for debugging output.

        Returns:
            tuple[bool, str | None]: (True, None) if capable, (False, reason) if not capable.
        """
        def debug_log(reason: str):
            if debug_model_key:
                logger.warning(f"Model '{debug_model_key}' capability check failed: {reason}")

        if force_free and model.pricing_in_dollar_per_1M_tokens is not None:
            reason = f"not free (costs ${model.pricing_in_dollar_per_1M_tokens} per 1M tokens)"
            debug_log(reason)
            return False, reason
        if has_vision and not model.has_vision:
            reason = "no vision support"
            debug_log(reason)
            return False, reason

        if model.model_key in self._model_limits:
            token_limit = self._model_limits[model.model_key]
            if len(chat) >= token_limit:
                reason = f"exceeds saved token limit ({token_limit} < {len(chat)} tokens)"
                debug_log(reason)
                return False, reason

        if model.get_context_window() < len(chat):
            reason = f"context window too small ({model.get_context_window()} < {len(chat)} tokens)"
            debug_log(reason)
            return False, reason
        if strengths and model.strengths:
            # Check if ALL of the required strengths are included in the model's strengths
            if not all(s.value in [ms.value for ms in model.strengths] for s in strengths):
                missing_strengths = [s.name for s in strengths if s.value not in [ms.value for ms in model.strengths]]
                reason = f"missing required strengths: {missing_strengths}"
                debug_log(reason)
                return False, reason
        if local != model.local:
            reason = f"local mismatch (required: {local}, model: {model.local})"
            debug_log(reason)
            return False, reason
        return True, None

    @classmethod
    def get_models(cls, preferred_models: List[str] = [], strengths: List[AIStrengths] = [], chat: Chat = Chat(), force_local: bool = False, force_free: bool = False, has_vision: bool = False, force_preferred_model: bool = False) -> List[Llm]:
        """
        Get a list of available models based on the given constraints.

        Args:
            preferred_models (List[str]): List of preferred model keys.
            strengths (List[AIStrengths]): The required strengths of the model.
            chat (Chat): The chat which the model will be processing.
            force_local (bool): Whether to force local models only.
            force_free (bool): Whether to force free models only.
            has_vision (bool): Whether to require models with vision capability.
            force_preferred_model (bool): Whether to only consider preferred models.

        Returns:
            List[Llm]: A list of available Llm instances that meet the specified criteria.
        """
        instance = cls()

        # Trigger dynamic discovery for comprehensive model access (e.g., LLM selector)
        # Also trigger when we have preferred models that might need discovery
        if not preferred_models or len(preferred_models) == 0 or force_preferred_model:
            # When no specific models requested OR when forcing preferred models, load all available (including dynamic discovery)
            instance._ensure_dynamic_models_loaded()

        available_models: List[Llm] = []

        # First try to find models with exact capability matches
        for model_key in preferred_models:
            logger.debug(f"Processing preferred model: {model_key}")
            if model_key:
                if model_key in instance.failed_models:
                    logger.debug(f"Skipping {model_key}: in failed_models list")
                    continue

                # Find the most specific match (exact match preferred over substring match)
                # This ensures fallback order is preserved correctly
                exact_match = next((model for model in instance.retry_models if model.model_key == model_key), None)
                if exact_match:
                    model = exact_match
                else:
                    # Fallback to substring match (for backward compatibility)
                    model = next((model for model in instance.retry_models if model_key in model.model_key), None)

                logger.debug(f"Found model object for {model_key}: {model}")
                if model:
                    capability_result, _ = instance.model_capable_check(model, chat, strengths, model.local, force_free, has_vision, debug_model_key=model_key)
                    logger.debug(f"Capability check for {model_key}: {capability_result}")
                    if capability_result:
                        available_models.append(model)
                else:
                    logger.debug(f"No model object found for {model_key}")

        # If no preferred models with exact capabilities, check all models
        if not available_models:
            for model in instance.retry_models:
                if model.model_key not in instance.failed_models and model.model_key not in [model.model_key for model in available_models]:
                    capable, _ = instance.model_capable_check(model, chat, strengths, model.local, force_free, has_vision)
                    if (not force_local or model.local) and capable:
                        available_models.append(model)

        # If still no models found, fallback to preferred models without strict capability matching
        if not available_models and strengths:
            # First check preferred models
            for model_key in preferred_models:
                if model_key and model_key not in instance.failed_models:
                    # Find the most specific match (exact match preferred over substring match)
                    exact_match = next((model for model in instance.retry_models if model.model_key == model_key), None)
                    if exact_match:
                        model = exact_match
                    else:
                        # Fallback to substring match (for backward compatibility)
                        model = next((model for model in instance.retry_models if model_key in model.model_key), None)

                    if model:
                        capable, _ = instance.model_capable_check(model, chat, strengths, model.local, force_free, has_vision)
                        if capable:
                            available_models.append(model)

            # Then check all models
            if not available_models:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and model.model_key not in [model.model_key for model in available_models]:
                        capable, _ = instance.model_capable_check(model, chat, strengths, model.local, force_free, has_vision)
                        if (not force_local or model.local) and capable:
                            available_models.append(model)

        return available_models

    @classmethod
    def get_provider_info(cls, model_key: str, colored_output: bool = False) -> tuple[str, str]:
        """
        Get provider info and emoji for a model_key.

        Args:
            model_key (str): The model key to get provider info for
            colored_output (bool): Whether to return colored provider info

        Returns:
            tuple[str, str]: (provider_info, provider_emoji) where provider_info is like " via <Provider>" or " on <host>"
        """
        try:
            models = cls.get_models([model_key])
            if not models:
                return "", ""

            model = models[0]
            provider_info = ""
            provider_emoji = ""

            if model.provider.__class__.__name__ == 'OllamaClient':
                # Get host info - use localhost if current_host is None
                host = getattr(model.provider.__class__, 'current_host', None) or "localhost:11434"
                provider_info = f" via <{host}>"
                provider_emoji = "🏠 "  # Local/home emoji
                if colored_output:
                    from termcolor import colored
                    provider_info = provider_info.replace(f"<{host}>", colored(f"<{host}>", "light_green", attrs=["bold"]))
            elif model.provider.__class__.__name__ != 'OllamaClient':
                # For non-Ollama providers, show the provider name with cloud emoji
                provider_name = model.provider.__class__.__name__.replace('API', '').replace('Client', '')
                provider_info = f" via <{provider_name}>"
                provider_emoji = "☁️  "  # Cloud emoji
                if colored_output:
                    from termcolor import colored
                    if provider_name == 'Google':
                        # Special Google logo colors: G(blue), o(red), o(yellow), g(green), l(blue), e(red)
                        google_colored = (colored('G', 'blue', attrs=['bold']) +
                                        colored('o', 'red', attrs=['bold']) +
                                        colored('o', 'yellow', attrs=['bold']) +
                                        colored('g', 'green', attrs=['bold']) +
                                        colored('l', 'blue', attrs=['bold']) +
                                        colored('e', 'red', attrs=['bold']))
                        provider_info = provider_info.replace(f"<{provider_name}>", f"<{google_colored}>")
                    else:
                        provider_info = provider_info.replace(provider_name, colored(provider_name, "light_blue", attrs=["bold"]))

            return provider_info, provider_emoji
        except Exception:
            return "", ""

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

        # Trigger dynamic discovery for comprehensive model access
        # Especially important when forcing preferred models that might need discovery/auto-download
        if not preferred_models or len(preferred_models) == 0 or force_preferred_model:
            instance._ensure_dynamic_models_loaded()

        # Debug print for large token counts
        if (len(chat) > 4000 and not force_free and not force_local) and g.VERBOSE_DEBUG:
            print(colored("DEBUG: len(chat) returned: " + str(len(chat)), "yellow"))

        # Try models in order of preference
        candidate_models = []

        # First try to find preferred model with exact capabilities
        for model_key in preferred_models:
            if (model_key not in instance.failed_models) and model_key:
                # Find the most specific match (exact match preferred over substring match)
                # This ensures fallback order is preserved correctly
                exact_match = next((model for model in instance.retry_models if model.model_key == model_key and (not has_vision or has_vision == model.has_vision)), None)
                if exact_match:
                    model = exact_match
                else:
                    # Fallback to substring match (for backward compatibility)
                    model = next((model for model in instance.retry_models if model_key in model.model_key and (not has_vision or has_vision == model.has_vision)), None)

                if model:
                    capable, _ = instance.model_capable_check(model, chat, strengths, model.local, False, has_vision, debug_model_key=model_key)
                    if capable:
                        candidate_models.append(model)

        # If no preferred candidates and force_preferred_model is True
        # Try auto-download for any Ollama models that might not be downloaded yet
        if not candidate_models and force_preferred_model:
            # Try to create dynamic Ollama models for auto-download
            # This handles cases where the model exists but isn't downloaded yet
            try:
                from core.providers.cls_ollama_interface import OllamaClient
                for model_key in preferred_models:
                    if model_key not in instance.failed_models:
                        # Only attempt auto-download for models that look like Ollama models
                        # (i.e., not containing / which indicates a cloud provider model)
                        if '/' not in model_key and ':' in model_key:
                            # Create a dynamic Ollama model for auto-download attempt
                            strengths = [AIStrengths.LOCAL]  # All Ollama models are local
                            if any(x in model_key.lower() for x in ['vision', 'vl', 'visual']):
                                strengths.append(AIStrengths.VISION)
                            if any(x in model_key.lower() for x in ['uncensored', 'abliterated', 'rp', 'roleplay']):
                                strengths.append(AIStrengths.UNCENSORED)
                            if any(x in model_key.lower() for x in ['lite', "2b", "3b", "4b"]):
                                strengths.append(AIStrengths.LITE)
                            if any(x in model_key.lower() for x in ['pro', "strong", "24b", "30b", "32b"]):
                                strengths.append(AIStrengths.PRO)

                            dynamic_model = Llm(OllamaClient(), model_key, None, None, strengths)
                            capable, _ = instance.model_capable_check(dynamic_model, chat, strengths, True, False, has_vision)
                            if capable:
                                candidate_models.append(dynamic_model)
            except ImportError:
                pass

        # Continue gathering candidates from other models if needed
        # Only allow fallback if force_preferred_model is False
        if not candidate_models and not force_preferred_model:
            # Search online models by exact capability next
            if not force_local:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and not model.local:
                        capable, _ = instance.model_capable_check(model, chat, strengths, local=False, force_free=force_free, has_vision=has_vision)
                        if capable:
                            candidate_models.append(model)

                # Add online models as fallback
                if not candidate_models and strengths:
                    for model in instance.retry_models:
                        if model.model_key not in instance.failed_models and not model.local:
                            capable, _ = instance.model_capable_check(model, chat, strengths, local=False, force_free=force_free, has_vision=has_vision)
                            if capable:
                                candidate_models.append(model)

            # Add local models by exact capability
            if not candidate_models or force_local:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and model.local:
                        capable, _ = instance.model_capable_check(model, chat, strengths, local=True, force_free=force_free, has_vision=has_vision)
                        if capable:
                            candidate_models.append(model)

            # Add local models as fallback
            if not candidate_models and strengths:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and model.local:
                        capable, _ = instance.model_capable_check(model, chat, strengths, local=True, force_free=force_free, has_vision=has_vision)
                        if capable:
                            candidate_models.append(model)

            # Last resort: try with empty chat to ignore context length
            if not candidate_models:
                for model in instance.retry_models:
                    if model.model_key not in instance.failed_models and model.local:
                        capable, _ = instance.model_capable_check(model, Chat(), strengths, local=True, force_free=force_free, has_vision=has_vision)
                        if capable:
                            candidate_models.append(model)

        # Return the first valid candidate
        return candidate_models[0] if candidate_models else None

    @classmethod
    async def _process_stream(
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
                            finished_response = await callback(token)
                            if finished_response and isinstance(finished_response, str):
                                break
                        elif (not hidden_reason) or g.DEBUG_MODE:
                            g.print_token(token_stream_painter.apply_color(token))
            else:
                logger.warning(f"Anthropic stream missing 'text_stream' attribute. Stream type: {type(stream).__name__}")
                raise Exception(f"Anthropic stream format error: expected 'text_stream' attribute on {type(stream).__name__}")
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
                            finished_response = await callback(token)
                            if finished_response and isinstance(finished_response, str):
                                break
                        elif (not hidden_reason) or g.DEBUG_MODE:
                            g.print_token(token_stream_painter.apply_color(token))
        # ! Google Gemini - IMPROVED HANDLING
        elif isinstance(provider, GoogleAPI):
            try:
                first_chunk_processed = False
                last_finish_reason = None
                last_safety_ratings = None

                for chunk in stream:  # chunk is a GenerateContentResponse
                    if not first_chunk_processed:
                        first_chunk_processed = True
                        # Minimal check for immediate prompt blocking on the first chunk
                        if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback and \
                           hasattr(chunk.prompt_feedback, 'block_reason') and chunk.prompt_feedback.block_reason:
                            block_reason = chunk.prompt_feedback.block_reason
                            logger.warning(f"Google API blocked prompt: {block_reason}")
                            raise Exception(f"Google API blocked the prompt (reason: {block_reason})")

                    # Track finish_reason and safety_ratings from all chunks
                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        candidate = chunk.candidates[0]
                        if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                            last_finish_reason = candidate.finish_reason
                        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                            last_safety_ratings = candidate.safety_ratings

                        # Check for safety blocking (finish_reason == 3 is SAFETY, 4 is RECITATION)
                        if last_finish_reason in [3, 4]:
                            reason_name = "SAFETY" if last_finish_reason == 3 else "RECITATION"
                            safety_details = ""
                            if last_safety_ratings:
                                blocked_categories = [
                                    str(rating.category).replace('HarmCategory.', '').replace('HARM_CATEGORY_', '')
                                    for rating in last_safety_ratings
                                    if hasattr(rating, 'blocked') and rating.blocked
                                ]
                                if blocked_categories:
                                    safety_details = f" - Blocked categories: {', '.join(blocked_categories)}"
                            logger.warning(f"Google API blocked generation: {reason_name}{safety_details}")
                            raise Exception(f"Google API content blocked: {reason_name}{safety_details}")

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
                            result_from_callback = await callback(token_from_this_chunk)
                            if result_from_callback and isinstance(result_from_callback, str):
                                finished_response = result_from_callback
                                break
                        elif (not hidden_reason) or g.DEBUG_MODE:
                            g.print_token(token_stream_painter.apply_color(token_from_this_chunk))

                # If callback signaled to finish early
                if finished_response and isinstance(finished_response, str):
                    return finished_response

                # Check if stream ended with no content but has blocking info
                if (not full_response or not full_response.strip()) and last_finish_reason in [3, 4]:
                    reason_name = "SAFETY" if last_finish_reason == 3 else "RECITATION"
                    safety_details = ""
                    if last_safety_ratings:
                        blocked_categories = [
                            str(rating.category).replace('HarmCategory.', '').replace('HARM_CATEGORY_', '')
                            for rating in last_safety_ratings
                            if hasattr(rating, 'blocked') and rating.blocked
                        ]
                        if blocked_categories:
                            safety_details = f" - Blocked categories: {', '.join(blocked_categories)}"
                    logger.warning(f"Google API blocked generation: {reason_name}{safety_details}")
                    raise Exception(f"Google API content blocked: {reason_name}{safety_details}")

            except StopIteration:
                pass  # Normal end of stream
        # ! Ollama/Groq
        elif provider.__class__.__name__ == 'OllamaClient' or isinstance(provider, GroqAPI):
            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                token = None
                if isinstance(provider, GroqAPI):
                    if hasattr(chunk, 'choices') and chunk.choices and \
                       hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        token = chunk.choices[0].delta.content
                elif provider.__class__.__name__ == 'OllamaClient':
                    if isinstance(chunk, dict):  # Ollama dictionary chunks
                        if 'error' in chunk:
                            # Raise a specific exception if the Ollama server returns an error in the stream
                            raise Exception(f"Ollama server error: {chunk['error']}")
                        token = chunk.get('message', {}).get('content', '') or chunk.get('response', '')
                    elif hasattr(chunk, 'message'):  # Ollama response object
                        if hasattr(chunk.message, 'content'):
                            token = chunk.message.content

                if token is not None:  # Ensure token is not None
                    full_response += token
                    if callback is not None:
                        finished_response = await callback(token)
                        if finished_response and isinstance(finished_response, str):
                            break
                    elif (not hidden_reason) or g.DEBUG_MODE:
                        g.print_token(token_stream_painter.apply_color(token))

        # Fallback for other unknown stream types (original logic)
        else:
            for chunk_item in stream:  # Renamed to avoid conflict
                token = str(chunk_item)  # Basic conversion
                if token:
                    full_response += token
                    if callback is not None:
                        finished_response = await callback(token)
                        if finished_response and isinstance(finished_response, str):
                            break
                    elif (not hidden_reason) or g.DEBUG_MODE:
                        g.print_token(token_stream_painter.apply_color(token))


        # If callback returned a final response string at any point (and broke the loop)
        if finished_response and isinstance(finished_response, str):
            return finished_response

        # Check for empty response before returning
        if not full_response or not full_response.strip():
            logger.warning("Model returned empty response (generation failure or content blocking)")
            raise Exception("Model returned empty response (generation failure or content blocking)")

        return full_response

    @classmethod
    async def _process_cached_response(
        cls,
        cached_completion: str,
        model: Llm,
        text_stream_painter: TextStreamPainter,
        hidden_reason: str,
        debug_title:str,
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
            cache_msg = f"{colored('Cache - ' + model.provider.__module__.split('.')[-1], 'green')} <{colored(model.model_key, 'green')}>"
            g.debug_log(cache_msg, "blue", force_print=True, prefix=f"[{debug_title}]")
            for char in cached_completion:
                if callback:
                    import asyncio
                    if asyncio.iscoroutinefunction(callback):
                        finished_response = await callback(char)
                    else:
                        finished_response = callback(char)
                    if finished_response and isinstance(finished_response, str):
                        return finished_response
                elif (not hidden_reason) or g.DEBUG_MODE:
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
                if model in instance.retry_models:
                    instance.retry_models.remove(model)
            return

        error_msg = str(e)
        prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
        provider_name = model.provider.__class__.__name__ if model else "Unknown"
        model_key = model.model_key if model else "unknown"

        # Store the failure reason for final error reporting
        instance._last_failure_reason = error_msg

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
            if model in instance.retry_models:
                instance.retry_models.remove(model)

        # Special handling for timeout issues - SHOW USER MESSAGE and allow retries
        if (isinstance(e, TimeoutException) or
            "request timed out" in error_msg.lower() or
            "timeout" in error_msg.lower() or
            "timed out" in error_msg.lower() or
            "inactivity" in error_msg.lower()):

            # Show timeout message to user with more context
            if "inactivity" in error_msg.lower():
                g.debug_log(f"⏱️ Model {model_key} timed out due to inactivity - will retry with extended timeout", "yellow", force_print=True, prefix=prefix)
            else:
                g.debug_log(f"⏱️ Model {model_key} timed out during processing - will retry with extended timeout", "yellow", force_print=True, prefix=prefix)

            # For timeout errors, don't permanently fail the model - just skip this attempt
            # This allows the model to be retried in future requests
            logger.debug(f"Timeout issue with model {model_key}: {e}")
            return

        # Special handling for rate limit exceptions - these are already handled by the provider
        if isinstance(e, RateLimitException):
            # Rate limit exceptions are already handled by the provider with appropriate user messages
            # Just log silently to debug level to avoid spam
            if model is not None:
                logger.debug(f"Rate limit issue with model {model_key}: {e}")
            return

        # Special handling for connection issues
        if ("connection" in error_msg.lower() or
            "rate_limit" in error_msg.lower()):

            # Show connection message to user
            if "connection" in error_msg.lower():
                g.debug_log(f"🌐 Connection issue with model {model_key} - trying next model", "yellow", force_print=True, prefix=prefix)
            else:
                g.debug_log(f"⚡ Rate limit reached for model {model_key} - trying next model", "yellow", force_print=True, prefix=prefix)

            # Log silently to file for debugging
            if model is not None:
                logger.info(f"Network/rate-limit issue with model {model_key}: {e}")
            return

        # Check if this error has already been logged by the Google API provider
        if "Google API" in error_msg and "error" in error_msg.lower():
            # This error was already logged by the Google API provider, don't log it again
            return

        # Provider-specific error handling
        if "OllamaClient" in provider_name:
            # Provide more detailed error messages for common Ollama issues
            if "timeout" in error_msg.lower():
                if "first token" in error_msg.lower():
                    g.debug_log(f"⏱️  Ollama-Api: Model {model_key} failed to start generation (3min timeout) - likely not installed or Ollama server issue", "red", is_error=True, prefix=prefix)
                elif "inactivity" in error_msg.lower():
                    g.debug_log(f"⏱️  Ollama-Api: Model {model_key} timed out during generation (stream stuck/slow)", "red", is_error=True, prefix=prefix)
                else:
                    g.debug_log(f"⏱️  Ollama-Api: Model {model_key} timed out during generation", "red", is_error=True, prefix=prefix)
            elif "empty response" in error_msg.lower():
                g.debug_log(f"❌ Ollama-Api: Model {model_key} returned empty response (generation failure or blocking)", "red", is_error=True, prefix=prefix)
            elif "No valid host found" in error_msg:
                g.debug_log(f"🌐 Ollama-Api: {error_msg} - check Ollama server connectivity", "yellow", prefix=prefix)
            elif "model not found" in error_msg.lower() or "404" in error_msg:
                g.debug_log(f"❓ Ollama-Api: Model {model_key} not found/installed - run 'ollama pull {model_key}' to install", "yellow", prefix=prefix)
            elif "connection refused" in error_msg.lower():
                g.debug_log("🚫 Ollama-Api: Connection refused to Ollama server - check if Ollama is running", "red", is_error=True, prefix=prefix)
            elif "no such model" in error_msg.lower():
                g.debug_log(f"❓ Ollama-Api: Model {model_key} not available - verify model name and installation", "yellow", prefix=prefix)
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
            # Check if this error has already been handled by the Google API provider
            if hasattr(e, 'already_logged') and getattr(e, 'already_logged'):
                # Error was already logged by the provider, don't log it again
                return

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
    async def chat_with_retry(
        cls,
        chat: Chat,
        model_key: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Alias for generate_completion with simplified parameters for backward compatibility."""
        preferred_models = [model_key] if model_key else []
        return await cls.generate_completion(
            chat=chat,
            preferred_models=preferred_models,
            temperature=temperature,
            **kwargs
        )

    @classmethod
    async def generate_completion(
        cls,
        chat: Chat|str,
        preferred_models: List[str] | List[Llm] = [],
        strengths: List[AIStrengths] = [],
        temperature: float = 0,
        base64_images: List[str] = [],
        force_local: bool = g.FORCE_LOCAL,
        force_free: bool = False,
        force_preferred_model: bool = not g.FORCE_LOCAL, # If force local is on we may as well be greedy
        hidden_reason: str = "",
        exclude_reasoning_tokens: bool = True,
        thinking_budget: Optional[int] = 4096,
        generation_stream_callback: Optional[Callable[[str], None]] = lambda char: print(colored(char, 'green'), end='')
    ) -> str:
        """
        Generate a completion response using the appropriate LLM.

        Args:
            chat (Chat|str): The conversation context as a Chat object or plain text string prompt.
            preferred_models (List[str] | List[Llm]): Model preference list - can be model keys (strings) or Llm objects. Router tries these first before fallbacks.
            strengths (List[AIStrengths]): Required model capabilities (e.g., vision, reasoning, coding). Empty list means no specific requirements.
            temperature (float): Controls randomness in responses (0.0-1.0). Default 0 for deterministic output. Higher values increase creativity.
            base64_images (List[str]): Base64-encoded images for vision-capable models. Automatically enables vision model selection when provided.
            force_local (bool): When True, restricts to locally-hosted models only (Ollama). Ignores cloud models entirely.
            force_free (bool): When True (default), prioritizes free tier models. When False, allows premium/paid models for complex tasks.
            force_preferred_model (bool): When True, only uses models from preferred_models list. When False, falls back to system defaults if preferred models fail.
            hidden_reason (str): When non-empty, suppresses live token streaming output to console. Used for background/silent completions.
            exclude_reasoning_tokens (bool): When True (default), strips reasoning tokens from response. When False, includes model's internal reasoning in output.
            thinking_budget (Optional[int]): Token budget for model reasoning (Gemini only). -1=dynamic, 0=disabled, >0=fixed limit. Default 4096.
            generation_stream_callback (Optional[Callable[[str], None]]): Function called with each streaming token. Default prints tokens in green. None disables streaming callbacks.

        Returns:
            str: The generated completion text with reasoning tokens excluded (unless exclude_reasoning_tokens=False).
        """
        instance = cls()
        cls.call_counter += 1

        # Check for override model first - if set, use it exclusively
        if g.override_llm:
            preferred_models = [g.override_llm]
            force_preferred_model = True  # Force using only the override model

        # Check global force flags
        if g.LLM:
            preferred_models = [g.LLM]

        # Force means force
        if g.FORCE_LOCAL:
            force_local = True

        def exclude_reasoning(response: str) -> str:
            if exclude_reasoning_tokens and ("</think>" in response or "</thinking>" in response):
                if "</think>" in response:
                    after_thinking = response.split("</think>")[1]
                    # If there's actual content after the reasoning tokens, return it
                    if after_thinking.strip():
                        return after_thinking
                    else:
                        # No content after reasoning tokens - return original response
                        # This preserves the reasoning for debugging while indicating the issue
                        return response
                elif "</thinking>" in response:
                    after_thinking = response.split("</thinking>")[1]
                    # If there's actual content after the reasoning tokens, return it
                    if after_thinking.strip():
                        return after_thinking
                    else:
                        # No content after reasoning tokens - return original response
                        return response
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
        # Infinite retries with linear backoff (1s, 2s, 3s, etc.)
        while True:
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

                # If no model is available, all available models have failed.
                if not model:
                    prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""

                    # Provide detailed information about what failed
                    failed_count = len(instance.failed_models)
                    available_count = len(instance.retry_models) + failed_count

                    if failed_count > 0:
                        failed_list = ", ".join(list(instance.failed_models)[:3])  # Show first 3 failed models
                        if failed_count > 3:
                            failed_list += f" (and {failed_count - 3} others)"

                        g.debug_log(f"All {available_count} available models failed. Recently failed: {failed_list}", "red", is_error=True, prefix=prefix)

                        # Store the failure reason for the final exception
                        instance._last_failure_reason = f"All {available_count} models failed (including: {failed_list})"
                    else:
                        # For single preferred model, be more specific about why it wasn't found
                        if force_preferred_model and len(preferred_models) == 1:
                            requested_model = preferred_models[0]

                            # Check if the model exists in the full model list but was filtered out
                            all_models = Llm.get_available_llms(include_dynamic=True)
                            model_exists = any(requested_model in model.model_key for model in all_models)

                            if model_exists:
                                # Model exists but was filtered out due to constraints
                                # Check what constraint caused the exclusion
                                exclusion_reasons = []
                                detailed_diagnostics = []
                                matching_model = next((model for model in all_models if requested_model in model.model_key), None)

                                if matching_model:
                                    # Use the actual model_capable_check method to get the real exclusion reason
                                    capable, exclusion_reason = instance.model_capable_check(
                                        matching_model, chat, strengths,
                                        local=force_local, force_free=force_free,
                                        has_vision=bool(base64_images)
                                    )

                                    if not capable:
                                        if exclusion_reason:
                                            exclusion_reasons.append(exclusion_reason)
                                        else:
                                            # Perform detailed diagnostic checks to understand why it was excluded
                                            # Check if model is in failed models list
                                            if matching_model.model_key in instance.failed_models:
                                                detailed_diagnostics.append("model previously failed and is in failed_models list")

                                            # Check if model is rate limited (enhanced rate limiting)
                                            try:
                                                from infrastructure.rate_limiting.cls_enhanced_rate_limit_tracker import enhanced_rate_limit_tracker
                                                token_cost = len(str(chat)) // 4  # Rough estimate
                                                can_request, rate_reason = enhanced_rate_limit_tracker.can_make_request(
                                                    matching_model.model_key, token_cost=token_cost, image_count=len(base64_images) if base64_images else 0
                                                )
                                                if not can_request:
                                                    detailed_diagnostics.append(f"rate limited: {rate_reason}")
                                            except Exception as rate_check_error:
                                                # If rate limiting check fails, note it but don't fail
                                                detailed_diagnostics.append(f"rate limiting check failed: {rate_check_error}")

                                            # Check pricing constraint
                                            if force_free and matching_model.pricing_in_dollar_per_1M_tokens is not None:
                                                detailed_diagnostics.append(f"not free (costs ${matching_model.pricing_in_dollar_per_1M_tokens} per 1M tokens)")

                                            # Check vision requirement
                                            if bool(base64_images) and not matching_model.has_vision:
                                                detailed_diagnostics.append("no vision support required for images")

                                            # Check local/remote constraint
                                            if force_local != matching_model.local:
                                                detailed_diagnostics.append(f"local constraint mismatch (required: {force_local}, model: {matching_model.local})")

                                            # Check token limits
                                            if matching_model.model_key in instance._model_limits:
                                                token_limit = instance._model_limits[matching_model.model_key]
                                                if len(chat) >= token_limit:
                                                    detailed_diagnostics.append(f"exceeds saved token limit ({token_limit} < {len(chat)} tokens)")

                                            # Check context window
                                            if matching_model.get_context_window() < len(chat):
                                                detailed_diagnostics.append(f"context window too small ({matching_model.get_context_window()} < {len(chat)} tokens)")

                                            # Check strengths requirement
                                            if strengths and matching_model.strengths:
                                                if not all(s.value in [ms.value for ms in matching_model.strengths] for s in strengths):
                                                    missing_strengths = [s.name for s in strengths if s.value not in [ms.value for ms in matching_model.strengths]]
                                                    detailed_diagnostics.append(f"missing required strengths: {missing_strengths}")

                                            # If we still don't know why, check additional factors
                                            if not detailed_diagnostics:
                                                # Check if it's in the retry_models list at all
                                                if matching_model not in instance.retry_models:
                                                    detailed_diagnostics.append("not in available retry_models list")
                                                else:
                                                    detailed_diagnostics.append("passed all capability checks but still excluded (possible race condition or logic error)")

                                # Combine all reasons
                                all_reasons = exclusion_reasons + detailed_diagnostics

                                if all_reasons:
                                    reason_str = ", ".join(all_reasons)
                                    g.debug_log(f"Model '{requested_model}' available but excluded: {reason_str}", "red", is_error=True, prefix=prefix)
                                    instance._last_failure_reason = f"Model '{requested_model}' excluded: {reason_str}"
                                else:
                                    # Model exists but no clear exclusion reason - allow it to be tried once
                                    g.debug_log(f"Model '{requested_model}' available but reason unclear - allowing attempt", "yellow", prefix=prefix)
                                    # Don't exclude, let it be tried - if it fails, the retry logic will handle it
                                    pass
                            else:
                                # Model truly doesn't exist - show relevant alternatives
                                if force_local:
                                    # For local requests, only show local models
                                    local_models = [m.model_key for m in instance.retry_models if m.local][:5]
                                    available_str = ", ".join(local_models)
                                    if len([m for m in instance.retry_models if m.local]) > 5:
                                        available_str += f" (and {len([m for m in instance.retry_models if m.local]) - 5} others)"
                                    model_type = "local"
                                else:
                                    # For general requests, show all models
                                    available_models = [m.model_key for m in instance.retry_models[:5]]
                                    available_str = ", ".join(available_models)
                                    if len(instance.retry_models) > 5:
                                        available_str += f" (and {len(instance.retry_models) - 5} others)"
                                    model_type = ""

                                g.debug_log(f"Model '{requested_model}' not found in {model_type} models", "red", is_error=True, prefix=prefix)
                                g.debug_log(f"Available {model_type} models: {available_str}", "yellow", prefix=prefix)
                                instance._last_failure_reason = f"Model '{requested_model}' not found in {model_type} models."
                        else:
                            # Build detailed criteria description
                            criteria_parts = []
                            if strengths:
                                criteria_parts.append(f"strengths: {[s.value for s in strengths]}")
                            if preferred_models:
                                criteria_parts.append(f"preferred_models: {preferred_models}")
                            if force_local:
                                criteria_parts.append("force_local: True")
                            if force_free:
                                criteria_parts.append("force_free: True")
                            if bool(base64_images):
                                criteria_parts.append("has_vision: True")
                            if force_preferred_model:
                                criteria_parts.append("force_preferred_model: True")

                            criteria_desc = ", ".join(criteria_parts) if criteria_parts else "no specific criteria"

                            # Get total available models for context
                            all_models = Llm.get_available_llms(include_dynamic=True)
                            total_models = len(all_models)
                            available_model_names = [m.model_key for m in all_models[:10]]  # Show first 10
                            available_str = ", ".join(available_model_names)
                            if total_models > 10:
                                available_str += f" (and {total_models - 10} others)"

                            detailed_message = (f"No models found matching criteria: {criteria_desc}. "
                                              f"Total models available: {total_models} "
                                              f"({available_str})")

                            g.debug_log(detailed_message, "yellow", prefix=prefix)
                            instance._last_failure_reason = detailed_message

                    break # Exit the while loop, the exception below will be raised.

                # Enable caching for all temperatures - behavior differs based on temperature
                enable_caching = True
                cls.last_used_model = model.model_key

                # --- FIX: Log BEFORE cache check to ensure all branch attempts are visible ---
                if not hidden_reason:
                    token_count = math.ceil(len(str(chat)) * 3/4)
                    message_count = len(chat.messages)
                    temp_str = "" if temperature == 0 or temperature is None else f" at temperature <{temperature}>"

                    # Get host/provider information with emojis
                    provider_info, provider_emoji = cls.get_provider_info(model.model_key, colored_output=True)

                    # Create colored version with distinct colors for text in brackets
                    base_msg = f"{provider_emoji}[Tokens: {token_count} | Messages: {message_count}] -> Calling model "
                    model_colored = colored(f"<{model.model_key}>", "light_blue", attrs=["bold"])
                    temp_colored = temp_str.replace(f"<{temperature}>", colored(f"<{temperature}>", "light_blue", attrs=["bold"])) if temp_str else ""
                    # provider_info is already colored from get_provider_info call above

                    full_msg = f"{base_msg}{model_colored}{temp_colored}{provider_info}"
                    logging.info(colored(full_msg, "cyan"))

                # Check for interrupt before proceeding with model call
                try:
                    from main import check_interrupt
                    if check_interrupt():
                        raise UserInterruptedException("Model generation interrupted by user before starting.")
                except ImportError:
                    pass  # Fallback if main import is not available

                # Check for cached completion (behavior varies by temperature)
                prefixed_chat = chat.deep_copy()

                cached_completion = instance._get_cached_completion(model.model_key, str(chat), base64_images, temperature)
                if cached_completion:
                    return exclude_reasoning(await cls._process_cached_response(
                        cached_completion, model, TextStreamPainter(), hidden_reason, prefixed_chat.debug_title, generation_stream_callback
                    ))

                try:
                    # Get the stream from the provider
                    if hasattr(model.provider, 'generate_response'):
                        # Only pass thinking_budget to Gemini/Google providers, others ignore it
                        provider_thinking_budget = thinking_budget if 'google' in model.provider.__class__.__module__.lower() else None
                        stream = model.provider.generate_response(prefixed_chat, model.model_key, temperature, hidden_reason, provider_thinking_budget)
                    else:
                        raise Exception(f"Provider {model.provider.__class__.__name__} does not support generate_response method")

                    # Check if stream is None before processing
                    if stream is None:
                        raise Exception(f"Model {model.model_key} returned None stream")

                    # Activity-based stream processing timeout
                    import signal
                    import time

                    class StreamActivityMonitor:
                        def __init__(self, timeout_seconds=60, first_token_timeout=10):
                            self.start_time = time.time()
                            self.last_activity = time.time()
                            self.timeout_seconds = timeout_seconds
                            self.first_token_timeout = first_token_timeout
                            self.total_chars = 0
                            self.first_token_received = False

                        def reset_activity(self):
                            self.last_activity = time.time()

                        def add_chars(self, char_count):
                            self.total_chars += char_count
                            if not self.first_token_received:
                                self.first_token_received = True
                            self.reset_activity()

                        def check_timeout(self):
                            current_time = time.time()
                            # Check first token timeout (10 seconds from start)
                            if not self.first_token_received and (current_time - self.start_time) > self.first_token_timeout:
                                return "first_token_timeout"
                            # Check inactivity timeout (after first token)
                            elif self.first_token_received and (current_time - self.last_activity) > self.timeout_seconds:
                                return "inactivity_timeout"
                            return None

                    # Determine timeouts based on provider type
                    inactivity_timeout = 60
                    first_token_timeout = 10  # Default 10 seconds for cloud providers

                    if model.provider.__class__.__name__ == 'OllamaClient':
                        # Ollama models need longer timeouts
                        first_token_timeout = 180  # 3 minutes for first token
                        try:
                            hosts = model.provider.reached_hosts
                            if len(hosts) > 0:
                                host = hosts[0]
                                if host in ("localhost", "127.0.0.1") or host in g.ollama_host_env:
                                    inactivity_timeout = 300
                        except Exception:
                            pass

                    stream_monitor = StreamActivityMonitor(inactivity_timeout, first_token_timeout)

                    def stream_timeout_handler(signum, frame):
                        timeout_type = stream_monitor.check_timeout()
                        if timeout_type:
                            if timeout_type == "first_token_timeout":
                                raise Exception(f"First token timeout after {stream_monitor.first_token_timeout} seconds for model {model.model_key}")
                            elif timeout_type == "inactivity_timeout":
                                raise Exception(f"Stream inactivity timeout after {stream_monitor.timeout_seconds} seconds for model {model.model_key} (got {stream_monitor.total_chars} chars)")
                        else:
                            # Reset the alarm for another check
                            signal.alarm(5)  # Check every 5 seconds for more responsive first token detection

                    # Track recent characters for repetition detection
                    recent_chars = []
                    repetition_threshold = 500  # Characters to analyze
                    max_char_repetition = 0.8  # 80% of same character = spam

                    # Enhanced stream callback that monitors activity and repetition
                    async def activity_aware_callback(chunk: str) -> str:
                        nonlocal recent_chars

                        if chunk:
                            stream_monitor.add_chars(len(chunk))

                            # Add characters to recent buffer
                            recent_chars.extend(list(chunk))

                            # Keep only last 500 characters for analysis
                            if len(recent_chars) > repetition_threshold:
                                recent_chars = recent_chars[-repetition_threshold:]

                            # Check for character spam once we have enough data
                            if len(recent_chars) >= repetition_threshold:
                                char_counts = {}
                                for char in recent_chars:
                                    char_counts[char] = char_counts.get(char, 0) + 1

                                # Find most frequent character and its percentage
                                if char_counts:
                                    max_count = max(char_counts.values())
                                    max_char = max(char_counts, key=lambda k: char_counts[k])
                                    repetition_ratio = max_count / len(recent_chars)

                                    # Detect character spam (ignore whitespace/newlines for this check)
                                    if repetition_ratio > max_char_repetition and max_char not in [' ', '\n', '\t']:
                                        raise Exception(f"Model repetition detected: '{max_char}' repeated {max_count}/{len(recent_chars)} times ({repetition_ratio:.1%}) in last {repetition_threshold} characters")

                        # Call original callback if provided
                        if generation_stream_callback:
                            import asyncio
                            if asyncio.iscoroutinefunction(generation_stream_callback):
                                return await generation_stream_callback(chunk)
                            else:
                                return generation_stream_callback(chunk)
                        return ""

                    # Set up periodic timeout checking for stream processing
                    signal.signal(signal.SIGALRM, stream_timeout_handler)
                    signal.alarm(5)  # Start checking after 5 seconds for first token timeout

                    try:
                        # Detect existing assistant message prefix before processing stream
                        full_response = await cls._process_stream(stream, model.provider, hidden_reason, activity_aware_callback)

                    finally:
                        signal.alarm(0)  # Clear the alarm

                    if (not full_response.endswith("\n") and not hidden_reason):
                        print()

                    if enable_caching:
                        # Cache the response
                        instance._update_cache(model.model_key, str(prefixed_chat), base64_images, full_response, temperature)

                    # Save the chat completion pair if requested
                    if not force_local:
                        instance._save_chat_completion_pair(prefixed_chat.to_openai(), full_response, model.model_key)


                    # Branch context completion is handled in main.py for MCT branches
                    # No need to print success status here as it's handled at the branch level

                    return exclude_reasoning(full_response)

                except KeyboardInterrupt:
                    # Explicitly catch Ctrl+C during model generation and convert to clean exception
                    raise UserInterruptedException("Model generation interrupted by user (Ctrl+C).")

            except (UserInterruptedException, StreamInterruptedException):
                # Handle stream interruption by cancelling any active generation
                # This now correctly triggers for BOTH user interrupts and tool detection.
                provider = model.provider if model else None
                if provider and hasattr(provider, 'cancel_current_generation'):
                    try:
                        provider.cancel_current_generation()
                        logger.debug(f"Cancelled generation for provider {provider.__class__.__name__}")
                    except Exception as cancel_error:
                        logger.warning(f"Failed to cancel generation for provider {provider.__class__.__name__}: {cancel_error}")
                elif provider:
                    # Provider exists but doesn't support cancellation
                    logger.debug(f"Provider {provider.__class__.__name__} does not support stream cancellation")

                # Re-raise exceptions that are part of the normal control flow
                # so they can be handled by the caller in main.py.
                raise
            except Exception as e:
                # Handle model errors and capture the specific failure reason

                # Debug vision-specific issues
                if g.VERBOSE_DEBUG:
                    print(f"🔍 ROUTER EXCEPTION: Model {model.model_key if model else 'unknown'}")
                    print(f"🔍 ROUTER ERROR: {str(e)}")
                    print(f"🔍 ROUTER ERROR TYPE: {type(e).__name__}")
                    if base64_images:
                        print(f"🔍 VISION REQUEST: {len(base64_images)} images")
                        for i, img in enumerate(base64_images):
                            print(f"🔍   IMAGE {i}: {len(img)} chars")
                    print(f"🔍 CHAT TYPE: {type(chat).__name__}")
                    if hasattr(chat, 'messages'):
                        print(f"🔍 CHAT MESSAGES: {len(chat.messages)}")

                cls._handle_model_error(e, model, instance, chat)

                # Global retry limit: 3 attempts total
                if retry_count >= 3:
                    prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                    failure_reason = getattr(instance, '_last_failure_reason', str(e))
                    g.debug_log("Model generation failed after 3 retry attempts", "red", is_error=True, prefix=prefix)
                    raise ModelGenerationFailedException(f"Model generation failed after 3 attempts: {failure_reason}")

                # Increment retry count
                retry_count += 1

                # Check if there's a suggested delay from the API (e.g., rate limiting)
                suggested_delay = 0
                try:
                    from infrastructure.rate_limiting.cls_enhanced_rate_limit_tracker import enhanced_rate_limit_tracker
                    suggested_delay = enhanced_rate_limit_tracker.get_suggested_retry_delay(model.model_key if model else "unknown")
                except Exception:
                    pass

                # Determine wait time: 10s, 15s, 20s for attempts 1, 2, 3
                base_delays = [10, 15, 20]
                base_wait = base_delays[retry_count - 1] if retry_count <= 3 else 20

                if suggested_delay > 0:
                    if suggested_delay < 30:
                        wait_time = suggested_delay + 1
                    else:
                        wait_time = suggested_delay
                else:
                    wait_time = base_wait

                # Show retry message to user with error details
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                error_msg = str(e)
                if suggested_delay > 0 and suggested_delay < 30:
                    g.debug_log(f"🔄 API suggested {suggested_delay:.0f}s wait, using {wait_time:.0f}s (attempt #{retry_count}/3) - Error: {error_msg}", "cyan", force_print=True, prefix=prefix)
                elif suggested_delay >= 30:
                    g.debug_log(f"🔄 Respecting API rate limit: waiting {wait_time:.0f}s (attempt #{retry_count}/3) - Error: {error_msg}", "cyan", force_print=True, prefix=prefix)
                else:
                    g.debug_log(f"🔄 Retrying in {wait_time:.0f}s (attempt #{retry_count}/3) - Error: {error_msg}", "cyan", force_print=True, prefix=prefix)

                # Wait with API-suggested delay or linear backoff
                import asyncio
                await asyncio.sleep(wait_time)

                # Reset failed models to allow retrying them
                instance.failed_models.clear()

                # Reset retry_models to the original list of available models
                # When force_preferred_model is True, only include preferred models in retry list
                if force_preferred_model and preferred_models:
                    # Only retry with preferred models, but still do capability checking
                    instance.retry_models = instance.get_models(strengths=strengths, preferred_models=preferred_models, force_local=force_local, force_free=force_free, has_vision=bool(base64_images), force_preferred_model=True)
                else:
                    # Normal retry behavior - use all available models
                    instance.retry_models = instance.get_models(strengths=strengths, preferred_models=preferred_models, force_local=force_local, force_free=force_free, has_vision=bool(base64_images))

        # This should never be reached due to infinite retry loop
        # If we get here, provide the specific failure reason stored earlier
        failure_reason = getattr(instance, '_last_failure_reason', "Unknown reason")
        raise Exception(f"Model generation failed: {failure_reason}")

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

            logger.debug(f"\nSaved chat completion pair to {filename}")
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

    @classmethod
    def _detect_assistant_prefix(cls, chat: Chat) -> str:
        """
        Detect if the latest assistant message already contains text that should be included
        as a prefix before processing the generated stream.

        Args:
            chat (Chat): The chat context to examine

        Returns:
            str: The existing assistant message prefix, or empty string if none
        """
        if not chat.messages:
            return ""

        # Check if the last message is from the assistant
        last_role, last_content = chat.messages[-1]
        if last_role.value == "assistant" and last_content:
            # Return the existing assistant content as prefix
            return last_content

        return ""

    @classmethod
    def get_last_used_model(cls) -> str:
        """
        Get the model key of the last model used for completion.

        Returns:
            str: The model key of the last used model, or 'unknown' if none
        """
        return cls.last_used_model if cls.last_used_model else 'unknown'
