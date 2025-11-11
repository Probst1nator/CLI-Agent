from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from termcolor import colored

from core.globals import g
from py_classes.unified_interfaces import AIProviderInterface

if TYPE_CHECKING:
    from core.chat import Chat

logger = logging.getLogger(__name__)


class HuggingFaceLocalClient(AIProviderInterface):
    """
    Implementation of the AIProviderInterface for local HuggingFace models.
    Supports automatic model downloading, loading, and streaming inference.
    """

    # Class-level state for model management
    _loaded_model = None
    _loaded_tokenizer = None
    _loaded_model_key: Optional[str] = None
    _model_lock = threading.Lock()

    # Stream cancellation support
    _current_task: Optional[threading.Thread] = None
    _cancellation_event: Optional[threading.Event] = None

    # Model cache directory
    _models_dir: Optional[Path] = None

    @classmethod
    def get_models_dir(cls) -> Path:
        """Get the directory where models are stored."""
        if cls._models_dir is None:
            models_dir_str = os.getenv("HF_LOCAL_MODELS_DIR", "~/.cli-agent/models/")
            cls._models_dir = Path(os.path.expanduser(models_dir_str))
            cls._models_dir.mkdir(parents=True, exist_ok=True)
        return cls._models_dir

    @classmethod
    def cancel_current_generation(cls):
        """
        Cancel any currently running generation by stopping the stream.
        """
        try:
            if cls._cancellation_event:
                cls._cancellation_event.set()
                logger.debug("HuggingFace Local stream cancellation event set")

            if cls._current_task and cls._current_task.is_alive():
                logger.debug("HuggingFace Local generation thread still running, will stop at next iteration")

        except Exception as e:
            logger.warning(f"Error cancelling HuggingFace Local generation: {e}")

    @staticmethod
    def is_model_downloaded(model_id: str) -> bool:
        """
        Check if a model is already downloaded.

        Args:
            model_id (str): HuggingFace model ID (e.g., "moonshotai/Moonlight-16B-A3B-Instruct")

        Returns:
            bool: True if model exists locally
        """
        try:
            from huggingface_hub import snapshot_download
            models_dir = HuggingFaceLocalClient.get_models_dir()

            # Try to get the model path using local_files_only
            # This will succeed if the model is cached, fail otherwise
            model_path = snapshot_download(
                repo_id=model_id,
                cache_dir=str(models_dir),
                local_files_only=True
            )
            return Path(model_path).exists()
        except Exception:
            return False

    @staticmethod
    def download_model(model_id: str, chat: Optional[Chat] = None) -> Path:
        """
        Download a model from HuggingFace Hub with progress tracking.

        Args:
            model_id (str): HuggingFace model ID
            chat (Optional[Chat]): Chat object for debug printing

        Returns:
            Path: Local path to downloaded model

        Raises:
            Exception: If download fails
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for downloading models. "
                "Install it with: pip install huggingface_hub"
            )

        models_dir = HuggingFaceLocalClient.get_models_dir()
        safe_model_name = model_id.replace("/", "--")
        model_path = models_dir / safe_model_name

        # Print download message
        prefix = chat.get_debug_title_prefix() if chat and hasattr(chat, 'get_debug_title_prefix') else ""
        if chat:
            g.debug_log(
                f"HuggingFace-Local: Downloading {model_id}...",
                "yellow",
                force_print=True,
                prefix=prefix
            )
        else:
            logger.info(colored(f"HuggingFace-Local: Downloading {model_id}...", "yellow"))

        try:
            # Get HF token if available (for gated models)
            hf_token = os.getenv("HF_TOKEN")

            # Download with progress bar
            downloaded_path = snapshot_download(
                repo_id=model_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,  # Copy files instead of symlinks for clarity
                token=hf_token,
                resume_download=True,  # Resume if partially downloaded
            )

            if chat:
                g.debug_log(
                    f"HuggingFace-Local: {model_id} downloaded successfully",
                    "green",
                    force_print=True,
                    prefix=prefix
                )
            else:
                logger.info(colored(f"HuggingFace-Local: {model_id} downloaded successfully", "green"))

            return Path(downloaded_path)

        except Exception as e:
            error_msg = f"Failed to download {model_id}: {str(e)}"
            if chat:
                g.debug_log(f"HuggingFace-Local: {error_msg}", "red", is_error=True, prefix=prefix)
            else:
                logger.error(error_msg)
            raise Exception(error_msg) from e

    @staticmethod
    def get_valid_model(model_id: str, chat: Optional[Chat] = None) -> tuple[Path, str]:
        """
        Get a valid model path, downloading if necessary.

        Args:
            model_id (str): HuggingFace model ID
            chat (Optional[Chat]): Chat object for debug printing

        Returns:
            tuple[Path, str]: (model_path, model_id)

        Raises:
            Exception: If model cannot be obtained
        """
        try:
            from huggingface_hub import snapshot_download
            models_dir = HuggingFaceLocalClient.get_models_dir()

            # Try to get model from cache first
            try:
                model_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(models_dir),
                    local_files_only=True
                )
                return Path(model_path), model_id
            except Exception:
                # Not in cache, download it
                model_path = HuggingFaceLocalClient.download_model(model_id, chat)
                return model_path, model_id

        except ImportError:
            raise ImportError(
                "huggingface_hub is required. Install it with: pip install huggingface_hub"
            )

    @staticmethod
    def load_model(model_id: str, chat: Optional[Chat] = None) -> tuple[Any, Any]:
        """
        Load a model and tokenizer into memory.

        Args:
            model_id (str): HuggingFace model ID
            chat (Optional[Chat]): Chat object for debug printing

        Returns:
            tuple[Any, Any]: (model, tokenizer)

        Raises:
            Exception: If loading fails
        """
        with HuggingFaceLocalClient._model_lock:
            # If this model is already loaded, return it
            if (HuggingFaceLocalClient._loaded_model_key == model_id and
                HuggingFaceLocalClient._loaded_model is not None):
                logger.debug(f"Model {model_id} already loaded, reusing")
                return HuggingFaceLocalClient._loaded_model, HuggingFaceLocalClient._loaded_tokenizer

            # Unload previous model if different
            if (HuggingFaceLocalClient._loaded_model_key is not None and
                HuggingFaceLocalClient._loaded_model_key != model_id):
                logger.debug(f"Unloading previous model {HuggingFaceLocalClient._loaded_model_key}")
                HuggingFaceLocalClient.unload_model()

            # Get model path (download if needed)
            model_path, _ = HuggingFaceLocalClient.get_valid_model(model_id, chat)

            prefix = chat.get_debug_title_prefix() if chat and hasattr(chat, 'get_debug_title_prefix') else ""
            if chat:
                g.debug_log(
                    f"HuggingFace-Local: Loading {model_id}...",
                    "yellow",
                    force_print=True,
                    prefix=prefix
                )
            else:
                logger.info(colored(f"HuggingFace-Local: Loading {model_id}...", "yellow"))

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                raise ImportError(
                    "transformers is required for loading models. "
                    "Install it with: pip install transformers accelerate"
                )

            try:
                # Get device setting
                device = os.getenv("HF_LOCAL_DEVICE", "auto")

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    trust_remote_code=True,
                )

                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype="auto",
                    device_map=device,
                    trust_remote_code=True,
                )

                # Cache the loaded model
                HuggingFaceLocalClient._loaded_model = model
                HuggingFaceLocalClient._loaded_tokenizer = tokenizer
                HuggingFaceLocalClient._loaded_model_key = model_id

                if chat:
                    g.debug_log(
                        f"HuggingFace-Local: {model_id} loaded successfully",
                        "green",
                        force_print=True,
                        prefix=prefix
                    )
                else:
                    logger.info(colored(f"HuggingFace-Local: {model_id} loaded successfully", "green"))

                return model, tokenizer

            except Exception as e:
                error_msg = f"Failed to load {model_id}: {str(e)}"
                if chat:
                    g.debug_log(f"HuggingFace-Local: {error_msg}", "red", is_error=True, prefix=prefix)
                else:
                    logger.error(error_msg)
                raise Exception(error_msg) from e

    @staticmethod
    def unload_model():
        """
        Unload the currently loaded model to free memory.
        """
        with HuggingFaceLocalClient._model_lock:
            if HuggingFaceLocalClient._loaded_model is not None:
                logger.debug(f"Unloading model {HuggingFaceLocalClient._loaded_model_key}")

                # Delete model and tokenizer
                del HuggingFaceLocalClient._loaded_model
                del HuggingFaceLocalClient._loaded_tokenizer

                HuggingFaceLocalClient._loaded_model = None
                HuggingFaceLocalClient._loaded_tokenizer = None
                HuggingFaceLocalClient._loaded_model_key = None

                # Force garbage collection
                import gc

                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    @staticmethod
    def generate_response(
        chat: Chat | str,
        model_key: str = "moonlight-16b-instruct",
        temperature: Optional[float] = None,
        silent_reason: str = "",
        thinking_budget: Optional[int] = None
    ) -> Any:
        """
        Generates a response using a local HuggingFace model.

        Args:
            chat (Union[Chat, str]): The chat object containing messages or a string prompt.
            model_key (str): The model identifier (short name or full HuggingFace ID).
            temperature (Optional[float]): The temperature setting for the model.
            silent_reason (str): If provided, suppresses output and shows this reason.
            thinking_budget (Optional[int]): Not used for HuggingFace models, kept for compatibility.

        Returns:
            Any: A stream object that yields response chunks.

        Raises:
            Exception: If there's an error generating the response, to be handled by the router.
        """
        # Import model registry
        from core.providers.huggingface_local_models import get_model_info

        # Convert string to Chat object if needed
        if isinstance(chat, str):
            from core.chat import Chat as ChatClass
            from core.chat import Role
            chat_obj = ChatClass()
            chat_obj.add_message(Role.USER, chat)
            chat_inner = chat_obj
        else:
            chat_inner = chat

        # Get model info from registry
        model_info = get_model_info(model_key)
        if not model_info:
            raise Exception(f"Model {model_key} not found in registry. Available models: {get_model_info()}")

        model_id = model_info["model_id"]

        # Default temperature
        if temperature is None:
            temperature = 0.7

        # Load model
        model, tokenizer = HuggingFaceLocalClient.load_model(model_id, chat_inner)

        # Log generation start
        if silent_reason:
            temp_str = "" if temperature == 0 else f" at temperature {temperature}"
            prefix = chat_inner.get_debug_title_prefix() if hasattr(chat_inner, 'get_debug_title_prefix') else ""
            g.debug_log(
                f"HuggingFace-Local: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is {colored('silently', 'green')} generating response{temp_str}...",
                force_print=True,
                prefix=prefix
            )
        else:
            temp_str = "" if temperature == 0 else f" at temperature {temperature}"
            prefix = chat_inner.get_debug_title_prefix() if hasattr(chat_inner, 'get_debug_title_prefix') else ""
            g.debug_log(
                f"HuggingFace-Local: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is generating response{temp_str}...",
                "green",
                force_print=True,
                prefix=prefix
            )

        # Convert chat to model format
        if hasattr(tokenizer, 'apply_chat_template'):
            # Use model's chat template
            messages = []
            for message in chat_inner.messages:
                role = message.role.value if hasattr(message.role, 'value') else str(message.role)
                messages.append({
                    "role": role,
                    "content": message.content
                })

            # Apply chat template and tokenize
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)
        else:
            # Fallback: simple concatenation
            prompt = "\n\n".join(
                f"{msg.role}: {msg.content}"
                for msg in chat_inner.messages
            )
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        # Create cancellation event
        HuggingFaceLocalClient._cancellation_event = threading.Event()

        try:
            # Generate with streaming
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Generation kwargs
            generation_kwargs = {
                "input_ids": input_ids,
                "streamer": streamer,
                "max_new_tokens": 2048,
                "temperature": temperature,
                "do_sample": temperature > 0,
            }

            # Start generation in separate thread
            generation_thread = threading.Thread(
                target=model.generate,
                kwargs=generation_kwargs
            )
            HuggingFaceLocalClient._current_task = generation_thread
            generation_thread.start()

            # Return cancellable stream
            return HuggingFaceLocalClient._create_cancellable_stream(
                streamer,
                generation_thread,
                model_key
            )

        except Exception as e:
            error_msg = f"HuggingFace Local error for {model_key}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e

    @classmethod
    def _create_cancellable_stream(cls, raw_stream, generation_thread, model_key: str):
        """
        Create a cancellable stream wrapper that monitors the cancellation event.
        """
        class CancellableHFStream:
            def __init__(self, stream, thread, cancellation_event, model_key):
                self.stream = stream
                self.thread = thread
                self.cancellation_event = cancellation_event
                self.model_key = model_key
                self._cancelled = False

            def __iter__(self):
                return self

            def __next__(self):
                # Check for cancellation
                if self.cancellation_event and self.cancellation_event.is_set():
                    if not self._cancelled:
                        self._cancelled = True
                        logger.debug(f"HuggingFace Local stream for {self.model_key} cancelled")
                        raise StopIteration("Stream cancelled")

                try:
                    # Get next token from stream
                    token = next(self.stream)
                    # Return in format compatible with existing stream processors
                    # Create a simple object with message.content attribute
                    class StreamChunk:
                        def __init__(self, content):
                            self.message = type('obj', (object,), {'content': content})()

                    return StreamChunk(token)

                except StopIteration:
                    # Natural end of stream
                    self.thread.join(timeout=1.0)
                    raise

        return CancellableHFStream(
            raw_stream,
            generation_thread,
            cls._cancellation_event,
            model_key
        )

    @staticmethod
    def get_downloaded_models() -> list[dict[str, Any]]:
        """
        Get list of downloaded models.

        Returns:
            List[Dict[str, Any]]: List of model information dictionaries
        """
        models_dir = HuggingFaceLocalClient.get_models_dir()
        downloaded = []

        if not models_dir.exists():
            return downloaded

        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                # Convert safe name back to model_id
                model_id = model_dir.name.replace("--", "/")

                # Check if valid model
                if (model_dir / "config.json").exists():
                    # Get size
                    size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())

                    downloaded.append({
                        "name": model_id,
                        "path": str(model_dir),
                        "size": size,
                    })

        return downloaded
