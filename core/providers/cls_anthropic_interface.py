import logging
import os
import threading
from typing import Any, Optional

from anthropic import Anthropic
from termcolor import colored

from core.chat import Chat, Role
from core.globals import g
from py_classes.unified_interfaces import AIProviderInterface

logger = logging.getLogger(__name__)

class AnthropicAPI(AIProviderInterface):
    """
    Implementation of the ChatClientInterface for the Anthropic API.
    """

    # Class-level cancellation support
    _cancellation_event = None
    _current_task = None
    _cancellation_lock = threading.Lock()

    @classmethod
    def cancel_current_generation(cls):
        """
        Cancel any currently running Anthropic generation.
        This sets a cancellation event that can be checked by streaming operations.
        """
        try:
            with cls._cancellation_lock:
                if cls._cancellation_event:
                    cls._cancellation_event.set()
                    logger.debug("Anthropic stream cancellation event set")

                # Note: Anthropic SDK doesn't support direct HTTP request cancellation
                # The cancellation will be handled at the stream iteration level

        except Exception as e:
            logger.warning(f"Error cancelling Anthropic generation: {e}")

    @staticmethod
    def generate_response(chat: Chat, model_key: str = "claude-3-5-sonnet-latest", temperature: float = 0.7, silent_reason: str = "", thinking_budget: Optional[int] = None) -> Any:
        """
        Generates a response using the Anthropic API.

        Args:
            chat (Chat): The chat object containing messages or a string prompt.
            model_key (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent_reason (str): Reason for suppressing print statements.
            thinking_budget (Optional[int]): Token budget for reasoning (ignored by Anthropic).

        Returns:
            Any: A stream object that yields response chunks.

        Raises:
            RateLimitError: If the API rate limit is exceeded.
            TimeoutError: If the request times out.
            Exception: For other errors, to be handled by the router.
        """
        # Configure the client (let any error here bubble up to the router)
        client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

        # Informational logging (not error handling)
        if silent_reason:
            temp_str = "" if temperature == 0 else f" at temperature {temperature}"
            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
            g.debug_log(f"Anthropic-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is {colored('silently', 'green')} generating response{temp_str}...", force_print=True, prefix=prefix)
        else:
            temp_str = "" if temperature == 0 else f" at temperature {temperature}"
            prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
            g.debug_log(f"Anthropic-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is generating response{temp_str}...", "green", force_print=True, prefix=prefix)

        system_content = ""
        anthropic_messages = []

        if isinstance(chat, Chat):
            for message in chat.messages:
                if message.role == Role.SYSTEM:
                    system_content = message.content
                    continue

                if message.role == Role.USER:
                    if message.images:
                        content = []
                        if message.content:
                            content.append({"type": "text", "text": message.content})

                        for image in message.images:
                            mime_type = "image/jpeg"
                            try:
                                import base64
                                image_data = base64.b64decode(image[:100])
                                if image_data.startswith(b'\x89PNG'):
                                    mime_type = "image/png"
                                elif image_data.startswith(b'GIF8'):
                                    mime_type = "image/gif"
                                elif image_data.startswith(b'\xff\xd8\xff'):
                                    mime_type = "image/jpeg"
                                elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:20]:
                                    mime_type = "image/webp"
                            except Exception:
                                pass

                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": image
                                }
                            })

                        anthropic_messages.append({"role": "user", "content": content})
                    else:
                        anthropic_messages.append({"role": "user", "content": message.content})

                elif message.role == Role.ASSISTANT:
                    anthropic_messages.append({"role": "assistant", "content": message.content})
        else:
            anthropic_messages.append({"role": "user", "content": chat})

        # Create the stream with system prompt if provided
        create_args = {
            "model": model_key,
            "messages": anthropic_messages,
            "max_tokens": 4096,
            "temperature": temperature,
            "stream": True
        }

        if system_content:
            create_args["system"] = system_content

        # Set up cancellation event for this generation
        AnthropicAPI._cancellation_event = threading.Event()

        # Let errors bubble up to the router for centralized handling
        # RateLimitError and TimeoutError will be handled by the router's _handle_model_error method
        raw_stream = client.messages.create(**create_args)

        # Return a cancellable stream wrapper
        return AnthropicAPI._create_cancellable_stream(raw_stream, model_key)

    @classmethod
    def _create_cancellable_stream(cls, raw_stream, model_key: str):
        """
        Create a cancellable stream wrapper that monitors the cancellation event.
        This allows for immediate interruption of Anthropic streaming responses.
        """
        class CancellableAnthropicStream:
            def __init__(self, stream, cancellation_event, model_key):
                self.stream = stream
                self.cancellation_event = cancellation_event
                self.model_key = model_key

            def __iter__(self):
                try:
                    for chunk in self.stream:
                        # Check for cancellation before yielding each chunk
                        if self.cancellation_event and self.cancellation_event.is_set():
                            logger.debug(f"Anthropic stream for {self.model_key} cancelled via interruption")
                            break
                        yield chunk
                finally:
                    # Cleanup: close the stream if possible
                    try:
                        if hasattr(self.stream, 'close'):
                            self.stream.close()
                    except Exception as e:
                        logger.debug(f"Error closing Anthropic stream: {e}")

            @property
            def text_stream(self):
                """Extract text from Anthropic streaming events"""
                def text_stream_generator():
                    """Parse Anthropic raw events and extract text content"""
                    try:
                        for event in self:
                            if self.cancellation_event and self.cancellation_event.is_set():
                                logger.debug(f"Anthropic text_stream for {self.model_key} cancelled via interruption")
                                break

                            # Extract text from RawContentBlockDeltaEvent
                            if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                                yield event.delta.text
                    except Exception as e:
                        logger.warning(f"Error in Anthropic text_stream extraction: {e}")
                        raise

                return text_stream_generator()

        return CancellableAnthropicStream(raw_stream, cls._cancellation_event, model_key)

    @staticmethod
    def custom_anthropic_call(model: str, messages: list, max_tokens: int = 1024, temperature: float = 0.7, system: str = None):
        """
        Direct Anthropic API call using Claude Code subscription auth.
        No API key needed - uses Claude Code authentication.

        Args:
            model (str): Model identifier (e.g., "claude-sonnet-4-5-20250929")
            messages (list): List of message dicts with 'role' and 'content' keys
            max_tokens (int): Maximum tokens to generate
            temperature (float): Temperature for generation
            system (str): Optional system prompt

        Returns:
            Message object from Anthropic SDK

        Example:
            from core.providers.cls_anthropic_interface import AnthropicAPI

            response = AnthropicAPI.custom_anthropic_call(
                model="claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Write hello world in Rust"}],
                max_tokens=1024
            )
            print(response.content[0].text)
        """
        client = Anthropic()

        create_args = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        if system:
            create_args["system"] = system

        return client.messages.create(**create_args)

