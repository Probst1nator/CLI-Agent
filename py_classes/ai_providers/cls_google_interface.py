import os
import time
from typing import Optional, List, Dict, Any, Union, Iterator, TypeVar, Generic, Literal, Tuple, cast
from collections.abc import Callable
import socket
import logging
from termcolor import colored
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, TooManyRequests
from google.generativeai.types import GenerateContentResponse
from py_classes.cls_text_stream_painter import TextStreamPainter
from py_classes.cls_chat import Chat, Role
from py_classes.unified_interfaces import AIProviderInterface
from py_classes.cls_rate_limit_tracker import rate_limit_tracker
from py_classes.globals import g
import base64
import re

logger = logging.getLogger(__name__)

# Define custom exception classes to differentiate error types
class TimeoutException(Exception):
    """Exception raised when a request to Google API times out."""
    pass

class RateLimitException(Exception):
    """Exception raised when Google API rate limits are exceeded."""
    pass

# Flag to track if API has been configured
_google_api_configured = False

class GoogleAPI(AIProviderInterface):
    """
    Implementation of the AIProviderInterface for the Google Gemini API.
    
    This class provides methods to interact with Google's Gemini models using 
    the Google Generative AI client library.
    """
    
    @staticmethod
    def _configure_api() -> None:
        """
        Configure the Google Generative AI API with the API key.
        
        Raises:
            Exception: If the API key is missing or invalid.
        """
        global _google_api_configured
        
        # Only configure if not already done
        if not _google_api_configured:
            api_key: Optional[str] = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise Exception("Google API key not found. Set the GOOGLE_API_KEY environment variable.")
            
            try:
                genai.configure(api_key=api_key)
                _google_api_configured = True
                logger.info("Google Generative AI API configured successfully")
            except Exception as e:
                _google_api_configured = False
                raise Exception(f"Failed to configure Google API: {e}")

    @staticmethod
    def generate_response(
        chat: Union[Chat, str], 
        model_key: str = "gemini-1.5-pro-latest", 
        temperature: float = 0, 
        silent_reason: str = ""
    ) -> Iterator[GenerateContentResponse]:
        """
        Generates a response using the Google Gemini API.
        
        Args:
            chat (Union[Chat, str]): The chat object containing messages or a string prompt.
            model_key (str): The model identifier (defaults to gemini-1.5-pro-latest).
            temperature (float): The temperature setting for the model (0.0 to 1.0).
            silent_reason (str): Reason for silence if applicable.
            
        Returns:
            Iterator[GenerateContentResponse]: A stream of response chunks from the Gemini API.
            
        Raises:
            RateLimitException: If the model is rate limited.
            TimeoutException: If the request times out.
            Exception: For other errors, to be handled by the router.
        """
        # Convert string to Chat object if needed
        if isinstance(chat, str):
            chat_obj = Chat()
            chat_obj.add_message(Role.USER, chat)
            chat = chat_obj

        # Get the prefix for debug logging
        prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""

        # Check if the model is rate limited
        if rate_limit_tracker.is_rate_limited(model_key):
            remaining_time = rate_limit_tracker.get_remaining_time(model_key)
            rate_limit_reason = f"rate limited (wait {remaining_time:.1f}s)"
            
            if not silent_reason:
                g.debug_log(f"Google-Api: {colored('<', 'yellow')}{colored(model_key, 'yellow')}{colored('>', 'yellow')} is {colored(rate_limit_reason, 'yellow')}", force_print=True, prefix=prefix)
            
            # Raise a silent rate limit exception
            raise RateLimitException(f"Model {model_key} is rate limited. Try again in {remaining_time:.1f} seconds")

        # Configure the API if not already done - let any error here bubble up to the router
        GoogleAPI._configure_api()
        
        # Get the appropriate model
        model = genai.GenerativeModel(model_key)
        
        # Set up the generation config with temperature
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            top_p=1.0,
            top_k=32,
            max_output_tokens=8192,
        )
        
        # Create Gemini-compatible messages
        gemini_messages = []
        
        if isinstance(chat, Chat):
            # Handle Chat object conversion to Gemini format
            system_prompt = chat.system_prompt if hasattr(chat, 'system_prompt') else None
            
            for message in chat.messages:
                # Extract role and content from the message tuple (role, content)
                message_role = message[0]  # Role enum
                message_content = message[1]  # Content string or list
                
                if message_role == Role.USER:
                    # Handle user messages, which might contain images
                    if isinstance(message_content, list):
                        # Multimodal content
                        content_parts = []
                        
                        for item in message_content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    content_parts.append(item.get("text", ""))
                                elif item.get("type") == "image":
                                    image_url = item.get("image_url", {}).get("url", "")
                                    if image_url.startswith("data:image"):
                                        # Handle base64 encoded images
                                        image_data = image_url.split(",")[1]
                                        mime_type = image_url.split(";")[0].split(":")[1]
                                        decoded_image = base64.b64decode(image_data)
                                        content_parts.append(genai.Part.from_data(decoded_image, mime_type=mime_type))
                                    elif image_url.startswith("http"):
                                        # Handle URL images
                                        content_parts.append(genai.Part.from_uri(image_url))
                            else:
                                # Simple text message
                                content_parts.append(str(item))
                                
                        gemini_messages.append({"role": "user", "parts": content_parts})
                    else:
                        # Simple text message
                        gemini_messages.append({"role": "user", "parts": [str(message_content)]})
                        
                elif message_role == Role.ASSISTANT:
                    # Assistant messages are always text
                    gemini_messages.append({"role": "model", "parts": [str(message_content)]})
                
                elif message_role == Role.SYSTEM:
                    # System messages are converted to user messages with a prefix
                    # We'll store them separately and handle them below
                    system_prompt = message_content
            
            # Add system prompt if present (prepend to the first user message)
            if system_prompt and gemini_messages and gemini_messages[0]["role"] == "user":
                first_message = gemini_messages[0]
                first_message["parts"].insert(0, f"System instruction: {system_prompt}\n\n")
        else:
            # Handle string input
            gemini_messages.append({"role": "user", "parts": [str(chat)]})
            
        # Print status message
        if silent_reason:
            temp_str = "" if temperature == 0 or temperature is None else f" at temperature {temperature}"
            g.debug_log(f"Google-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is {colored('silently', 'green')} generating response{temp_str}...", force_print=True, prefix=prefix)
        else:
            temp_str = "" if temperature == 0 or temperature is None else f" at temperature {temperature}"
            g.debug_log(f"Google-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is generating response{temp_str}...", "green", force_print=True, prefix=prefix)
        
        # Generate streaming response - let errors bubble up to the router
        # except for rate limit errors, which we'll handle here
        try:
            response = model.generate_content(
                gemini_messages,
                generation_config=generation_config,
                stream=True
            )
            
            return response
        except Exception as e:
            # Check if this is a rate limit error (usually contains "quota" in the error message)
            error_str = str(e).lower()
            if "quota" in error_str or "rate" in error_str:
                try:
                    # Extract retry time if possible (default to 60 seconds if not found)
                    retry_seconds = 60
                    retry_matches = re.findall(r"retry in (\d+)", error_str)
                    if retry_matches:
                        retry_seconds = int(retry_matches[0])
                    
                    # Update rate limit tracker
                    rate_limit_tracker.update_rate_limit(model_key, retry_seconds)
                    
                    error_msg = f"Google-Api: Rate limit reached for {colored('<' + model_key + '>', 'red')}: {e}"
                    g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
                    raise RateLimitException(f"Rate limit exceeded: {e}")
                except Exception as e2:
                    # If there's an error in the rate limit handling, just proceed with regular error
                    raise Exception(f"Google API error: {e} (rate limit handling failed: {e2})")
            # For all other errors, let them bubble up to the router
            raise

    @staticmethod
    def generate_embeddings(
        text: Union[str, List[str]], 
        model: str = "embedding-001", 
        chat: Optional[Chat] = None
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """
        Generates embeddings for the given text using the Google Gemini API.
        
        Args:
            text (Union[str, List[str]]): The text or list of texts to generate embeddings for.
            model (str): The embedding model to use.
            chat (Optional[Chat]): The chat object for debug printing.
            
        Returns:
            Optional[Union[List[float], List[List[float]]]]: The generated embedding(s) or None if an error occurs.
        """
        try:
            # Configure the API if not already done
            GoogleAPI._configure_api()
            
            # Print status message
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                if isinstance(text, list):
                    g.debug_log(f"Google-Api: {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')} is generating {len(text)} embeddings...", force_print=True, prefix=prefix)
                else:
                    g.debug_log(f"Google-Api: {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')} is generating embedding...", force_print=True, prefix=prefix)
                    
            # Determine task type for embedding (can influence quality)
            task_type = "RETRIEVAL_DOCUMENT"
                    
            if isinstance(text, str):
                # Single text case
                result = genai.embed_content(
                    model=model,
                    content=text,
                    task_type=task_type
                )
                return result["embedding"]
            else:
                # List of texts case
                embeddings = []
                for t in text:
                    result = genai.embed_content(
                        model=model,
                        content=t,
                        task_type=task_type
                    )
                    embeddings.append(result["embedding"])
                return embeddings
                
        except Exception as e:
            error_msg = f"Google API embedding error: {e}"
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
            logger.error(error_msg)
            return None
