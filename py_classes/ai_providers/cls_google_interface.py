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
from py_classes.cls_custom_coloring import CustomColoring
from py_classes.cls_chat import Chat, Role
from py_classes.unified_interfaces import AIProviderInterface
from py_classes.cls_rate_limit_tracker import rate_limit_tracker

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
            Exception: For other errors.
        """
        # Convert string to Chat object if needed
        if isinstance(chat, str):
            chat_obj = Chat()
            chat_obj.add_message(Role.USER, chat)
            chat = chat_obj
            
        debug_print = AIProviderInterface.create_debug_printer(chat)

        # Check if the model is rate limited
        if rate_limit_tracker.is_rate_limited(model_key):
            remaining_time = rate_limit_tracker.get_remaining_time(model_key)
            rate_limit_reason = f"rate limited (wait {remaining_time:.1f}s)"
            
            if not silent_reason:
                debug_print(f"Google-Api: {colored('<', 'yellow')}{colored(model_key, 'yellow')}{colored('>', 'yellow')} is {colored(rate_limit_reason, 'yellow')}", force_print=True)
            
            # Raise a silent rate limit exception
            raise RateLimitException(f"Model {model_key} is rate limited. Try again in {remaining_time:.1f} seconds")

        try:
            # Configure the API if not already done
            GoogleAPI._configure_api()
            
            # Create generation config
            generation_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
            
            # Create model instance
            model = genai.GenerativeModel(model_key)
            
            # Convert Chat to Gemini format
            gemini_messages = chat.to_gemini()
            
            # Check if we're using a 2.5+ model, which may have stricter role requirements
            is_newer_model = "2.5" in model_key or "3" in model_key
            
            # For newer models, ensure we're only using "user" and "model" roles
            if is_newer_model:
                # Make a copy of the messages to modify
                fixed_messages = []
                
                # Find system message if any
                system_content = None
                for msg in gemini_messages:
                    if msg.get("role") == "system":
                        system_content = "\n".join([part.get("text", "") for part in msg.get("parts", []) if "text" in part])
                    else:
                        fixed_messages.append(msg)
                
                # If we found a system message, add it to the first user message
                if system_content and fixed_messages:
                    for msg in fixed_messages:
                        if msg.get("role") == "user":
                            for part in msg.get("parts", []):
                                if "text" in part:
                                    part["text"] = f"{system_content}\n\n{part['text']}"
                                    break
                            break
                
                # Ensure all roles are valid
                for msg in fixed_messages:
                    if msg.get("role") not in ["user", "model"]:
                        if msg.get("role") == "assistant":
                            msg["role"] = "model"
                        else:
                            msg["role"] = "user"  # Default to user
                
                gemini_messages = fixed_messages
                
            # Print status message
            if silent_reason:
                temp_str = "" if temperature == 0 or temperature is None else f" at temperature {temperature}"
                debug_print(f"Google-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is {colored('silently', 'green')} generating response{temp_str}...", force_print=True)
            else:
                temp_str = "" if temperature == 0 or temperature is None else f" at temperature {temperature}"
                debug_print(f"Google-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is generating response{temp_str}...", "green", force_print=True)
            
            # Generate streaming response
            response = model.generate_content(
                gemini_messages,
                generation_config=generation_config,
                stream=True
            )
            
            return response

        except (socket.timeout, socket.error, TimeoutError) as e:
            # Handle timeout errors
            debug_print(f"Google API timeout error: {e}", "red", is_error=True)
            raise TimeoutException(f"Request timed out: {e}")
            
        except (ResourceExhausted, ServiceUnavailable, TooManyRequests) as e:
            # Handle rate limit errors
            debug_print(f"Google API rate limit error: {e}", "red", is_error=True)
            
            # Try to parse retry after information (if available)
            retry_after: int = 60  # Default to 60 seconds
            error_message = str(e)
            if "retry after" in error_message.lower():
                try:
                    # Try to extract the retry after time
                    retry_after_part = error_message.lower().split("retry after")[1].strip()
                    if "seconds" in retry_after_part:
                        retry_after = int(retry_after_part.split("seconds")[0].strip())
                    elif "minutes" in retry_after_part:
                        retry_after = int(retry_after_part.split("minutes")[0].strip()) * 60
                except (IndexError, ValueError):
                    pass
            
            # Update rate limit tracker
            rate_limit_tracker.update_rate_limit(model_key, retry_after)
            
            # Raise exception
            raise RateLimitException(f"Rate limit exceeded: {e}")
            
        except Exception as e:
            # Handle other errors
            error_msg = f"Google API error: {e}"
            debug_print(error_msg, "red", is_error=True)
            raise Exception(error_msg)
            
    @staticmethod
    def generate_embeddings(
        text: Union[str, List[str]], 
        model: str = "embedding-001", 
        chat: Optional[Chat] = None
    ) -> Optional[Union[List[float], List[List[float]]]]:
        """
        Generates embeddings for the given text(s) using the specified Gemini embedding model.
        
        Args:
            text (Union[str, List[str]]): The input text or list of texts to generate embeddings for.
            model (str): The embedding model to use.
            chat (Optional[Chat]): The chat object for debug printing.
        
        Returns:
            Optional[Union[List[float], List[List[float]]]]: The generated embedding(s) as a list of floats or 
                                                           list of list of floats, or None if an error occurs.
        
        Raises:
            Exception: For errors during embedding generation.
        """
        if not text or (isinstance(text, str) and len(text.strip()) < 3):
            return None
            
        if isinstance(text, list) and (len(text) == 0 or all(len(t.strip()) < 3 for t in text if isinstance(t, str))):
            return None
            
        debug_print = AIProviderInterface.create_debug_printer(chat)
        
        # Check if the model is rate limited
        if rate_limit_tracker.is_rate_limited(model):
            remaining_time = rate_limit_tracker.get_remaining_time(model)
            rate_limit_reason = f"rate limited (wait {remaining_time:.1f}s)"
            
            if debug_print:
                debug_print(f"Google-Api Embeddings: {colored('<', 'yellow')}{colored(model, 'yellow')}{colored('>', 'yellow')} is {colored(rate_limit_reason, 'yellow')}", force_print=True)
            
            # Raise a silent rate limit exception
            raise RateLimitException(f"Model {model} is rate limited. Try again in {remaining_time:.1f} seconds")
            
        try:
            # Configure the API if not already done
            GoogleAPI._configure_api()
            
            # Print status message
            if debug_print:
                if isinstance(text, list):
                    debug_print(f"Google-Api: {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')} is generating {len(text)} embeddings...", force_print=True)
                else:
                    debug_print(f"Google-Api: {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')} is generating embedding...", force_print=True)
                    
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
                # List of texts case - process in batches to avoid rate limits
                embeddings: List[List[float]] = []
                batch_size = 5  # Process 5 at a time to avoid rate limits
                
                for i in range(0, len(text), batch_size):
                    batch = text[i:i+batch_size]
                    
                    for item in batch:
                        if isinstance(item, str) and len(item.strip()) >= 3:
                            result = genai.embed_content(
                                model=model,
                                content=item,
                                task_type=task_type
                            )
                            embeddings.append(result["embedding"])
                        else:
                            # Add empty list for invalid items to maintain position
                            embeddings.append([])
                            
                    # Print progress if debug_print is available
                    if debug_print:
                        debug_print(f"Processed {min(i+batch_size, len(text))}/{len(text)} embeddings", force_print=True)
                
                return embeddings
            
        except (ResourceExhausted, ServiceUnavailable, TooManyRequests) as e:
            # Handle rate limit errors
            if debug_print:
                debug_print(f"Google API rate limit error: {e}", "red", is_error=True)
            
            # Default retry time
            retry_after = 60  # Default to 60 seconds
            
            # Update rate limit tracker
            rate_limit_tracker.update_rate_limit(model, retry_after)
            
            # Raise exception
            raise RateLimitException(f"Rate limit exceeded: {e}")
            
        except Exception as e:
            # Handle other errors
            error_msg = f"Google API embedding error: {e}"
            if debug_print:
                debug_print(error_msg, "red", is_error=True)
            raise Exception(error_msg)
