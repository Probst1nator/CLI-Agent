import os
from typing import Optional, Tuple
from groq import Groq
from termcolor import colored
from py_classes.cls_custom_coloring import CustomColoring
from py_classes.cls_chat import Chat
from py_classes.cls_ai_provider_interface import ChatClientInterface
from py_classes.cls_rate_limit_tracker import rate_limit_tracker
import socket
import logging

logger = logging.getLogger(__name__)

# Define custom exception classes to differentiate error types
class TimeoutException(Exception):
    """Exception raised when a request to Groq API times out."""
    pass

class RateLimitException(Exception):
    """Exception raised when Groq API rate limits are exceeded."""
    pass

class GroqAPI(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the Groq API.
    """

    @staticmethod
    def generate_response(chat: Chat, model: str, temperature: float = 0.7, silent_reason: str = False) -> Optional[str]:
        """
        Generates a response using the Groq API.
        Args:
            chat (Chat): The chat object containing messages.
            model (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent_reason (str): Reason for silence if applicable.
        Returns:
            Optional[str]: The generated response, or None if an error occurs.
        """
        debug_print = ChatClientInterface.create_debug_printer(chat)

        # Check if the model is rate limited
        if rate_limit_tracker.is_rate_limited(model):
            remaining_time = rate_limit_tracker.get_remaining_time(model)
            rate_limit_reason = f"rate limited (wait {remaining_time:.1f}s)"
            
            if not silent_reason:
                debug_print(f"Groq-Api: <{colored(model, 'yellow')}> is {colored(rate_limit_reason, 'yellow')}", force_print=True)
            
            # Raise a silent rate limit exception
            raise RateLimitException(f"Model {model} is rate limited. Try again in {remaining_time:.1f} seconds")

        try:
            client = Groq(api_key=os.getenv('GROQ_API_KEY'), timeout=3.0, max_retries=0)
            chat_completion = client.chat.completions.create(
                messages=chat.to_groq(), model=model, temperature=temperature, stream=True, stop="</s>"
            )
            if silent_reason:
                debug_print(f"Groq-Api: <{colored(model, 'green')}> is {colored('silently ' + silent_reason, 'green')}", force_print=True)
            else:
                debug_print(f"Groq-Api: <{colored(model, 'green')}> is generating response...", "green", force_print=True)
            full_response = ""
            token_keeper = CustomColoring()
            for chunk in chat_completion:
                token = chunk.choices[0].delta.content
                if token:
                    full_response += token
                    if not silent_reason:
                        debug_print(token_keeper.apply_color(token), end="", with_title=False)
            if not silent_reason:
                debug_print("", with_title=False)
            return full_response
        except (socket.timeout, socket.error, TimeoutError) as e:
            # Handle timeout-related errors silently without printing
            raise TimeoutException(f"❌ Request timed out: {e}")
        except Exception as e:
            # Check if this is a rate limit error (429)
            error_str = str(e)
            if "Error code: 429" in error_str or "rate_limit_exceeded" in error_str:
                try:
                    try_again_seconds = int(error_str.split("Please try again in")[1].split("ms")[0].strip()) / 1000
                except (IndexError, ValueError):
                    # If parsing fails, use a default value
                    try_again_seconds = 60.0  # Default to 60 seconds if parsing fails
                    logger.warning(f"Failed to parse rate limit wait time, using default of {try_again_seconds}s")
                
                # Update the rate limit tracker
                rate_limit_tracker.update_rate_limit(model, try_again_seconds)
                
                # Handle rate limit errors silently
                raise RateLimitException(f"❌ Rate limit exceeded: {e}")
            # Handle other types of errors
            raise Exception(f"Groq-Api error: {e}")


    @staticmethod
    def transcribe_audio(filepath: str, model: str = "whisper-large-v3-turbo", language: Optional[str] = None, silent_reason: str = False, chat: Optional[Chat] = None) -> Optional[Tuple[str, str]]:
        """
        Transcribes an audio file using Groq's Whisper implementation.
        Args:
            filepath (str): The path to the audio file.
            model (str): The Whisper model to use.
            language (str): The language of the audio (default: "auto" for auto-detection).
            silent_reason (str): Reason for silence if applicable.
            chat (Optional[Chat]): Chat object for debug printing with title.
        Returns:
            Optional[Tuple[str, str]]: A tuple containing the transcribed text and detected language, or None if an error occurs.
        """
        debug_print = ChatClientInterface.create_debug_printer(chat)
        
        # Check if the model is rate limited
        if rate_limit_tracker.is_rate_limited(model):
            remaining_time = rate_limit_tracker.get_remaining_time(model)
            rate_limit_reason = f"rate limited (wait {remaining_time:.1f}s)"
            
            if not silent_reason and debug_print:
                debug_print(f"Groq-Api: <{colored(model, 'yellow')}> is {colored(rate_limit_reason, 'yellow')}", force_print=True)
            
            # Raise a silent rate limit exception
            raise RateLimitException(f"Model {model} is rate limited. Try again in {remaining_time:.1f} seconds")
        
        try:
            client = Groq(api_key=os.getenv('GROQ_API_KEY'), timeout=3.0, max_retries=0)
            
            if not silent_reason:
                debug_print(f"Groq-Api: Transcribing audio using <{colored(model, 'green')}>...", force_print=True)
            
            with open(filepath, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(filepath, file.read()),
                    model=model,
                    response_format="verbose_json",
                    language=language
                )
            
            if not silent_reason:
                debug_print(colored("Transcription complete.", 'green'), force_print=True)
            
            return transcription.text, transcription.language

        except (socket.timeout, socket.error, TimeoutError) as e:
            # Handle timeout-related errors silently without printing
            raise TimeoutException(f"❌ Request timed out: {e}")
        except Exception as e:
            # Check if this is a rate limit error (429)
            error_str = str(e)
            if "Error code: 429" in error_str or "rate_limit_exceeded" in error_str:
                try:
                    try_again_seconds = int(error_str.split("Please try again in")[1].split("ms")[0].strip()) / 1000
                except (IndexError, ValueError):
                    # If parsing fails, use a default value
                    try_again_seconds = 60.0  # Default to 60 seconds if parsing fails
                    logger.warning(f"Failed to parse rate limit wait time, using default of {try_again_seconds}s")
                
                # Update the rate limit tracker
                rate_limit_tracker.update_rate_limit(model, try_again_seconds)
                
                # Handle rate limit errors silently
                raise RateLimitException(f"❌ Rate limit exceeded: {e}")
            
            # Handle other errors without printing
            error_msg = f"Groq-Api audio transcription error: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    # The token counting method is commented out in your original code,
    # so I'm leaving it as is.
    
    # @staticmethod
    # def count_tokens(text: str, model: str) -> int:
    #     """
    #     Counts the number of tokens in the given text for the specified model.
    #     Args:
    #     text (str): The input text.
    #     model (str): The model identifier.
    #     Returns:
    #     int: The number of tokens in the input text.
    #     """
    #     encoding = tiktoken.encoding_for_model(model)
    #     tokens = encoding.encode(text)
    #     return len(tokens)