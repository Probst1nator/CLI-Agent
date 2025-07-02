import os
import time
from typing import Optional, List, Dict, Any, Union, Iterator, TypeVar, Generic, Literal, Tuple, cast
from collections.abc import Callable
import socket
import logging
from termcolor import colored
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, TooManyRequests
import google.generativeai as genai
from google.generativeai import types
from py_classes.cls_text_stream_painter import TextStreamPainter
from py_classes.cls_chat import Chat, Role
from py_classes.unified_interfaces import AIProviderInterface
from py_classes.cls_rate_limit_tracker import rate_limit_tracker
from py_classes.globals import g
import base64
import re
# Import audio utility functions
from py_methods.utils_audio import save_binary_file, convert_to_wav, parse_audio_mime_type
import mimetypes
# Import exceptions from Groq interface to avoid duplication
from py_classes.ai_providers.cls_groq_interface import TimeoutException, RateLimitException

logger = logging.getLogger(__name__)

# Flag to track if API has been configured
_gemini_api_configured = False

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
        global _gemini_api_configured
        
        # Only configure if not already done
        if not _gemini_api_configured:
            api_key: Optional[str] = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise Exception("Google API key not found. Set the GEMINI_API_KEY environment variable.")
            
            try:
                genai.configure(api_key=api_key)
                _gemini_api_configured = True
                logger.info("Google Generative AI API configured successfully")
            except Exception as e:
                _gemini_api_configured = False
                raise Exception(f"Failed to configure Google API: {e}")

    @staticmethod
    def generate_response(
        chat: Union[Chat, str], 
        model_key: str = "gemini-1.5-pro-latest", 
        temperature: float = 0, 
        silent_reason: str = "", 
        thinking_budget: Optional[int] = None
    ) -> Any:  # Return type changed to Any to avoid circular imports
        """
        Generates a response using the Google Gemini API.
        
        Args:
            chat (Union[Chat, str]): The chat object containing messages or a string prompt.
            model_key (str): The model identifier (defaults to gemini-1.5-pro-latest).
            temperature (float): The temperature setting for the model (0.0 to 1.0).
            silent_reason (str): Reason for silence if applicable.
            thinking_budget (Optional[int]): Token budget for model's internal reasoning process.
                                           Use -1 for dynamic, 0 to disable, or positive integer for fixed budget.
            
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
        
        # Set up the generation config with temperature and thinking budget
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            top_p=1.0,
            top_k=32,
            max_output_tokens=8192,
        )
        
        # Add thinking config if thinking_budget is specified and model supports it
        if thinking_budget is not None and ("2.5" in model_key or "2.0" in model_key):
            try:
                generation_config.thinking_config = genai.types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )
            except Exception as thinking_error:
                # If thinking config fails, continue without it
                if not silent_reason:
                    g.debug_log(f"Google-Api: Warning - thinking config not supported: {thinking_error}", "yellow", prefix=prefix)
        elif thinking_budget is not None and not ("2.5" in model_key or "2.0" in model_key):
            # Warn if thinking budget is requested but model doesn't support it
            if not silent_reason:
                g.debug_log(f"Google-Api: Warning - thinking budget not supported by model {model_key}", "yellow", prefix=prefix)
        
        # Use the Chat's to_gemini method to create Gemini-compatible messages
        if isinstance(chat, Chat):
            gemini_messages = chat.to_gemini()
        else:
            # Simple string input case (should not happen due to conversion above, but just in case)
            gemini_messages = [{"role": "user", "parts": [str(chat)]}]
            
        # Print status message
        if silent_reason:
            temp_str = "" if temperature == 0 or temperature is None else f" at temperature {temperature}"
            g.debug_log(f"Google-Api: {colored('<' + model_key + '>', 'green')} is using {colored('<gemini-api>', 'green')} for: {colored('<' + silent_reason + '>', 'yellow')}{temp_str}", force_print=True, with_title=False, prefix=prefix)
        else:
            temp_str = "" if temperature == 0 or temperature is None else f" at temperature {temperature}"
            g.debug_log(f"Google-Api: {colored('<' + model_key + '>', 'green')} is using {colored('<gemini-api>', 'green')}{temp_str} to generate a response...", force_print=True, prefix=prefix)
        
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
            # Check if this is a quota exceeded error (429 error with specific message)
            error_str = str(e)
            if "429" in error_str and "quota" in error_str.lower() and "exceeded" in error_str.lower():
                # Extract retry delay from the error message
                retry_seconds = 60  # default
                retry_matches = re.findall(r"retry_delay\s*{\s*seconds:\s*(\d+)", error_str)
                if retry_matches:
                    retry_seconds = int(retry_matches[0])
                
                # Update rate limit tracker
                rate_limit_tracker.update_rate_limit(model_key, retry_seconds)
                
                # Show only a single line warning for quota exceeded
                if not silent_reason:
                    g.debug_log(f"⚡ Quota exceeded for {model_key} - retry in {retry_seconds}s", "yellow", force_print=True, prefix=prefix)
                
                # Mark the exception as already logged and raise as RateLimitException
                setattr(e, 'already_logged', True)
                raise RateLimitException(f"Model {model_key} quota exceeded. Try again in {retry_seconds} seconds")
            
            # Check if this is a general rate limit error
            elif "quota" in error_str.lower() or "rate" in error_str.lower():
                retry_seconds = 60
                retry_matches = re.findall(r"retry in (\d+)", error_str)
                if retry_matches:
                    retry_seconds = int(retry_matches[0])
                
                # Update rate limit tracker
                rate_limit_tracker.update_rate_limit(model_key, retry_seconds)
                
                # Show only a single line warning for rate limits
                if not silent_reason:
                    g.debug_log(f"⚡ Rate limit reached for {model_key} - retry in {retry_seconds}s", "yellow", force_print=True, prefix=prefix)
                
                # Mark the exception as already logged and raise as RateLimitException
                setattr(e, 'already_logged', True)
                raise RateLimitException(f"Model {model_key} is rate limited. Try again in {retry_seconds} seconds")
            
            # Log other errors here so we don't get duplicate logs
            if not silent_reason:
                error_msg = f"\nGoogle-Api: Failed to generate response with model {model_key}: {e}"
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
                
                # Mark the exception as already logged so the router won't log it again
                setattr(e, 'already_logged', True)
                
            # Re-raise the original exception
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
            # Keep returning None for embeddings as this appears to be the expected behavior
            # This is different from generate_response and generate_speech which re-raise exceptions
            return None

    @staticmethod
    def generate_speech(
        text: str,
        output_file: str = "output.wav",
        model: str = "gemini-2.5-flash-preview-tts",
        temperature: float = 1.0,
        chat: Optional[Chat] = None,
        speaker_config: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generates speech from text using Google's TTS API with multi-speaker support.
        
        Args:
            text (str): The text to convert to speech.
            output_file (str): The name of the output file to save the audio to.
            model (str): The TTS model to use.
            temperature (float): The temperature setting for the model.
            chat (Optional[Chat]): The chat object for debug printing.
            speaker_config (Optional[List[Dict[str, str]]]): List of speaker configurations.
                Each dictionary should contain 'speaker' and 'voice' keys.
                If None, default voices will be used.
                
        Returns:
            str: The path to the saved audio file.
            
        Raises:
            Exception: If the API is not configured or if there is an error during generation.
        """
        try:
            # Imports are now at the top of the file
            
            # Configure the API if not already done
            GoogleAPI._configure_api()
            
            # Print status message
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(f"Google-Api: {colored('<', 'green')}{colored(model, 'green')}{colored('>', 'green')} is generating speech...", force_print=True, prefix=prefix)
            
            # Create the client
            client = genai.Client( # Using genai from top-level import
                api_key=os.environ.get("GEMINI_API_KEY"),
            )
            
            # Default speaker configuration if none provided
            if not speaker_config:
                speaker_config = [
                    {"speaker": "Chloe", "voice": "Kore"},
                    {"speaker": "Liam", "voice": "Iapetus"}
                ]
            
            # Create speaker voice configs
            speaker_voice_configs = []
            for config in speaker_config:
                speaker_voice_configs.append(
                    types.SpeakerVoiceConfig( # Using types from top-level import
                        speaker=config["speaker"],
                        voice_config=types.VoiceConfig( # Using types from top-level import
                            prebuilt_voice_config=types.PrebuiltVoiceConfig( # Using types from top-level import
                                voice_name=config["voice"]
                            )
                        ),
                    )
                )
            
            # Set up content for TTS
            contents = [
                types.Content( # Using types from top-level import
                    role="user",
                    parts=[
                        types.Part.from_text(text=text), # Using types from top-level import
                    ],
                ),
            ]
            
            # Configure TTS generation
            generate_content_config = types.GenerateContentConfig( # Using types from top-level import
                temperature=temperature,
                response_modalities=["audio"],
                speech_config=types.SpeechConfig( # Using types from top-level import
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig( # Using types from top-level import
                        speaker_voice_configs=speaker_voice_configs
                    ),
                ),
            )
            
            # Generate speech content
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue
                
                if chunk.candidates[0].content.parts[0].inline_data:
                    inline_data = chunk.candidates[0].content.parts[0].inline_data
                    data_buffer = inline_data.data
                    file_extension = mimetypes.guess_extension(inline_data.mime_type)
                    
                    if file_extension is None:
                        file_extension = ".wav"
                        data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)
                    
                    # Ensure output_file has the correct extension
                    base_name, existing_ext = os.path.splitext(output_file)
                    if not existing_ext:
                        output_file = f"{base_name}{file_extension}"
                    
                    # Save the file
                    return save_binary_file(output_file, data_buffer)
                else:
                    # Handle text response
                    if chunk.text and chat:
                        prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                        g.debug_log(f"TTS Response: {chunk.text}", force_print=True, prefix=prefix)
            
            # If we get here without returning a file, raise an exception
            raise Exception("No audio data received from the API")
            
        except Exception as e:
            error_msg = f"Google API speech generation error: {e}"
            if chat:
                prefix = chat.get_debug_title_prefix() if hasattr(chat, 'get_debug_title_prefix') else ""
                g.debug_log(error_msg, "red", is_error=True, prefix=prefix)
            logger.error(error_msg)
            # Mark the exception as already logged so the router won't log it again
            setattr(e, 'already_logged', True)
            # Re-raise the original exception
            raise