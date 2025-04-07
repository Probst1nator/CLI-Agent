from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from py_classes.cls_chat import Chat
from py_classes.cls_debug_utils import get_debug_title_prefix, DEBUG_TITLE_FORMAT
from termcolor import colored
import logging

# Configure logger with a NullHandler by default
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Remove any existing handlers and set up console handler to only show ERROR or higher
for handler in logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        logger.removeHandler(handler)

# Add a console handler that only shows ERROR level and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
logger.addHandler(console_handler)

class ChatClientInterface(ABC):
    @staticmethod
    def get_debug_title_prefix(chat: Chat) -> str:
        """
        Get a formatted prefix string for debug messages that includes the chat's debug title if available.
        
        Args:
            chat (Chat): The chat whose debug_title should be included
            
        Returns:
            str: The formatted prefix string
        """
        return get_debug_title_prefix(chat)
    
    @staticmethod
    def create_debug_printer(chat: Optional[Chat] = None):
        """
        Creates a debug printer function that includes the chat's debug title in each print.
        Also logs messages to the logger with appropriate log levels.
        
        Args:
            chat (Chat): The chat object whose debug_title should be included in prints.
            
        Returns:
            function: A function that can be used for debug printing with the chat title.
        """
        def debug_print(message: str, color: str = None, end: str = '\n', with_title: bool = True, is_error: bool = False, force_print: bool = False) -> None:
            """
            Print debug information with chat title prefix and logging.
            
            Args:
                message (str): The message to print
                color (str, optional): Color for the message
                end (str): End character
                with_title (bool): Whether to include the chat title
                is_error (bool): Whether this is an error message
                force_print (bool): Force printing to console even for info messages
            """

            if with_title and chat:
                prefix = ChatClientInterface.get_debug_title_prefix(chat)
                log_message = f"{prefix}{message}"
                
                # Log to appropriate logger level (ignoring color)
                if is_error:
                    logger.error(log_message)
                    # For errors, always print to console
                    if color:
                        print(colored(log_message, color), end=end)
                    else:
                        print(log_message, end=end)
                else:
                    # For info level, log to logger
                    logger.info(log_message)
                    # Only print to console if forced
                    if force_print:
                        if color:
                            print(colored(log_message, color), end=end)
                        else:
                            print(log_message, end=end)
            else:
                # For character-by-character printing, don't log to the logger
                # But still print to console
                if color:
                    print(colored(message, color), end=end)
                else:
                    print(message, end=end)
        return debug_print
    
    @abstractmethod
    def generate_response(self, chat: Chat, model: str, temperature: float, tools: Optional[List[Dict[str, Any]]] = None, silent_reason: str = "") -> Optional[str]:
        """
        Generates a response based on the provided chat and model.

        Args:
            chat (Chat): The chat object containing messages.
            model (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent (bool): Whether to suppress print statements.

        Returns:
            Optional[str]: The generated response, or None if an error occurs.
        """
        pass