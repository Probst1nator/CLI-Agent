from anthropic import Anthropic, RateLimitError
import os
import time
from typing import Dict, List, Optional, Any, Union
from collections.abc import Callable
from termcolor import colored
from py_classes.cls_chat import Chat
from py_classes.unified_interfaces import AIProviderInterface
import socket
import json
import logging
import base64

from py_classes.cls_text_stream_painter import TextStreamPainter
from py_classes.cls_chat import Role

class AnthropicAPI(AIProviderInterface):
    """
    Implementation of the ChatClientInterface for the Anthropic API.
    """

    @staticmethod
    def generate_response(chat: Union[Chat, str], model_key: str = "claude-3-5-sonnet-latest", temperature: float = 0.7, silent_reason: str = "") -> Any:
        """
        Generates a response using the Anthropic API.

        Args:
            chat (Union[Chat, str]): The chat object containing messages or a string prompt.
            model_key (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent_reason (str): Reason for suppressing print statements.

        Returns:
            Any: A stream object that yields response chunks.
        """
        # Convert string to Chat object if needed
        if isinstance(chat, str):
            from py_classes.cls_chat import Chat, Role
            chat_obj = Chat()
            chat_obj.add_message(Role.USER, chat)
            chat = chat_obj
            
        debug_print = AIProviderInterface.create_debug_printer(chat)
        try:
            client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=3.0, max_retries=2)
            
            if silent_reason:
                temp_str = "" if temperature == 0 else f" at temperature {temperature}"
                debug_print(f"Anthropic-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is {colored('silently', 'green')} generating response{temp_str}...", force_print=True)
            else:
                temp_str = "" if temperature == 0 else f" at temperature {temperature}"
                debug_print(f"Anthropic-Api: {colored('<', 'green')}{colored(model_key, 'green')}{colored('>', 'green')} is generating response{temp_str}...", "green", force_print=True)

            l_chat = Chat()
            l_chat.messages = chat.messages

            system_message = ""
            if l_chat.messages[0][0] == Role.SYSTEM:
                system_message = l_chat.messages[0][1]
                l_chat.messages = l_chat.messages[1:]

            return client.messages.stream(
                model=model_key,
                max_tokens=4096,
                system=system_message,
                messages=l_chat.to_groq(),
                temperature=temperature,
            )

        except Exception as e:
            error_msg = f"Anthropic API error: {e}"
            debug_print(error_msg, "red", is_error=True)
            raise Exception(error_msg)

    # @staticmethod
    # def count_tokens(text: str, model: str = "claude-3-5-sonnet-latest-20240620") -> int:
    #     """
    #     Counts the number of tokens in the given text for the specified model.

    #     Args:
    #         text (str): The input text.
    #         model (str): The model identifier.

    #     Returns:
    #         int: The number of tokens in the input text.
    #     """
    #     encoding = tiktoken.encoding_for_model(model)
    #     tokens = encoding.encode(text)
    #     return len(tokens)
