import os
from typing import Optional
from anthropic import Anthropic

from termcolor import colored
from cls_custom_coloring import CustomColoring
from interface.cls_chat import Chat, Role
import traceback
import tiktoken

from interface.cls_chat_client_interface import ChatClientInterface

class AnthropicChat(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the Anthropic API.
    """

    @staticmethod
    def generate_response(chat: Chat, model: str = "claude-3-5-sonnet-20240620", temperature: float = 0.7, silent: bool = False) -> Optional[str]:
        """
        Generates a response using the Anthropic API.

        Args:
            chat (Chat): The chat object containing messages.
            model (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent (bool): Whether to suppress print statements.

        Returns:
            Optional[str]: The generated response, or None if an error occurs.
        """
        try:
            client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=3.0, max_retries=2)
            
            if "claude-3-5-sonnet" in model or not model:
                model = "claude-3-5-sonnet-20240620"
            
            if not silent:
                print("Anthropic API: <" + colored(model,"green") + "> is generating response...")

            l_chat = Chat()
            l_chat.messages = chat.messages

            system_message = ""
            if l_chat.messages[0][0] == Role.SYSTEM:
                system_message = l_chat.messages[0][1]
                l_chat.messages = l_chat.messages[1:]

            with client.messages.stream(
                model=model,
                max_tokens=4096,
                system=system_message,
                messages=l_chat.to_groq_format(),
                temperature=temperature,
            ) as stream:
                full_response = ""
                token_keeper = CustomColoring()
                for token in stream.text_stream:
                    if not silent:
                        print(token_keeper.apply_color(token), end="")
                    full_response += token
                if not silent:
                    print()
                return full_response
        except Exception as e:
            print(f"Anthropic API error: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def count_tokens(text: str, model: str = "claude-3-5-sonnet-20240620") -> int:
        """
        Counts the number of tokens in the given text for the specified model.

        Args:
            text (str): The input text.
            model (str): The model identifier.

        Returns:
            int: The number of tokens in the input text.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"Token counting error: {e}")
            return 0
