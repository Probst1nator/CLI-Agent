import os
from typing import Optional
from anthropic import Anthropic

from termcolor import colored
from py_classes.cls_custom_coloring import CustomColoring
from py_classes.cls_chat import Chat, Role
from py_classes.cls_ai_provider_interface import ChatClientInterface

class AnthropicAPI(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the Anthropic API.
    """

    @staticmethod
    def generate_response(chat: Chat, model: str = "claude-3-5-sonnet-latest", temperature: float = 0.7, silent_reason: str = False) -> Optional[str]:
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
        debug_print = ChatClientInterface.create_debug_printer(chat)
        try:
            client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=3.0, max_retries=2)
            
            
            if silent_reason:
                debug_print(f"Anthropic-Api: <{colored(model, 'green')}> is {colored('silently', 'green')} generating response...", force_print=True)
            else:
                debug_print(f"Anthropic-Api: <{colored(model, 'green')}> is generating response...", "green", force_print=True)

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
                messages=l_chat.to_groq(),
                temperature=temperature,
            ) as stream:
                full_response = ""
                token_keeper = CustomColoring()
                for token in stream.text_stream:
                    if not silent_reason:
                        debug_print(token_keeper.apply_color(token), end="", with_title=False)
                    full_response += token
                if not silent_reason:
                    debug_print("", with_title=False)
                return full_response
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
