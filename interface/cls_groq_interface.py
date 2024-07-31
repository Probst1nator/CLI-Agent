import os
from typing import Optional
from groq import Groq
from termcolor import colored
from cls_custom_coloring import CustomColoring
from interface.cls_chat import Chat

from interface.cls_chat_client_interface import ChatClientInterface

class GroqChat(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the Groq API.
    """
    

    @staticmethod
    def generate_response(chat: Chat, model: str, temperature: float = 0.7, silent: bool = False) -> Optional[str]:
        """
        Generates a response using the Groq API.

        Args:
            chat (Chat): The chat object containing messages.
            model (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent (bool): Whether to suppress print statements.

        Returns:
            Optional[str]: The generated response, or None if an error occurs.
        """
        try:
            client = Groq(api_key=os.getenv('GROQ_API_KEY'), timeout=3.0, max_retries=2)
            if not silent:
                print(f"Groq-Api: <{colored(model, 'green')}> is generating response...")

            chat_completion = client.chat.completions.create(
                messages=chat.to_groq_format(), model=model, temperature=temperature, stream=True, stop="</s>", max_tokens=8000
            )
            

            full_response = ""
            token_keeper = CustomColoring()
            for chunk in chat_completion:
                token = chunk.choices[0].delta.content
                if token:
                    if not silent:
                        print(token_keeper.apply_color(token), end="")
                    full_response += token
            if not silent:
                print()

            return full_response

        except Exception as e:
            raise Exception(f"Groq-Api error: {e}")

    # @staticmethod
    # def count_tokens(text: str, model: str) -> int:
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

