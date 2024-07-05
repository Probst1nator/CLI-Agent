from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
from termcolor import colored
from interface.cls_chat import Chat
from cls_custom_coloring import CustomColoring
import tiktoken
import os

from interface.cls_chat_client_interface import ChatClientInterface

load_dotenv()

class OpenAIChat(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the OpenAI API.
    """

    @staticmethod
    def generate_response(chat: Chat, model: str = "gpt-4o", temperature: float = 0.7, silent: bool = False) -> Optional[str]:
        """
        Generates a response using the OpenAI API.

        Args:
            chat (Chat): The chat object containing messages.
            model (str): The model identifier.
            temperature (float): The temperature setting for the model.
            silent (bool): Whether to suppress print statements.

        Returns:
            Optional[str]: The generated response, or None if an error occurs.
        """
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            if not silent:
                print("OpenAI API: <" + colored(model, "green") + "> is generating response...")

            stream = client.chat.completions.create(
                model=model,
                messages=chat.to_openai_chat(),
                temperature=temperature,
                stream=True
            )

            full_response = ""
            token_keeper = CustomColoring()
            for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    if not silent:
                        print(token_keeper.apply_color(token), end="")
                    full_response += token
            if not silent:
                print()
            return full_response

        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

    # @staticmethod
    # def count_tokens(text: str, model: str = "gpt-4o") -> int:
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

