from abc import ABC, abstractmethod
from typing import Optional
from interface.cls_chat import Chat

class ChatClientInterface(ABC):
    @abstractmethod
    def generate_response(self, chat: Chat, model: str, temperature: float, silent: bool) -> Optional[str]:
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

    # @abstractmethod
    # def count_tokens(self, text: str, model: str) -> int:
    #     """
    #     Counts the number of tokens in the given text for the specified model.

    #     Args:
    #         text (str): The input text.
    #         model (str): The model identifier.

    #     Returns:
    #         int: The number of tokens in the input text.
    #     """
    #     pass
