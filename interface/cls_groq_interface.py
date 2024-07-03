import os
from dotenv import load_dotenv
from groq import Groq
from termcolor import colored
from interface.cls_chat import Chat
from cls_custom_coloring import CustomColoring

# Load the environment variables from .env file
load_dotenv()

class GroqChat:
    @staticmethod
    def generate_response(chat: Chat, model: str, temperature: float = 0.7, silent: bool = False) -> str:
        """
        Generates a response using the Groq API based on the provided model and messages.

        :param chat: The chat object containing messages.
        :param model: The model string to use for generating the response.
        :param temperature: The temperature setting for the model.
        :param silent: Whether to suppress printing output.
        :return: A string containing the generated response.
        """
        try:
            # Initialize the Groq client with an API key
            client = Groq(api_key=os.getenv('GROQ_API_KEY'), timeout=3.0, max_retries=2)
            if not silent:
                print(f"Groq-Api: <{colored(model, 'green')}> is generating response...")

            # Create a chat completion with the provided model and messages
            chat_completion = client.chat.completions.create(
                messages=chat.to_groq_format(), model=model, temperature=temperature, stream=True, stop="</s>"
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
            print(f"Groq-Api error: {e}")
            return None
