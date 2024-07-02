import os
from openai import OpenAI
from dotenv import load_dotenv
from termcolor import colored
from interface.cls_chat import Chat
from cls_custom_coloring import CustomColoring
import tiktoken

# Load the environment variables from .env file
load_dotenv()

class OpenAIChat:
    @staticmethod
    def generate_response(chat: Chat, model: str = "gpt-4o", temperature: float = 0.7, silent: bool = False) -> str:
        """
        Generates a response using the OpenAI API based on the provided model and messages, with error handling and retries.

        :param model: The model string to use for generating the response.
        :param messages: A list of message dictionaries with 'role' and 'content' keys.
        :return: A string containing the generated response.
        """

        try:
            # Initialize the OpenAI client with an API key
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            if not silent:
                print("OpenAI API: <" + colored(model, "green") + "> is generating response...")

            # Create a chat completion with the provided model and messages
            stream = client.chat.completions.create(
                model=model,
                messages=chat.to_openai_chat(), # type: ignore
                temperature=temperature,
                stream=True
            )

            # Create a generator for the stream
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
            print(f"OpenAI API error: {e}")
            return None

    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4o") -> int:
        """
        Counts the tokens in the given text using the specified model's tokenizer.

        :param text: The text to tokenize.
        :param model: The model string to use for tokenizing the text.
        :return: An integer representing the number of tokens.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"Token counting error: {e}")
            return None
