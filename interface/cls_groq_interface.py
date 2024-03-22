import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq

from interface.cls_chat import Chat

# Load the environment variables from .env file
load_dotenv()

class GroqChat:
    @staticmethod
    def generate_response(chat: Chat, model: str = "mixtral-8x7b-32768", temperature: float = 0.7) -> str:
        """
        Generates a response using the Groq API based on the provided model and messages, with error handling and retries.

        :param model: The model string to use for generating the response.
        :param messages: A list of message dictionaries with 'role' and 'content' keys.
        :return: A string containing the generated response.
        """
        try:
            # Initialize the Groq client with an API key
            client = Groq(api_key=os.getenv('GROQ_API_KEY'))
            
            # Create a chat completion with the provided model and messages
            chat_completion = client.chat.completions.create(messages=chat.to_groq_format(), model=model, temperature=temperature)
            
            # If successful, extract and return the content of the first choice's message
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq Api error: {e}")
            return None