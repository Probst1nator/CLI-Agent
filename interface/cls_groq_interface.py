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
        max_retries = 8 # Will max out at waiting about 5h, will give up after a sum of about 10h
        retry_delay = 150  # Delay in seconds (2.5 minutes)
        
        for attempt in range(max_retries):
            try:
                # Initialize the Groq client with an API key
                client = Groq(api_key=os.getenv('GROQ_API_KEY'))
                
                # Create a chat completion with the provided model and messages
                chat_completion = client.chat.completions.create(messages=chat.to_groq_format(), model=model, temperature=temperature)
                
                # If successful, extract and return the content of the first choice's message
                return chat_completion.choices[0].message.content
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = retry_delay * 2
                else:
                    raise Exception("Max retries reached. Unable to generate response.") from e