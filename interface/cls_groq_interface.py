import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from groq import Groq
from termcolor import colored

from interface.cls_chat import Chat

# Load the environment variables from .env file
load_dotenv()

class GroqChat:

    saved_block_delimiters: str = ""
    color_red: bool = False
    
    def apply_color(self, string: str) -> str:
        if ("`" in string):
            self.saved_block_delimiters += string
            string = ""
        if (self.saved_block_delimiters.count("`")>=3):
            self.color_red = not self.color_red
            string = colored(self.saved_block_delimiters, "red")
            self.saved_block_delimiters = ""
        elif (len(self.saved_block_delimiters)>=3):
            string = colored(self.saved_block_delimiters, "light_blue")
            self.saved_block_delimiters = ""
        elif (self.color_red):
            string = colored(string, "red")
        elif (not self.color_red):
            string = colored(string, "light_blue")
        return string
    
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
            print("Groq-Api is generating response...")
            # Create a chat completion with the provided model and messages
            chat_completion = client.chat.completions.create(messages=chat.to_groq_format(), model=model, temperature=temperature, stream=True, stop="</s>")
            
            # Create a generator for the stream
            # generator = client.chat.completions.with_streaming_response.create(stream=True, messages=chat.to_groq_format(), model=model, temperature=temperature)

            # Iterate through the stream and print each token
            full_response = ""
            token_keeper = GroqChat()
            for chunk in chat_completion:
                token = chunk.choices[0].delta.content
                if (token):
                    print(token_keeper.apply_color(token), end="")
                    full_response += token
            print()
                
            # If successful, extract and return the content of the first choice's message
            return full_response
            # return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq-Api error: {e}")
            return None