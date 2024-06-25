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
    def generate_response(chat: Chat, model: str = "llama3-70b-8192", temperature: float = 0.7, silent: bool = False) -> str:
        """
        Generates a response using the Groq API based on the provided model and messages, with error handling and retries.

        :param model: The model string to use for generating the response.
        :param messages: A list of message dictionaries with 'role' and 'content' keys.
        :return: A string containing the generated response.
        """
        
        if ("mixtral" in model):
            model = "mixtral-8x7b-32768"
        elif("70b"in model):
            model = "llama3-70b-8192"
        elif("llama3" in model):
            model = "llama3-8b-8192"
        elif("gemma" in model):
            model = "gemma-7b-it"
        
        try:
            # Initialize the Groq client with an API key
            client = Groq(api_key=os.getenv('GROQ_API_KEY'), timeout=3.0, max_retries=2)
            if not silent:
                print("Groq-Api: <" + colored(model,"green") + "> is generating response...")
            # Create a chat completion with the provided model and messages
            chat_completion = client.chat.completions.create(messages=chat.to_groq_format(), model=model, temperature=temperature, stream=True, stop="</s>")
            
            # Create a generator for the stream
            # generator = client.chat.completions.with_streaming_response.create(stream=True, messages=chat.to_groq_format(), model=model, temperature=temperature)

            # Iterate through the stream and print each token
            full_response = ""
            token_keeper = CustomColoring()
            for chunk in chat_completion:
                token = chunk.choices[0].delta.content
                if (token):
                    if not silent:
                        print(token_keeper.apply_color(token), end="")
                    full_response += token
            if not silent:
                print()
                
            # If successful, extract and return the content of the first choice's message
            return full_response
            # return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq-Api error: {e}")
            if ("70b" in model):
                print(colored("Retrying with mixtral-8x7b-32768 model...", "yellow"))
                return GroqChat.generate_response(chat, model="mixtral-8x7b-32768", temperature=temperature)
            elif("llama3" in model and "70b" not in model):
                print(colored("Retrying with llama3-8b-8192 model...", "yellow"))
                return GroqChat.generate_response(chat, model="llama3-8b-8192", temperature=temperature)
            elif("gemma" in model):
                print(colored("Retrying with gemma-7b-it model...", "yellow"))
                return GroqChat.generate_response(chat, model="gemma-7b-it", temperature=temperature)
            return None