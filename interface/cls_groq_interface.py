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
        
        model_mapping = {
            "70b": "llama3-70b-8192",
            "llama3": "llama3-8b-8192",
            "mixtral": "mixtral-8x7b-32768",
            "gemma": "gemma-7b-it"
        }

        # Determine the initial model based on the mapping
        for key, mapped_model in model_mapping.items():
            if key in model:
                model = mapped_model
                break

        # Create a list of models to retry, excluding the initial model
        retry_models = [m for m in model_mapping.values() if m != model]
        failed_models = set()

        while True:
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
                failed_models.add(model)

                if not retry_models:
                    return None

                # Get the next model to try that hasn't failed yet
                model = next((m for m in retry_models if m not in failed_models), None)
                if not model:
                    return None
                retry_models.remove(model)
                print(colored(f"Retrying with {model} model...", "yellow"))

        return None
