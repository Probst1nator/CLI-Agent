import os
from typing import Optional, Tuple
from groq import Groq
from termcolor import colored
from py_classes.cls_custom_coloring import CustomColoring
from py_classes.cls_chat import Chat
from py_classes.cls_ai_provider_interface import ChatClientInterface

class GroqAPI(ChatClientInterface):
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
            if silent:
                print(f"Groq-Api: <{colored(model, 'green')}> is {colored('silently', 'green')} generating response...")
            else:
                print(f"Groq-Api: <{colored(model, 'green')}> is generating response...")
            chat_completion = client.chat.completions.create(
                messages=chat.to_groq(), model=model, temperature=temperature, stream=True, stop="</s>"
            )
            full_response = ""
            token_keeper = CustomColoring()
            for chunk in chat_completion:
                token = chunk.choices[0].delta.content
                if token:
                    full_response += token
                    if not silent:
                        print(token_keeper.apply_color(token), end="")
            if not silent:
                print()
            return full_response
        except Exception as e:
            raise Exception(f"Groq-Api error: {e}")


    @staticmethod
    def transcribe_audio(filepath: str, model: str = "whisper-large-v3", language: str = "auto", silent: bool = False) -> Optional[Tuple[str, str]]:
        """
        Transcribes an audio file using Groq's Whisper implementation.
        Args:
            filename (str): The path to the audio file.
            model (str): The Whisper model to use (default: "whisper-large-v3").
            language (str): The language of the audio (default: "auto" for auto-detection).
            silent (bool): Whether to suppress print statements.
        Returns:
            Optional[Tuple[str, str]]: A tuple containing the transcribed text and detected language, or None if an error occurs.
        """
        try:
            client = Groq(api_key=os.getenv('GROQ_API_KEY'), timeout=30.0, max_retries=2)
            if not silent:
                print(f"Groq-Api: Transcribing audio using <{colored(model, 'green')}>...")
            
            with open(filepath, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(filepath, file.read()),
                    model=model,
                    response_format="verbose_json",
                    language=language
                )
            
            if not silent:
                print(colored("Transcription complete.", 'green'))
            
            return transcription.text, transcription.language

        except Exception as e:
            raise Exception(f"Groq-Api audio transcription error: {e}")

    # The token counting method is commented out in your original code,
    # so I'm leaving it as is.
    
    # @staticmethod
    # def count_tokens(text: str, model: str) -> int:
    #     """
    #     Counts the number of tokens in the given text for the specified model.
    #     Args:
    #     text (str): The input text.
    #     model (str): The model identifier.
    #     Returns:
    #     int: The number of tokens in the input text.
    #     """
    #     encoding = tiktoken.encoding_for_model(model)
    #     tokens = encoding.encode(text)
    #     return len(tokens)