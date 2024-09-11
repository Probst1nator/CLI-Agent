import tempfile
from typing import List, Optional
from openai import OpenAI
from termcolor import colored
from py_classes.cls_chat import Chat
from py_classes.cls_custom_coloring import CustomColoring
import speech_recognition as sr
import os
from py_classes.globals import g

from py_classes.cls_ai_provider_interface import ChatClientInterface

class OpenAIAPI(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the OpenAI API.
    """

    @staticmethod
    def generate_response(chat: Chat, model: str = "gpt-4o", temperature: float = 0.7, silent: bool = False, base64_images: List[str] = []) -> Optional[str]:
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
            
            if silent:
                print(f"OpenAI-Api: <{colored(model, 'green')}> is {colored('silently', 'green')} generating response...")
            else:
                print(f"OpenAI-Api: <{colored(model, 'green')}> is generating response...")

            stream = client.chat.completions.create(
                model=model,
                messages=chat.to_openai(),
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

    @staticmethod
    def transcribe_audio(audio_data: sr.AudioData, language: str = "", model: str = "whisper-1") -> tuple[str,str]:
        """
        Transcribes an audio file using the OpenAI Whisper API.

        Args:
            audio_data (sr.AudioData): The audio data object.
            model (str): The model identifier for Whisper.

        Returns:
            tuple[str,str]: (transcribed text from the audio file, language)
        """
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=g.PROJ_VSCODE_DIR_PATH) as temp_audio_file:
                temp_audio_file.write(audio_data.get_wav_data())
                temp_audio_file_path = temp_audio_file.name

            with open(temp_audio_file_path, 'rb') as audio_file:
                response = client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    response_format="verbose_json",
                    language=language
                )
            
            no_speech_prob = response.segments[0]['no_speech_prob']
            if (no_speech_prob > 0.7):
                return "", "english"
            
            language = response.language
            
            return response.text, language

        except Exception as e:
            raise Exception(f"OpenAI Whisper API error: {e}")

        finally:
            if os.path.exists(temp_audio_file_path):
                os.remove(temp_audio_file_path)