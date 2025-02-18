import tempfile
from typing import Any, Dict, List, Optional
from openai import OpenAI
from termcolor import colored
from py_classes.cls_chat import Chat
from py_classes.cls_custom_coloring import CustomColoring
import speech_recognition as sr
import os
from py_classes.globals import g

from py_classes.cls_ai_provider_interface import ChatClientInterface

class HumanAPI(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the OpenAI API.
    """

    @staticmethod
    def generate_response(chat: Chat, base64_images: List[str] = [], tools: Optional[List[Dict[str,Any]]] = None, model: str = "human", temperature=0.0, silent_reason: str = False ) -> Optional[str]:
        """
        Generates a response by prompting the human in front of the terminal .

        Args:
            chat (Chat): The chat object containing messages.
            base64_images (List[str]): The images as base64 strings
            tools: (Optional[List[Dict[str,Any]]]) The tools available for use
            model (str): Unused
            temperature: (float): Unused 
            silent (bool): Unused
        Returns:
            Optional[str]: The generated response, or None if an error occurs.
        """
        try:
            print(colored(f"Human-Api: User is asked for a response...", "green"))
            print(colored(("# " * 20) + "CHAT BEGIN" + (" #" * 20), "yellow"))
            chat.print_chat()
            print(colored(("# " * 20) + "CHAT STOP" + (" #" * 21), "yellow"))
            
            print(colored("# # # Enter your multiline response. Type '--f' on a new line when finished.", "blue"))
            lines = []
            while True:
                line = input()
                if line == "--f":
                    break
                lines.append(line)
            full_response = "\n".join(lines)

            full_response = ""
            token_keeper = CustomColoring()
            for character in full_response:
                print(token_keeper.apply_color(character), end="")
            print()
            return full_response

        except Exception as e:
            raise Exception(f"Human-API error: {e}")

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

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=g.PROJ_PERSISTENT_STORAGE_PATH) as temp_audio_file:
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