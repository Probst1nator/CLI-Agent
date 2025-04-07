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
        debug_print = ChatClientInterface.create_debug_printer(chat)
        try:
            debug_print(f"Human-Api: User is asked for a response...", "green", force_print=True)
            debug_print(colored(("# " * 20) + "CHAT BEGIN" + (" #" * 20), "yellow"), force_print=True)
            chat.print_chat()
            debug_print(colored(("# " * 20) + "CHAT STOP" + (" #" * 21), "yellow"), force_print=True)
            
            debug_print(colored("# # # Enter your multiline response. Type '--f' on a new line when finished.", "blue"), force_print=True)
            lines = []
            while True:
                line = input()
                if line == "--f":
                    break
                lines.append(line)
            full_response = "\n".join(lines)

            token_keeper = CustomColoring()
            for character in full_response:
                debug_print(token_keeper.apply_color(character), end="", with_title=False)
            debug_print("", with_title=False)
            return full_response

        except Exception as e:
            error_msg = f"Human-API error: {e}"
            debug_print(error_msg, "red", is_error=True)
            raise Exception(error_msg)

    @staticmethod
    def transcribe_audio(audio_data: sr.AudioData, language: str = "", model: str = "whisper-1", chat: Optional[Chat] = None) -> tuple[str,str]:
        """
        Transcribes an audio file using the OpenAI Whisper API.

        Args:
            audio_data (sr.AudioData): The audio data object.
            model (str): The model identifier for Whisper.
            language (str): The language of the audio.
            chat (Optional[Chat]): The chat object for debug printing.

        Returns:
            tuple[str,str]: (transcribed text from the audio file, language)
        """
        debug_print = None
        if chat:
            debug_print = ChatClientInterface.create_debug_printer(chat)
            
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            if debug_print:
                debug_print(f"Human-Api: Using OpenAI to transcribe audio using <{colored(model, 'green')}>...", force_print=True)

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
                if debug_print:
                    debug_print("No speech detected", force_print=True)
                return "", "english"
            
            language = response.language
            
            if debug_print:
                debug_print(f"Transcription complete. Detected language: {language}", "green", force_print=True)
                
            return response.text, language

        except Exception as e:
            error_msg = f"OpenAI Whisper API error: {e}"
            if debug_print:
                debug_print(error_msg, "red", is_error=True)
            else:
                logger.error(error_msg)
            raise Exception(error_msg)

        finally:
            if 'temp_audio_file_path' in locals() and os.path.exists(temp_audio_file_path):
                os.remove(temp_audio_file_path)