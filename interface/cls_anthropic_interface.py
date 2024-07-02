import os
from typing import Optional
from anthropic import Anthropic
from dotenv import load_dotenv
from cls_custom_coloring import CustomColoring
from interface.cls_chat import Chat, Role
from termcolor import colored
import traceback

load_dotenv()

class AnthropicChat:
    @staticmethod
    def generate_response(chat: Chat, model: str = "claude-3-5-sonnet-20240620", temperature: float = 0.7, silent: bool = False) -> Optional[str]:
        """
        Generates a response using the Anthropic API based on the provided model and messages, with error handling and retries.

        :param chat: An instance of a chat class containing the messages.
        :param model: The model string to use for generating the response.
        :param temperature: Controls the randomness of the response.
        :param silent: If True, suppresses print statements.
        :return: A string containing the generated response or None if an error occurs.
        """
        try:
            client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'), timeout=3.0, max_retries=2)
            
            if "claude-3-5-sonnet" in model or not model:
                model = "claude-3-5-sonnet-20240620"
            
            if not silent:
                print("Anthropic API: <" + colored(model,"green") + "> is generating response...")

            l_chat = Chat()
            l_chat.messages = chat.messages

            system_message = ""
            if l_chat.messages[0][0] == Role.SYSTEM:
                system_message = l_chat.messages[0][1]
                l_chat.messages = l_chat.messages[1:]
    
            with client.messages.stream(
                model=model,
                max_tokens=4096, # sadly this is currently the maximum allowed
                system=system_message,
                messages=l_chat.to_groq_format(),
                temperature=temperature,
            ) as stream:
                full_response = ""
                token_keeper = CustomColoring()
                for token in stream.text_stream:
                    if not silent:
                        print(token_keeper.apply_color(token), end="")
                    full_response += token
                if not silent:
                    print()
                return full_response
        except Exception as e:
            print(f"Anthropic API error: {e}")
            traceback.print_exc()
            return None
