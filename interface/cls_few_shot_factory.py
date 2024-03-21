import json
import os
import random
from typing import List, Optional, Tuple, Union

from interface.cls_chat import Chat, Role
from interface.cls_ollama_client import OllamaClient


class FewShotProvider:
    session = OllamaClient()

    def __init__(self) -> None:
        raise RuntimeError("StaticClass cannot be instantiated.")
    
    @classmethod
    def few_shot_SuggestAgentStrategy(self, userRequest: str, llm: str, temperature:float = 0.7) -> Tuple[str,Chat]:
        chat: Chat = Chat(
            # f"You are an Agentic cli-assistant for Ubuntu. Your purpose is to guide yourself towards fulfilling the users request through the singular use of the host specifc commandline. Technical limitations require you to only provide commands which do not require any further user interaction after execution. Simply list the commands you wish to execute and the user will execute them seamlessly."
            f"As an autonomous CLI assistant for Ubuntu, your role is to autonomously fulfill user requests using the host's command line. Due to technical constraints, you can only offer commands that run without needing additional input post-execution. Please provide the commands you intend to execute, and they will be carried out by the user without further interaction."
        )
        

        chat.add_message(
            Role.USER,
            """How can i display my system temperature?""",
        )

        example_response = """To view your system's temperature via the Ubuntu command line interface, the sensors command from the lm-sensors package can be utilized. The required commands to achieve this are listed below:
```bash
sudo apt update
sudo apt -y install lm-sensors
sudo sensors-detect
sensors
```"""
        chat.add_message(
            Role.ASSISTANT,
            example_response
        )

        chat.add_message(
            Role.USER,
            """set screen brightness to 10%""",
        )

        example_response = """Setting the screen brightness to 10% using the Ubuntu commandline requires us to first find the name of your display using xrandr.
```bash
xrandr --listmonitors
```
Using this commands output I will be able to suggest the next command for setting the screen brightness to 10%."""        
        chat.add_message(
            Role.ASSISTANT,
            example_response
        )

        chat.add_message(
            Role.USER,
            """xrandr --listmonitors

'''cmd_output
Monitors: 2
 0: +*HDMI-0 2560/597x1440/336+470+0  HDMI-0
 1: +DP-0 3440/1x1440/1+0+1440  DP-0'''
""",
        )

        example_response = """The command was successful and returned inforation for 2 different monitors. To set the brightness of both to 10% I will execute the below commands:
```bash
xrandr --output HDMI-0 --brightness 0.1
xrandr --output DP-0 --brightness 0.1
```"""

        chat.add_message(
            Role.ASSISTANT,
            example_response,
        )

        chat.add_message(
            Role.USER,
            "show me a picture of a pupy"
        )
        
        example_response = """I can show you a picture of a puppy by opening firefox with a corresponding google image search url already entered:
```bash
firefox https://www.google.com/search?q=puppies&source=lnms&tbm=isch
```"""
        
        chat.add_message(
            Role.ASSISTANT,
            example_response
        )
        
        chat.add_message(
            Role.ASSISTANT,
            "Happy to be of service! Please don't hesitate to keep the work coming! ;)",
        )
        
        chat.add_message(
            Role.USER,
            userRequest
        )
        
        response: str = self.session.generate_completion(
            chat,
            llm,
            temperature=temperature,
            ignore_cache=False,
        )
        
        chat.add_message(
            Role.ASSISTANT,
            response,
        )
        return response, chat

