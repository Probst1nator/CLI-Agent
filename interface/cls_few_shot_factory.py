import json
import os
import random
from typing import List, Optional, Tuple, Union

from interface.cls_chat import Chat, Role
from interface.cls_ollama_client import OllamaClient
from tooling import run_command, select_and_execute_commands

# class Agent:
#     tools:List[str] = []

#     def __init__():
#         pass
    
#     def chat(prompt:str) -> str:
#         few_shot_chat:Chat = Chat("You are an ai-agent. Use strategic step by step planning to advance your current state toward optimal actions.")
#         user_prompt: str = "Please pick you next action to take:"
#         few_shot_chat.add_message(Role.USER, user_prompt)
#         OllamaClient().generate_completion(few_shot_chat)
    
    
class FewShotProvider:
    session = OllamaClient()

    def __init__(self) -> None:
        raise RuntimeError("StaticClass cannot be instantiated.")
    
    @classmethod
    def few_shot_TextToTerm(self, prompt: str, **kwargs) -> str:
        chat: Chat = Chat("You are a text to keyword converting engine. Summarize the user given text into a term fit for google search.")
        chat.add_message(Role.USER, "I would like to know more about search engine optimization.")
        chat.add_message(Role.ASSISTANT, "search engine optimization")
        chat.add_message(Role.USER, "What is todays weather like in Nuremberg?")
        chat.add_message(Role.ASSISTANT, "Nürnberg Wetter")
        chat.add_message(Role.USER, "Tell me the latest developments in artificial intelligence.")
        chat.add_message(Role.ASSISTANT, "latest AI developments")
        chat.add_message(Role.USER, prompt)
        response: str = self.session.generate_completion(
            chat,
            "mixtral",
            temperature=0.6,
            **kwargs
        )
        
        return response
        
            
    @classmethod
    def few_shot_CmdAgent(self, userRequest: str, llm: str, temperature:float = 0.7, **kwargs) -> Tuple[str,Chat]:
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

        chat.add_message(
            Role.ASSISTANT,
            """The command was successful and returned information for 2 different monitors. Now we can set them to 10% by executing these commands:
```bash
xrandr --output HDMI-0 --brightness 0.1
xrandr --output DP-0 --brightness 0.1
```""",
        )

        chat.add_message(
            Role.USER,
            "show me a puppy"
        )
        
        chat.add_message(
            Role.ASSISTANT,
            """I can show you a picture of a puppy by opening firefox with a corresponding google image search url already entered:
```bash
firefox https://www.google.com/search?q=puppies&source=lnms&tbm=isch
```"""
        )
        
        chat.add_message(
            Role.USER,
            "Thanks"
        )
        
        chat.add_message(
            Role.ASSISTANT,
            "Happy to be of service. Is there anything else I may help you with?",
        )
        
        chat.add_message(
            Role.USER,
            "What is the current working directory?"
        )
        
        chat.add_message(
            Role.ASSISTANT,
            """To find the working directory execute the following:
```bash
pwd
```"""  ),
        
        chat.add_message(
            Role.USER,
            select_and_execute_commands(["pwd"], True) + "\n\nThanks, can you also show the files in this directory?"
        )
        
        chat.add_message(
            Role.ASSISTANT,
            """Of course! To list the files in the working directory simply execute this:
```bash
ls
```"""  ),
        
        chat.add_message(
            Role.USER,
            select_and_execute_commands("ls", True)
        )
        
        chat.add_message(
            Role.ASSISTANT,
            "The command ran successfully. Anything else you need help with?",
        )
        
        chat.add_message(
            Role.USER,
            userRequest
        )
        
        response: str = self.session.generate_completion(
            chat,
            llm,
            temperature=temperature,
            **kwargs
        )
        
        chat.add_message(
            Role.ASSISTANT,
            response,
        )
        return response, chat