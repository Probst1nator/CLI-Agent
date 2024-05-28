import json
import os
import random
from typing import List, Tuple, Union

from interface.cls_chat import Chat, Role
from interface.cls_ollama_client import OllamaClient
from tooling import run_command, select_and_execute_commands

def programm_of_thought(streaming_response: str) -> str:
    bash_count: int = streaming_response.count("```bash\n")
    bash_out_count: int = streaming_response.count("```bash_out\n")
    signifiers_count = streaming_response.count("```")
    if bash_count > bash_out_count and signifiers_count % 2 == 0:
        last_bash_block = streaming_response.rsplit("```bash\n", 1)[-1]
        last_bash_block = last_bash_block.split("```")[0]
        cmds = last_bash_block.strip().split("\n")
        output = ""
        for cmd in cmds:
            output += run_command(cmd, verbose=False, include_cmd=False) + "\n"
        output = output.strip()
        return f"\n```bash_out\n{output}\n```\n"
    return ""

class FewShotProvider:
    session = OllamaClient()

    def __init__(self) -> None:
        raise RuntimeError("StaticClass cannot be instantiated.")
    
    @classmethod
    def few_shot_TextToTerm(cls, prompt: str, **kwargs) -> str:
        chat = Chat("You are a text to keyword converting engine. Summarize the user given text into a term fit for google search.")
        chat.add_message(Role.USER, "I would like to know more about search engine optimization.")
        chat.add_message(Role.ASSISTANT, "search engine optimization")
        chat.add_message(Role.USER, "What is today's weather like in Nuremberg?")
        chat.add_message(Role.ASSISTANT, "Nürnberg Wetter")
        chat.add_message(Role.USER, "Tell me the latest developments in artificial intelligence.")
        chat.add_message(Role.ASSISTANT, "latest AI developments")
        chat.add_message(Role.USER, prompt)
        response = cls.session.generate_completion(
            chat,
            "mixtral",
            temperature=0.6,
            **kwargs
        )
        return response
    
    @classmethod
    def few_shot_CmdAgent(cls, userRequest: str, llm: str, temperature: float = 0.7, **kwargs) -> Tuple[str, Chat]:
        chat = Chat(
            "Designed for autonomy, this Ubuntu command line interface (CLI) assistant intelligently addresses user queries by crafting optimized, non-interactive shell commands that execute independently. It progresses systematically, preemptively gathering vital CLI information to ensure the creation of perfectly structured and easily executable instructions."
        )

        chat.add_message(Role.USER, "How can I display my system temperature?")
        example_response = """To view your system's temperature via the Ubuntu command line interface, the sensors command from the lm-sensors package can be utilized. The required commands to achieve this are listed below:
```bash
sudo apt update
sudo apt -y install lm-sensors
sudo sensors-detect
sensors
```"""
        chat.add_message(Role.ASSISTANT, example_response)

        chat.add_message(Role.USER, "set screen brightness to 10%")
        example_response = """Setting the screen brightness to 10% using the Ubuntu command line requires us to first find the name of your display using xrandr.
```bash
xrandr --listmonitors

Using this command's output, I will be able to suggest the next command for setting the screen brightness to 10%."""
        chat.add_message(Role.ASSISTANT, example_response)

        chat.add_message(Role.USER, """xrandr --listmonitors

'''cmd_output
Monitors: 2
0: +*HDMI-0 2560/597x1440/336+470+0 HDMI-0
1: +DP-0 3440/1x1440/1+0+1440 DP-0'''
""")
        chat.add_message(Role.ASSISTANT, """The command was successful and returned information for 2 different monitors. Now we can set them to 10% by executing these commands:

```bash
xrandr --output HDMI-0 --brightness 0.1
xrandr --output DP-0 --brightness 0.1
```""")

        chat.add_message(Role.USER, "show me a puppy")
        chat.add_message(Role.ASSISTANT, """I can show you a picture of a puppy by opening firefox with a corresponding google image search url already entered:
```bash
firefox https://www.google.com/search?q=puppies&source=lnms&tbm=isch
```""")
        
        chat.add_message(Role.USER, "Thanks")
        chat.add_message(Role.ASSISTANT, "Happy to be of service. Is there anything else I may help you with?")
        
        chat.add_message(Role.USER, "Setup a config file for running syncthing automatically in the background")
        chat.add_message(Role.ASSISTANT, '''To setup syncthing as you specified, I will create a configuration file for autostarting Syncthing. Here is the command:
```bash
echo -e "[Desktop Entry]
Type=Application
Exec=syncthing
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true" > ~/.config/autostart/syncthing.desktop
```''')

        chat.add_message(Role.USER, '''```bash_out
echo -e "[Desktop Entry]
Type=Application
Exec=syncthing
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true" > ~/.config/autostart/syncthing.desktop # Executed Successfully```''' + "\nThanks! What is the current working directory?")

        chat.add_message(Role.ASSISTANT, """To find the working directory we'll execute this command:
```bash
pwd
```""")

        chat.add_message(Role.USER, select_and_execute_commands(["pwd"], True, False) + "\n\nThanks, can you also show the files in this directory?")
        chat.add_message(Role.ASSISTANT, """Of course! To list the files in the working directory, let's run the following command:
```bash
ls
```""")
        
        chat.add_message(Role.USER, select_and_execute_commands(["ls"], True, False) + "\n\nThank you! Please now calculate the sum of 5 and 10")
        chat.add_message(Role.ASSISTANT, '''Sure! To perform a calculation, I will follow these steps:
1. I will implement the calculation in a python script. 
2. Then I will run the python script and view its output.
```bash
echo -e "# Define a function to calculate the sum of two numbers
def calculate_sum(a, b):
    return a + b

# Calculate the sum of 5 and 10
result = calculate_sum(5, 10)
print(result)" > ./sum_calc.py
```
```bash
python3 sum_calc.py
```''')
        chat.add_message(Role.USER, """```bash_out
15
```""")

        chat.add_message(Role.ASSISTANT, "The command ran successfully, anything else you need help with?")
        chat.add_message(Role.USER, userRequest)
        response = cls.session.generate_completion(
            chat,
            llm,
            temperature=temperature,
            **kwargs
        )
        
        chat.add_message(Role.ASSISTANT, response)
        return response, chat

    @classmethod
    def deep_thinking_module(cls, chat: Chat, llm: str) -> str:
        ideas = []
        for _ in range(3):
            ideas.append(cls.session.generate_completion(chat, llm, temperature=0.6, verbose=False, ignore_cache=True))
        
        master_chat = Chat("The assistant considers the different thoughts to implement a strategic plan to approach best solutions.")
        master_chat.add_message(Role.USER, "What are the different ideas you have?")
        master_chat.add_message(Role.ASSISTANT, "\n".join([f"```idea {i+1}\n{thought}\n```" for i, thought in enumerate(ideas)]))
        master_chat.add_message(Role.USER, "Please come up with an optimal plan to solve the problem.")
        master_thought = cls.session.generate_completion(
            master_chat, 
            llm, 
            temperature=0.6, 
            start_response_with="Let's think about this, step by step.\n", 
            include_start_response_str=False, 
            token_stream_func=programm_of_thought
        )
        return master_thought
    
    @classmethod
    def plan_and_execute(cls, userRequest: str, llm: str, temperature: float = 0.7, **kwargs) -> List[str]:
        chat = Chat(
            "You are an AI assistant that can plan and execute a series of steps to fulfill user requests effectively. Break down the task into manageable steps and ensure each step is clear and executable."
        )

        chat.add_message(Role.USER, "How can I set up a new user on Ubuntu and give them sudo privileges?")
        example_response = """To set up a new user on Ubuntu and give them sudo privileges, follow these steps:
1. Create a new user
2. Add the user to the sudo group
```bash
sudo adduser newuser
sudo usermod -aG sudo newuser
```"""
        chat.add_message(Role.ASSISTANT, example_response)
        chat.add_message(Role.USER, userRequest)
        
        response = cls.session.generate_completion(
            chat,
            llm,
            temperature=temperature,
            **kwargs
        )

        chat.add_message(Role.ASSISTANT, response)

        commands = cls.extract_commands(response)
        for command in commands:
            run_command(command)
        
        return commands
    
    @staticmethod
    def extract_commands(plan: str) -> List[str]:
        commands = []
        lines = plan.split('\n')
        for line in lines:
            if line.startswith('```bash'):
                command = line.strip('```bash').strip()
                commands.append(command)
        return commands