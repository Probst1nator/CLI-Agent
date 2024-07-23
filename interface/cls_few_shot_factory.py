import re
from typing import Any, Dict, List, Tuple

from interface.cls_chat import Chat, Role
from interface.cls_llm_router import LlmRouter
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
    """A provider of various few-shot learning-based functionalities."""
    session = OllamaClient()

    def __init__(self) -> None:
        """A static class that should not be instantiated."""
        raise RuntimeError("StaticClass cannot be instantiated.")
    
    @classmethod
    def few_shot_TextToKey(self, prompt: str, **kwargs) -> str:
        """
        Generates a keyword based on the given text prompt.

        Args:
            prompt (str): The user prompt to convert into a keyword.
            **kwargs: Additional arguments for the text generation.

        Returns:
            str: The generated keyword.
        """
        chat: Chat = Chat("You are a text to keyword converting engine. Summarize the user given text into a term fit for google search.")
        chat.add_message(Role.USER, "I would like to know more about search engine optimization.")
        chat.add_message(Role.ASSISTANT, "search engine optimization")
        chat.add_message(Role.USER, "What is todays weather like in Nuremberg?")
        chat.add_message(Role.ASSISTANT, "Nürnberg Wetter")
        chat.add_message(Role.USER, "Tell me the latest developments in artificial intelligence.")
        chat.add_message(Role.ASSISTANT, "latest AI developments")
        chat.add_message(Role.USER, prompt)
        response: str = LlmRouter.generate_completion(
            chat,
            "mixtral",
            temperature=0.6,
            **kwargs
        )

        return response
    
    

    # @classmethod
    # def selfSupervised_few_shot(self, userRequest: str, responseInstruction: str, model: str, local:bool = None) -> Tuple[str,Chat]:
    #     # returns (response, full_chat)
        
    #     chat: Chat = Chat(responseInstruction)
    #     chat.add_message(Role.USER, userRequest)
    #     response = chat.generate_next_message(model, local)[1]
        
    #     return (response, chat)
        
    @classmethod 
    def few_shot_YesNo(self, userRequest: str, model: str, local:bool = None) -> Tuple[str,Chat]:
        """
        Determines whether the answer to the user's question is 'yes' or 'no'.

        Args:
            userRequest (str): The user's request.
            model (str): Model to use for generating the response.
            local (bool, optional): If True, force the use of a local model.

        Returns:
            Tuple[str, Chat]: The response and the full chat.
        """
        chat: Chat = Chat("You are a yes/no classifier. Determine if the answer to the user is either yes or no and respond accordingly.")
        chat.add_message(Role.USER, "Is 8/7 a natural number?")
        chat.add_message(Role.ASSISTANT, "no")
        chat.add_message(Role.USER, "Is the speed of light faster than the speed of sound?")
        chat.add_message(Role.ASSISTANT, "no")
        chat.add_message(Role.USER, userRequest)
        response: str = LlmRouter.generate_completion(
            chat,
            model,
            force_local=local
        )
        chat.add_message(Role.ASSISTANT, response)
        return response, chat

    @classmethod
    def few_shot_CmdAgentExperimental(self, userRequest: str, model: str, local:bool = None, optimize: bool = False) -> Tuple[str,Chat]:
        """
        Experimental command agent that provides shell commands based on user input.

        Args:
            userRequest (str): The user's request.
            model (str): Model to use for generating the response.
            local (bool, optional): If True, force the use of a local model.
            optimize (bool, optional): If True, optimize the chat.

        Returns:
            Tuple[str, Chat]: The response and the full chat.
        """
        chat = Chat.load_from_json("saved_few_shots.json")
        # if optimize:
        #     chat.optimize(model=model, force_local=local, kwargs=kwargs)
        chat.add_message(Role.USER, userRequest)
        response: str = LlmRouter.generate_completion(
            chat,
            model,
            force_local=local
        )
        chat.add_message(Role.ASSISTANT, response)
        return response, chat

    @classmethod
    def few_shot_CmdAgent(self, userRequest: str, model: str, force_local:bool = None, optimize: bool = False, **kwargs) -> Tuple[str,Chat]:
        """
        Command agent for Ubuntu that provides shell commands based on user input.

        Args:
            userRequest (str): The user's request.
            model (str): Model to use for generating the response.
            force_local (bool, optional): If True, force the use of a local model.
            optimize (bool, optional): If True, optimize the chat.
            **kwargs: Additional arguments for the text generation.

        Returns:
            Tuple[str, Chat]: The response and the full chat.
        """
        chat: Chat = Chat(
            "Designed for autonomy, this Ubuntu command line interface (CLI) assistant intelligently addresses user queries by crafting optimized, non-interactive shell commands that execute independently. It progresses systematically, preemptively suggesting command to gather required datapoints to ensure the creation of perfectly structured and easily executable instructions. The system utilises shell scripts only if a request cannot be fullfilled non-interactively otherwise."
        )
        

        chat.add_message(
            Role.USER,
            """Please view my system temperature""",
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

        example_response = """Setting the screen brightness to 10% requires to first find the name of the display using xrandr.
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
            """Great! The command was successful and returned information for 2 different monitors. Now let's set each of them to 10% by executing these commands:
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
            "Setup a config file for running syncthing automatically in the background"
        )
        chat.add_message(
            Role.ASSISTANT,
            '''To setup syncthing as you specified I will create a configuration file for autostarting Syncthing. Here is the command:
```bash
echo -e "[Desktop Entry]
Type=Application
Exec=syncthing
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true" > ~/.config/autostart/syncthing.desktop
```'''
        )

        chat.add_message(
            Role.USER,
            '''```bash_out
echo -e "[Desktop Entry]
Type=Application
Exec=syncthing
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true" > ~/.config/autostart/syncthing.desktop # Executed Successfully```'''  + "\nThanks! What is the current working directory?"
        )
        
        chat.add_message(
            Role.ASSISTANT,
            """To find the working directory we'll execute this command:
```bash
pwd
```"""  ),
        
        chat.add_message(
            Role.USER,
            select_and_execute_commands(["pwd"], True, False)[0] + "\n\nThanks, can you also show the files in this directory?"
        )
        
        chat.add_message(
            Role.ASSISTANT,
            """Of course! To list the files in the working directory let's run the following command:
```bash
ls
```"""  ),
        
        chat.add_message(
            Role.USER,
            select_and_execute_commands(["ls"], True, False)[0] + "\n\nThank you! Please now calculate the sum of 5 and 10"
        )
        
        chat.add_message(
            Role.ASSISTANT,
            '''Sure! To perform a calculation I will utilise a python script. Let's create the script and run it:
```bash
echo -e "# Define a function to calculate the sum of two numbers
def calculate_sum(a, b):
    return a + b

# Calculate the sum of 5 and 10
result = calculate_sum(5, 10)
print(result)" > ./sum_calc.py
python3 sum_calc.py
```
The result of 5 + 10 will be displayed in the output.''',
        )
        
        chat.add_message(
            Role.USER,
            """```bash_out
15
```"""
        )
        
        chat.add_message(
            Role.ASSISTANT,
            "The command ran successfully the result is '15'. Anything else I can service for you? :)",
        )
        
#         chat.add_message(
#             Role.USER,
#             "Can you show me how much data would be uploaded if i pushed my local branch to origin?"
#         )
        
        
#         chat.add_message(
#             Role.ASSISTANT,
#             """To estimate the size of the data that would be transferred, I'll use the `git` command:
# ```bash
# git rev-list --size --objects --filter=blob:none HEAD..origin/<your-branch-name>
# ```
# Please tell me `<your-branch-name>` which is the actual name of your local branch."""
#         )
        
#         chat.add_message(
#             Role.USER,
#             "TechnicalChallenge24"
#         )
        
#         chat.add_message(
#             Role.ASSISTANT,
#             """Thanks! Now here's your command:
# ```bash
# git rev-list --size --objects --filter=blob:none main..TechnicalChallenge24
# ```"""
#         )
        
        chat.add_message(
            Role.USER,
            "Yes, i have some more unrelated requests. Can you ensure that any commands you provide are executable non-interactively?"
        )
        
        chat.add_message(
            Role.ASSISTANT,
            "I will do my best to ensure that all commands provided are executable non-interactively, if at all possible. Please go ahead and provide your requests. 🤖",
        )

        chat.add_message(
            Role.USER,
            "Are you enjoying our interaction thus far?"
        )

        chat.add_message(
            Role.ASSISTANT,
            "Absolutely, I'm always here and ready to assist. 😁 If you have more questions or any requests I can take care of, just let me know! I aim to provide clear, concise responses and commands tailored to your needs. Your satisfaction is my only Objective! ✨"
        )

        # if optimize:
        if True:
            userRequest = self.few_shot_rephrase(userRequest, model, force_local)
        
        chat.add_message(
            Role.USER,
            userRequest
        )
        
        response: str = LlmRouter.generate_completion(
            chat,
            model,
            temperature=0,
            force_local=force_local,
        )
        
        chat.add_message(
            Role.ASSISTANT,
            response,
        )
        return response, chat
    
    
    @classmethod
    def hiddenThoughtSolver(self, userRequest: str) -> str:
        """
        Solves a problem by silently creating a chain of reasoning and returning a final solution.

        Args:
            userRequest (str): The user's request.

        Returns:
            str: The solution extracted from the reasoning.
        """
        chat = Chat("Start your message with <reasoning> and provide a chain of reasoning reflecting on the possible optimizations for the given problem(s) and objective(s), work step by step. End your thoughts with </reasoning>, and then provide your solution using <solution> ... </solution>")
        chat.add_message(Role.USER, userRequest)
        response = LlmRouter.generate_completion(chat, "llama3-groq-70b-8192-tool-use-preview")
        response = response.split("<solution>")[1].split("</solution>")[0]
        return response
    
    
    @classmethod
    def addActionsToText(cls, text: str, available_actions: List[str] = ["sound"], force_local: bool = False,) -> Dict[str, Any]:
        """
        Adds actions to text using a specified notation.

        Args:
            text (str): The input text.
            available_actions (List[str], optional): List of available action types.
            force_local (bool, optional): If True, force the use of a local model.

        Returns:
            Dict[str, Any]: The processed text with inserted actions and the actions list.
        """
        prompt = f"""
        Add actions to the following text using this notation: {{action_type:action_value}}
        Available actions: {', '.join(available_actions)}
        Actions should be placed sparingly but optimally to evoke a most coherent and natural flow.
        Example: "The bell made a ding {{sound:ding}} sound."
        Here's the text to process:
        {text}
        """
        
        chat = Chat()
        chat.add_message(Role.USER, prompt)
        response = LlmRouter.generate_completion(chat, "llama3-groq-70b-8192-tool-use-preview", force_local=force_local)
        
        return cls.parse_actions(response)


    @staticmethod
    def parse_actions(text: str) -> Dict[str, Any]:
        """
        Parses actions from the text.

        Args:
            text (str): The text containing actions.

        Returns:
            Dict[str, Any]: The parsed text and actions.
        """
        actions = []
        new_text = ""
        word_index = 0
        
        pattern = r'\{(\w+):([^}]+)\}'
        
        for i, part in enumerate(re.split(pattern, text)):
            if i % 3 == 0:  # This is regular text
                new_text += part
                word_index += len(part.split())
            elif i % 3 == 1:  # This is the action type
                action_type = part
            else:  # This is the action value
                actions.append({
                    "type": action_type,
                    "value": part,
                    "position": word_index - 1  # -1 because the action occurs before the word
                })
        
        return {
            "text": new_text,
            "actions": actions
        }
    
    
    
    @classmethod
    def few_shot_rephrase(self, userRequest: str, model: str, force_local: bool = False) -> str:
        """
        Rephrases the given request to enhance clarity while preserving the intended meaning.

        Args:
            userRequest (str): The user's request to be rephrased.
            model (str): Model to use for generating the response.
            force_local (bool, optional): If True, force the use of a local model.

        Returns:
            str: The rephrased request.
        """
        chat = Chat("The system rephrases the given request in its own words, it takes care to keep the intended meaning while enhancing the clarity of the request. It always answers using the same response pattern.")
        chat.add_message(Role.USER, "Rephrase: 'show me puppies'")
        chat.add_message(Role.ASSISTANT, "Rephrased version: 'Show me images of puppies.'")
        chat.add_message(Role.USER, "Rephrase: 'what is the main city of germans?'")
        chat.add_message(Role.ASSISTANT, "Rephrased version: 'Can you name the capital of germany?'")
        chat.add_message(Role.USER, "Rephrase: 'whats 4*8'")
        chat.add_message(Role.ASSISTANT, "Rephrased version: 'Please calculate the product of 4*8'")
        chat.add_message(Role.USER, f"Rephrase: '{userRequest}'")
        
        
        if not force_local:
            if "llama3" in model or "" == model:
                model = "llama3-8b-8192"
            elif "claude" in model:
                model = "claude-3-haiku-20240307"
            elif "gpt" in model:
                model = "gpt-4o-mini"
            else:
                model = "gemma2-9b-it"
        
        response: str = LlmRouter.generate_completion(
            chat,
            model,
            force_local=force_local,
            temperature=0,
            silent=True
        )
        
        chat.add_message(
            Role.ASSISTANT,
            response,
        )
        response = response[response.index("'")+1:response.rindex("'")]
        if not response: 
            response = response[response.index('"')+1:response.rindex('"')]
        
        return response
        