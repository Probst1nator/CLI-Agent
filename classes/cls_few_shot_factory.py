import re
from typing import Any, Dict, List, Tuple

from termcolor import colored

from classes.cls_chat import Chat, Role
from classes.cls_llm_router import AIStrengths, LlmRouter
from classes.ai_providers.cls_ollama_interface import OllamaClient
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
    def few_shot_TextToQuery(self, text: str) -> str:
        """
        Generates a search query based on the given text.
        
        Args:
        text (str): The user text to convert into a search query.
        
        Returns:
        str: The generated search query.
        """
        chat: Chat = Chat("You are a search query generator. Analyze the user's input, whether it's a question, statement, or description, and provide a concise, effective search query suitable for a search engine like Google. Focus on capturing the main topic or intent, using keywords and phrases that would yield relevant results. Avoid unnecessary words and aim for clarity and specificity.")
        
        chat.add_message(Role.USER, "I've been experiencing frequent headaches and dizziness lately, especially when I stand up quickly. It's been going on for about two weeks now. Should I be concerned? What might be causing this?")
        chat.add_message(Role.ASSISTANT, "frequent headaches dizziness standing up causes")
        
        chat.add_message(Role.USER, "The rise of remote work has dramatically changed the landscape of modern offices. Many companies are now adopting hybrid models, allowing employees to split their time between home and office. This shift is impacting everything from real estate decisions to team collaboration strategies.")
        chat.add_message(Role.ASSISTANT, "impact of remote work on office space and collaboration")
        
        chat.add_message(Role.USER, "When will the suntime in bavaria be on 12 hours long?")
        chat.add_message(Role.ASSISTANT, "equinox date in Bavaria daylight hours 12 hours")
        
        chat.add_message(Role.USER, "Wann wird die Sonnenscheindauer in Bayern 12 Stunden lang sein?")
        chat.add_message(Role.ASSISTANT, "Tagundnachtgleiche in Bayern Sonnenscheindauer 12 Stunden")
        
        chat.add_message(Role.USER, text)
        response: str = LlmRouter.generate_completion(chat, strength=AIStrengths.FAST)
        return response
    
    
    # @classmethod
    # def selfSupervised_few_shot(self, userRequest: str, responseInstruction: str, model: str, local:bool = None) -> Tuple[str,Chat]:
    #     # returns (response, full_chat)
        
    #     chat: Chat = Chat(responseInstruction)
    #     chat.add_message(Role.USER, userRequest)
    #     response = chat.generate_next_message(model, local)[1]
        
    #     return (response, chat)
        
    @classmethod 
    def few_shot_YesNo(self, userRequest: str, local:bool = None) -> Tuple[str,Chat]:
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
            strength=AIStrengths.FAST,
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
            [model],
            force_local=local
        )
        chat.add_message(Role.ASSISTANT, response)
        return response, chat

    @classmethod
    def few_shot_CmdAgent(self, userRequest: str, preferred_model_keys: List[str], force_local:bool = None, silent: bool = False) -> Tuple[str,Chat]:
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
            "Designed for autonomy, this Ubuntu CLI-Assistant autonomously addresses user queries by crafting optimized, non-interactive shell commands that execute independently. It progresses systematically, preemptively suggesting command to gather required datapoints to ensure the creation of perfectly structured and easily executable instructions. The system utilises shell scripts only if a request cannot be fullfilled non-interactively otherwise."
        )
        
        chat.add_message(
            Role.USER,
            """set screen brightness to 10%""",
        )

        chat.add_message(
            Role.ASSISTANT,
            """Setting the screen brightness to 10% requires to first find the name of the display using xrandr.
```bash
xrandr --listmonitors
```
Using the terminals response I will be able to suggest the next command for setting the screen brightness to 10%."""
        )

        chat.add_message(
            Role.USER,
            """xrandr --listmonitors

'''cmd_response
Monitors: 2
 0: +*HDMI-0 2560/597x1440/336+470+0  HDMI-0
 1: +DP-0 3440/1x1440/1+0+1440  DP-0'''
""",
        )

        chat.add_message(
            Role.ASSISTANT,
            """The command was successful and returned information for 2 different monitors. Let's set both of them to 10% with these commands:
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
            """I can show you a picture of a puppy by opening firefox with a corresponding image search url already entered:
```bash
firefox https://www.google.com/search?q=puppies&source=lnms&tbm=isch
```"""
        )
        
        chat.add_message(
            Role.USER,
            "Thanks. Whats our current working directory?"
        )

        
        chat.add_message(
            Role.ASSISTANT,
            """Let's execute this command to find the working directory:
```bash
pwd
```"""  ),
        

        chat.add_message(
            Role.USER,
            select_and_execute_commands(["pwd"], True, False)[0] + "\n\nGreat thanks!"
        )
        
        if (len(select_and_execute_commands(["tree -d -L 3 ."], True, False)[0])/4<2000): # extensive directory overview, the magic number serves as cutoff for too big directories
            chat.add_message(
                Role.USER,
                "Great! Can you show me a tree for all files and folders in current directory up to 3 levels deep?"
            )
            chat.add_message(
                Role.ASSISTANT,
                """To create a tree view of the current directory up to 3 levels deep, we can use the `tree` command:
```bash
tree -d -L 3 .
```"""
            )
            chat.add_message(
                Role.USER,
                select_and_execute_commands(["tree -d -L 3 ."], True, False)[0]
            )
            chat.add_message(
                Role.ASSISTANT,
                "Great! The tree view was successfully generated. Is there anything else I can help you with? 🌳"
            )
        else: # more minimal directory overview
            chat.add_message(
                Role.USER,
                "Great! Now list the 20 most recently modified files and folders in the current directory, include hidden files and directories"
            )
            chat.add_message(
                Role.ASSISTANT,
                """Sure. we can achieve this by using the `ls` command with the `-1hFt` options:
```bash
ls -1hFt | head -n 20
```"""
            )
        
            chat.add_message(
                Role.USER,
                select_and_execute_commands(["ls -1hFt | head -n 20"], True, False)[0]
            )
            chat.add_message(
                Role.ASSISTANT,
                "The 20 most recently modified files and folders were successfully listed. Is there anything else I can help you with? 📂"
            )
        
        
        chat.add_message(
            Role.USER,
            "please find the running cli-agent process"
        )
        chat.add_message(
            Role.ASSISTANT,
        """To identify and locate the running cli-agent process, we can use the `pgrep` command. I'll ignore casing for simplicity:
```bash
pgrep -aif "cli-agent"
```
This command will search for any running processes that match the pattern "cli-agent" and display their process IDs and corresponding command lines.""")
        
        chat.add_message(
            Role.USER,
            select_and_execute_commands(['pgrep -aif "cli-agent"'], True, False)[0] + "\n\nlook its you!"
        )
        
        chat.add_message(
            Role.ASSISTANT,
            "That's meta! " + LlmRouter.generate_completion(chat, preferred_model_keys, silent=True) + " 🤖"
        )

        if len(userRequest)<400 and not "if (" in userRequest and not "{" in userRequest: # ensure userRequest contains no code snippet
            userRequest = self.few_shot_rephrase(userRequest, preferred_model_keys, force_local, silent=True)
                
        chat.add_message(
            Role.USER,
            userRequest
        )

        response: str = LlmRouter.generate_completion(
            chat,
            preferred_model_keys,
            force_local=force_local,
            silent=silent
        )
        
        applied_hardcoded_fixes = False
        if "```" in response and not "```bash" in response:
            applied_hardcoded_fixes = True
            parts = response.split("```")
            for i in range(len(parts)):
                if i % 2 == 1:  # Odd-indexed parts
                    parts[i] = "bash\n" + parts[i]
            response = "```".join(parts)
        if ("apt" in response and "install" in response and not "-y" in response):
            applied_hardcoded_fixes = True
            response = response.replace("apt install", "apt install -y")
            response = response.replace("apt-get install", "apt-get install -y")
        if applied_hardcoded_fixes:
            print(colored("DEBUG: Applied hardcoded fixes to the response.", "yellow"))
        
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
        response = LlmRouter.generate_completion(chat, ["llama3-groq-70b-8192-tool-use-preview"])
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
        response = LlmRouter.generate_completion(chat, ["llama3-groq-70b-8192-tool-use-preview"], force_local=force_local)
        
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
    def few_shot_rephrase(self, userRequest: str, preferred_model_keys: List[str] = [], force_local: bool = False, silent: bool = True) -> str:
        """
        Rephrases the given request to enhance clarity while preserving the intended meaning.

        Args:
            userRequest (str): The user's request to be rephrased.
            model (str): Model to use for generating the response.
            force_local (bool, optional): If True, force the use of a local model.

        Returns:
            str: The rephrased request.
        """
        try:
            chat = Chat("The system rephrases the given request in its own words, it takes care to keep the intended meaning while enhancing the clarity of the request. It always answers using the same response pattern.")
            chat.add_message(Role.USER, "Rephrase: 'show me puppies'")
            chat.add_message(Role.ASSISTANT, "Rephrased version: 'Show me images of puppies'")
            chat.add_message(Role.USER, "Rephrase: 'what is the main city of germans?'")
            chat.add_message(Role.ASSISTANT, "Rephrased version: 'Can you name the capital of germany?'")
            chat.add_message(Role.USER, "Rephrase: 'whats 4*8'")
            chat.add_message(Role.ASSISTANT, "Rephrased version: 'Please calculate the product of 4*8'")
            chat.add_message(Role.USER, "Rephrase: 'hi'")
            chat.add_message(Role.ASSISTANT, "Rephrased version: 'Hi there!'")
            chat.add_message(Role.USER, f"Rephrase: '{userRequest}'")
            
            preferred_rephrase_model_keys = []
            for preferred_model_key in preferred_model_keys:
                if not force_local:
                    if "llama" in preferred_model_key or "" == preferred_model_key:
                        preferred_rephrase_model_keys.append("llama3-8b-8192")
                    elif "claude" in preferred_model_key:
                        preferred_rephrase_model_keys.append("claude-3-haiku-20240307")
                    elif "gpt" in preferred_model_key:
                        preferred_rephrase_model_keys.append("gpt-4o-mini")
                    else:
                        preferred_rephrase_model_keys.append("gemma2-9b-it")
            
            response: str = LlmRouter.generate_completion(
                chat,
                preferred_model_keys,
                force_local=force_local,
                temperature=0,
                silent=silent
            )
            
            chat.add_message(
                Role.ASSISTANT,
                response,
            )
            response = response[response.index("'")+1:response.rindex("'")]
            if not response: 
                response = response[response.index('"')+1:response.rindex('"')]
            
            return response
        except Exception as e:
            if not silent:
                print(colored(f"DEBUG: few_shot_rephrase failed with response: {response}", "yellow"))
            return userRequest