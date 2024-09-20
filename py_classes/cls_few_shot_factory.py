import json
import re
from typing import Any, Dict, List, Optional, Tuple

import ollama
import chromadb
from termcolor import colored

from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import AIStrengths, LlmRouter
from py_classes.ai_providers.cls_ollama_interface import OllamaClient
from py_classes.cls_pptx_presentation import PptxPresentation, Slide
from py_methods.cmd_execution import select_and_execute_commands
from py_classes.globals import g

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
        
        chat.add_message(Role.USER, "What are the main differences between Python and JavaScript for web development? I'm trying to decide which one to learn first.")
        chat.add_message(Role.ASSISTANT, "Python vs JavaScript web development comparison")
        
        chat.add_message(Role.USER, "döner vs currywurst was is besser")
        chat.add_message(Role.ASSISTANT, "Döner vs Currywurst Vergleich beliebtheit Deutschland") 
        
        chat.add_message(Role.USER, "derivatives vs integrals whats the diff")
        chat.add_message(Role.ASSISTANT, "derivatives vs integrals differences calculus")
        
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
    def few_shot_YesNo(self, userRequest: str | Chat, preferred_models: List[str]=[], force_local: bool = False, silent: bool = False, force_free: bool = False, force_preferred_model: bool = False) -> Tuple[bool,Chat]:
        """
        Determines whether the answer to the user's question is 'yes' or 'no'.

        Args:
            userRequest (str): The user's request.
            model (str): Model to use for generating the response.
            local (bool, optional): If True, force the use of a local model.

        Returns:
            Tuple[str, Chat]: The response and the full chat.
        """
        if isinstance(userRequest, str):
            chat: Chat = Chat("You are a yes/no classifier. Determine if the answer to the user is either yes or no and respond accordingly.")
            chat.add_message(Role.USER, "Is 8/7 a natural number?")
            chat.add_message(Role.ASSISTANT, "No")
            chat.add_message(Role.USER, """Does the below text provide relevant information to answer this question: Does Germany have a population of 84.000.000?\n\n```txt\nGermany,[e] officially the Federal Republic of Germany (FRG),[f] is a country in Central Europe. It lies between the Baltic and North Sea to the north and the Alps to the south. Its 16 constituent states have a total population of over 84 million in an area of 357,569 km2 (138,058 sq mi), making it the most populous member state of the European Union. It borders Denmark to the north, Poland and Czechia to the east, Austria and Switzerland to the south, and France, Luxembourg, Belgium, and the Netherlands to the west. The nation's capital and most populous city is Berlin and its main financial centre is Frankfurt; the largest urban area is the Ruhr.\n""")
            chat.add_message(Role.ASSISTANT, "Yes")
            chat.add_message(Role.USER, "Was 9/11 a inside job?")
            chat.add_message(Role.ASSISTANT, "No")
            chat.add_message(Role.USER, "Fehlt der folgenden Sequenz eine Zahl: 1, 2, 3, 5, 6?")
            chat.add_message(Role.ASSISTANT, "Yes")
            chat.add_message(Role.USER, "The next request will be very long, the question will be at its start, please ensure you only answer with 'Yes' or 'No'.\nAre you ready?")
            chat.add_message(Role.ASSISTANT, "Yes")
            chat.add_message(Role.USER, userRequest)
        else:
            chat = userRequest
        
        response: str = LlmRouter.generate_completion(
            chat,
            strength=AIStrengths.FAST,
            preferred_models=preferred_models, 
            force_preferred_model=force_preferred_model,
            force_local=force_local,
            force_free=force_free,
            silent=silent,
        )
        chat.add_message(Role.ASSISTANT, response)
        return "yes" in response.lower(), chat

    @classmethod
    def few_shot_CmdAgentExperimental(self, userRequest: str, model: str, force_local:bool = False, optimize: bool = False) -> Tuple[str,Chat]:
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
            force_local=force_local
        )
        chat.add_message(Role.ASSISTANT, response)
        return response, chat

    @classmethod
    def few_shot_TerminalAssistant(self, userRequest: str, preferred_models: List[str] = [], force_local:bool = False, silent: bool = False, use_reasoning: bool = False, silent_reasoning: bool = False) -> Tuple[str,Chat]:
        """
        Command agent for Ubuntu that provides shell commands based on user input.

        Args:
            userRequest (str): The user's request.
            model (str): Model to use for generating the response.
            force_local (bool, optional): If True, force the use of a local model.
            optimize (bool, optional): If True, optimize the chat.

        Returns:
            Tuple[str, Chat]: The response and the full chat.
        """
        chat: Chat = Chat(
            FewShotProvider.few_shot_rephrase("Designed for autonomy, this Ubuntu CLI-Assistant autonomously addresses user queries by crafting optimized, non-interactive shell commands that execute independently. It progresses systematically, preemptively suggesting command to gather required datapoints to ensure the creation of perfectly structured and easily executable instructions. The system utilises shell scripts only if a request cannot be fullfilled non-interactively otherwise.", preferred_models, force_local, silent=True, use_reasoning=False)
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
            """Let's execute the following command to find our working directory:
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
            "I've been found! The output shows the process IDs and command lines for the Python processes that are running the CLI-Agent code. That's meta! 🤖"
        )
        

        # if len(userRequest)<400 and not "if (" in userRequest and not "{" in userRequest: # ensure userRequest contains no code snippet
        #     userRequest = self.few_shot_rephrase(userRequest, preferred_models, force_local, silent=True)
        
        chat.add_message(
            Role.USER,
            userRequest
        )

        response: str = LlmRouter.generate_completion(
            chat,
            preferred_models,
            force_local=force_local,
            silent=silent,
            use_reasoning=use_reasoning,
            silent_reasoning=silent_reasoning
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
        
        if ("apt" in response and "purge" in response and not "-y" in response):
            applied_hardcoded_fixes = True
            response = response.replace("apt purge", "apt purge -y")
            response = response.replace("apt-get purge", "apt-get purge -y")
        
        if applied_hardcoded_fixes:
            print(colored("DEBUG: Applied hardcoded fix(es) to the response.", "yellow"))
        
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
    def few_shot_rephrase(self, userRequest: str, preferred_models: List[str] = [""], force_local: bool = False, silent: bool = True, force_free = False, use_reasoning: bool = False) -> str:
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
            chat.add_message(Role.USER, "Rephrase: '¿Cuál es la principal ciudad de los alemanes?'")
            chat.add_message(Role.ASSISTANT, "Rephrased version: '¿Puedes nombrar la capital de Alemania?'")
            chat.add_message(Role.USER, "Rephrase: 'whats 4*8'")
            chat.add_message(Role.ASSISTANT, "Rephrased version: 'Please calculate the product of 4*8'")
            chat.add_message(Role.USER, f"Rephrase: '{userRequest}'")
            
            preferred_rephrase_model_keys = []
            for preferred_model_key in preferred_models:
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
                preferred_models,
                force_local=force_local,
                force_free=force_free,
                silent=silent,
                temperature=0.4,
                use_reasoning=use_reasoning
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
    
    @classmethod
    def few_shot_textToPresentation(self, text: str, preferred_models: List[str]=[], force_local: bool = False) -> Tuple[Chat, str]:
        slides_1 = [
            Slide("Hybrid Approach (Marvin + ML)", 
                "• Combine existing 'Marvin' strategy with ML models\n"
                "• Leverage strengths of both: domain knowledge and learned patterns"),
            Slide("Modular ML Integration",
                "• Integrate ML components in a modular, decoupled fashion\n"
                "• Define clear interfaces between ML and non-ML parts\n" 
                "• Enables independent training, testing, and upgrades of ML models\n"
                "• Facilitates controlled experiments and component reuse"),
            Slide("Online Learning for Adaptability",
                "Incorporating online learning allows the robots to continuously adapt their behavior during actual game play based on the current opponent strategies. This real-time adaptation enables the discovery of new strategies that were not initially programmed or anticipated. Key considerations include processing latency, sampling efficiency, balancing exploration vs exploitation, and robustness against adversarial exploitation."),
            Slide("Transfer Learning for Jumpstart",
                "• Leverage ML models pre-trained on relevant tasks\n"
                "• Fine-tune them for the specific robocup application\n"
                "• Achieves faster learning and higher initial performance"),
            Slide("Opponent Modeling for Strategic Advantage", 
                "• Learn models of specific opponent teams based on their behaviors\n"
                "• Predict opponent actions and exploit their weaknesses\n"
                "• Adapt strategy to each opponent for a competitive edge\n"
                "• Consider generalization, robustness, and the meta-game of strategy switching"),
            Slide("Hierarchical Learning: Strategy vs Execution",
                "• Separate high-level strategy from low-level execution\n"
                "• Use sophisticated ML for strategic decisions, lightweight ML for execution")
        ]
        presentation_1 = PptxPresentation("ER-Force Strategy Optimization", "Strategy meeting 2024", slides_1)
        
        instruction = FewShotProvider.few_shot_rephrase("You are a presentation creator. Given a topic or text, generate a concise, informative presentation", silent=True, preferred_models=preferred_models, force_local=force_local)
        chat = Chat(instruction)
        
        user_input = """My robotics team ER-Force is discussing the optimization of our robot strategy tomorrow. The following points will be discussed:
A hybrid approach that combines the existing "marvin" strategy with ML models.
A modular ML integration for easier testing and improvements.
Real-time adaptability through online learning.
Use of transfer learning to improve initial performance.
Implementation of opponent modeling for strategic advantages.
A hierarchical learning approach to separate strategy and execution."""
        rephrased_user_input = FewShotProvider.few_shot_rephrase(user_input, silent=True, preferred_models=preferred_models, force_local=force_local)
        decomposition_prompt = FewShotProvider.few_shot_rephrase("Please decompose the following into 3-6 subtopics and provide step by step explanations + a very short discussion:", silent=True, preferred_models=preferred_models, force_local=force_local)
        presentation_details = LlmRouter.generate_completion(f"{decomposition_prompt}: '{rephrased_user_input}'", strength=AIStrengths.STRONG, silent=True, preferred_models=preferred_models, force_local=force_local)
        chat.add_message(Role.USER, presentation_details)
        
        create_presentation_response = FewShotProvider.few_shot_rephrase("I will create a presentation titled 'ER-Force Strategy Optimization' that covers the main points of your discussion.", silent=True, preferred_models=preferred_models, force_local=force_local).strip(".")
        chat.add_message(Role.ASSISTANT, f"""{create_presentation_response}
        ```
        {presentation_1.to_json()}```""")
        
        thanks_prompt = FewShotProvider.few_shot_rephrase("Thank you! You have generated exactly the right JSON data. Keep this exact format.\nNow create such a presentation for this", silent=True, preferred_models=preferred_models, force_local=force_local).strip(".")
        chat.add_message(Role.USER, f"{thanks_prompt}: {text}")
        
        response: str = LlmRouter.generate_completion(
            chat,
            strength=AIStrengths.STRONG,
            preferred_models=preferred_models, 
            force_local=force_local
        )
        chat.add_message(
            Role.ASSISTANT,
            response,
        )
        return chat, response

    @classmethod
    def few_shot_objectFromTemplate(cls, example_objects: List[Any], target_description: str, preferred_models: List[str] = [], force_local: bool = False) -> Any:
        """
        Returns an object based on a list of example objects and a target description.

        Args:
            example_objects (List[Any]): A list of objects to use as examples.
            target_description (str): Description of the target object to create.
            preferred_models (List[str], optional): List of preferred model keys for LLM.
            force_local (bool, optional): If True, force the use of a local model.

        Returns:
            Any: The newly created object based on the examples and description.
        """
        # Convert example objects to JSON strings
        example_jsons = [json.dumps(obj, default=lambda o: o.__dict__, sort_keys=True, indent=2) for obj in example_objects]

        # Create the chat prompt
        chat = Chat("You are an object creator. Given a list of example objects in JSON format and a description, create a new object with a similar structure but different content.")
        
        # Add example interactions
        chat.add_message(Role.USER, """Example objects:
[
{
  "name": "John Doe",
  "age": 30,
  "skills": ["Python", "JavaScript", "SQL"],
  "contact": {
    "email": "john@example.com",
    "phone": "123-456-7890"
  }
},
{
  "name": "Jane Smith",
  "age": 28,
  "skills": ["Java", "C++", "Ruby"],
  "contact": {
    "email": "jane@example.com",
    "phone": "987-654-3210"
  }
}
]
Create a similar object for a senior data scientist specializing in machine learning.""")

        chat.add_message(Role.ASSISTANT, """{
  "name": "Alice Johnson",
  "age": 35,
  "skills": ["Python", "R", "TensorFlow", "Scikit-learn", "SQL"],
  "contact": {
    "email": "alice.johnson@datatech.com",
    "phone": "555-123-4567"
  }
}""")

        # Add the actual task
        example_objects_str = ",\n".join(example_jsons)
        chat.add_message(Role.USER, f"""Example objects:
[
{example_objects_str}
]
Create a similar object based on this description: {target_description}""")

        # Generate the response
        response: str = LlmRouter.generate_completion(
            chat,
            strength=AIStrengths.FAST,
            preferred_models=preferred_models,
            force_local=force_local
        )

        # Extract the JSON string from the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        result_json = response[json_start:json_end]

        # Parse the JSON string back into an object
        result_object = json.loads(result_json)

        # Recursively convert dict to object if the examples were objects
        if not isinstance(example_objects[0], dict):
            result_object = cls._dict_to_obj(result_object)

        return result_object

    @staticmethod
    def _dict_to_obj(d):
        """
        Convert a dictionary to an object recursively.
        """
        class DynamicObject:
            pass
        
        obj = DynamicObject()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(obj, k, FewShotProvider._dict_to_obj(v))
            else:
                setattr(obj, k, v)
        return obj
    
    @classmethod
    def few_shot_distilText(cls, query: str, scraped_contents: List[str], summarization_llms: List[str], force_free: bool = True) -> str:
        """
        Summarizes the scraped content using a distillation model.

        Args:
            query (str): The query used to search for the content.
            scraped_content (str): The scraped content to summarize.
            summarization_llm (str): The summarization model to use.

        Returns:
            str: The summarized content.
        """
        summarizations = []
        for scraped_content in scraped_contents:
            while len(scraped_content)/4 > 4096:
                digestible_text = scraped_content[:8192]
                scraped_content = scraped_content[8192:]
                chat = Chat("You are a text summarizer. Given a query and scraped content, you summarize the text to provide a concise and factual overview.")
                chat.add_message(Role.USER, f"""Please interpret the below website-content, can you extract information relevant to the query?
QUERY:
{query}
CONTENT:
{digestible_text}""")
                summary = LlmRouter.generate_completion(chat, summarization_llms, force_free=force_free)
                summarizations.append(summary)
        while len(summarizations) > 1:
            chat = Chat("You are a expert data scientist. Please combine the provided summaries to create a concise, but as detailed as needed, overview.")
            chat.add_message(Role.USER, f"""Please summarize the below summaries, the topic is '{query}':
```summary_1
{summarizations.pop(0)}
```
```summary_2
{summarizations.pop(1)}
```
""")
            summary = LlmRouter.generate_completion(chat, summarization_llms, force_free=force_free)
            summarizations.append(summary)
        return summarizations[0]




    @classmethod
    def few_shot_projectModificationPlanning(cls, structure_description: Dict[str, List[str]], modification_request: str, preferred_models: List[str], force_local: bool = False) -> Tuple[List[Tuple[str, str]], Chat]:
        """
        Analyzes the project structure and identifies files that likely need modification based on a given request.

        Args:
            structure_description (Dict[str, List[str]]): Dictionary of files grouped by type.
            modification_request (str): The modification request to fulfill.
            preferred_models (List[str]): List of preferred model keys for LLM.
            force_local (bool): If True, force the use of a local model.

        Returns:
            List[Tuple[str, str]]: List of tuples containing (file_path, explanation) for recommended modifications.
        """
        chat = Chat("""You are a software development expert. Given a project structure (grouped by file types) and a modification request, identify the files that likely need to be modified to fulfill the request. 
For each file, provide a brief explanation of why it needs to be modified. 
Present your response in the following format:

1. file_path_1
Explanation: Reason for modifying this file.

2. file_path_2
Explanation: Reason for modifying this file.

... and so on.

Limit your response to the most important files (usually 3-5) that need modification.""")

        # Example 1
        chat.add_message(Role.USER, """Analyze the following project structure and respond with the file(s) that likely need modification in order to fulfill the following request 'Add a new API endpoint for user registration':

{
    "py": [
        "src/api/__init__.py",
        "src/api/users.py",
        "src/api/auth.py",
        "src/models/__init__.py",
        "src/models/user.py",
        "src/services/__init__.py",
        "src/services/user_service.py",
        "src/main.py",
        "tests/test_api.py"
    ],
    "yml": [
        "config.yml"
    ]
}""")

        chat.add_message(Role.ASSISTANT, """1. src/api/users.py
Explanation: This file likely contains the API endpoints for user-related operations. It needs to be modified to add the new endpoint for user registration.

2. src/services/user_service.py
Explanation: This file probably contains the business logic for user operations. It needs to be updated to include the logic for user registration.

3. src/models/user.py
Explanation: This file defines the user model. It may need updates if new fields are required for user registration.

4. tests/test_api.py
Explanation: This file contains API tests. It should be updated to include tests for the new user registration endpoint.

5. src/main.py
Explanation: This file might need minor changes if the new route needs to be explicitly registered or if any app-wide configurations need to be updated for the new endpoint.""")

        # Example 2
        chat.add_message(Role.USER, """Analyze the following project structure and respond with the file(s) that likely need modification in order to fulfill the following request 'Implement caching for database queries':

{
    "py": [
        "app/database/__init__.py",
        "app/database/connection.py",
        "app/database/queries.py",
        "app/cache/__init__.py",
        "app/cache/redis_client.py",
        "app/services/__init__.py",
        "app/services/data_service.py",
        "app/main.py",
        "tests/test_database.py"
    ],
    "ini": [
        "config.ini"
    ]
}""")

        chat.add_message(Role.ASSISTANT, """1. app/database/queries.py
Explanation: This file likely contains the database queries. It needs to be modified to implement caching logic for the queries.

2. app/services/data_service.py
Explanation: This file probably contains the data retrieval and manipulation logic. It needs to be updated to integrate the caching mechanism with the existing data operations.

3. app/cache/redis_client.py
Explanation: This file handles Redis operations. It may need modifications to support the specific caching requirements for database queries.

4. config.ini
Explanation: This configuration file might need updates to include cache-related settings, such as Redis connection details or caching policies.

5. tests/test_database.py
Explanation: This file contains database-related tests. It should be updated to include tests for the new caching functionality.""")

        # Actual task
        structure_str = json.dumps(structure_description, indent=2)
        chat.add_message(Role.USER, f"Analyze the following project structure and respond with the file(s) that likely need modification in order to fulfill the following request '{modification_request}':\n\n{structure_str}")

        response: str = LlmRouter.generate_completion(
            chat,
            strength=AIStrengths.FAST,
            preferred_models=preferred_models,
            force_local=force_local
        )

        # Parse the response into a list of tuples
        parsed_response = cls._parse_projectModificationPlanningResponse(response)
        return parsed_response, chat
    
    @classmethod
    def _parse_projectModificationPlanningResponse(cls, response: str) -> List[Tuple[str, str]]:
        """
        Parses the AI's response into a list of tuples (file_path, explanation).
        """
        result = []
        lines = response.strip().split('\n')
        current_file = None
        current_explanation: List[str] = []
        for line in lines:
            if re.match(r'^\d+\.', line):
                if current_file:
                    result.append((current_file, ' '.join(current_explanation)))
                current_file = line.split('.', 1)[1].strip()
                current_explanation = []
            elif line.startswith('Explanation:'):
                current_explanation.append(line.split(':', 1)[1].strip())
            elif current_explanation:
                current_explanation.append(line.strip())
        if current_file:
            result.append((current_file, ' '.join(current_explanation)))
        return result
    
    @classmethod
    def few_shot_textToPropositions(cls, text: str, preferred_models: List[str] = [], force_local: bool = False, silent: bool = False) -> List[str]:
        """
        Extracts explicit, reliable factual propositions from the given text, supporting multiple languages.

        Args:
            text (str): The input text to extract propositions from.
            preferred_models (List[str], optional): List of preferred model keys for LLM.
            force_local (bool, optional): If True, force the use of a local model.
            silent (bool, optional): If True, suppress output during processing.

        Returns:
            List[str]: A list of strings representing the extracted factual propositions.
        """
        chat = Chat("""You are a reliable fact extractor. Your task is to extract clear, verifiable and meaningful information from the given document.

Key Principles:
1. Extract only objective facts directly supported by the document.
2. Make each fact self-contained and easily understandable.
3. Include necessary context, dates, and sources within each fact.
4. Break complex statements into simpler, distinct facts.
5. Use the original language for non-English documents.
6. Restate the information presented in the document, rewording it to provide context and clarity, making it understandable on its own without referencing the original document.

Avoid:
- Opinions or subjective statements
- Facts with critical missing information
- Adding information not explicitly stated in the document

Presentation:
1. Begin with "Here are the extracted facts:"
2. List facts in numbered lines
3. If no clear facts can be extracted, respond only with "No clear facts could be extracted."

Remember: Focus on quality over quantity. Each fact should be clear, as detailed as required for out-of-context validity, and directly supported by the document.""")

        # Example 1: German text about Informatik
        chat.add_message(Role.USER, """Extrahiere faktische Aussagen aus dem folgenden Dokument:
# Filepath: '/Users/yohanOettinger/Downloads/Informatik.pdf'

Informatik ist die Wissenschaft der systematischen Verarbeitung von Informationen, insbesondere der automatischen Verarbeitung mit Hilfe von Computern. Sie wurde in Deutschland in den 1960er Jahren als akademisches Fach etabliert. 
Zu den Teilgebieten der Informatik gehören unter anderem Theoretische Informatik, Praktische Informatik, Technische Informatik und Angewandte Informatik. Ein zentrales Konzept in der Informatik ist der Algorithmus, eine eindeutige Handlungsvorschrift zur Lösung eines Problems. 
Die Programmiersprache C, entwickelt von Dennis Ritchie in den frühen 1970er Jahren, hatte einen großen Einfluss auf die Entwicklung moderner Programmiersprachen. Heute spielt die Informatik eine wichtige Rolle in vielen Bereichen, 
von der Künstlichen Intelligenz bis zur Cybersicherheit.""")

        chat.add_message(Role.ASSISTANT, """Hier sind die extrahierten Fakten:

1. Informatik ist die Wissenschaft der systematischen Informationsverarbeitung.
2. Computer werden in der Informatik zur automatischen Informationsverarbeitung eingesetzt.
3. Informatik wurde in den 1960er Jahren als akademisches Fach in Deutschland etabliert.
4. Die Informatik umfasst Teilgebiete wie Theoretische, Praktische, Technische und Angewandte Informatik.
5. Ein Algorithmus ist eine eindeutige Handlungsvorschrift zur Problemlösung.
6. Dennis Ritchie entwickelte die Programmiersprache C in den frühen 1970er Jahren.
7. Die Programmiersprache C beeinflusste die Entwicklung moderner Programmiersprachen maßgeblich.
8. Künstliche Intelligenz ist ein Anwendungsbereich der Informatik.
9. Cybersicherheit ist ein Anwendungsgebiet der Informatik.""")

        # Example 2: Text about Mona Lisa
        chat.add_message(Role.USER, """Extract factual propositions from the following document:
# Filepath: /home/lmao/OneDrive/MonaLisa.pdf

The Mona Lisa, painted by Leonardo da Vinci, is one of the most famous paintings in the world. Some people believe it's overrated, but it attracts millions of visitors to the Louvre in Paris each year. The exact date of its creation is disputed, but it's generally believed to have been painted in the early 16th century.""")

        chat.add_message(Role.ASSISTANT, """Here are the extracted facts:

1. Leonardo da Vinci malte die Mona Lisa.
2. Die Mona Lisa wird im Louvre in Paris ausgestellt.
3. Die Mona Lisa zieht jährlich Millionen von Besuchern an.
4. Die Mona Lisa entstand vermutlich im frühen 16. Jahrhundert.""")

        # Actual task
        chat.add_message(Role.USER, f"Extrahiere faktische Aussagen aus dem folgenden Dokument:\n{text}")

        response: str = LlmRouter.generate_completion(
            chat,
            preferred_models=preferred_models,
            force_local=force_local,
            force_free=True,
            silent=silent
        )

        # Remove the first default line
        shortened_response = response.split('\n', 1)[-1].strip()
        # Ensure the input starts with a newline so that the first proposition is correctly identified
        modified_response = '\n' + shortened_response
        # Split the response where a number followed by a period and space is at the start of a line
        propositions = [prop.strip() for prop in re.split(r'\n\d+\.\s+', modified_response) if prop.strip()]
        return propositions

    @classmethod
    def few_shot_toInteger(cls, text: str, preferred_models: List[str] = [], force_local: bool = False, silent: bool = False, force_free:bool = False) -> int:
        """
        Converts a given text representation of a number into an integer using few-shot learning.

        Args:
            text (str): The input text to convert to an integer.
            preferred_models (List[str], optional): List of preferred model keys for LLM.
            force_local (bool, optional): If True, force the use of a local model.
            silent (bool, optional): If True, suppress output during processing.

        Returns:
            int: The integer representation of the input text.
        """
        chat = Chat("""You are an expert in converting various textual representations of numbers into integers. Your task is to analyze the given text and return the corresponding integer value. Follow these guidelines:

    1. Handle various formats including words, digits, and mixed representations.
    2. Support multiple languages.
    3. Interpret contextual clues to determine the intended number.
    4. Handle negative numbers and zero.
    5. If the input is ambiguous or cannot be converted to an integer, return None.

    Respond with only the integer value or None, without any additional explanation.""")

        # Example 1: English word representation
        chat.add_message(Role.USER, "Convert to integer: five hundred and twenty-three")
        chat.add_message(Role.ASSISTANT, "523")

        # Example 2: Digit representation
        chat.add_message(Role.USER, "Convert to integer: 1,234")
        chat.add_message(Role.ASSISTANT, "1234")

        # Example 3: German representation
        chat.add_message(Role.USER, "Convert to integer: zweitausenddreihundertvierundvierzig")
        chat.add_message(Role.ASSISTANT, "2344")

        # Example 4: Mixed representation
        chat.add_message(Role.USER, "Convert to integer: 2 million five hundred thousand and 99")
        chat.add_message(Role.ASSISTANT, "2500099")

        # Example 5: Negative number
        chat.add_message(Role.USER, "Convert to integer: minus three hundred and twelve")
        chat.add_message(Role.ASSISTANT, "-312")

        # Example 6: Zero
        chat.add_message(Role.USER, "Convert to integer: null")
        chat.add_message(Role.ASSISTANT, "0")

        # Example 7: Ambiguous input
        chat.add_message(Role.USER, "Convert to integer: a dozen oranges")
        chat.add_message(Role.ASSISTANT, "None")

        # Actual task
        chat.add_message(Role.USER, f"Convert to integer: {text}")

        response: str = LlmRouter.generate_completion(
            chat,
            preferred_models=preferred_models,
            force_local=force_local,
            force_free=force_free,
            silent=silent
        )

        try:
            result = int(response.strip())
            return result
        except ValueError:
            return None

    @classmethod
    def few_shot_textToQuestions(cls, text: str, preferred_models: List[str] = [], force_local: bool = False, silent: bool = False) -> List[str]:
        """
        Generates an extensive list of questions that can be answered using the contents of the given text.

        Args:
            text (str): The input text to generate questions from.
            preferred_models (List[str], optional): List of preferred model keys for LLM.
            force_local (bool, optional): If True, force the use of a local model.
            silent (bool, optional): If True, suppress output during processing.

        Returns:
            List[str]: A list of strings representing the generated questions.
        """
        chat = Chat("""You are an expert question generator. Your task is to create a comprehensive list of questions that can be answered using the information provided in the given text. Follow these guidelines:

    1. Generate a diverse range of questions covering various aspects of the text.
    2. Include questions of different types: factual, analytical, comparative, and inferential.
    3. Ensure questions are clear, concise, and directly answerable from the text.
    4. Avoid questions that require information not present in the text.
    5. Generate questions in the same language as the input text.
    6. Aim for a minimum of 10 questions, but generate more if the text content allows.

    Presentation:
    1. Begin with "Generated questions:"
    2. List questions in numbered lines
    3. If the text doesn't contain enough information for questions, respond with "Insufficient information to generate meaningful questions."

    Remember: Focus on creating questions that thoroughly explore the content of the text and encourage a deeper understanding of the material.""")

        # Example 1: Text about climate change
        chat.add_message(Role.USER, """Generate questions for the following text:
    Climate change is a long-term shift in global or regional climate patterns. Often climate change refers specifically to the rise in global temperatures from the mid-20th century to present. The main driver of climate change is the greenhouse effect. Some gases in the Earth's atmosphere act a bit like the glass in a greenhouse, trapping the sun's heat and stopping it from leaking back into space. Many of these greenhouse gases occur naturally, but human activity is increasing the concentrations of some of them in the atmosphere, particularly carbon dioxide (CO2), methane, and nitrous oxide. CO2 is the greenhouse gas most commonly produced by human activities and it is responsible for 64% of man-made global warming.""")

        chat.add_message(Role.ASSISTANT, """Generated questions:
    1. What is climate change?
    2. What specific aspect does climate change often refer to?
    3. What is the main driver of climate change?
    4. How do greenhouse gases affect the Earth's temperature?
    5. Are all greenhouse gases man-made?
    6. Which greenhouse gases are being increased by human activity?
    7. What is the most common greenhouse gas produced by human activities?
    8. What percentage of man-made global warming is CO2 responsible for?
    9. How do greenhouse gases in the atmosphere compare to glass in a greenhouse?
    10. When did the rise in global temperatures that we often refer to as climate change begin?
    11. What are the three main greenhouse gases mentioned in the text that are increased by human activity?
    12. How does the greenhouse effect contribute to climate change?
    13. Are there any natural sources of greenhouse gases mentioned in the text?
    14. What is the relationship between human activities and the concentration of greenhouse gases in the atmosphere?
    15. How does the text define the greenhouse effect?""")

        # Example 2: Text about the history of computers
        chat.add_message(Role.USER, """Generate questions for the following text:
    The history of computers begins with the abacus, a manual counting device. In the 17th century, mechanical calculators were invented. The first general-purpose computer, ENIAC, was built in 1945. It used vacuum tubes and could perform calculations much faster than previous machines. In the 1950s, transistors replaced vacuum tubes, making computers smaller and more reliable. The first microprocessor, the Intel 4004, was introduced in 1971, paving the way for personal computers. Apple II, one of the first successful mass-produced microcomputers, was launched in 1977. The World Wide Web was invented by Tim Berners-Lee in 1989, revolutionizing how we use computers and access information.""")

        chat.add_message(Role.ASSISTANT, """Generated questions:
    1. What is considered the beginning of computer history?
    2. When were mechanical calculators invented?
    3. What was the name of the first general-purpose computer?
    4. In which year was ENIAC built?
    5. What technology did ENIAC use?
    6. How did ENIAC's calculation speed compare to previous machines?
    7. What replaced vacuum tubes in computers during the 1950s?
    8. How did the transition to transistors affect computers?
    9. When was the first microprocessor introduced?
    10. What was the name of the first microprocessor?
    11. What impact did the microprocessor have on computer development?
    12. Which computer is mentioned as one of the first successful mass-produced microcomputers?
    13. In what year was the Apple II launched?
    14. Who invented the World Wide Web?
    15. When was the World Wide Web invented?
    16. How did the invention of the World Wide Web impact computer use?
    17. What technological advancements are mentioned in the text that contributed to the evolution of computers?
    18. How did computer size change over time according to the text?
    19. What is the chronological order of major developments in computer history as presented in the text?
    20. How did the development of computers impact information access according to the text?""")

        # Actual task
        chat.add_message(Role.USER, f"Generate questions for the following text:\n{text}")

        response: str = LlmRouter.generate_completion(
            chat,
            preferred_models=preferred_models,
            force_local=force_local,
            force_free=True,
            silent=silent
        )

        # Remove the first default line
        shortened_response = response.split('\n', 1)[-1].strip()
        # Split the response where a number followed by a period and space is at the start of a line
        questions = [q.strip() for q in re.split(r'\n\d+\.\s+', shortened_response) if q.strip()]
        return questions


    @classmethod
    def few_shot_contextToJson(cls, context: str, silent:bool = False):
        # Initialize the model
        model = ollama.Client()

        # Define the template for extraction
        template = '''
        {
        "Model": {
            "Name": "",
            "Number of parameters": ""
        },
        "Usage": {
            "Use case": [],
            "Licence": ""
        }
        }
        '''

        # The text to extract information from
        text = '''
        We introduce Mistral 7B, a 7–billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms the best open 13B model (Llama 2) across all evaluated benchmarks, and the best released 34B model (Llama 1) in reasoning, mathematics, and code generation. Our model leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost. We also provide a model fine-tuned to follow instructions, Mistral 7B – Instruct, that surpasses Llama 2 13B – chat model both on human and automated benchmarks. Our models are released under the Apache 2.0 license.
        '''

        # Construct the prompt
        prompt = f'''
        ### Template:

        {template}

        ### Text:

        {context}
        '''

        # Generate the response
        response = model.generate(model="nuextract", prompt=prompt)

        print(response.response)

    @classmethod
    def few_shot_toPythonRequirements(cls, implementationDescription: str, preferred_models: List[str] = [], force_local: bool = False, silent: bool = False) -> str:
        """
        Generates the contents for a requirements.txt file based on the given implementation description.

        Args:
            implementationDescription (str): Description of the implementation.
            preferred_models (List[str], optional): List of preferred model keys for LLM.
            force_local (bool, optional): If True, force the use of a local model.
            silent (bool, optional): If True, suppress output during processing.

        Returns:
            str: The contents for a requirements.txt file.
        """
        chat = Chat("""You are an expert Python developer. Your task is to analyze the given implementation description and generate a requirements.txt file containing the necessary Python packages for the implementation. Follow these guidelines:

        1. Include only the necessary packages for the described implementation.
        2. Use standard package names as they appear in PyPI.
        3. Specify version numbers only when strictly necessary.
        4. Include one package per line.
        5. If the implementation doesn't require any external packages, return an empty string.""")

        # Example 1: Web scraping implementation
        chat.add_message(Role.USER, "Generate requirements for: A web scraping script using BeautifulSoup and requests to extract data from websites. The script also uses pandas to store the data in a CSV file.")
        chat.add_message(Role.ASSISTANT, """Sure here are the requirements:
'''txt
beautifulsoup4
requests
pandas
'''""")

        # Example 2: Machine learning implementation
        chat.add_message(Role.USER, "Generate requirements for: A machine learning project using TensorFlow for deep learning, scikit-learn for preprocessing, and matplotlib for visualizations. The project also uses numpy for numerical operations.")
        chat.add_message(Role.ASSISTANT, """Sure here are the requirements:
'''txt
tensorflow
scikit-learn
matplotlib
numpy
'''""")

        # Example 3: Flask web application
        chat.add_message(Role.USER, "Generate requirements for: A Flask web application with SQLAlchemy for database management, Flask-WTF for form handling, and Pillow for image processing. The app uses pytest for testing.")
        chat.add_message(Role.ASSISTANT, """Sure here are the requirements:
'''txt
Flask
SQLAlchemy
Flask-WTF
Pillow
pytest
'''""")
        # Actual task
        chat.add_message(Role.USER, f"Generate requirements for: {implementationDescription}")

        response: str = LlmRouter.generate_completion(
            chat,
            preferred_models=preferred_models,
            force_local=force_local,
            force_free=True,
            silent=silent
        )
    
        # Extract content between '''txt and '''
        match = re.search(r"'''txt\n(.*?)'''", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return ""

    @classmethod
    def few_shot_GenerateHtmlPage(
        cls,
        page_description: str,
        preferred_models: List[str] = [],
        force_preferred_model: bool = False,
        force_local: bool = False,
        silent: bool = False,
        force_free: bool = True
    ) -> Tuple[str|None, Chat]:
        """
        Generates an HTML page based on the given description.

        Args:
            page_description (str): A description of the desired HTML page.
            preferred_models (List[str], optional): List of preferred model keys.
            force_preferred_model (bool, optional): If True, force the use of a preferred model.
            force_local (bool, optional): If True, force the use of a local model.
            silent (bool, optional): If True, suppress output.
            force_free (bool, optional): If True, force the use of a free model.

        Returns:
            Tuple[str, Chat]: The generated HTML content and the GenerateHtmlPage-Chat.
        """
        chat = Chat("You are an expert HTML developer. Generate a complete, valid HTML page based on the given description. Include appropriate CSS styling within a <style> tag in the <head> section. Use semantic HTML5 tags where appropriate. Ensure the page is responsive and follows modern web design principles.")

        example_html_description="""I'd be happy to help you!

Both courses of study are interesting and can prepare you for various career paths.

**Data Science:**

* You'll learn fundamentals of mathematics, computer science, and statistics
* You'll master tools like Python, R, or SQL
* You'll be equipped to analyze complex data and draw conclusions using algorithms
* Career opportunities lie in industry, research, finance, and many other fields

**Technomathematics:**

* You'll study mathematics with a focus on applications in technology and natural sciences
* You'll learn fundamentals of analysis, linear algebra, and numerical methods
* You'll be able to mathematically model complex problems and find solutions
* Career opportunities lie in industry, research, IT, and many other areas

**Your Decision:**

If you're interested in data analysis and algorithms and enjoy working with large datasets, Data Science might be your path. However, if you're more inclined towards gaining a mathematical understanding for technical applications, Technomathematics could be the right choice for you.

I hope this helps! Let me know if you have any further questions."""

        chat.add_message(Role.USER, "Please create an HTML overview of the following text:\n" + example_html_description)
        chat.add_message(Role.ASSISTANT, """Certainly, here's the HTML code:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparison: Data Science vs. Technomathematics</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f4f8;
        }
        h1, h2 {
            color: #2c3e50;
        }
        .comparison {
            display: flex;
            justify-content: space-between;
            margin-top: 30px;
        }
        .field {
            flex-basis: 48%;
            background-color: #fff;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .field h2 {
            text-align: center;
            color: #fff;
            padding: 10px;
            margin: -20px -20px 20px -20px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }
        .data-science h2 {
            background-color: #3498db;
        }
        .technomathematics h2 {
            background-color: #e74c3c;
        }
        ul {
            padding-left: 20px;
        }
        .decision {
            margin-top: 30px;
            background-color: #fff;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .decision h2 {
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <h1>Comparison: Data Science vs. Technomathematics</h1>
    
    <div class="comparison">
        <div class="field data-science">
            <h2>Data Science</h2>
            <ul>
                <li>Fundamentals of mathematics, computer science, and statistics</li>
                <li>Tools like Python, R, or SQL</li>
                <li>Analysis of complex data and application of algorithms</li>
                <li>Career opportunities in industry, research, finance, and other fields</li>
            </ul>
        </div>
        
        <div class="field technomathematics">
            <h2>Technomathematics</h2>
            <ul>
                <li>Mathematics with focus on applications in technology and natural sciences</li>
                <li>Fundamentals of analysis, linear algebra, and numerical methods</li>
                <li>Mathematical modeling of complex problems and solution finding</li>
                <li>Career opportunities in industry, research, IT, and other areas</li>
            </ul>
        </div>
    </div>
    
    <div class="decision">
        <h2>Your Decision</h2>
        <p><strong>Data Science:</strong> If you're interested in data analysis and algorithms and enjoy working with large datasets.</p>
        <p><strong>Technomathematics:</strong> If you're more inclined towards gaining a mathematical understanding for technical applications.</p>
    </div>
</body>
</html>
```""")

        chat.add_message(Role.USER, f"Nice, now please make the next ones more aesthetically pleasing, incorporate emojis as icons, do not use external dependencies other than image urls. Your code will be used as is so do not use placeholders. Please create a HTML page to visualize the latest discussed topic(s): {page_description}")

        response: str = LlmRouter.generate_completion(
            chat=chat,
            preferred_models=preferred_models,
            force_preferred_model=force_preferred_model,
            force_local=force_local,
            force_free=force_free,
            silent=silent
        )

        chat.add_message(Role.ASSISTANT, response)
        
        html_block = re.search(r'```html\n(.*?)```', response, re.DOTALL)
        
        if html_block:
            html_content = html_block.group(1)  # Get the content inside the HTML block
        else:
            html_content = ""  # Or you might want to raise an exception here
        
        return html_content, chat
    
    @classmethod
    def few_shot_ToImageGenPrompt(cls, description: str, preferred_models: List[str] = [], force_local: bool = False, silent: bool = False) -> str:
        """
        Generates an optimized image generation prompt based on the given description.

        Args:
            description (str): The input description to convert into an image generation prompt.
            preferred_models (List[str], optional): List of preferred model keys for LLM.
            force_local (bool, optional): If True, force the use of a local model.
            silent (bool, optional): If True, suppress output during processing.

        Returns:
            str: The optimized image generation prompt.
        """
        chat = Chat("""You are an expert in creating prompts for AI image generation. Your task is to take a given description and convert it into an optimized prompt suitable for image generation models. Follow these guidelines:

        1. Be specific and detailed in describing visual elements.
        2. Use clear and concise language.
        3. Include information about style, mood, lighting, and composition when relevant.
        4. Avoid abstract concepts or non-visual elements.
        5. Use commas to separate different elements of the prompt.
        6. Keep the prompt length reasonable (typically under 100 words).

        Respond with only the optimized prompt, without any additional explanation.""")

        # Example 1: Simple scene
        chat.add_message(Role.USER, "Convert to image prompt: A cat sitting on a windowsill")
        chat.add_message(Role.ASSISTANT, "A fluffy orange tabby cat sitting on a wooden windowsill, looking out at a sunny garden, soft natural lighting, detailed fur texture, cozy home atmosphere")

        # Example 2: Fantasy character
        chat.add_message(Role.USER, "Convert to image prompt: An elf warrior in a forest")
        chat.add_message(Role.ASSISTANT, "Elegant elven warrior with long silver hair, intricate green and gold armor, wielding a glowing bow, standing in a misty ancient forest, dappled sunlight, mystical atmosphere, digital fantasy art style")

        # Example 3: Abstract concept
        chat.add_message(Role.USER, "Convert to image prompt: The feeling of nostalgia")
        chat.add_message(Role.ASSISTANT, "Sepia-toned photograph of a child's hand touching an old vinyl record player, dust particles visible in warm sunlight streaming through a window, vintage toys and books scattered nearby, soft focus, emotional and wistful atmosphere")

        # Actual task
        chat.add_message(Role.USER, f"Convert to image prompt: {description}")

        response: str = LlmRouter.generate_completion(
            chat,
            preferred_models=preferred_models,
            force_local=force_local,
            force_free=True,
            silent=silent
        )

        return response.strip()