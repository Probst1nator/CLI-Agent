import os
import json
from typing import Dict, Any

from agent.utils_manager.util_base import UtilBase
from .deprecated.todos import TodosUtil

class ArchitectNewUtil(UtilBase):
    """
    A Util to implement a new util.
    """
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["create tool", "new utility", "implement feature", "build function", "add capability", "constraint", "limited", "lack", "alternative", "outside the box", "automate", "integrate", "workflow", "pipeline", "orchestrate", "deploy", "setup", "configure", "install", "docker", "container", "download model", "huggingface", "ollama", "multi-step", "complex task", "no existing tool", "custom solution", "specialized", "technical integration", "fix error", "debug tool", "error handling", "missing functionality", "need utility", "build fixer", "create debugger"],
            "use_cases": [
                "I need a tool that can connect to a websocket. Can you build it?",
                "Create a new utility for managing my calendar.",
                "Architect a new util to interact with the Spotify API.",
                "This interaction would be a lot more efficient if I automated some steps inside of a util.",
                "A util could be created for this interface.",
                "Find the latest agent LLMs under 30GB on HuggingFace, download and deploy to Ollama.",
                "Download a model from HuggingFace and set it up in my local environment.",
                "Automate the process of downloading, converting, and deploying ML models.",
                "Create a pipeline to sync data between multiple APIs automatically.",
                "Build a tool that monitors system resources and sends alerts.",
                "I need to orchestrate a complex multi-step deployment workflow."
            ],
            "arguments": {
                "requirements": "A detailed natural language description of what the new utility should do.",
                "util_name": "The desired class name for the new utility, in CamelCase (e.g., 'WebsocketUtil')."
            },
            "code_examples": [
                {
                    "description": "Create a websocket utility",
                    "code": """<python>
from utils.architectnewutil import ArchitectNewUtil
result = ArchitectNewUtil.run("A util which can connect to a websocket and send/receive messages", "WebsocketUtil")
print(result)
</python>"""
                },
                {
                    "description": "Build an API integration utility",
                    "code": """<python>
from utils.architectnewutil import ArchitectNewUtil
result = ArchitectNewUtil.run("Create a utility to interact with the Spotify API for playlist management", "SpotifyUtil")
print(result)
</python>"""
                }
            ]
        }

    @staticmethod
    def _run_logic(requirements: str, util_name: str) -> str:
        """
        Implement a new util based on passed requirements.

        Args:
            requirements (str): The requirements for the new util. (e.g. A util which can connect to a websocket and send and receive messages)
            util_name (str): The name of the new util. (e.g. WebsocketUtil)

        Returns:
            str: A JSON string with a 'result' key containing the new util,
                 or an 'error' key on failure.
        """
        try:
            utils_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))

            # Manually add the 'architectnewutil:' prefix to each task
            TodosUtil._run_logic("add", task=f"architectnewutil: To implement a new util for a needed function, break down the requirements into manageable components that can be implemented via python and update the todos as needed: {requirements}")
            TodosUtil._run_logic("add", task="architectnewutil: Research, implement and test the planned component(s) one by one in python, do not yet write to a new util file.")
            TodosUtil._run_logic("add", task="architectnewutil: Identify a minimal set of args for a run method for the util to ensure it is easy and minimal to use")
            TodosUtil._run_logic("add", task=f"architectnewutil: Consolidate the components into a single util and write it to {utils_folder}")
            TodosUtil._run_logic("add", task="architectnewutil: Comprehensively test the new util in and update the todos as you progress")
            
            return json.dumps(TodosUtil._run_logic("list"), indent=2)

        except Exception as e:
            return json.dumps({"error": f"An unexpected error ocurred in ArchitectNewUtil: {str(e)}"}, indent=2)


# Module-level run function for CLI-Agent compatibility
def run(requirements: str, util_name: str) -> str:
    """
    Module-level wrapper for ArchitectNewUtil._run_logic() to maintain compatibility with CLI-Agent.
    
    Args:
        requirements (str): Requirements for the new utility
        util_name (str): Name for the new utility
        
    Returns:
        str: JSON string with result or error
    """
    return ArchitectNewUtil._run_logic(requirements=requirements, util_name=util_name)
