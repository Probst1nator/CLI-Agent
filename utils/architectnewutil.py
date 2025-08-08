import os
import json
from typing import List, Dict, Any

from py_classes.cls_util_base import UtilBase
from utils.todos import TodosUtil

class ArchitectNewUtil(UtilBase):
    """
    A Util to implement a new util.
    """
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["create tool", "new utility", "implement feature", "build function", "add capability", "constraint", "limited", "lack", "alternative", "outside the box", "automate", "integrate", "workflow", "pipeline", "orchestrate", "deploy", "setup", "configure", "install", "docker", "container", "download model", "huggingface", "ollama", "multi-step", "complex task", "no existing tool", "custom solution", "specialized", "technical integration"],
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
            }
        }

    @staticmethod
    def run(requirements: str, util_name: str) -> str:
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
            # read all files in the utils folder
            utils_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
            existing_utils_contents: List[str] = []
            for file in os.listdir(utils_folder):
                if not file.startswith("_"):
                    with open(os.path.join(utils_folder, file), "r") as f:
                        content = f.read()
                        existing_utils_contents.append(content)
            TodosUtil.run("add", f"To implement a new util for a needed function, break down the requirements into manageable components that can be implemented via python and update the todos as needed: {requirements}")
            TodosUtil.run("add", "Research, implement and test the planned component(s) one by one in python, do not yet write to a new util file.")
            TodosUtil.run("add", "Identify a minimal set of args for a run method for the util to ensure it is easy and minimal to use")
            TodosUtil.run("add", f"Consolidate the components into a single util and write it to {utils_folder}")
            TodosUtil.run("add", "Comprehensively test the new util in and update the todos as you progress")
            
            return json.dumps(TodosUtil.run("list"), indent=2)

        except Exception as e:
            return json.dumps({"error": f"An unexpected error ocurred in ArchitectNewUtil: {str(e)}"}, indent=2)