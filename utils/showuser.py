from typing import Dict, Any
import markpickle
from py_classes.cls_util_base import UtilBase
from utils.todos import TodosUtil

class ShowUser(UtilBase):
    """
    A utility showing the user a html site optionally with images and animations
    """

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["display html", "show web page", "render html", "user interface", "graphical output"],
            "use_cases": [
                "Show the user a formatted report in HTML.",
                "Display a web page with images and charts.",
                "Render an interactive HTML file for the user."
            ],
            "arguments": {
                "path": "The absolute or relative path to the html file."
            },
            "code_examples": [
                {
                    "description": "Show an HTML file to the user",
                    "code": "from utils.showuser import ShowUser\nresult = ShowUser.run(path='report.html')"
                }
            ]
        }


    @staticmethod
    def _run_logic(
        path: str
    ) -> str:
        """
        Create and show a html site to the user

        Args:
            path (str): The absolute or relative path to the html file.

        Returns:
            Todos: overview of the updated todo list.
        """
        try:

            TodosUtil.run("add", f"showuser: Create your html site to show to the user at {path}")
            TodosUtil.run("add", f"showuser: Use the systems default browser to show {path} to the user.")
            
            return TodosUtil.run("list")

        except Exception as e:
            error_result = {"error": f"Could not write to file {path}. Reason: {e}"}
            return markpickle.dumps(error_result)


# Module-level run function for CLI-Agent compatibility
def run(path: str) -> str:
    """
    Module-level wrapper for ShowUser._run_logic() to maintain compatibility with CLI-Agent.
    
    Args:
        path (str): Path to a file or directory
        
    Returns:
        str: Markdown string with result or error
    """
    return ShowUser._run_logic(path=path)