import json
from typing import Dict, Any
from py_classes.cls_util_base import UtilBase
from utils.todos import TodosUtil

class EditFile(UtilBase):
    """
    A robust utility for creating or modifying files by specifying line ranges.
    This tool is the preferred method for all file writing and modification operations.
    """
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["edit file", "modify file", "change file", "write file", "update file", "create file", "file modification", "code editing", "text editing", "file writer", "save file", "file changes", "refactor code", "fix code", "update code", "file operations"],
            "use_cases": [
                "Edit a Python script to fix a bug.",
                "Modify a configuration file to change settings.",
                "Create a new file with specific content.",
                "Update source code with improvements.",
                "Change text in a documentation file.",
                "Fix syntax errors in existing code.",
                "Add new functions to an existing script."
            ],
            "arguments": {
                "path": "The absolute or relative path to the file to be edited or created."
            },
            "code_examples": [
                {
                    "description": "Edit an existing Python file",
                    "code": """```python
from utils.editfile import EditFile
result = EditFile.run("main.py")
print(result)
```"""
                },
                {
                    "description": "Create or modify a configuration file",
                    "code": """```python
from utils.editfile import EditFile
result = EditFile.run("config.json")
print(result)
```"""
                }
            ]
        }

    @staticmethod
    def _run_logic(
        path: str
    ) -> str:
        """
        Applies an edit to a file.

        Args:
            path (str): The absolute or relative path to the file.

        Returns:
            str: A JSON string with a 'result' key containing a success message and
                 the absolute path of the file, or an 'error' key on failure.
        """
        try:
            # Manually add the 'editfile:' prefix to each task
            TodosUtil._run_logic("add", task=f"editfile: Show all modifications you're suggesting to apply to {path} to the user, include minimal snippets and filepaths.")
            TodosUtil._run_logic("add", task="editfile: Convert the modifications into search and replace operations and add a todo for each find and replace operation to the ToDosUtil")
            TodosUtil._run_logic("add", task="editfile: Use your native python interpreter to carefully perform the search and replace operations, always verify your applied changes are correct with character-level precision.")
            
            return json.dumps(TodosUtil._run_logic("list"), indent=2)

        except Exception as e:
            error_result = {"error": f"Could not write to file {path}. Reason: {e}"}
            return json.dumps(error_result, indent=2)


# Module-level run function for CLI-Agent compatibility
def run(path: str) -> str:
    """
    Module-level wrapper for EditFile._run_logic() to maintain compatibility with CLI-Agent.
    
    Args:
        path (str): The absolute or relative path to the file to be edited or created
        
    Returns:
        str: JSON string with result or error
    """
    return EditFile._run_logic(path=path)
