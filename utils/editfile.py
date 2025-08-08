import json
from py_classes.cls_util_base import UtilBase
from utils.todos import TodosUtil

class EditFile(UtilBase):
    """
    A robust utility for creating or modifying files by specifying line ranges.
    This tool is the preferred method for all file writing and modification operations.
    """

    @staticmethod
    def run(
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

            TodosUtil.run("add", f"editfile: Show all modifications you*re suggesting to apply to {path} to the user, include minimal snippets and filepaths.")
            TodosUtil.run("add", "editfile: Convert the modifications into search and replace operations and add a todo for each find and replace operation to the ToDosUtil")
            TodosUtil.run("add", "editfile: Use your native python interpreter to carefully perform the search and replace operations, always verify your applied changes are correct with character-level precision.")
            
            return json.dumps(TodosUtil.run("list"), indent=2)

        except Exception as e:
            error_result = {"error": f"Could not write to file {path}. Reason: {e}"}
            return json.dumps(error_result, indent=2)