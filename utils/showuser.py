import json
from py_classes.cls_util_base import UtilBase
from utils.todos import TodosUtil

class ShowUser(UtilBase):
    """
    A utility showing the user a html site optionally with images and animations
    """

    @staticmethod
    def run(
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
            
            return json.dumps(TodosUtil.run("list"), indent=2)

        except Exception as e:
            error_result = {"error": f"Could not write to file {path}. Reason: {e}"}
            return json.dumps(error_result, indent=2)