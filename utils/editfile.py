import os
from typing import Literal, Optional
from py_classes.cls_chat import Chat

class EditFile:
    """
    A utility for performing various editing operations on a file.
    """

    @staticmethod
    async def run(
        filepath: Optional[str] = None,
        
    ) -> str:
        """
        Creates or edits a file and can access the current context chat

        Args:
            filepath: The path to the file to edit

        Returns:
            A result message
        """
        context_chat = Chat.load_from_json()
        
        if not filepath:
            return "Error: filepath is required"
        
        content = ""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    content = f.read()                
            except Exception as e:
                return f"Error reading file {filepath}: {e}"

        edit_main = context_chat.deep_copy()