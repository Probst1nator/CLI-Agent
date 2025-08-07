import os
import json

from py_classes.cls_util_base import UtilBase

class ViewFile(UtilBase):
    """
    A utility to read the content of a file.
    This is the preferred method for viewing file contents.
    """

    @staticmethod
    def run(path: str, max_chars: int = 8000) -> str:
        """
        Reads the content of a file and returns it as a string.

        Args:
            path (str): The absolute or relative path to the file to be read.
            max_chars (int): The maximum number of characters to read from the start of the file.
                             Defaults to 8000.

        Returns:
            str: A JSON string with a 'result' key containing the file content,
                 or an 'error' key on failure.
        """
        try:
            if not os.path.exists(path):
                return json.dumps({"error": f"File not found: {path}"}, indent=2)

            if not os.path.isfile(path):
                return json.dumps({"error": f"Path is not a file: {path}"}, indent=2)

            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_chars)
            
            abs_path = os.path.abspath(path)
            file_size = os.path.getsize(path)
            truncated = file_size > max_chars
            
            message = f"Successfully read content from {abs_path}."
            if truncated:
                message += f" Truncated to first {max_chars} chars (total size: {file_size} bytes)."

            result = {
                "result": {
                    "status": "Success",
                    "message": message,
                    "path": abs_path,
                    "content": content,
                    "truncated": truncated
                }
            }
            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Failed to read file '{path}': {str(e)}"}, indent=2)