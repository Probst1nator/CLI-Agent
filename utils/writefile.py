import os
import json
from py_classes.cls_util_base import UtilBase
from typing import Literal

class WriteFile(UtilBase):
    """
    A robust utility for creating, overwriting, or appending to files.
    This tool is the preferred method for all file writing operations.
    """

    @staticmethod
    def run(
        path: str,
        content: str,
        mode: Literal['w', 'a'] = 'w',
        create_dirs: bool = True
    ) -> str:
        """
        Writes or appends content to a file.

        Args:
            path (str): The absolute or relative path to the file.
            content (str): The content to write to the file.
            mode (Literal['w', 'a']): 'w' for write/overwrite, 'a' for append.
            create_dirs (bool): If True, automatically creates parent directories.

        Returns:
            str: A JSON string with a 'result' key containing the absolute path on success,
                 or an 'error' key on failure.
        """
        try:
            if create_dirs:
                parent_dir = os.path.dirname(path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)

            with open(path, mode, encoding='utf-8') as f:
                f.write(content)

            abs_path = os.path.abspath(path)
            action = "Appended to" if mode == 'a' else "Wrote to"
            
            result = {
                "result": {
                    "status": "Success",
                    "message": f"{action} file successfully.",
                    "path": abs_path
                }
            }
            return json.dumps(result, indent=2)

        except Exception as e:
            error_result = {"error": f"Could not write to file {path}. Reason: {e}"}
            return json.dumps(error_result, indent=2) 