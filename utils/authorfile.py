import os
from py_classes.cls_util_base import UtilBase
from typing import Literal

class AuthorFile(UtilBase):
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
            mode (Literal['w', 'a']): The write mode.
                'w' for write/overwrite (default).
                'a' for append.
            create_dirs (bool): If True, automatically creates parent directories.

        Returns:
            str: A confirmation message indicating success or an error string.
        """
        try:
            if create_dirs:
                # Ensure the parent directory exists
                parent_dir = os.path.dirname(path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)

            # Write the content to the file using the specified mode
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)

            action = "Appended to" if mode == 'a' else "Successfully wrote to"
            return f"{action} file: {os.path.abspath(path)}"

        except Exception as e:
            return f"Error: Could not write to file {path}. Reason: {e}" 