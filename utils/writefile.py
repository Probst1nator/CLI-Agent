import os
import json
from py_classes.cls_util_base import UtilBase

class WriteFile(UtilBase):
    """
    A robust utility for creating or modifying files by specifying line ranges.
    This tool is the preferred method for all file writing and modification operations.
    """

    @staticmethod
    def run(
        path: str,
        content: str,
        start_line: int = 0,
        end_line: int = -1,
        create_dirs: bool = True
    ) -> str:
        """
        Writes or replaces content in a file within a specified line range.
        If the file does not exist, it will be created.

        Args:
            path (str): The absolute or relative path to the file.
            content (str): The content to write to the file.
            start_line (int): The 0-indexed line number at which to start the insertion/replacement.
                              To append, set this to a value greater than or equal to the file's line count.
            end_line (int): The 0-indexed line number at which to end the replacement (exclusive).
                            A value of -1 signifies replacing until the end of the file.
                            To insert content without deleting, set end_line equal to start_line.
            create_dirs (bool): If True, automatically creates parent directories for the file.

        Returns:
            str: A JSON string with a 'result' key containing a success message and
                 the absolute path of the file, or an 'error' key on failure.
        """
        try:
            if create_dirs:
                parent_dir = os.path.dirname(path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)

            existing_lines = []
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    existing_lines = f.readlines()

            # Prepare the new content to be inserted
            content_lines = content.splitlines(True)

            # Normalize start_line for appending
            if start_line > len(existing_lines):
                start_line = len(existing_lines)

            # Determine the lines that will come after the inserted content
            lines_after = []
            if end_line != -1 and end_line < len(existing_lines):
                lines_after = existing_lines[end_line:]

            # Combine the parts: lines before + new content + lines after
            updated_content = existing_lines[:start_line] + content_lines + lines_after

            # Write the modified content back to the file
            with open(path, 'w', encoding='utf-8') as f:
                f.writelines(updated_content)

            abs_path = os.path.abspath(path)
            result = {
                "result": {
                    "status": "Success",
                    "message": f"Successfully modified file at path: {abs_path}",
                    "path": abs_path
                }
            }
            return json.dumps(result, indent=2)

        except Exception as e:
            error_result = {"error": f"Could not write to file {path}. Reason: {e}"}
            return json.dumps(error_result, indent=2)