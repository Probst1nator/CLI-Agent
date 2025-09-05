# tools/editfile.py
"""
This file implements the 'editfile' tool. It allows the agent to perform a
find-and-replace operation on a file, resolving paths relative to the current
working directory.
"""
import os
import re

class editfile:
    @staticmethod
    def get_delim() -> str:
        return 'editfile'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "editfile",
            "description": "Finds and replaces ALL occurrences of a string in a specified file. Paths are resolved relative to the current working directory. Requires `<filepath>`, `<find>`, and `<replace>` tags.",
            "example": "<editfile><filepath>./src/app.js</filepath><find>const old_variable = 1;</find><replace>const new_variable = 2;</replace></editfile>"
        }

    @staticmethod
    def run(content: str) -> str:
        """
        Parses XML-like input to find and replace text in a file.
        This operation replaces all occurrences of the find string.
        """
        try:
            filepath_match = re.search(r'<filepath>(.*?)</filepath>', content, re.DOTALL)
            find_match = re.search(r'<find>(.*?)</find>', content, re.DOTALL)
            replace_match = re.search(r'<replace>(.*?)</replace>', content, re.DOTALL)

            if not all([filepath_match, find_match, replace_match]):
                return "Error: Invalid format. Input must contain <filepath>, <find>, and <replace> tags."

            filepath = filepath_match.group(1).strip()
            find_str = find_match.group(1)
            replace_str = replace_match.group(1)

            if not filepath:
                return "Error: The <filepath> tag cannot be empty."

            # Resolve the path relative to the current working directory for safety and predictability.
            full_path = os.path.abspath(filepath)

            if not os.path.isfile(full_path):
                return f"Error: File not found at '{full_path}' (resolved from '{filepath}')"

            with open(full_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            if find_str not in original_content:
                return f"Error: The string to find was not found in the file '{full_path}'. No changes were made."

            # Note: str.replace() replaces all occurrences. To replace only the first,
            # use: new_content = original_content.replace(find_str, replace_str, 1)
            new_content = original_content.replace(find_str, replace_str)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return f"Successfully edited file: {full_path}"

        except Exception as e:
            return f"Error executing editfile tool: {str(e)}"