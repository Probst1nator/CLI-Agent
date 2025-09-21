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
            "description": "Finds and replaces ALL occurrences of a string in a specified file, or edits specific lines. Supports line-specific editing with <line> tag. Paths are resolved relative to the current working directory. Requires `<filepath>`, and either `<find>`+`<replace>` tags OR `<line>`+`<content>` tags.",
            "example": "<editfile><filepath>./src/app.js</filepath><search>const old_variable = 1;</search><replace>const new_variable = 2;</replace></editfile> or <editfile><filepath>./src/main.py</filepath><line>15</line><replace>print('new line')</replace></editfile>"
        }

    @staticmethod
    def run(content: str) -> str:
        """
        Parses XML-like input to find and replace text in a file, or edit specific lines.
        Supports:
        1. search/replace: replaces all occurrences of the find string
        2. Line-specific: replaces content of a specific line number
        """
        try:
            filepath_match = re.search(r'<filepath>(.*?)</filepath>', content, re.DOTALL)
            search_match = re.search(r'<search>(.*?)</search>', content, re.DOTALL)
            line_match = re.search(r'<line>(.*?)</line>', content, re.DOTALL)
            replace_match = re.search(r'<replace>(.*?)</replace>', content, re.DOTALL)

            if not filepath_match:
                return "Error: Invalid format. Input must contain <filepath> tag."

            # Check for line-specific editing mode
            if line_match and replace_match:
                return editfile._edit_specific_line(filepath_match, line_match, replace_match)
            elif search_match and replace_match:
                return editfile._find_and_replace(filepath_match, search_match, replace_match)
            else:
                return "Error: Invalid format. Input must contain either (<find> and <replace>) OR (<line> and <content>) tags."

        except Exception as e:
            return f"Error executing editfile tool: {str(e)}"

    @staticmethod
    def _find_and_replace(filepath_match, find_match, replace_match):
        """Handle traditional find-and-replace editing."""
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

    @staticmethod
    def _edit_specific_line(filepath_match, line_match, content_match):
        """Handle line-specific editing."""
        filepath = filepath_match.group(1).strip()
        line_str = line_match.group(1).strip()
        new_content = content_match.group(1)

        if not filepath:
            return "Error: The <filepath> tag cannot be empty."

        try:
            line_number = int(line_str)
            if line_number < 1:
                return "Error: Line number must be 1 or greater."
        except ValueError:
            return f"Error: Invalid line number '{line_str}'. Must be an integer."

        # Resolve the path relative to the current working directory
        full_path = os.path.abspath(filepath)

        if not os.path.isfile(full_path):
            return f"Error: File not found at '{full_path}' (resolved from '{filepath}')"

        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        total_lines = len(lines)
        if line_number > total_lines:
            return f"Error: Line {line_number} exceeds file length ({total_lines} lines). Use insertline tool to add new lines."

        # Replace the specific line (convert to 0-based indexing)
        old_line = lines[line_number - 1].rstrip('\n\r')
        lines[line_number - 1] = new_content + '\n'

        with open(full_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return f"Successfully edited line {line_number} in {full_path}:\nOld: {old_line}\nNew: {new_content}"