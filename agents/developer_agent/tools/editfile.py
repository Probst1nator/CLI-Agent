# tools/editfile.py
"""
Tool to edit files using find-and-replace operations.
"""
import os
import re


class editfile:
    """Edit existing files using find-and-replace."""

    @staticmethod
    def get_delim() -> str:
        """Delimiter for this tool."""
        return 'editfile'

    @staticmethod
    def get_tool_info() -> dict:
        """Tool documentation for the agent's system prompt."""
        return {
            "name": "editfile",
            "description": (
                "Edit a file by finding and replacing text. Supports regex patterns. "
                "Can replace first occurrence or all occurrences. "
                "Always verify changes by reading the file after editing."
            ),
            "example": (
                "<editfile>\n"
                "<filepath>./src/app.py</filepath>\n"
                "<find>old_function_name</find>\n"
                "<replace>new_function_name</replace>\n"
                "<all>true</all>\n"
                "</editfile>"
            )
        }

    @staticmethod
    def run(content: str) -> str:
        """
        Edits a file by finding and replacing text.

        Args:
            content: XML containing filepath, find, replace, and optional all/regex tags.

        Returns:
            Success message with replacement count or error message.
        """
        try:
            # Parse XML
            filepath_match = re.search(r'<filepath>(.*?)</filepath>', content, re.DOTALL)
            find_match = re.search(r'<find>(.*?)</find>', content, re.DOTALL)
            replace_match = re.search(r'<replace>(.*?)</replace>', content, re.DOTALL)
            all_match = re.search(r'<all>(.*?)</all>', content, re.DOTALL)
            regex_match = re.search(r'<regex>(.*?)</regex>', content, re.DOTALL)

            if not filepath_match:
                return "Error: Invalid format. Input must contain a <filepath> tag."

            if not find_match:
                return "Error: Invalid format. Input must contain a <find> tag."

            if not replace_match:
                return "Error: Invalid format. Input must contain a <replace> tag."

            filepath = filepath_match.group(1).strip()
            find_text = find_match.group(1)
            replace_text = replace_match.group(1)
            replace_all = all_match.group(1).strip().lower() == 'true' if all_match else False
            use_regex = regex_match.group(1).strip().lower() == 'true' if regex_match else False

            if not filepath:
                return "Error: Filepath cannot be empty."

            if not find_text:
                return "Error: Find text cannot be empty."

            # Resolve the path
            full_path = os.path.abspath(filepath)

            if not os.path.isfile(full_path):
                return f"Error: File not found at '{full_path}'"

            # Read the file
            with open(full_path, 'r', encoding='utf-8') as f:
                original_content = f.read()

            # Perform replacement
            if use_regex:
                try:
                    if replace_all:
                        new_content, count = re.subn(find_text, replace_text, original_content)
                    else:
                        new_content, count = re.subn(find_text, replace_text, original_content, count=1)
                except re.error as e:
                    return f"Error: Invalid regex pattern: {e}"
            else:
                if replace_all:
                    new_content = original_content.replace(find_text, replace_text)
                    count = original_content.count(find_text)
                else:
                    new_content = original_content.replace(find_text, replace_text, 1)
                    count = 1 if find_text in original_content else 0

            if count == 0:
                return f"No occurrences of '{find_text}' found in '{full_path}'"

            # Write the modified content back
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return f"Successfully replaced {count} occurrence(s) in '{full_path}'"

        except PermissionError:
            return f"Error: Permission denied for '{full_path}'"
        except Exception as e:
            return f"Error editing file: {str(e)}"
