# tools/writefile.py
"""
Tool to create or overwrite files with specified content.
"""
import os
import re


class writefile:
    """Write content to a file, creating it if it doesn't exist."""

    @staticmethod
    def get_delim() -> str:
        """Delimiter for this tool."""
        return 'writefile'

    @staticmethod
    def get_tool_info() -> dict:
        """Tool documentation for the agent's system prompt."""
        return {
            "name": "writefile",
            "description": (
                "Create a new file or overwrite an existing file with specified content. "
                "WARNING: Overwrites existing files without confirmation. Use with caution."
            ),
            "example": (
                "<writefile>\n"
                "<filepath>./src/new_file.py</filepath>\n"
                "<content>\n"
                "def hello():\n"
                "    print('Hello, world!')\n"
                "</content>\n"
                "</writefile>"
            )
        }

    @staticmethod
    def run(content: str) -> str:
        """
        Writes content to a file specified in the input string.

        Args:
            content: XML containing filepath and content tags.

        Returns:
            Success message or error message.
        """
        try:
            # Parse XML
            filepath_match = re.search(r'<filepath>(.*?)</filepath>', content, re.DOTALL)
            content_match = re.search(r'<content>(.*?)</content>', content, re.DOTALL)

            if not filepath_match:
                return "Error: Invalid format. Input must contain a <filepath> tag."

            if not content_match:
                return "Error: Invalid format. Input must contain a <content> tag."

            filepath = filepath_match.group(1).strip()
            file_content = content_match.group(1)

            if not filepath:
                return "Error: Filepath cannot be empty."

            # Resolve the path relative to the current working directory
            full_path = os.path.abspath(filepath)

            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            # Write the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(file_content)

            return f"Successfully wrote {len(file_content)} characters to '{full_path}'"

        except PermissionError:
            return f"Error: Permission denied writing to '{full_path}'"
        except Exception as e:
            return f"Error writing file: {str(e)}"
