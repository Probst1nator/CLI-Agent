"""
A utility for editing files with various modes: overwrite, search-replace, and line-replace.
"""
import os
from typing import Dict, Any, Literal, Optional
import markpickle

# Assuming UtilBase is a simple base class, we can define a mock for stand-alone use
class UtilBase:
    @staticmethod
    def run(*args, **kwargs):
        # This will now call the refactored run function with explicit arguments
        return EditFile._run_logic(*args, **kwargs)

class EditFile(UtilBase):
    """
    A robust utility for creating or modifying files with multiple edit modes:
    1. overwrite: Replaces the entire file content.
    2. search_replace: Finds and replaces text occurrences.
    3. line_replace: Replaces a specific range of lines.

    This tool is the preferred method for all programmatic file writing and modification.
    """

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["edit file", "modify file", "change file", "write file", "update file", "create file", "file modification", "code editing", "text editing", "file writer", "save file", "file changes", "refactor code", "fix code", "update code", "file operations", "search and replace", "overwrite file"],
            "use_cases": [
                "Atomically replace the entire content of a configuration file.",
                "Perform a search-and-replace to rename a variable across a script.",
                "Insert a new function into a file at a specific line range.",
                "Create a new file from scratch with initial content.",
                "Remove a block of deprecated code by replacing it with an empty string.",
                "Correct a typo in a documentation file."
            ],
            # --- MODIFIED: Arguments are now explicit ---
            "arguments": {
                "path": "The absolute or relative path to the file to be edited or created.",
                "edit_mode": "The operation to perform. Must be one of: 'overwrite', 'search_replace', 'line_replace'.",
                "new_content": "The new content for the file. Required for 'overwrite' mode; used by 'line_replace' mode.",
                "search_string": "The text to find. Required for 'search_replace' mode.",
                "replace_with": "The text to replace with. Required for 'search_replace' mode.",
                "start_line": "The starting line number (1-based, inclusive). Required for 'line_replace' mode.",
                "end_line": "The ending line number (1-based, inclusive). Required for 'line_replace' mode."
            },
            "code_examples": [
                {
                    "description": "Example 1: Overwrite a file with new content (creates the file if it doesn't exist).",
                    "code": """```python
from utils.editfile import EditFile
result = EditFile.run(
    path="config.json",
    edit_mode="overwrite",
    new_content='{"setting": "value", "enabled": true}'
)
print(result)
```"""
                },
                {
                    "description": "Example 2: Search and replace a string in a Python file.",
                    "code": """```python
from utils.editfile import EditFile
result = EditFile.run(
    path="main.py",
    edit_mode="search_replace",
    search_string="old_function_name",
    replace_with="new_function_name"
)
print(result)
```"""
                },
                {
                    "description": "Example 3: Replace lines 10 through 15 in a file with a new function.",
                    "code": """```python
from utils.editfile import EditFile
new_code = 'def new_feature():\\n    print("Hello from the new feature!")'
result = EditFile.run(
    path="script.py",
    edit_mode="line_replace",
    start_line=10,
    end_line=15,
    new_content=new_code
)
print(result)
```"""
                }
            ]
        }

    @staticmethod
    def _run_logic(
        path: str,
        edit_mode: Literal["overwrite", "search_replace", "line_replace"],
        # --- MODIFIED: Replaced **kwargs with explicit optional arguments ---
        new_content: Optional[str] = None,
        search_string: Optional[str] = None,
        replace_with: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> str:
        """
        Applies a specified edit to a file.

        Args:
            path (str): The path to the file.
            edit_mode (str): The mode of editing ('overwrite', 'search_replace', 'line_replace').
            new_content (Optional[str]): The content for 'overwrite' or 'line_replace'.
            search_string (Optional[str]): The search term for 'search_replace'.
            replace_with (Optional[str]): The replacement text for 'search_replace'.
            start_line (Optional[int]): The start line for 'line_replace'.
            end_line (Optional[int]): The end line for 'line_replace'.

        Returns:
            str: A Markdown string with a success or error message.
        """
        try:
            abs_path = os.path.abspath(path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            # --- Read current content (if needed) ---
            original_content = ""
            if os.path.exists(abs_path) and edit_mode != "overwrite":
                with open(abs_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()

            # --- Process based on edit_mode ---
            if edit_mode == "overwrite":
                # --- MODIFIED: Directly use the named argument ---
                if new_content is None:
                    raise ValueError("'new_content' is required for 'overwrite' mode.")

                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                message = f"Successfully overwrote file at '{abs_path}'."

            elif edit_mode == "search_replace":
                # --- MODIFIED: Directly use the named arguments ---
                if search_string is None or replace_with is None:
                    raise ValueError("'search_string' and 'replace_with' are required for 'search_replace' mode.")

                if not os.path.exists(abs_path):
                    raise FileNotFoundError(f"File not found at '{abs_path}'. Cannot perform search and replace.")

                new_file_content = original_content.replace(search_string, replace_with)

                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(new_file_content)
                message = f"Successfully performed search and replace in '{abs_path}'."

            elif edit_mode == "line_replace":
                # --- MODIFIED: Directly use the named arguments ---
                if start_line is None or end_line is None:
                    raise ValueError("'start_line' and 'end_line' are required for 'line_replace' mode.")
                if not isinstance(start_line, int) or not isinstance(end_line, int) or start_line <= 0 or end_line < start_line:
                    raise ValueError("Invalid line numbers. 'start_line' must be > 0 and 'end_line' >= 'start_line'.")

                if not os.path.exists(abs_path):
                    raise FileNotFoundError(f"File not found at '{abs_path}'. Cannot perform line replacement.")

                # Ensure new_content is a string, even if None was passed
                content_for_replacement = new_content if new_content is not None else ""
                lines = original_content.splitlines(True) # keepends=True

                # Adjust for 0-based indexing
                start_idx = start_line - 1
                end_idx = end_line

                if start_idx > len(lines):
                    raise ValueError(f"Invalid 'start_line': {start_line} is beyond the end of the file ({len(lines)} lines).")

                # Construct the new file content
                pre_lines = lines[:start_idx]
                post_lines = lines[end_idx:]

                # Add a newline if the content to insert doesn't end with one
                if content_for_replacement and not content_for_replacement.endswith('\n'):
                    content_for_replacement += '\n'

                final_content = "".join(pre_lines) + content_for_replacement + "".join(post_lines)

                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(final_content)
                message = f"Successfully replaced lines {start_line}-{end_line} in '{abs_path}'."

            else:
                raise ValueError(f"Invalid 'edit_mode': {edit_mode}. Must be one of 'overwrite', 'search_replace', 'line_replace'.")

            success_result = {"result": message, "path": abs_path}
            return markpickle.dumps(success_result)

        except (ValueError, FileNotFoundError, IOError, Exception) as e:
            error_result = {"error": f"Failed to edit file '{path}'. Reason: {e}"}
            return markpickle.dumps(error_result)


# --- MODIFIED: Module-level run function signature now matches _run_logic ---
def run(
    path: str,
    edit_mode: Literal["overwrite", "search_replace", "line_replace"],
    new_content: Optional[str] = None,
    search_string: Optional[str] = None,
    replace_with: Optional[str] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> str:
    """
    Module-level wrapper for EditFile._run_logic() to maintain compatibility with CLI-Agent.

    Args:
        path (str): The path to the file.
        edit_mode (str): The mode of editing ('overwrite', 'search_replace', 'line_replace').
        new_content (Optional[str]): The content for 'overwrite' or 'line_replace'.
        search_string (Optional[str]): The search term for 'search_replace'.
        replace_with (Optional[str]): The replacement text for 'search_replace'.
        start_line (Optional[int]): The start line for 'line_replace'.
        end_line (Optional[int]): The end line for 'line_replace'.

    Returns:
        str: Markdown string with result or error
    """
    return EditFile._run_logic(
        path=path,
        edit_mode=edit_mode,
        new_content=new_content,
        search_string=search_string,
        replace_with=replace_with,
        start_line=start_line,
        end_line=end_line
    )
