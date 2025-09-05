# tools/primitives/todos.py
"""
This file implements the 'todos' tool. It allows the agent to maintain a
persistent checklist to track its progress on a task.
"""
import os
from typing import Optional
from core.globals import g

class todos:
    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool."""
        return 'todos'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool."""
        return {
            "name": "todos",
            "description": "Manages a persistent task list. Use it to break down complex goals and track your progress. Provide the complete, updated list every time.",
            "example": "<todos>\n- [x] Step 1: List files.\n- [ ] Step 2: Read the relevant file.\n- [ ] Step 3: Analyze its content.\n</todos>"
        }

    @staticmethod
    def _get_storage_path() -> str:
        """Returns the full path to the dedicated todos markdown file."""
        return os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, "todos.md")

    @staticmethod
    def run(content: str) -> str:
        """Saves the provided todo list content to a persistent file."""
        try:
            storage_path = todos._get_storage_path()
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            with open(storage_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return "Todos list updated successfully."
        except IOError as e:
            return f"Error saving todos: {str(e)}"

    @staticmethod
    def get_next_unchecked_task() -> Optional[str]:
        """
        Reads the todos file and returns the first unchecked task.
        An unchecked task is a line that starts with "- [ ]".
        """
        try:
            storage_path = todos._get_storage_path()
            if not os.path.exists(storage_path):
                return None

            with open(storage_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line.startswith("- [ ]"):
                        # Remove the checkbox prefix to get the task text.
                        return stripped_line[len("- [ ]"):].strip()
            return None # No unchecked tasks were found
        except IOError as e:
            # In a real application, you might want to log this error
            return f"Error reading todos: {str(e)}"