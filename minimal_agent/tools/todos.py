# tools/todos.py
"""
Tool for managing a persistent task list.
"""
import os

class todos:
    @staticmethod
    def get_delim() -> str:
        """Delimiter for this tool."""
        return 'todos'

    @staticmethod
    def get_tool_info() -> dict:
        """Tool documentation for the agent's system prompt."""
        return {
            "name": "todos",
            "description": "Manages a persistent task list. Use it to break down complex goals and track your progress. Provide the complete, updated list every time.",
            "example": "<todos>\n- [x] Step 1: List files.\n- [ ] Step 2: Read the relevant file.\n- [ ] Step 3: Analyze its content.\n</todos>"
        }

    @staticmethod
    def _get_storage_path() -> str:
        """Returns the full path to the todos markdown file."""
        storage_dir = os.path.expanduser("~/.minimal-agent")
        os.makedirs(storage_dir, exist_ok=True)
        return os.path.join(storage_dir, "todos.md")

    @staticmethod
    def run(content: str) -> str:
        """Saves the provided todo list content to a persistent file."""
        try:
            storage_path = todos._get_storage_path()
            with open(storage_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return "Todos list updated successfully."
        except OSError as e:
            return f"Error saving todos: {str(e)}"
