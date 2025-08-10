# utils/todos.py
import re
import os
from typing import Literal, Optional, List, Dict, Any

from py_classes.cls_util_base import UtilBase
# Import the globals to access the persistent storage path
from py_classes.globals import g

class TodosUtil(UtilBase):
    """
    A utility for managing a to-do list.
    It allows an agent to list, add, complete, uncomplete, edit, remove, reorder, and clear tasks.
    The to-do list is stored persistently in '.cliagent/todos.md'.
    The information returned by actions is minimal, focusing on confirmation and key stats.
    """

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["task list", "todo", "to-do", "manage tasks", "checklist", "add task", "complete task", "goals", "objectives", "workflow steps", "pipeline stages", "deployment steps", "installation process", "setup checklist", "project milestones", "track progress", "next steps", "action items"],
            "use_cases": [
                "Add 'write documentation' to my to-do list.",
                "What are my current tasks?",
                "Mark the first task as complete.",
                "Clear my entire to-do list.",
                "Track the steps for deploying this model to production.",
                "Create a checklist for setting up the development environment.",
                "Add the Docker container setup steps to my task list.",
                "What's the next step in this complex workflow?"
            ],
            "arguments": {
                "action": "The operation to perform (e.g., 'list', 'add', 'complete').",
                "index": "The 1-based index of the task to act upon.",
                "task": "The text of the task to add or edit."
            },
            "code_examples": [
                {
                    "description": "Add a new task to the todo list. This prints the result and returns None.",
                    "code": """```python
from utils.todos import TodosUtil
TodosUtil.run("add", task="Review the documentation")
```"""
                },
                {
                    "description": "List all current todos. This prints the list and returns None.",
                    "code": """```python
from utils.todos import TodosUtil
TodosUtil.run("list")
```"""
                },
                {
                    "description": "Mark a task as complete. This prints the result and returns None.",
                    "code": """```python
from utils.todos import TodosUtil
TodosUtil.run("complete", index=1)
```"""
                }
            ]
        }
    
    @staticmethod
    def _get_storage_path() -> str:
        """Returns the full path to the dedicated todos markdown file."""
        return os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, "todos.md")

    @staticmethod
    def _load_todos() -> List[Dict[str, Any]]:
        """
        Loads the to-do list from the dedicated .cliagent/todos.md file.
        Returns an empty list if the file does not exist.
        """
        storage_path = TodosUtil._get_storage_path()
        if not os.path.exists(storage_path):
            return []
        
        todos = []
        try:
            with open(storage_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_stripped = line.strip()
                    # Parse markdown todo items: "- [x] Task" or "- [ ] Task"
                    todo_match = re.match(r'^- \[([ x])\] (.+)$', line_stripped)
                    if todo_match:
                        completed = todo_match.group(1) == 'x'
                        task = todo_match.group(2)
                        todos.append({"task": task, "completed": completed})
            return todos
        except IOError:
            return []

    @staticmethod
    def _save_todos(todos: List[Dict[str, Any]]) -> None:
        """
        Saves the to-do list by overwriting the dedicated .cliagent/todos.md file.
        """
        storage_path = TodosUtil._get_storage_path()
        
        # Build the new file content
        new_todos_lines = ["# Todos"]
        for todo in todos:
            completed_marker = "x" if todo.get("completed", False) else " "
            task = todo.get("task", "")
            new_todos_lines.append(f"- [{completed_marker}] {task}")
        
        new_content = "\n".join(new_todos_lines)

        try:
            # Ensure the directory exists before writing
            os.makedirs(os.path.dirname(storage_path), exist_ok=True)
            with open(storage_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except IOError as e:
            # In a real-world scenario, you might want to log this error
            print(f"Error saving todos to {storage_path}: {e}")

    @staticmethod
    def get_list_as_str() -> str:
        """
        Returns the formatted to-do list as a string, without printing it.
        This is intended for programmatic use where the output needs to be captured.
        """
        # Load the current todos from the file
        todos = TodosUtil._load_todos()
        # Use the existing formatting logic and return the result
        return TodosUtil._format_todos_md(todos)

    @staticmethod
    def _format_todos_md(todos: List[Dict[str, Any]], header: str = "### Current To-Do List (TodosUtil)") -> str:
        """
        Formats the to-do list into a clean, human-readable markdown string.
        """
        if not todos:
            return "Your to-do list is empty."
        
        formatted_list = [header]
        for i, item in enumerate(todos, 1):
            status_icon = "[x]" if item.get('completed', False) else "[ ]"
            task_description = item.get('task', 'No task description')
            formatted_list.append(f"{i}. {status_icon} {task_description}")
        return "\n".join(formatted_list)

    @staticmethod
    def _run_logic(
        action: Literal['list', 'add', 'complete', 'uncomplete', 'edit', 'remove', 'reorder', 'clear'],
        index: Optional[int] = None,
        task: Optional[str] = None,
        new_index: Optional[int] = None
    ) -> str:
        """
        Manages the to-do list through various actions and returns a confirmation string.
        """
        try:
            todos = TodosUtil._load_todos()

            if action == 'list':
                return TodosUtil._format_todos_md(todos)

            elif action == 'add':
                if not task or not task.strip():
                    return "**Error:** 'add' action requires a non-empty 'task' argument."

                if any(existing_todo.get('task') == task for existing_todo in todos):
                    existing_todo = next(t for t in todos if t.get('task') == task)
                    todos.remove(existing_todo)
                    todos.append(existing_todo)
                    message = f"Task already existed. Moved to the end (task {len(todos)})."
                else:
                    todos.append({"task": task, "completed": False})
                    message = f"Task added at index {len(todos)}."
                
                TodosUtil._save_todos(todos)
                remaining_count = sum(1 for t in todos if not t.get('completed', False))
                return f"**Success:** {message} Total todos: {len(todos)} ({remaining_count} remaining)."

            elif action == 'clear':
                count = len(todos)
                if count == 0:
                    return "The to-do list is already empty."
                TodosUtil._save_todos([])
                return f"**Success:** Cleared all {count} tasks."

            if index is None:
                return f"**Error:** The '{action}' action requires an 'index' argument."
            
            if not isinstance(index, int) or not (1 <= index <= len(todos)):
                return f"**Error:** Invalid index: {index}. Must be an integer between 1 and {len(todos)}."
            
            idx_0_based = index - 1

            if action in ['complete', 'uncomplete']:
                is_completing = action == 'complete'
                current_task = todos[idx_0_based]
                status_text = "completed" if is_completing else "uncompleted"
                
                if current_task['completed'] == is_completing:
                    return f"**Info:** Task {index} is already {status_text}."
                
                current_task['completed'] = is_completing
                TodosUtil._save_todos(todos)
                remaining_count = sum(1 for t in todos if not t.get('completed', False))
                return f"**Success:** Marked task {index} as {status_text}. {remaining_count} tasks remaining."

            elif action == 'edit':
                if not task or not task.strip():
                    return "**Error:** The 'edit' action requires a non-empty 'task' argument."
                todos[idx_0_based]['task'] = task
                TodosUtil._save_todos(todos)
                return f"**Success:** Edited task {index}. Total todos: {len(todos)}."

            elif action == 'remove':
                todos.pop(idx_0_based)
                TodosUtil._save_todos(todos)
                remaining_count = sum(1 for t in todos if not t.get('completed', False))
                return f"**Success:** Removed task {index}. Total todos: {len(todos)} ({remaining_count} remaining)."

            elif action == 'reorder':
                if new_index is None:
                    return "**Error:** The 'reorder' action requires a 'new_index' argument."
                if not isinstance(new_index, int) or not (1 <= new_index <= len(todos)):
                     return f"**Error:** Invalid new_index: {new_index}. Must be an integer between 1 and {len(todos)}."
                
                new_idx_0_based = new_index - 1
                moved_task = todos.pop(idx_0_based)
                todos.insert(new_idx_0_based, moved_task)
                TodosUtil._save_todos(todos)
                return f"**Success:** Moved task from position {index} to {new_index}. Total todos: {len(todos)}."
            
            else:
                return f"**Error:** Unknown action: '{action}'."

        except Exception as e:
            return f"**Error:** An unexpected error occurred in TodosUtil: {str(e)}"
            
    @staticmethod
    def run(
        action: Literal['list', 'add', 'complete', 'uncomplete', 'edit', 'remove', 'reorder', 'clear'],
        index: Optional[int] = None,
        task: Optional[str] = None,
        new_index: Optional[int] = None
    ) -> None:
        """
        Executes a to-do list action.
        This method prints the outcome of the action directly to the console and always returns None.
        """
        result_string = TodosUtil._run_logic(action=action, index=index, task=task, new_index=new_index)
        if result_string:
            print(result_string)


# Module-level run function for CLI-Agent compatibility
def run(action: Literal['list', 'add', 'complete', 'uncomplete', 'edit', 'remove', 'reorder', 'clear'], index: Optional[int] = None, task: Optional[str] = None, new_index: Optional[int] = None) -> None:
    """
    Module-level wrapper for TodosUtil.run() to maintain compatibility with CLI-Agent.
    This function prints the output directly and always returns None.
    """
    TodosUtil.run(action=action, index=index, task=task, new_index=new_index)