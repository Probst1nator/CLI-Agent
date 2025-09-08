# utils/todos.py
import re
import os
from typing import Literal, Optional, List, Dict, Any

from agent.utils_manager.util_base import UtilBase
# Import the globals to access the persistent storage path
from core.globals import g

class TodosUtil(UtilBase):
    """
    A utility for managing a to-do list with PRINT-ONLY behavior.
    
    CRITICAL: The run() method ALWAYS returns None and prints output to console.
    DO NOT attempt to capture return values from run() - use get_list_as_str() for data access.
    
    Features:
    - Persistent storage in '.cliagent/todos.md'
    - Actions: list, add, complete, uncomplete, edit, remove, reorder, clear
    - Print-based feedback with minimal confirmation messages
    
    Data Access Patterns:
    - For side effects: TodosUtil.run(action, ...) -> None (prints to console)
    - For data retrieval: TodosUtil.get_list_as_str() -> str (returns formatted list)
    - For raw data: TodosUtil._load_todos() -> List[Dict] (internal use)
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
                    "description": "WRONG: Don't assign run() output - it always returns None!",
                    "code": """<python>
# ❌ INCORRECT - This assigns None!
from utils.todos import TodosUtil
todos = TodosUtil.run("list")  # todos will be None!
print(todos)  # prints: None
</python>"""
                },
                {
                    "description": "CORRECT: Use run() for side effects (printing), get_list_as_str() for data",
                    "code": """<python>
# ✅ CORRECT - Side effect usage
from utils.todos import TodosUtil
TodosUtil.run("add", task="Review the documentation")  # Prints confirmation
TodosUtil.run("list")  # Prints formatted todo list

# ✅ CORRECT - Data retrieval usage
todos_text = TodosUtil.get_list_as_str()  # Returns string
print(f"Current todos: {todos_text}")
</python>"""
                },
                {
                    "description": "Complete a task - prints confirmation, returns None",
                    "code": """<python>
from utils.todos import TodosUtil
# This prints success message and returns None
TodosUtil.run("complete", index=1)
</python>"""
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
        DATA ACCESS METHOD: Returns formatted todo list as string (does not print).
        
        This is the CORRECT way to programmatically access todo list data.
        Unlike run(), this method returns actual data instead of None.
        
        Use this when you need to:
        - Capture todo list content in a variable
        - Process or analyze the todo list
        - Include todo list in other output
        - Check todo list status programmatically
        
        Example:
            todos_text = TodosUtil.get_list_as_str()  # Returns string
            if "empty" in todos_text:
                print("No todos found")
        
        Returns:
            str: Formatted markdown-style todo list or "Your to-do list is empty."
        """
        # Load the current todos from the file
        todos = TodosUtil._load_todos()
        # Use the existing formatting logic and return the result
        return TodosUtil._format_todos_md(todos)

    @staticmethod
    def _format_todos_md(todos: List[Dict[str, Any]], header: str = "### Current To-Do List (TodosUtil)") -> str:
        """
        Internal formatter: Converts todo data to human-readable markdown string.
        
        This method RETURNS formatted strings (does not print).
        Used internally by both run() and get_list_as_str() methods.
        
        Args:
            todos: List of todo dictionaries
            header: Optional header text for the formatted output
            
        Returns:
            str: Formatted markdown with numbered tasks or empty message
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
        Internal method that performs todo operations and RETURNS formatted strings.
        
        This method does the actual work and returns strings for:
        - Success/error messages
        - Formatted todo lists
        - Status confirmations
        
        Used by run() (which prints the result) and potentially by other methods
        that need the string output without printing.
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
        PRINT-ONLY todo action executor - ALWAYS returns None!
        
        WARNING: This method has side effects (prints to console) but NO return value.
        Do NOT attempt to capture its output with assignment:
        
        ❌ WRONG: todos = TodosUtil.run("list")  # todos will be None!
        ✅ CORRECT: TodosUtil.run("list")         # Prints list to console
        
        For programmatic access to todo data, use:
        - get_list_as_str() -> returns formatted string
        - _load_todos() -> returns raw data (internal)
        
        Args:
            action: Operation to perform
            index: 1-based task index (when required)
            task: Task description (for add/edit)
            new_index: New position (for reorder)
            
        Returns:
            None (always! output goes to console via print())
        """
        result_string = TodosUtil._run_logic(action=action, index=index, task=task, new_index=new_index)
        if result_string:
            print(result_string)


# Module-level run function for CLI-Agent compatibility
def run(action: Literal['list', 'add', 'complete', 'uncomplete', 'edit', 'remove', 'reorder', 'clear'], index: Optional[int] = None, task: Optional[str] = None, new_index: Optional[int] = None) -> None:
    """
    Module-level PRINT-ONLY wrapper - ALWAYS returns None!
    
    This is a direct wrapper around TodosUtil.run() for CLI-Agent compatibility.
    
    CRITICAL: Like TodosUtil.run(), this function:
    - Prints output to console
    - Always returns None
    - Should NOT be used for data capture
    
    Usage:
        ❌ WRONG: result = run("list")  # result will be None!
        ✅ CORRECT: run("list")         # Prints to console
        
    For data access, use: TodosUtil.get_list_as_str()
    """
    TodosUtil.run(action=action, index=index, task=task, new_index=new_index)