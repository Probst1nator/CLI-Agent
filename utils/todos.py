import json
import re
import os
from typing import Literal, Optional, List, Dict, Any

from py_classes.cls_util_base import UtilBase

class TodosUtil(UtilBase):
    """
    A utility for managing a to-do list.
    It allows an agent to list, add, complete, uncomplete, edit, remove, reorder, and clear tasks.
    The to-do list is stored persistently in a dedicated section within the 'QWEN.md' file.
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
            }
        }
    
    # --- Constants for persistent storage in QWEN.md ---
    _STORAGE_FILE = "QWEN.md"
    _START_MARKER = "<!-- TODOS_START -->"
    _END_MARKER = "<!-- TODOS_END -->"

    @staticmethod
    def _load_todos() -> List[Dict[str, Any]]:
        """
        Loads the to-do list from the dedicated JSON block in QWEN.md.
        Returns an empty list if the file or block does not exist, or if JSON is corrupt.
        """
        if not os.path.exists(TodosUtil._STORAGE_FILE):
            return []
        
        try:
            with open(TodosUtil._STORAGE_FILE, 'r', encoding='utf-8') as f:
                content = f.read()

            # Regex to find the JSON content within the markers and a markdown code block
            pattern = re.compile(
                f"{re.escape(TodosUtil._START_MARKER)}\\s*```json\n(.*?)\n```\\s*{re.escape(TodosUtil._END_MARKER)}", 
                re.DOTALL
            )
            match = pattern.search(content)

            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            
            return []  # No to-do block found

        except (IOError, json.JSONDecodeError):
            # Return empty list on read error or if JSON is malformed
            return []

    @staticmethod
    def _save_todos(todos: List[Dict[str, Any]]) -> None:
        """
        Saves the to-do list to the dedicated JSON block in QWEN.md.
        It preserves all other content in the file.
        """
        file_content = ""
        if os.path.exists(TodosUtil._STORAGE_FILE):
            try:
                with open(TodosUtil._STORAGE_FILE, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except IOError:
                pass # Will proceed with empty content if read fails

        # Prepare the new to-do block content
        json_string = json.dumps(todos, indent=2)
        new_block = (
            f"{TodosUtil._START_MARKER}\n"
            f"```json\n{json_string}\n```\n"
            f"{TodosUtil._END_MARKER}"
        )

        # Regex to find an existing block
        pattern = re.compile(
            f"{re.escape(TodosUtil._START_MARKER)}.*?{re.escape(TodosUtil._END_MARKER)}", 
            re.DOTALL
        )

        if pattern.search(file_content):
            # If block exists, replace it
            updated_content = pattern.sub(new_block, file_content)
        else:
            # If no block exists, append it to the end of the file
            updated_content = file_content.strip() + f"\n\n{new_block}\n"
        
        try:
            with open(TodosUtil._STORAGE_FILE, 'w', encoding='utf-8') as f:
                f.write(updated_content)
        except IOError as e:
            # In a real-world scenario, you might want to log this error
            print(f"Error saving todos to {TodosUtil._STORAGE_FILE}: {e}")


    @staticmethod
    def _format_todos(todos: List[Dict[str, Any]]) -> str:
        """Formats the to-do list for a clean, human-readable display."""
        if not todos:
            return "Your to-do list is empty."
        
        formatted_list = ["To-Do List:"]
        for i, item in enumerate(todos, 1):
            status_icon = "[x]" if item.get('completed', False) else "[ ]"
            task_description = item.get('task', 'No task description')
            formatted_list.append(f"{i}. {status_icon} {task_description}")
        return "\n".join(formatted_list)

    @staticmethod
    def run(
        action: Literal['list', 'add', 'complete', 'uncomplete', 'edit', 'remove', 'reorder', 'clear'],
        index: Optional[int] = None,
        task: Optional[str] = None,
        new_index: Optional[int] = None
    ) -> str:
        """
        Manages a to-do list in QWEN.md through various actions.

        Args:
            action: The operation to perform.
                'list': Shows all tasks.
                'add': Adds a new task. Requires `task`.
                'complete': Marks a task as complete. Requires `index`.
                'uncomplete': Marks a task as not complete. Requires `index`.
                'edit': Changes the text of a task. Requires `index` and `task`.
                'remove': Deletes a task. Requires `index`.
                'reorder': Moves a task to a new position. Requires `index` and `new_index`.
                'clear': Removes all tasks from the list.
            index: The 1-based index of the task to act on.
            task: The text content for a new or edited task.
            new_index: The new 1-based position for a task being reordered.

        Returns:
            A JSON string with a 'result' key on success, or an 'error' key on failure.
        """
        try:
            todos = TodosUtil._load_todos()

            if action == 'list':
                return json.dumps({"result": {
                    "status": "Success",
                    "task_count": len(todos),
                    "tasks": todos
                }}, indent=2)

            elif action == 'add':
                if not task or not task.strip():
                    return json.dumps({"error": "'add' action requires a non-empty 'task' argument."})
                todos.append({"task": task, "completed": False})
                TodosUtil._save_todos(todos)
                return json.dumps({"result": {
                    "status": "Success",
                    "message": f"Added new task: '{task}' at index {len(todos)}."
                }}, indent=2)

            elif action == 'clear':
                count = len(todos)
                TodosUtil._save_todos([])
                return json.dumps({"result": {
                    "status": "Success",
                    "message": f"Cleared all {count} tasks from the list."
                }}, indent=2)

            # --- Actions below require an index ---
            if index is None:
                return json.dumps({"error": f"'{action}' action requires an 'index' argument."})
            
            if not isinstance(index, int) or not (1 <= index <= len(todos)):
                return json.dumps({"error": f"Invalid index: {index}. Must be an integer between 1 and {len(todos)}."})
            
            idx_0_based = index - 1

            if action in ['complete', 'uncomplete']:
                is_completing = action == 'complete'
                if todos[idx_0_based]['completed'] == is_completing:
                    status_text = "already completed" if is_completing else "not completed"
                    message = f"Task {index} ('{todos[idx_0_based]['task']}') is {status_text}."
                else:
                    todos[idx_0_based]['completed'] = is_completing
                    TodosUtil._save_todos(todos)
                    status_text = "completed" if is_completing else "uncompleted"
                    message = f"Marked task {index} ('{todos[idx_0_based]['task']}') as {status_text}."
                return json.dumps({"result": {"status": "Success", "message": message}}, indent=2)

            elif action == 'edit':
                if not task or not task.strip():
                    return json.dumps({"error": "'edit' action requires a non-empty 'task' argument."})
                old_task = todos[idx_0_based]['task']
                todos[idx_0_based]['task'] = task
                TodosUtil._save_todos(todos)
                return json.dumps({"result": {
                    "status": "Success",
                    "message": f"Edited task {index}. Old: '{old_task}'. New: '{task}'."
                }}, indent=2)

            elif action == 'remove':
                removed_task = todos.pop(idx_0_based)
                TodosUtil._save_todos(todos)
                return json.dumps({"result": {
                    "status": "Success",
                    "message": f"Removed task {index}: '{removed_task['task']}'."
                }}, indent=2)

            elif action == 'reorder':
                if new_index is None:
                    return json.dumps({"error": "'reorder' action requires a 'new_index' argument."})
                if not isinstance(new_index, int) or not (1 <= new_index <= len(todos)):
                     return json.dumps({"error": f"Invalid new_index: {new_index}. Must be an integer between 1 and {len(todos)}."})
                
                new_idx_0_based = new_index - 1
                moved_task = todos.pop(idx_0_based)
                todos.insert(new_idx_0_based, moved_task)
                TodosUtil._save_todos(todos)
                return json.dumps({"result": {
                    "status": "Success",
                    "message": f"Moved task '{moved_task['task']}' from position {index} to {new_index}."
                }}, indent=2)
            
            else:
                # This case should ideally not be reached due to Literal typing
                return json.dumps({"error": f"Unknown action: '{action}'."})

        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred in TodosUtil: {str(e)}"})