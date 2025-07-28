import os
import json
from typing import Literal, Optional, List, Dict, Any

from py_classes.cls_util_base import UtilBase
from py_classes.globals import g

class TodosUtil(UtilBase):
    """
    A utility for managing a to-do list.
    It allows an agent to list, add, complete, uncomplete, edit, remove, reorder, and clear tasks.
    The to-do list is stored only in memory and will be lost when the process ends.
    """
    
    # Class variable to store todos in RAM
    _todos: List[Dict[str, Any]] = []

    @staticmethod
    def _load_todos() -> List[Dict[str, Any]]:
        """Returns the current todos from RAM."""
        return TodosUtil._todos

    @staticmethod
    def _save_todos(todos: List[Dict[str, Any]]) -> None:
        """Saves the to-do list to RAM."""
        TodosUtil._todos = todos

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
        Manages a to-do list in RAM through various actions.

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
                # formatted_list = TodosUtil._format_todos(todos)
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