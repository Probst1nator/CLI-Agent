import json
from abc import ABC, abstractmethod
from typing import Any, Type, Dict

class UtilBase(ABC):
    """
    Abstract base class for implementing utility functionality.
    
    This class defines the interface that all utility implementations 
    must follow. The public `run` method contains a guard to prevent 
    re-execution if tasks for the utility are already pending. Subclasses
    must implement the `_run_logic` method for their specific behavior.
    """

    @classmethod
    def run(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Public entry point for executing a utility.
        
        This method contains a guard that checks for pending to-do items
        specific to this utility. If found, it blocks execution and returns
        an error. Otherwise, it calls the utility's specific implementation.
        """
        # Local import to avoid circular dependency issues
        import importlib.util
        import os
        # From: agent/utils_manager/util_base.py -> project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        todos_path = os.path.join(project_root, 'tools', 'main_cli_agent', 'utils', '.deprecated', 'todos.py')
        todos_spec = importlib.util.spec_from_file_location("todos", todos_path)
        todos_module = importlib.util.module_from_spec(todos_spec)
        todos_spec.loader.exec_module(todos_module)
        TodosUtil = todos_module.TodosUtil

        util_name = cls.get_name(cls)
        
        # TodosUtil is foundational and must be exempt from the guard.
        if util_name == "todosutil":
            return cls._run_logic(*args, **kwargs)

        try:
            all_todos = TodosUtil._load_todos()
            pending_util_todos = [
                todo for todo in all_todos
                if not todo.get('completed', False) and todo.get('task', '').startswith(f"{util_name}:")
            ]

            if pending_util_todos:
                error_message = (
                    f"Execution of '{util_name}' is blocked. Please complete the following "
                    f"pending tasks for this utility first:"
                )
                formatted_todos = TodosUtil._format_todos_md(pending_util_todos)
                return json.dumps({
                    "error": error_message,
                    "pending_tasks": formatted_todos
                }, indent=2)

        except Exception as e:
            print(f"Warning: Run guard check failed for {util_name}. Reason: {e}")

        # If the guard passes, execute the actual utility logic
        return cls._run_logic(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def _run_logic(*args: Any, **kwargs: Any) -> Any:
        """
        The actual implementation of the utility execution logic.
        
        This method must be overridden by subclasses to provide the
        specific functionality of the utility. Each subclass will
        define its own specific arguments.
        """
        pass

    @staticmethod
    def get_name(util_cls: Type['UtilBase']) -> str:
        """Get the name of a utility class."""
        return util_cls.__name__.lower()
    
    @staticmethod
    def get_description(util_cls: Type['UtilBase']) -> str:
        """Get the description of a utility class from its docstring."""
        import inspect
        docstring = inspect.getdoc(util_cls)
        return docstring or "No description available"
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Get comprehensive metadata for the utility."""
        return {"keywords": [], "use_cases": [], "arguments": {}}