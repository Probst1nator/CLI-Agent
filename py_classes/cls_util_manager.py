import importlib
import inspect
import os
from typing import Dict, List, Type, Optional, Set
import logging
from pathlib import Path
from termcolor import colored

from py_classes.cls_util_base import UtilBase

class UtilsManager:
    def __init__(self):
        # Get the absolute path of the project root directory (parent of py_classes)
        self.project_root = Path(__file__).parent.parent.absolute()
        self.utils_directory = "utils"
        self.utils_path = self.project_root / self.utils_directory
        self.utils: List[Type[UtilBase]] = []
        self.util_history: List[Type[UtilBase]] = []
        print(colored(f"Initializing UtilsManager with directory: {self.utils_path}", "green"))
        self._load_utils()

    def _load_utils(self) -> None:
        """Dynamically load all utility modules from the utils directory"""
        if not self.utils_path.exists():
            os.makedirs(self.utils_path)
            # Create __init__.py to make it a package
            (self.utils_path / "__init__.py").touch()

        print(colored(f"Scanning for utilities in: {self.utils_path}", "green"))
        for file in self.utils_path.glob("*.py"):
            if file.name.startswith("_"):
                continue

            try:
                print(colored(f"Loading utility from file: {file}", "cyan"))
                # Convert path to module name (e.g., utils/my_util.py -> utils.my_util)
                module_name = f"{self.utils_directory}.{file.stem}"
                module = importlib.import_module(module_name)

                # Find all classes in the module that inherit from UtilBase
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, UtilBase) and 
                        obj != UtilBase):
                        try:
                            # Add utility class directly to the list
                            util_name = UtilBase.get_name(obj)
                            print(colored(f"Successfully loaded utility: {util_name}", "green"))
                            self.utils.append(obj)
                        except Exception as util_error:
                            print(colored(f"Error loading utility {name}: {str(util_error)}", "red"))

            except Exception as e:
                print(colored(f"Error loading utility from {file}: {str(e)}", "red"))

    def get_util(self, name: str) -> Type[UtilBase]:
        """Get a utility by name and add it to the utility history"""
        # Search through utilities to find matching name
        name = name.lower()
        for util_cls in self.utils:
            if UtilBase.get_name(util_cls) == name:
                # Add to utility history
                self.util_history.append(util_cls)
                
                # Limit history size if needed
                if len(self.util_history) > 10:  # Keep only the last 10 utilities
                    self.util_history = self.util_history[-10:]
                
                return util_cls
        
        raise KeyError(f"Utility '{name}' not found")

    def get_utils(self) -> List[Type[UtilBase]]:
        """Get all available utilities
        
        Returns:
            List of utility classes
        """
        return self.utils.copy()
    
    def get_util_names(self) -> List[str]:
        """Get names of all available utilities
        
        Returns:
            List of utility names
        """
        return [UtilBase.get_name(util_cls) for util_cls in self.utils]

    def get_available_utils_info(self, util_names: Optional[List[str]] = None) -> str:
        """Generate a prompt that describes available utilities.
        
        Args:
            util_names: Optional list of utility names to include. If None, includes all available utilities.
            
        Returns:
            str: Formatted prompt describing the specified utilities.
        """
        prompt = "Custom available utilities:\n"
        
        # Get all currently available utilities
        available_utils = self.get_utils()
        
        # If specific utilities were requested, filter for those
        if util_names is not None:
            util_names = [name.lower() for name in util_names]
            available_utils = [
                util_cls for util_cls in available_utils 
                if UtilBase.get_name(util_cls) in util_names
            ]
        
        i = 0
        # Format the utilities into the prompt
        for util_cls in available_utils:
            util_name = UtilBase.get_name(util_cls)
            
            # Get the signature and docstring of the run method if available
            try:
                # Try to find the run method
                if hasattr(util_cls, 'run'):
                    run_method = util_cls.run
                    run_docstring = inspect.getdoc(run_method) or "No method documentation available"
                    signature = inspect.signature(run_method)
                    
                    # Format the method signature with docstring and example showing static usage
                    prompt += f"""# Custom Utility {i}
## Name: {util_name}                    
## Usage: 
```python
from utils.{util_name} import {util_cls.__name__}
{util_cls.__name__}.run{signature} # Example usage
```
## Docstring:
\"\"\"
{run_docstring}
\"\"\""""
                else:
                    prompt += "```python\n# No run method found\n```\n"
            except Exception:
                prompt += "```python\n# Could not retrieve method signature\n```\n"
            
            prompt += f"\n"
            i += 1
            
        return prompt

    def reload_utils(self) -> None:
        """Reload all utilities from the utils directory"""
        self.utils.clear()
        self._load_utils() 
        