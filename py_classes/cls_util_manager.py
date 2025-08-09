import importlib
import inspect
import os
from typing import List, Type, Optional, Dict, Any
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
        # Initializing UtilsManager silently
        self._load_utils()
        
        # Initialize vector database for smart tool retrieval
        self.vector_db = None
        self._init_vector_db()

    def _load_utils(self) -> None:
        """Dynamically load all utility modules from the utils directory"""
        if not self.utils_path.exists():
            os.makedirs(self.utils_path)
            # Create __init__.py to make it a package
            (self.utils_path / "__init__.py").touch()

        # Scanning for utilities silently
        for file in self.utils_path.glob("*.py"):
            if file.name.startswith("_"):
                continue

            try:
                # Loading utility from file silently
                # Convert path to module name (e.g., utils/my_util.py -> utils.my_util)
                module_name = f"{self.utils_directory}.{file.stem}"
                module = importlib.import_module(module_name)

                # Find all classes in the module that inherit from UtilBase
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, UtilBase) and 
                        obj != UtilBase):
                        try:
                            # Successfully loaded utility silently
                            self.utils.append(obj)
                        except Exception as util_error:
                            print(colored(f"Error loading utility {name}: {str(util_error)}", "red"))

            except Exception as e:
                # Only show critical errors, suppress common dependency issues
                if "No module named" not in str(e):
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
                    run_docstring = "\n".join("# " + line for line in run_docstring.strip().splitlines())
                    signature = inspect.signature(run_method)
                    
                    # Format the method signature with docstring and example showing static usage
                    prompt += f"""# Custom Utility {i}
## Name: {util_name}                    
## Usage: 
```python
from utils.{util_name} import {util_cls.__name__}
{run_docstring}
{util_cls.__name__}.run{signature} # Example usage
```
"""
                else:
                    prompt += "```python\n# No run method found\n```\n"
            except Exception:
                prompt += "```python\n# Could not retrieve method signature\n```\n"
            
            prompt += "\n"
            i += 1
            
        return prompt

    def _init_vector_db(self) -> None:
        """Initialize the vector database for smart tool retrieval"""
        try:
            from py_classes.cls_vector_db import ToolVectorDB
            self.vector_db = ToolVectorDB()
            
            # Add all loaded utilities to the vector database
            for util_cls in self.utils:
                self.vector_db.add_tool(util_cls)
                
        except Exception as e:
            # If vector DB initialization fails, continue without it
            if os.environ.get('CLAUDE_CODE_DEBUG') == '1':
                print(f"Warning: Could not initialize vector database: {e}")
            self.vector_db = None

    def get_relevant_utils(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get utilities most relevant to a query using vector search.
        
        Args:
            query: The search query describing what the user wants to do
            top_k: Number of relevant utilities to return
            
        Returns:
            List of utility information dictionaries
        """
        if self.vector_db is None:
            # Fallback to returning all utilities if vector DB is not available
            return [
                {
                    "name": UtilBase.get_name(util_cls),
                    "score": 1.0,
                    "class": util_cls,
                    "metadata": getattr(util_cls, 'get_metadata', lambda: {})()
                } 
                for util_cls in self.utils[:top_k]
            ]
        
        results = self.vector_db.search(query, top_k)
        
        # Record the search for learning purposes
        if results and len(results) > 0:
            # For now, record the first result as selected for learning
            self.vector_db.record_tool_selection(results[0]["name"], query)
        
        return results
    
    def get_relevant_guidance(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Get relevant guidance hints for a query.
        
        Args:
            query: The search query
            top_k: Number of guidance hints to return
            
        Returns:
            List of relevant guidance hints
        """
        if self.vector_db is None:
            return []
        
        return self.vector_db.get_relevant_guidance(query, top_k)

    def get_relevant_tools_prompt(self, query: str, top_k: int = 3) -> str:
        """
        Generates a combined prompt with relevant tools and guidance based on a query.
        This is used to augment the user's prompt to guide the agent.

        Args:
            query (str): The user's request or query.
            top_k (int): The number of top relevant items (tools and guidance) to suggest.

        Returns:
            str: A formatted string to be prepended to the main prompt, or an empty string if no relevant items are found.
        """
        # Fetch relevant tools and guidance from the vector DB
        top_k_tools = top_k
        top_k_guidance = max(1, top_k - 1) # Get slightly fewer guidance hints

        relevant_tools = self.get_relevant_utils(query, top_k=top_k_tools)
        relevant_guidance = self.get_relevant_guidance(query, top_k=top_k_guidance)

        prompt: str = ""

        if relevant_tools:
            # Filter tools by a confidence score to reduce noise, but allow more tools through
            high_confidence_tools = [tool for tool in relevant_tools if tool.get('score', 0) > 0.3]
            if high_confidence_tools:
                tool_suggestions = []
                for tool in high_confidence_tools:
                    tool_name = tool['name']
                    tool_class = tool['class']
                    
                    # Get concise signature and summary from _run_logic method
                    try:
                        if hasattr(tool_class, '_run_logic'):
                            run_logic_method = tool_class._run_logic
                            run_logic_signature = inspect.signature(run_logic_method)
                            run_logic_docstring = inspect.getdoc(run_logic_method) or "No documentation available"
                            
                            # Extract just the first line of docstring for concise summary
                            summary = run_logic_docstring.split('\n')[0].strip()
                            
                            # Get the module name for proper import pattern
                            module_name = tool_class.__module__.replace('utils.', '')
                            
                            # Format tool information with import pattern and usage  
                            # Simplify the signature display by removing complex type annotations
                            simplified_sig = str(run_logic_signature).replace('typing.', '').replace('Optional[', '').replace(']', '?')
                            tool_line = f"- **{tool_name}**: `import utils.{module_name} as {module_name}; {module_name}.run{simplified_sig}` - {summary}"
                        else:
                            # Fallback if no _run_logic method
                            description = tool_class.get_description(tool_class).splitlines()[0]
                            tool_line = f"- **{tool_name}**: {description}"
                    except Exception:
                        # Fallback for any errors
                        description = tool_class.get_description(tool_class).splitlines()[0]
                        tool_line = f"- **{tool_name}**: {description} (Signature unavailable)"
                    
                    tool_suggestions.append(tool_line)
                
                prompt = "\n".join(tool_suggestions)

        if relevant_guidance:
            # Use a higher confidence threshold for guidance to ensure it's very relevant.
            high_confidence_guidance = [g for g in relevant_guidance if g.get('score', 0) > 0.6]
            if high_confidence_guidance:
                guidance_hints = [f"- {g.get('guidance_text', g.get('text', ''))}" for g in high_confidence_guidance]
                prompt = "\n".join(guidance_hints)

        if not prompt:
            return ""

        return prompt

    def reload_utils(self) -> None:
        """Reload all utilities from the utils directory"""
        self.utils.clear()
        self._load_utils()
        # Reinitialize vector DB with new utilities
        self._init_vector_db()
        