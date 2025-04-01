import importlib
import inspect
import os
from typing import Dict, List, Type, Optional
import logging
from pathlib import Path
from termcolor import colored

from .cls_base_tool import BaseTool

class ToolManager:
    def __init__(self):
        # Get the absolute path of the project root directory (parent of py_classes)
        self.project_root = Path(__file__).parent.parent.absolute()
        self.tools_directory = "py_tools"
        self.tools_path = self.project_root / self.tools_directory
        self.tools: Dict[str, Type[BaseTool]] = {}
        print(colored(f"Initializing ToolManager with directory: {self.tools_path}", "green"))
        self._load_tools()

    def _load_tools(self) -> None:
        """Dynamically load all tool modules from the tools directory"""
        if not self.tools_path.exists():
            os.makedirs(self.tools_path)
            # Create __init__.py to make it a package
            (self.tools_path / "__init__.py").touch()

        print(colored(f"Scanning for tools in: {self.tools_path}", "green"))
        for file in self.tools_path.glob("*.py"):
            if file.name.startswith("_"):
                continue

            try:
                print(colored(f"Loading tool from file: {file}", "cyan"))
                # Convert path to module name (e.g., py_tools/my_tool.py -> py_tools.my_tool)
                module_name = f"{self.tools_directory}.{file.stem}"
                module = importlib.import_module(module_name)

                # Find all classes in the module that inherit from BaseTool
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseTool) and 
                        obj != BaseTool):
                        try:
                            tool_instance = obj()
                            self.tools[tool_instance.metadata.name] = obj
                            print(colored(f"Successfully loaded tool: {tool_instance.metadata.name}", "green"))
                        except Exception as tool_error:
                            print(colored(f"Error instantiating tool {name}: {str(tool_error)}", "red"))

            except Exception as e:
                print(colored(f"Error loading tool from {file}: {str(e)}", "red"))

    def get_tool(self, name: str) -> Type[BaseTool]:
        """Get a tool by name"""
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")
        return self.tools[name]

    def get_all_tools(self) -> List[Type[BaseTool]]:
        """Get all available tools"""
        return list(self.tools.values())

    def get_tools_prompt(self, tool_names: Optional[List[str]] = None, include_details: bool = False) -> str:
        """Generate a prompt that describes available tools.
        
        Args:
            tool_names: Optional list of tool names to include. If None, includes all tools.
            
        Returns:
            str: Formatted prompt describing the specified tools.
        """
        prompt = "Available tools:\n```\n"
        for tool_cls in self.tools.values():
            tool = tool_cls()
            if tool_names is None or tool.metadata.name in tool_names:
                metadata = tool.metadata
                prompt += f"Tool: {metadata.name}\n"
                prompt += f"Description: {metadata.description}\n"
                prompt += f"run(...) parameters: {metadata.parameters}\n"
                if include_details:
                    prompt += f"Detailed description: {metadata.detailed_description}\n"
                    prompt += f"Example usage: {metadata.example_usage}\n\n"
        prompt += "```"
        return prompt

    def reload_tools(self) -> None:
        """Reload all tools from the tools directory"""
        self.tools.clear()
        self._load_tools() 