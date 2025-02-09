import importlib
import inspect
import os
from typing import Dict, List, Type
import logging
from pathlib import Path
from termcolor import colored

from .cls_base_tool import BaseTool

class ToolManager:
    def __init__(self, tools_directory: str = "py_tools"):
        self.tools_directory = tools_directory
        self.tools: Dict[str, Type[BaseTool]] = {}
        print(colored(f"Initializing ToolManager with directory: {tools_directory}", "green"))
        self._load_tools()

    def _load_tools(self) -> None:
        """Dynamically load all tool modules from the tools directory"""
        tools_path = Path(self.tools_directory)
        if not tools_path.exists():
            os.makedirs(tools_path)
            # Create __init__.py to make it a package
            (tools_path / "__init__.py").touch()

        print(colored(f"Scanning for tools in: {tools_path}", "green"))
        for file in tools_path.glob("*.py"):
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

    def get_tools_prompt(self) -> str:
        """Generate a prompt that describes all available tools"""
        prompt = "Available tools:\n\n"
        for tool_cls in self.tools.values():
            tool = tool_cls()
            metadata = tool.metadata
            prompt += f"Tool: {metadata.name}\n"
            prompt += f"Description: {metadata.description}\n"
            prompt += f"Parameters: {metadata.parameters}\n"
            prompt += f"Required parameters: {metadata.required_params}\n"
            prompt += f"Example usage: {metadata.example_usage}\n\n"
        return prompt

    def reload_tools(self) -> None:
        """Reload all tools from the tools directory"""
        self.tools.clear()
        self._load_tools() 