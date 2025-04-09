import importlib
import inspect
import os
from typing import Dict, List, Type, Optional, Set
import logging
from pathlib import Path
from termcolor import colored

from py_classes.cls_base_tool import BaseTool, ToolResponse

class ToolManager:
    def __init__(self):
        # Get the absolute path of the project root directory (parent of py_classes)
        self.project_root = Path(__file__).parent.parent.absolute()
        self.tools_directory = "agentic_tools"
        self.tools_path = self.project_root / self.tools_directory
        self.default_tools: Dict[str, Type[BaseTool]] = {}
        self.followup_tools: Dict[str, Type[BaseTool]] = {}
        self.tool_history: List[Type[BaseTool]] = []
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
                # Convert path to module name (e.g., agentic_tools/my_tool.py -> agentic_tools.my_tool)
                module_name = f"{self.tools_directory}.{file.stem}"
                module = importlib.import_module(module_name)

                # Find all classes in the module that inherit from BaseTool
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseTool) and 
                        obj != BaseTool):
                        try:
                            tool_instance = obj()
                            metadata = tool_instance.metadata
                            
                            # Classify tool based on is_followup_only flag
                            if metadata.is_followup_only:
                                self.followup_tools[metadata.name] = obj
                                print(colored(f"Successfully loaded followup tool: {metadata.name}", "green"))
                            else:
                                self.default_tools[metadata.name] = obj
                                print(colored(f"Successfully loaded default tool: {metadata.name}", "green"))
                                
                        except Exception as tool_error:
                            print(colored(f"Error instantiating tool {name}: {str(tool_error)}", "red"))

            except Exception as e:
                print(colored(f"Error loading tool from {file}: {str(e)}", "red"))

    def get_tool(self, name: str) -> Type[BaseTool]:
        """Get a tool by name and add it to the tool history"""
        # Look in both default and followup tools
        if name in self.default_tools:
            tool = self.default_tools[name]
        elif name in self.followup_tools:
            tool = self.followup_tools[name]
        else:
            raise KeyError(f"Tool '{name}' not found")
        
        # Add to tool history
        self.tool_history.append(tool)
        
        # Limit history size if needed
        if len(self.tool_history) > 10:  # Keep only the last 10 tools
            self.tool_history = self.tool_history[-10:]
            
        return tool

    def get_all_tools(self) -> List[Type[BaseTool]]:
        """Get all default available tools (excluding followup-only tools)"""
        return list(self.default_tools.values())
    
    def get_available_tools(self, include_followups: bool = False) -> Dict[str, Type[BaseTool]]:
        """Get tools that are currently available for use.
        
        Args:
            include_followups: Whether to include followup tools in the results
            
        Returns:
            Dictionary mapping tool names to tool classes
        """
        tools = self.default_tools.copy()
        
        if include_followups:
            # Get followup tools from recent history
            available_followup_tools = self._get_available_followup_tools()
            # Add available followup tools to the result
            for name in available_followup_tools:
                if name in self.followup_tools:
                    tools[name] = self.followup_tools[name]
                    
        return tools

    def _get_available_followup_tools(self) -> Set[str]:
        """Get the set of available followup tool names based on recent tool history"""
        available_followups = set()
        
        # Look at recently used tools to find suggested followups
        for tool_class in self.tool_history:
            tool_instance = tool_class()
            followup_tools = tool_instance.metadata.default_followup_tools
            available_followups.update(followup_tools)
            
        return available_followups

    def get_tools_prompt(self, tool_names: Optional[List[str]] = None, include_details: bool = False) -> str:
        """Generate a prompt that describes available tools.
        
        Args:
            tool_names: Optional list of tool names to include. If None, includes all available tools.
            
        Returns:
            str: Formatted prompt describing the specified tools.
        """
        prompt = "```\n"
        
        # Get all currently available tools
        available_tools = self.get_available_tools(include_followups=True)
        
        # If specific tools were requested, filter for those
        if tool_names is not None:
            filtered_tools = {}
            for name in tool_names:
                if name in available_tools:
                    filtered_tools[name] = available_tools[name]
            available_tools = filtered_tools
        
        # Format the tools into the prompt
        for name, tool_cls in available_tools.items():
            tool = tool_cls()
            metadata = tool.metadata
            prompt += f"Tool: {metadata.name}\n"
            prompt += f"Description: {metadata.description}\n"
            
            # Extract method signature from the constructor string
            constructor_lines = metadata.constructor.strip().split('\n')
            
            method_signature = ""
            if include_details:
                method_signature = constructor_lines
            else:
                # The first line should be the method signature (def run(...))
                method_signature = constructor_lines[0].strip()
                # Remove the "def " part to get just the method name and parameters
                if method_signature.startswith("def "):
                    method_signature = method_signature[4:]
            
            prompt += f"```tool_code\n{metadata.name}.{method_signature}\n```\n"
            
            prompt += f"\n"
            
        prompt += "```"
        return prompt

    def reload_tools(self) -> None:
        """Reload all tools from the tools directory"""
        self.default_tools.clear()
        self.followup_tools.clear()
        self._load_tools()

    def add_followup_tool_to_response(self, tool_response: ToolResponse, tool: Type[BaseTool]) -> ToolResponse:
        """Add suggested followup tools to the tool response if applicable.
        
        Args:
            tool_response: The ToolResponse from a tool execution.
            tool: The tool class that generated the response.
            
        Returns:
            Updated ToolResponse with followup_tool list if applicable.
        """
        # Only add followup tools if the response was successful
        if tool_response["status"] == "success":
            tool_instance = tool()
            followup_tools = tool_instance.metadata.default_followup_tools
            
            if followup_tools and len(followup_tools) > 0:
                # Verify each tool exists and filter to only include valid tools
                verified_tools = []
                for suggested_tool in followup_tools:
                    if suggested_tool in self.default_tools or suggested_tool in self.followup_tools:
                        verified_tools.append(suggested_tool)
                
                if verified_tools:
                    tool_response["followup_tools"] = verified_tools
        
        return tool_response 