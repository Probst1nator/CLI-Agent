"""
Tool discovery and documentation generation for CLI-Agent.

This module contains functions for dynamically discovering tools and generating
their documentation, extracted from main.py to provide reusable tool management.
"""

import asyncio
import importlib
import logging
from pathlib import Path

from core.globals import g


def _check_getuserfiles_available() -> bool:
    """
    Check if getuserfiles GUI is available and has content by testing WebSocket connection.
    
    Returns:
        True if getuserfiles is available and has content, False otherwise
    """
    try:
        import websockets
        
        async def test_connection():
            try:
                async with websockets.connect("ws://localhost:8765", open_timeout=1) as websocket:
                    content = await asyncio.wait_for(websocket.recv(), timeout=2)
                    return content and not content.startswith("Error:")
            except:
                return False
        
        return asyncio.run(test_connection())
    except:
        return False


def _generate_tool_documentation() -> str:
    """
    Dynamically discovers all tools from the tools directory and formats
    their documentation for the system prompt.
    
    Returns:
        Formatted documentation string for all available tools
        
    Side Effects:
        Sets g.tool_classes with discovered tool classes
    """
    docs_string = "3.  **ACT**: Decide on a tool and execute it to pass your current step. You can make use of the following tools:\n\n"
    
    # Import and discover tools from the tools directory
    tools_dir = Path(__file__).parent.parent.parent / "tools"
    tool_classes = {}
    
    if tools_dir.exists():
        for file in tools_dir.glob("*.py"):
            if file.name.startswith("_") or file.stem == "__init__":
                continue
            
            try:
                # Import the tool module
                module_name = f"tools.{file.stem}"
                module = importlib.import_module(module_name)
                
                # Find the tool class (should match the filename)
                tool_class_name = file.stem
                if hasattr(module, tool_class_name):
                    tool_class = getattr(module, tool_class_name)
                    if hasattr(tool_class, 'get_delim') and hasattr(tool_class, 'get_tool_info'):
                        # Skip getuserfiles if GUI is not available or has no content
                        if tool_class_name == 'getuserfiles' and not _check_getuserfiles_available():
                            continue
                        tool_classes[tool_class.get_delim()] = tool_class
            except Exception as e:
                logging.warning(f"Could not load tool from {file}: {e}")
    
    # Sort tools to ensure a consistent order in the prompt
    sorted_tool_names = sorted(tool_classes.keys())

    for tool_name in sorted_tool_names:
        tool_class = tool_classes[tool_name]
        try:
            info = tool_class.get_tool_info()
            # Use four spaces for indentation in the final prompt
            docs_string += (
                f"    - **`{info['name']}`**: {info['description']}\n"
                f"      *Example*: `{info['example']}`\n\n"
            )
        except Exception as e:
            logging.warning(f"Could not generate documentation for tool '{tool_name}': {e}")
    
    # Store tool_classes globally for later use. This is a critical side-effect
    # that makes the tools available to the rest of the application.
    g.tool_classes = tool_classes
    return docs_string


# Helper function for backwards compatibility
def generate_tool_documentation() -> str:
    """
    Public interface for tool documentation generation.
    
    Returns:
        Formatted documentation string for all available tools
    """
    return _generate_tool_documentation()


def check_getuserfiles_available() -> bool:
    """
    Public interface for checking getuserfiles availability.
    
    Returns:
        True if getuserfiles is available, False otherwise
    """
    return _check_getuserfiles_available()