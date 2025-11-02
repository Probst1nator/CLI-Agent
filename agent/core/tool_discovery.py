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

        async def test_connection() -> bool:
            try:
                async with websockets.connect("ws://localhost:8765", open_timeout=1) as websocket:
                    content = await asyncio.wait_for(websocket.recv(), timeout=2)
                    return bool(content and not content.startswith("Error:"))
            except:
                return False

        return asyncio.run(test_connection())
    except:
        return False


def _check_homeassistant_available() -> bool:
    """
    Check if Home Assistant is available and reachable by testing connectivity.

    Returns:
        True if Home Assistant is reachable and authenticated, False otherwise
    """
    try:
        # Import the Home Assistant API module
        from tools.optional.homeassistant import _HomeAssistantAPI

        # Test reachability using the existing method
        is_reachable, message = _HomeAssistantAPI.check_reachability()
        return is_reachable
    except Exception as e:
        logging.debug(f"Home Assistant availability check failed: {e}")
        return False


def _generate_tool_documentation(disabled_tools: list[str] = []) -> str:
    """
    Dynamically discovers all tools from the tools directory and formats
    their documentation for the system prompt.

    Returns:
        Formatted documentation string for all available tools

    Side Effects:
        Sets g.tool_classes with discovered tool classes
        Sets g.loaded_tool_info for startup display
    """
    docs_string = "3.  **ACT**: Decide on a tool and execute it to pass your current step. You can make use of the following tools:\n\n"

    # Import and discover tools from the tools directory and optional subdirectory
    tools_dir = Path(__file__).parent.parent.parent / "tools"
    tool_classes = {}
    failed_tools = []
    skipped_tools = []

    # Scan main tools directory and subdirectories
    tool_directories = [
        (tools_dir, "tools"),
        (tools_dir / "optional", "tools.optional"),
        (tools_dir / "primitives", "tools.primitives")
    ]

    for directory, module_prefix in tool_directories:
        if not directory.exists():
            continue

        for file in directory.glob("*.py"):
            if file.name.startswith("_") or file.stem == "__init__":
                continue

            try:
                # Import the tool module with appropriate prefix
                module_name = f"{module_prefix}.{file.stem}"
                module = importlib.import_module(module_name)

                # Find the tool class (should match the filename)
                tool_class_name = file.stem
                if hasattr(module, tool_class_name):
                    tool_class = getattr(module, tool_class_name)
                    if hasattr(tool_class, 'get_delim') and hasattr(tool_class, 'get_tool_info'):
                        # Track disabled status but DON'T skip registration
                        # (tools must be in g.tool_classes to be detected and blocked during execution)
                        is_disabled = tool_class_name in disabled_tools

                        if is_disabled:
                            skipped_tools.append(f"{tool_class_name} (Disabled)")

                        # Special handling for optional tools
                        if tool_class_name == 'getuserfiles' and not _check_getuserfiles_available():
                            skipped_tools.append(f"{tool_class_name} (GUI not connected)")
                            continue
                        elif tool_class_name == 'homeassistant':
                            # Initialize Home Assistant and check availability
                            if hasattr(tool_class, 'initialize'):
                                tool_class.initialize()
                            if not _check_homeassistant_available():
                                skipped_tools.append(f"{tool_class_name} (Home Assistant not reachable)")
                                continue

                        tool_classes[tool_class.get_delim()] = tool_class
                    else:
                        failed_tools.append(f"{tool_class_name} (missing required methods)")
                else:
                    failed_tools.append(f"{file.stem} (class not found)")
            except Exception as e:
                failed_tools.append(f"{file.stem} ({str(e)})")
                logging.debug(f"Could not load tool from {file}: {e}")

    # Sort tools to ensure a consistent order in the prompt
    sorted_tool_names = sorted(tool_classes.keys())

    for tool_name in sorted_tool_names:
        tool_class = tool_classes[tool_name]
        try:
            # Skip disabled tools in the system prompt (LLM shouldn't try to use them)
            # But they're still in tool_classes for detection/blocking during execution
            if tool_name in disabled_tools:
                continue

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

    # Store tool info for startup display
    tool_info = {
        'loaded_tools': sorted_tool_names,
        'failed_tools': failed_tools,
        'skipped_tools': skipped_tools,
        'tool_count': len(tool_classes),
        'gui_tools': []
    }

    # Check for GUI tools that are loaded
    for tool_name in sorted_tool_names:
        # Currently no tools are specifically GUI-only
        pass

    # Add skipped GUI tools
    if skipped_tools:
        tool_info['gui_tools'].extend([tool for tool in skipped_tools if 'GUI' in tool])

    g.loaded_tool_info = tool_info
    return docs_string




