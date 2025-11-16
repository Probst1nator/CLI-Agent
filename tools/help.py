# tools/help.py
"""
This file implements the 'help' tool. It provides help documentation for all available tools.
"""

import importlib
import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure proper imports when running as module or script
if __name__ == "__main__":
    # Add project root to path for direct execution
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from shared.utils.tool_logging import get_logger_for_tool_module

logger = get_logger_for_tool_module(__name__)


class help:
    """
    A tool to provide help documentation for all available tools.
    """

    @staticmethod
    def get_delim() -> str:
        return "help"

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool."""
        return {
            "name": "help",
            "description": (
                "Provides help documentation for tools. "
                "Use <help>tool_name</help> for detailed help on a specific tool, "
                "or <help/> to list all available tools."
            ),
            "example": ("<help>papersearch</help>\n<help>bash</help>\n<help/>"),
        }

    @staticmethod
    def _discover_tools() -> dict[str, object]:
        """
        Discover all available tools in the tools directory.

        Returns:
            Dictionary mapping tool names to their class objects
        """
        tools = {}
        tools_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            for filename in os.listdir(tools_dir):
                # Skip non-Python files and special files
                if not filename.endswith(".py") or filename.startswith("_"):
                    continue

                tool_name = filename[:-3]  # Remove .py extension

                # Skip the help tool itself
                if tool_name == "help":
                    continue

                try:
                    # Import the module
                    module = importlib.import_module(f"tools.{tool_name}")

                    # Get the class (should match the filename)
                    if hasattr(module, tool_name):
                        tool_class = getattr(module, tool_name)

                        # Verify it has the required methods
                        if hasattr(tool_class, "get_delim") and hasattr(
                            tool_class, "get_tool_info"
                        ):
                            tools[tool_name] = tool_class
                            logger.debug(f"Discovered tool: {tool_name}")
                        else:
                            logger.warning(f"Tool {tool_name} missing required methods")
                    else:
                        logger.warning(
                            f"Module {tool_name} does not have class {tool_name}"
                        )

                except Exception as e:
                    logger.warning(f"Failed to import tool {tool_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to discover tools: {e}", exc_info=True)

        return tools

    @staticmethod
    def _format_tool_list(tools: dict[str, object]) -> str:
        """
        Format a list of all available tools.

        Args:
            tools: Dictionary of tool name to tool class

        Returns:
            Formatted string with tool list
        """
        if not tools:
            return "No tools available."

        # Sort tools alphabetically
        sorted_tools = sorted(tools.items())

        lines = [
            "╔══════════════════════════════════════════════════════════════════════════════╗",
            "║                          AVAILABLE TOOLS                                      ║",
            "╚══════════════════════════════════════════════════════════════════════════════╝",
            "",
            f"Total tools available: {len(tools)}",
            "",
            "To get detailed help for a specific tool, use: <help>tool_name</help>",
            "",
            "─" * 80,
            "",
        ]

        for tool_name, tool_class in sorted_tools:
            try:
                info = tool_class.get_tool_info()
                name = info.get("name", tool_name)
                description = info.get("description", "No description available")

                # Get the delimiter
                delim = (
                    tool_class.get_delim()
                    if hasattr(tool_class, "get_delim")
                    else tool_name
                )

                lines.append(f"📦 {name.upper()} (tag: <{delim}>)")
                lines.append(f"   {description}")
                lines.append("")

            except Exception as e:
                logger.warning(f"Failed to get info for tool {tool_name}: {e}")
                lines.append(f"📦 {tool_name.upper()}")
                lines.append("   Error retrieving tool information")
                lines.append("")

        lines.append("─" * 80)
        lines.append("")
        lines.append(
            "Use <help>tool_name</help> for detailed documentation and examples."
        )

        return "\n".join(lines)

    @staticmethod
    def _get_detailed_help(tool_name: str, tool_class: object) -> str:
        """
        Get detailed help for a specific tool.

        Args:
            tool_name: Name of the tool
            tool_class: Tool class object

        Returns:
            Detailed help string
        """
        try:
            # Try to get detailed help first
            if hasattr(tool_class, "get_detailed_tool_info"):
                detailed_info = tool_class.get_detailed_tool_info()
                return detailed_info

            # Fallback to basic tool info
            info = tool_class.get_tool_info()
            name = info.get("name", tool_name)
            description = info.get("description", "No description available")
            example = info.get("example", "No examples available")

            # Get the delimiter
            delim = (
                tool_class.get_delim()
                if hasattr(tool_class, "get_delim")
                else tool_name
            )

            # Format basic info nicely
            lines = [
                "╔══════════════════════════════════════════════════════════════════════════════╗",
                f"║  {name.upper().center(76)}  ║",
                "╚══════════════════════════════════════════════════════════════════════════════╝",
                "",
                "DESCRIPTION:",
                f"  {description}",
                "",
                "USAGE TAG:",
                f"  <{delim}>...</{delim}>",
                "",
                "EXAMPLES:",
                "",
            ]

            # Add examples
            for line in example.split("\n"):
                lines.append(f"  {line}")

            lines.append("")
            lines.append("─" * 80)
            lines.append("")
            lines.append("Note: This tool does not have detailed documentation yet.")
            lines.append(f"For implementation details, see: tools/{tool_name}.py")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Failed to get help for tool {tool_name}: {e}", exc_info=True)
            return f"Error: Failed to retrieve help for tool '{tool_name}': {str(e)}"

    @staticmethod
    def run(content: str) -> str:
        """
        Provides help documentation for tools.

        Args:
            content: Tool name or empty for full list

        Returns:
            Help documentation string
        """
        try:
            # Parse the tool name from content
            tool_name = content.strip()

            # Discover all available tools
            tools = help._discover_tools()

            if not tools:
                return "Error: No tools could be discovered."

            # If no tool name specified, return full list
            if not tool_name:
                logger.info("Listing all available tools")
                return help._format_tool_list(tools)

            # Get detailed help for specific tool
            if tool_name not in tools:
                available = ", ".join(sorted(tools.keys()))
                return (
                    f"Error: Tool '{tool_name}' not found.\n\n"
                    f"Available tools: {available}\n\n"
                    f"Use <help/> to see the full list with descriptions."
                )

            logger.info(f"Retrieving detailed help for tool: {tool_name}")
            return help._get_detailed_help(tool_name, tools[tool_name])

        except Exception as e:
            logger.error(f"Unexpected error in help tool: {e}", exc_info=True)
            return f"An unexpected error occurred in the help tool: {str(e)}"


# Allow direct execution for testing
if __name__ == "__main__":
    # Test listing all tools
    print("=== TESTING TOOL LIST ===")
    result = help.run("")
    print(result)

    print("\n" + "=" * 80)
    print("=== TESTING SPECIFIC TOOL HELP ===")
    result = help.run("papersearch")
    print(result)

    print("\n" + "=" * 80)
    print("=== TESTING INVALID TOOL ===")
    result = help.run("nonexistent_tool")
    print(result)
