# tools/paralleltools.py
"""
This file implements the 'paralleltools' tool. It allows an agent to execute
multiple arbitrary tools in parallel and return their results in a single
context block with markdown-style section headers.
"""

import asyncio
from typing import Any

from termcolor import colored

from core.globals import g


class paralleltools:
    """
    A tool to execute multiple tools in parallel and aggregate their results.
    """

    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool."""
        return 'paralleltools'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool."""
        return {
            "name": "paralleltools",
            "description": (
                "Executes multiple tools in parallel and returns their combined results "
                "in a single context block with markdown section headers. Place any valid "
                "tool invocations inside the paralleltools tags using their native XML format. "
                "Results are returned with headers like '## toolname' for easy parsing. "
            ),
            "example": (
                "<paralleltools>\n"
                "  <readfile>./src/main.py</readfile>\n"
                "  <grepsearch>\n"
                "    <pattern>async def</pattern>\n"
                "    <path>./src</path>\n"
                "  </grepsearch>\n"
                "  <bash>git status</bash>\n"
                "</paralleltools>"
            )
        }

    @staticmethod
    def _extract_tool_calls(content: str) -> list[tuple[str, str]]:
        """
        Extract all tool invocations from the paralleltools content.
        Returns a list of (tool_name, tool_content) tuples.

        This uses a simpler approach: find opening tags at the root level
        and extract their full content including nested tags.
        """
        tool_calls = []

        # Remove the outer paralleltools tags to get inner content
        content = content.strip()

        # Find all top-level tool tags
        # We need to handle nested tags, so we can't use simple regex
        # Instead, we'll parse iteratively
        i = 0
        while i < len(content):
            # Skip whitespace
            while i < len(content) and content[i].isspace():
                i += 1

            if i >= len(content):
                break

            # Look for opening tag
            if content[i] == '<':
                # Find tag name
                tag_start = i + 1
                tag_end = tag_start
                while tag_end < len(content) and content[tag_end] not in ('>', ' ', '\t', '\n'):
                    tag_end += 1

                if tag_end >= len(content):
                    break

                tag_name = content[tag_start:tag_end]

                # Skip if this is a closing tag
                if tag_name.startswith('/'):
                    i = tag_end + 1
                    continue

                # Find the matching closing tag
                closing_tag = f'</{tag_name}>'
                tag_content_start = content.find('>', tag_end) + 1

                # Count nested tags to find the right closing tag
                depth = 1
                search_pos = tag_content_start
                content_end = tag_content_start

                while depth > 0 and search_pos < len(content):
                    open_match = content.find(f'<{tag_name}', search_pos)
                    close_match = content.find(closing_tag, search_pos)

                    if close_match == -1:
                        break

                    if open_match != -1 and open_match < close_match:
                        depth += 1
                        search_pos = open_match + len(tag_name) + 1
                    else:
                        depth -= 1
                        if depth == 0:
                            content_end = close_match
                        search_pos = close_match + len(closing_tag)

                if content_end > tag_content_start:
                    tool_content = content[tag_content_start:content_end].strip()
                    tool_calls.append((tag_name, tool_content))
                    i = content_end + len(closing_tag)
                else:
                    i = tag_end + 1
            else:
                i += 1

        return tool_calls

    @staticmethod
    async def _execute_tool(tool_name: str, tool_content: str, tool_class: Any) -> str:
        """
        Execute a single tool and return its result.
        Handles both sync and async tools.
        """
        try:
            # Check if the tool's run method is async
            result = tool_class.run(tool_content)

            # If it's a coroutine, await it
            if asyncio.iscoroutine(result):
                result = await result

            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    @staticmethod
    async def run(content: str) -> str:
        """
        Parses the input to extract tool invocations, executes them in parallel,
        and returns aggregated results with markdown headers.
        """
        try:
            # Extract all tool calls from the content
            tool_calls = paralleltools._extract_tool_calls(content)

            if not tool_calls:
                return "Error: No tool invocations found inside <paralleltools> tags."

            # Get available tools from global registry
            if not hasattr(g, 'tool_classes') or not g.tool_classes:
                return "Error: Tool registry not initialized. No tools available."

            # Prepare tasks for parallel execution
            tasks = []
            task_metadata = []  # Track (tool_name, tool_class) for each task

            for tool_name, tool_content in tool_calls:
                # Check for recursive paralleltools calls
                if tool_name.lower() == 'paralleltools':
                    # Prevent recursion - return error message
                    async def make_recursion_error():
                        return "Error: Nested paralleltools calls are not allowed. Use paralleltools only at the top level."
                    tasks.append(make_recursion_error())
                    task_metadata.append((tool_name, None))
                    continue

                # Look up the tool class
                tool_class = g.tool_classes.get(tool_name)

                if not tool_class:
                    # Tool not found - still create a task that returns an error
                    async def make_error(name=tool_name):
                        return f"Error: Tool '{name}' not found in registry."
                    tasks.append(make_error())
                    task_metadata.append((tool_name, None))
                else:
                    # Create task for this tool
                    task = paralleltools._execute_tool(tool_name, tool_content, tool_class)
                    tasks.append(task)
                    task_metadata.append((tool_name, tool_class))

            # Execute all tools in parallel
            print(colored(f"🔧 Executing {len(tasks)} tools in parallel...", "cyan"))
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Format results with markdown headers
            output_parts = []

            for (tool_name, tool_class), result in zip(task_metadata, results, strict=False):
                # Add section header
                output_parts.append(f"## {tool_name}")

                # Handle exceptions that weren't caught in _execute_tool
                if isinstance(result, Exception):
                    output_parts.append(f"Error: {str(result)}")
                else:
                    output_parts.append(str(result))

                # Add spacing between sections
                output_parts.append("")

            # Join all parts and return
            return "\n".join(output_parts).strip()

        except Exception as e:
            return f"Fatal error in paralleltools: {str(e)}"
