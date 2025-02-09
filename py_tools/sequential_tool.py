from typing import Any, Dict, List
import json
from termcolor import colored
import re

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse
from py_classes.cls_tool_manager import ToolManager
from py_classes.cls_chat import Role, Chat
from py_classes.cls_llm_router import LlmRouter

class SequentialTool(BaseTool):
    @staticmethod
    def _sanitize_json_value(value: Any) -> Any:
        """Sanitize JSON values to prevent formatting issues."""
        if isinstance(value, str):
            # Replace newlines with spaces and remove control characters
            value = re.sub(r'\s+', ' ', value)
            value = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', value)
            return value.strip()
        elif isinstance(value, dict):
            return {k: SequentialTool._sanitize_json_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [SequentialTool._sanitize_json_value(item) for item in value]
        return value

    def _sanitize_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize a single step in the sequence."""
        return self._sanitize_json_value(step)

    def _generate_result_summary(self, first_result: Dict[str, Any], intended_second_step: Dict[str, Any]) -> str:
        """Generate a concise summary of the first step's result in context of the intended second step."""
        summary_chat = Chat()
        summary_chat.add_message(
            Role.SYSTEM,
            "You are an AI assistant that creates concise, relevant summaries of tool execution results."
        )
        
        summary_chat.add_message(
            Role.USER,
            f"""Given:
1. Result from first tool execution:
```json
{json.dumps(first_result, indent=2)}
```

2. Intended next step:
```json
{json.dumps(intended_second_step, indent=2)}
```

Please provide a concise summary of the first tool's results that would be relevant for deciding whether and how to execute the intended second step.
Focus only on information that would impact the decision about the second step.
Keep the summary brief and focused."""
        )
        
        try:
            return LlmRouter.generate_completion(
                summary_chat
            )
        except Exception as e:
            print(colored(f"Error generating summary: {str(e)}", "red"))
            return f"Error generating summary: {str(e)}"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="sequential",
            description="Execute a tool and prepare for a subsequent tool execution based on the results. Use this when you need to chain operations together.",
            parameters={
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of why this sequence is needed"
                },
                "tool": {
                    "type": "string",
                    "description": "The tool to execute"
                },
                "web_query": {
                    "type": "string",
                    "description": "The search query if using web_search"
                },
                "command": {
                    "type": "string",
                    "description": "The command if using bash"
                },
                "title": {
                    "type": "string",
                    "description": "The title if using python"
                },
                "requirements": {
                    "type": "string",
                    "description": "The requirements if using python"
                },
                "next_tool": {
                    "type": "string",
                    "description": "The tool to be used in the subsequent step"
                },
                "next_title": {
                    "type": "string",
                    "description": "Title for the subsequent operation"
                }
            },
            required_params=["tool", "next_tool", "next_title"],
            example_usage="""
            {
                "reasoning": "Search for latest AGI papers and prepare for creating a summary presentation",
                "tool": "web_search",
                "web_query": "latest papers on AGI",
                "next_tool": "python",
                "next_title": "Create AGI Presentation"
            }
            """
        )

    @property
    def prompt_template(self) -> str:
        return """Use the sequential tool to execute a tool and prepare for a subsequent operation. The first tool executes as specified, while the subsequent step is evaluated based on the results.

Example:
{
    "reasoning": "Search for latest AGI papers and prepare for creating a summary presentation",
    "tool": "web_search",
    "web_query": "latest papers on AGI",
    "next_tool": "python",
    "next_title": "Create AGI Presentation"
}"""

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """Execute the tool and provide a summary for deciding on the subsequent step."""
        if not self.validate_params(params):
            return self.format_response(
                "Invalid parameters provided",
                status="error",
                error="Missing required parameters: tool, next_tool, next_title"
            )

        try:
            # Initialize tool manager
            tool_manager = ToolManager()
            
            # Get tool and execute
            tool_name = params["tool"]
            
            # Extract tool-specific parameters
            tool_params = {}
            if tool_name == "web_search" and "web_query" in params:
                tool_params["web_query"] = params["web_query"]
            elif tool_name == "bash" and "command" in params:
                tool_params["command"] = params["command"]
            elif tool_name == "python" and "title" in params and "requirements" in params:
                tool_params["title"] = params["title"]
                tool_params["requirements"] = params["requirements"]
                tool_params["reasoning"] = params.get("reasoning")
            
            print(colored(f"\nExecuting tool: {tool_name}", "cyan"))
            print(colored(f"Reasoning: {params.get('reasoning', 'No specific reasoning provided')}", "cyan"))

            try:
                tool = tool_manager.get_tool(tool_name)()
            except KeyError:
                return self.format_response(
                    "Invalid tool specified",
                    status="error",
                    error=f"Tool '{tool_name}' not found"
                )

            # Execute the tool
            result = await tool.execute(tool_params)

            # Check for errors
            if result.get("status") == "error":
                return self.format_response(
                    "Error in tool execution",
                    status="error",
                    error=f"Tool '{tool_name}' failed: {result.get('error')}"
                )

            # Prepare next step info
            next_step = {
                "tool": params["next_tool"],
                "title": params["next_title"]
            }

            # Generate summary of results and next step
            result_summary = self._generate_result_summary(result, next_step)

            return self.format_response(
                reasoning=params.get("reasoning"),
                status="success",
                tool_result=result,
                next_step=next_step,
                result_summary=result_summary
            )

        except Exception as e:
            return self.format_response(
                "Error executing sequential tool",
                status="error",
                error=str(e)
            )