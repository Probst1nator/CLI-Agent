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
            description="Execute multiple tools in sequence, passing results between them as needed. Use this when you need to chain multiple operations together.",
            parameters={
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of why this sequence of tools is needed"
                },
                "steps": {
                    "type": "array",
                    "description": "List of tool calls to execute in sequence. Each step should be a complete tool call object with all required parameters.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "params": {"type": "object"}
                        }
                    }
                }
            },
            required_params=["steps"],
            example_usage="""
            {
                "reasoning": "Search for latest AGI papers and create a summary presentation.",
                "tool": "sequential",
                "steps": [
                    {
                        "tool": "web_search",
                        "reasoning": "Find the latest papers on AGI.",
                        "query": "latest papers on AGI"
                    },
                    {
                        "tool": "python",
                        "reasoning": "Create a presentation based on the search results.",
                        "title": "agi_presentation.py",
                        "requirements": "Create a presentation in python using streamlit. The presentation should be interactive and allow the user to grasp the presented implications in the latest papers on AGI."
                    }
                ]
            }
            """
        )

    @property
    def prompt_template(self) -> str:
        return """Use the sequential tool to execute tools in sequence. The first step executes as specified, while subsequent steps are evaluated based on previous results.

Example:
{
    "reasoning": "Search for latest AGI papers and create a summary presentation.",
    "tool": "sequential",
    "steps": [
        {
            "tool": "web_search",
            "reasoning": "Find the latest papers on AGI",
            "query": "latest papers on AGI"
        },
        {
            "tool": "python",
            "reasoning": "Create a presentation based on the search results",
            "title": "agi_presentation.py",
            "requirements": "Create a streamlit presentation based on the found papers"
        }
    ]
}"""

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """Execute the first tool and provide a summary for deciding on subsequent steps."""
        if not self.validate_params(params):
            return self.format_response(
                "Invalid parameters provided",
                status="error",
                error="Missing required parameters: reasoning and steps"
            )

        try:
            steps = params["steps"]
            if not isinstance(steps, list) or len(steps) < 2:
                return self.format_response(
                    "Invalid steps parameter",
                    status="error",
                    error="At least two steps must be provided"
                )

            # Sanitize steps
            sanitized_steps = [self._sanitize_step(step) for step in steps]
            first_step = sanitized_steps[0]
            next_step = sanitized_steps[1]

            # Initialize tool manager
            tool_manager = ToolManager()
            
            # Execute first step
            if not isinstance(first_step, dict) or "tool" not in first_step:
                return self.format_response(
                    "Invalid first step",
                    status="error",
                    error="First step must be a dictionary with at least a 'tool' key"
                )

            tool_name = first_step["tool"]
            print(colored(f"\nExecuting first step: {tool_name}", "cyan"))
            print(colored(f"Reasoning: {first_step.get('reasoning', 'No specific reasoning provided')}", "cyan"))

            try:
                tool = tool_manager.get_tool(tool_name)()
            except KeyError:
                return self.format_response(
                    "Invalid tool in first step",
                    status="error",
                    error=f"Tool '{tool_name}' not found"
                )

            # Execute first tool
            first_result = await tool.execute(first_step)

            # Check for errors in first step
            if first_result.get("status") == "error":
                return self.format_response(
                    "Error in first step",
                    status="error",
                    error=f"Tool '{tool_name}' failed: {first_result.get('error')}"
                )

            # Generate summary of results and next step
            result_summary = self._generate_result_summary(first_result, next_step)

            return self.format_response(
                reasoning=params["reasoning"],
                status="success",
                first_step_result=first_result,
                next_step=next_step,
                result_summary=result_summary,
                remaining_steps=sanitized_steps[2:] if len(sanitized_steps) > 2 else []
            )

        except Exception as e:
            return self.format_response(
                "Error executing sequential tool",
                status="error",
                error=str(e)
            )