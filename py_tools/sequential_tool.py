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

    def _generate_result_summary(self, first_result: Dict[str, Any], subsequent_intent: str) -> str:
        """Generate a concise summary of the first step's result in context of the subsequent intent."""
        summary_chat = Chat(debug_title="Sequential Tool Result Summary")
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

2. Subsequent intent:
"{subsequent_intent}"

Please provide a concise summary of the first tool's results that would be relevant for achieving this subsequent intent.
Focus only on information that would be useful for the intended next step.
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
                "first_tool_call": {
                    "type": "object",
                    "description": "Complete configuration for the first tool to execute",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "description": "The tool to execute first"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Reasoning for this specific tool usage"
                        }
                    }
                },
                "subsequent_intent": {
                    "type": "string",
                    "description": "A clear statement of what should be done next with the results, including a suggestion of which tool to use (e.g., 'Use python tool to create a visualization of the search results', 'Use reply tool to provide a summary of the findings', etc.)"
                }
            },
            required_params=["first_tool_call", "subsequent_intent"],
            example_usage="""
            {
                "reasoning": "Search for latest AGI papers and prepare for creating a summary presentation",
                "first_tool_call": {
                    "tool": "web_search",
                    "reasoning": "Need to gather current information about AGI developments",
                    "web_query": "latest papers on AGI developments 2024"
                },
                "subsequent_intent": "Use python tool to create a visual presentation summarizing the key AGI research findings"
            }
            """
        )

    @property
    def prompt_template(self) -> str:
        return """Use the sequential tool to execute a tool and prepare for a subsequent operation. The first tool executes as specified, while the subsequent step is determined based on the results.

Example:
{
    "reasoning": "Search for latest AGI papers and prepare for creating a summary presentation",
    "first_tool_call": {
        "tool": "web_search",
        "reasoning": "Need to gather current information about AGI developments",
        "web_query": "latest papers on AGI developments 2024"
    },
    "subsequent_intent": "Use python tool to create a visual presentation summarizing the key AGI research findings"
}"""

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """Execute the tool and provide a summary for deciding on the subsequent step."""
        if not self.validate_params(params):
            return self.format_response(
                "Invalid parameters provided",
                status="error",
                error="Missing required parameters: first_tool_call, subsequent_intent"
            )

        try:
            # Initialize tool manager
            tool_manager = ToolManager()
            
            # Get first tool call parameters
            tool_params = params["first_tool_call"]
            tool_name = tool_params["tool"]
            
            print(colored(f"\nExecuting tool: {tool_name}", "cyan"))
            print(colored(f"Reasoning: {params.get('reasoning', 'No specific reasoning provided')}", "cyan"))
            print(colored(f"Subsequent intent: {params['subsequent_intent']}", "cyan"))

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

            # Get subsequent intent
            subsequent_intent = params["subsequent_intent"]

            # Generate summary of results and next step
            result_summary = self._generate_result_summary(result, subsequent_intent)

            return self.format_response(
                reasoning=params.get("reasoning"),
                status="success",
                tool_result=result,
                subsequent_intent=subsequent_intent,
                result_summary=result_summary
            )

        except Exception as e:
            return self.format_response(
                "Error executing sequential tool",
                status="error",
                error=str(e)
            )