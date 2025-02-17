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
            "You are an AI assistant that creates intent driven summaries of tool execution results."
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

Please provide a factual summary of the first tool's results that would be relevant for achieving the subsequent intent.
Include a comprehensive summary of the information for the intended next step."""
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
            detailed_description="""Use this tool when you need to:
- Chain multiple operations together
- Use the output of one tool as input for another
- Create multi-step workflows
- Process results before the next step

Perfect for tasks like:
- Search and summarize workflows
- Data gathering
- Complex multi-step automations
- Processing and transforming data across tools""",
            parameters={
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
                    "description": "A clear statement of what should be done next with the results, including a suggestion of which tool(s) may be appropriate (e.g., 'Perform another web search to also retrieve information about x or use reply tool to provide a summary on the findings about y', etc.)"
                }
            },
            required_params=["first_tool_call", "subsequent_intent"],
            example_usage="""
{
    "reasoning": "Before achieving y I should optimally call tool_1 as this will allow me to achieve x relevant to y",
    "tool": "sequential",
    "parameters": {
        "first_tool_call": {
            "tool": "tool_1",
            "reasoning": "Reasoning for using tool_1 in the context of x",
            "parameters": {
                "tool_param_1": "tool_param_1",
                "tool_param_2": "tool_param_2"
            }
        },
        "subsequent_intent": "Use summary of tool_1 to do x enabling meto achieve y"
    }
}
"""
        )

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """Execute the tool and provide a summary for deciding on the subsequent step."""
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameters: first_tool_call, subsequent_intent"
            )

        try:
            # Initialize tool manager
            tool_manager = ToolManager()
            
            # Get first tool call parameters
            tool_params = params["parameters"]["first_tool_call"]
            tool_name = tool_params["tool"]
            
            print(colored(f"\nExecuting tool: {tool_name}", "cyan"))
            print(colored(f"Reasoning: {params.get('reasoning', 'No specific reasoning provided')}", "cyan"))
            print(colored(f"Subsequent intent: {params['parameters'].get('subsequent_intent', 'No specific subsequent intent provided')}", "cyan"))

            try:
                tool = tool_manager.get_tool(tool_name)()
            except KeyError:
                return self.format_response(
                    status="error",
                    summary=f"Tool '{tool_name}' not found"
                )

            # Execute the tool
            result = await tool.execute(tool_params)

            # Check for errors
            if result.get("status") == "error":
                return self.format_response(
                    status="error",
                    summary=f"Tool '{tool_name}' failed: {result.get('error')}"
                )

            # Get subsequent intent
            subsequent_intent = params["parameters"]["subsequent_intent"]

            # Generate summary of results and next step
            result_summary = f"The {result['tool']} tool executed successfully with the following result: {result['summary']}\n\nNext: {subsequent_intent}"

            return self.format_response(
                status="success",
                summary=result_summary
            )

        except Exception as e:
            return self.format_response(
                status="error",
                summary=f"Error executing sequential tool: {str(e)}"
            )