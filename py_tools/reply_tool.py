from typing import Any, Dict

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse

class ReplyTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="reply",
            description="Provide a direct response to the user's query",
            detailed_description="""Use this tool when you need to:
- Provide direct answers to questions
- Ask for clarification
- Explain concepts or ideas
Perfect for:
- Simple questions that don't need other tools
- Requesting more information from the user
- Summarizing information
- Interacting with the user at all""",
            parameters={
                "message": {
                    "type": "string",
                    "description": "The response content to send to the user"
                }
            },
            example_usage="""I need to explain to the user why their Python code is raising an IndexError exception.
```tool_code
reply.run(message="The IndexError in your code is occurring because you're trying to access an element at an index that doesn't exist in your list. Python lists are zero-indexed, so for a list with n elements, valid indices are 0 to n-1. Check the length of your list and make sure you're not attempting to access beyond its bounds.")
```"""
        )

    async def run(self, params: Dict[str, Any]) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: 'message'"
            )
        
        return self.format_response(
            status="success",
            summary=params["parameters"]["message"]
        ) 