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
            required_params=["message"],
            example_usage="""
{
    "reasoning": "Enough information is present to faithfully reply directly. The reply should include a, b and c as this will provide a comprehensive answer to the user's prompt about x and y.",
    "tool": "reply",
    "parameters": {
        "message": "Summarizing a, b and c, and their relevance to x and y."
    }
}
"""
        )

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: 'message'"
            )
        
        return self.format_response(
            status="success",
            summary=params["parameters"]["message"]
        ) 