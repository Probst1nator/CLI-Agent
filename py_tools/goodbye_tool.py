from typing import Any, Dict

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse

class GoodbyeTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="goodbye",
            description="End the conversation or system interaction",
            detailed_description="""Use this tool when you need to:
- Acknowledge a user's request to exit

Perfect for:
- User requests to end interaction""",
            parameters={
                "reply": {
                    "type": "string",
                    "description": "The farewell message to send"
                }
            },
            required_params=["reply"],
            example_usage="""
{
    "reasoning": "User has requested to end the conversation",
    "tool": "goodbye",
    "parameters": {
        "reply": "Friendly and quirky farewell message referencing the conversation concisely"
    }
}
"""
        )

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: reply"
            )
        
        return self.format_response(
            status="success",
            summary=params["parameters"]["reply"]
        ) 