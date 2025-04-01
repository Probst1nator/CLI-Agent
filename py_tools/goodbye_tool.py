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
            example_usage="""The user has asked to end the conversation, so I'll acknowledge their request with a friendly farewell.
```tool_code
goodbye.run(reply="Thanks for chatting with me today! I've enjoyed helping you with your Python project. If you have any more questions in the future, feel free to start another conversation. Have a great day!")
```"""
        )

    async def run(self, params: Dict[str, Any]) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: reply"
            )
        
        return self.format_response(
            status="success",
            summary=params["parameters"]["reply"]
        ) 