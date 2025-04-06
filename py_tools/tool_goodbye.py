from typing import Any, Dict

from py_tools.cls_base_tool import BaseTool, ToolMetadata, ToolResponse
from py_classes.cls_chat import Chat

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
            constructor="""
def run(reply: str) -> None:
    \"\"\"End the conversation with a farewell message.
    
    Args:
        reply: The farewell message to send
    \"\"\"
"""
        )

    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: reply"
            )
        
        return self.format_response(
            status="success",
            summary=params["parameters"]["reply"]
        ) 