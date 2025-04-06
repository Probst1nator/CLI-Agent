from typing import Any, Dict

from py_tools.cls_base_tool import BaseTool, ToolMetadata, ToolResponse
from py_classes.cls_chat import Chat

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
            constructor="""
def run(message: str) -> None:
    \"\"\"Send a message directly to the user.
    
    Args:
        message: The response content to send to the user
    \"\"\"
"""
        )

    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: 'message'"
            )
        
        return self.format_response(
            status="success",
            summary=params["parameters"]["message"]
        ) 