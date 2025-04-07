from typing import Any, Dict

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse, ToolStatus
from py_classes.cls_chat import Chat

class GoodbyeTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="goodbye",
            description="End the current conversation with the user",
            detailed_description="""Use this tool when:
- The user explicitly asks to end the conversation
- The task is complete and there's nothing more to say
- The user says 'bye', 'goodbye', 'exit', etc.""",
            constructor="""
def run(message: str = "Goodbye! I hope I was helpful. Feel free to start a new conversation anytime.") -> None:
    \"\"\"End the current conversation with the user.
    
    Args:
        message: Optional custom goodbye message
    \"\"\"
"""
        )

    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        # Get the message parameter or use default
        parameters = params.get("parameters", {})
        message = parameters.get("message", "Goodbye! I hope I was helpful. Feel free to start a new conversation anytime.")
        
        # Mark the chat as complete
        context_chat.complete = True
        
        return self.format_response(
            status=ToolStatus.SUCCESS,
            summary=message
        ) 