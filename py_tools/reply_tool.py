from typing import Any, Dict

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse

class ReplyTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="reply",
            description="Provide a direct response to the user's query",
            parameters={
                "reply": {
                    "type": "string",
                    "description": "The response content to send to the user"
                }
            },
            required_params=["reply"],
            example_usage="""
            {
                "reasoning": "A direct response is appropriate here",
                "tool": "reply",
                "reply": "Here's the information you requested..."
            }
            """
        )

    @property
    def prompt_template(self) -> str:
        return """
        Use the reply tool when you can directly answer the user's query without needing other tools.
        Always include clear reasoning for why a direct response is sufficient.
        The user can only see the reply, not the reasoning, so make sure the reply is clear and complete.
        
        Example:
        User: "What is Python?"
        Response: {
            "reasoning": "This is a general knowledge question about Python that can be answered directly",
            "tool": "reply",
            "reply": "Python is a high-level, interpreted programming language..."
        }
        """

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                "Invalid parameters provided",
                status="error",
                error="Missing required parameter: reply"
            )
        
        return self.format_response(
            reasoning=params.get("reasoning", "Direct response provided"),
            reply=params["reply"]
        ) 