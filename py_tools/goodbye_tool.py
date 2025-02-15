from typing import Any, Dict

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse

class GoodbyeTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="goodbye",
            description="End the conversation or system interaction",
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
                "reply": "Goodbye! Thank you for using the CLI agent."
            }
            """
        )

    @property
    def prompt_template(self) -> str:
        return """
        Use the goodbye tool when:
        1. The user explicitly requests to end the conversation
        2. The user wants to exit the system
        3. All requested tasks have been completed and it's appropriate to end
        
        Always provide a polite farewell message.
        
        Example:
        User: "Goodbye" or "exit"
        Response: {
            "reasoning": "User has requested to end the conversation",
            "tool": "goodbye",
            "reply": "Thank you for your time. Have a great day!"
        }
        """

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                error="Missing required parameter: reply"
            )
        
        return self.format_response(
            reasoning=params.get("reasoning", "Ending conversation as requested"),
            reply=params["reply"]
        ) 