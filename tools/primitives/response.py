# tools/response.py
"""
This file implements the 'response' tool. It is a structural tool used by the
agent to deliver its final, conversational answer to the user, signaling the
end of its operational turn.
"""

class response:
    """
    A structural tool to wrap the agent's final conversational output.
    This tool performs no action and adds no content to the agent's context.
    Its presence indicates that the agent's turn is complete.
    """
    
    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool, used for parsing agent output."""
        return 'response'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool for the agent's system prompt."""
        return {
            "name": "response",
            "description": "Delivers the final, conversational answer to the user. Use this after all other tools have been used to complete your turn.",
            "example": "<response>I have successfully read the file and here is the summary.</response>"
        }

    @staticmethod
    def run(content: str = "") -> None:
        """
        The response tool is a structural placeholder. It performs no action and
        returns no output to the agent's context. Its content is the final
        message delivered to the user.
        """
        # This tool is a no-op by design. The main loop will display the
        # content within the <response> tags to the user, and this tool's
        # execution produces no further output for the agent's context.
        return None