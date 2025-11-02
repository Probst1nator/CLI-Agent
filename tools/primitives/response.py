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
    def run(content: str = "") -> str:
        """
        The response tool delivers the final message to the user.

        In GUI/API mode: Broadcasts content as a text message to frontend.
        In CLI mode: Returns empty string (main loop displays content directly).

        Args:
            content: The text content to display to the user

        Returns:
            The content string in GUI/API mode, empty string in CLI mode
        """
        try:
            import logging

            # Try to broadcast to frontend if visualresponse server is available
            # We check if the server exists rather than relying on flags, since flags
            # may not be properly set in containerized/multi-user setups
            try:
                import time

                from tools.visualresponse import visualresponse

                session = visualresponse._get_session()
                if session and session.is_healthy():
                    # Server is running, so we're in GUI/API mode
                    payload = {
                        'type': 'assistant_message',
                        'action': 'display',
                        'message': content,
                        'id': int(time.time() * 1000000),
                        'timestamp': time.time()
                    }

                    # Use sync send_update which handles async properly
                    session.send_update(payload)

                    logging.info(f"Response tool: Broadcasted text message to frontend ({len(content)} chars)")

                    # Return content so it gets added to chat context
                    return content

            except Exception as e:
                logging.debug(f"Response tool: Failed to broadcast to frontend: {e}")

            # In CLI mode or if broadcast failed, return empty string
            # Main loop will display the content from the <response> tags directly
            return ""

        except Exception as e:
            # If anything fails, maintain backward compatibility
            logging.error(f"Response tool: Unexpected error: {e}")
            return ""