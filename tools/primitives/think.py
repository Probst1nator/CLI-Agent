# tools/primitives/think.py
"""
This file implements the 'think' tool. It serves as a cognitive step for
the agent, allowing it to record its thought process, plans, and observations
without producing an operational side effect.
"""

class think:
    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool, used for parsing agent output."""
        return 'think'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool for the agent's system prompt."""
        return {
            "name": "think",
            "description": "A cognitive tool for reasoning, planning, and self-correction. Use this to formulate a plan, reflect on previous actions, and decide what to do next. It produces no direct output.",
            "example": "<think>The previous command failed. I need to check the file permissions before trying again.</think>"
        }

    @staticmethod
    def run(content: str) -> None:
        """
        The think tool is a cognitive placeholder. It performs no action and
        returns no output to the agent's context. Its content is for the
        agent's own reasoning process, visible in the full chat history.
        """
        # This tool is a no-op by design. Its purpose is to structure the agent's
        # thought process within the conversation log.
        return None