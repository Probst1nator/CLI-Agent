# tools/think.py
"""
Cognitive tool for the agent to record its thought process, plans, and observations.
"""

class think:
    @staticmethod
    def get_delim() -> str:
        """Delimiter for this tool."""
        return 'think'

    @staticmethod
    def get_tool_info() -> dict:
        """Tool documentation for the agent's system prompt."""
        return {
            "name": "think",
            "description": "A cognitive tool for reasoning, planning, and self-correction. Use this to formulate a plan, reflect on previous actions, and decide what to do next. It produces no direct output.",
            "example": "<think>The previous command failed. I need to check the file permissions before trying again.</think>"
        }

    @staticmethod
    def run(content: str) -> None:
        """
        The think tool is a cognitive placeholder. It performs no action and
        returns no output to the agent's context.
        """
        return None
