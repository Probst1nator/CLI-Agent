"""Help command implementation."""

from __future__ import annotations
from typing import List, Any, TYPE_CHECKING
from termcolor import colored

from agent.commands.base_command import BaseCommand, CommandResponse, CommandResult

if TYPE_CHECKING:
    import argparse

class HelpCommand(BaseCommand):
    """Command to display help information."""

    @property
    def aliases(self) -> List[str]:
        return ['/h', '/help']

    @property
    def description(self) -> str:
        return 'Show help and available commands'

    async def execute(
        self,
        args: argparse.Namespace,
        parts: List[str],
        **kwargs: Any
    ) -> CommandResponse:
        """Execute help display."""

        # Get command handler from kwargs if available
        command_handler = kwargs.get('command_handler')

        if command_handler:
            help_text = command_handler.get_help_text()
        else:
            # Fallback help text
            help_text = self._get_basic_help()

        print(colored("# CLI-Agent Help", "blue", attrs=["bold"]))
        print(help_text)
        print()
        print(colored("Type any command for assistance or start chatting!", "cyan"))

        return CommandResponse(
            result=CommandResult.CONTINUE,
            message="Help displayed"
        )

    def _get_basic_help(self) -> str:
        """Get basic help text as fallback."""
        return """
Available commands:
  /h, /help: Show help and available commands
  /img, /screenshot: Take a screenshot for visual analysis
  /m, /message: Enter multiline input mode
  /r, /regenerate: Regenerate the last AI response
  /l, /local: Toggle local-only LLM mode
  /llm: Open LLM selection interface
  /perms, /permissions: Manage tool execution permissions
  /e, /exit: Exit CLI-Agent
  /s, /save: Save current LLM selection as default
  /clear: Clear chat history and start fresh
  /a, /auto: Toggle automatic code execution mode
  /v, /voice: Toggle voice input/output mode
  /o, /online: Toggle online mode for web-enabled features
  /p, /print: Print complete chat history
  /fo, /full_output: Show full model output instead of summary
  /unsay: Remove last user message and AI response
  /compact: Compact conversation (create summary)
        """