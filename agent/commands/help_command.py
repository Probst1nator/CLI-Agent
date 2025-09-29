"""Help command implementation."""

from __future__ import annotations
from typing import List, Any, TYPE_CHECKING
from termcolor import colored

from agent.commands.base_command import BaseCommand, CommandResponse, CommandResult

if TYPE_CHECKING:
    pass

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
        parts: List[str],
        **kwargs: Any
    ) -> CommandResponse:
        """Execute help display."""

        # Get command handler from kwargs if available, or create one from context
        command_handler = kwargs.get('command_handler')

        if not command_handler:
            # Create a command handler from our context to get the enhanced help
            try:
                from agent.commands.command_handler import CommandHandler
                command_handler = CommandHandler(self.context)
            except Exception as e:
                self.logger.warning(f"Could not create command handler for help: {e}")
                command_handler = None

        if command_handler:
            help_text = command_handler.get_help_text()
        else:
            # Ultimate fallback help text
            help_text = self._get_basic_help()

        print(colored("# CLI-Agent Help", "blue", attrs=["bold"]))
        print(help_text)
        print()
        print(colored("💡 Tip:", "yellow", attrs=["bold"]) + colored(" Type any command for assistance or start chatting!", "cyan"))

        return CommandResponse(
            result=CommandResult.CONTINUE,
            message="Help displayed"
        )

    def _get_basic_help(self) -> str:
        """Get basic help text as fallback with colors."""
        help_lines = [
            colored("Available commands:", "yellow", attrs=["bold"]),
            f"  {colored('/h, /help', 'green', attrs=['bold'])}: {colored('Show help and available commands', 'white')}",
            f"  {colored('/img, /screenshot', 'green', attrs=['bold'])}: {colored('Take a screenshot for visual analysis', 'white')}",
            f"  {colored('/m, /message', 'green', attrs=['bold'])}: {colored('Enter multiline input mode', 'white')}",
            f"  {colored('/r, /regenerate', 'green', attrs=['bold'])}: {colored('Regenerate the last AI response', 'white')}",
            f"  {colored('/l, /local', 'green', attrs=['bold'])}: {colored('Toggle local-only LLM mode', 'white')}",
            f"  {colored('/llm', 'green', attrs=['bold'])}: {colored('Open LLM selection interface', 'white')}",
            f"  {colored('/perms, /permissions', 'green', attrs=['bold'])}: {colored('Manage tool execution permissions', 'white')}",
            f"  {colored('/e, /exit', 'green', attrs=['bold'])}: {colored('Exit CLI-Agent', 'white')}",
            f"  {colored('/s, /save', 'green', attrs=['bold'])}: {colored('Save current LLM selection as default', 'white')}",
            f"  {colored('/clear', 'green', attrs=['bold'])}: {colored('Clear chat history and start fresh', 'white')}",
            f"  {colored('/yolo, /y', 'green', attrs=['bold'])}: {colored('Toggle YOLO mode (automatic code execution)', 'white')}",
            f"  {colored('/v, /voice', 'green', attrs=['bold'])}: {colored('Toggle voice input/output mode', 'white')}",
            f"  {colored('/o, /online', 'green', attrs=['bold'])}: {colored('Toggle online mode for web-enabled features', 'white')}",
            f"  {colored('/p, /print', 'green', attrs=['bold'])}: {colored('Print complete chat history', 'white')}",
            f"  {colored('/fo, /full_output', 'green', attrs=['bold'])}: {colored('Show full model output instead of summary', 'white')}",
            f"  {colored('/unsay', 'green', attrs=['bold'])}: {colored('Remove last user message and AI response', 'white')}",
            f"  {colored('/compact', 'green', attrs=['bold'])}: {colored('Compact conversation (create summary)', 'white')}"
        ]
        return "\n".join(help_lines)