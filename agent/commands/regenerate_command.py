"""Regenerate command implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from termcolor import colored

from agent.commands.base_command import BaseCommand, CommandResponse, CommandResult

if TYPE_CHECKING:
    pass

class RegenerateCommand(BaseCommand):
    """Command to regenerate the last AI response."""

    @property
    def aliases(self) -> list[str]:
        return ['/r', '/regenerate']

    @property
    def description(self) -> str:
        return 'Regenerate the last AI response'

    async def execute(
        self,
        parts: list[str],
        **kwargs: Any
    ) -> CommandResponse:
        """Execute regeneration."""

        context_chat = kwargs.get('context_chat')

        if not context_chat or len(context_chat.messages) < 2:
            self.logger.error(colored("# cli-agent: No chat history found, cannot regenerate.", "red"))
            return CommandResponse(
                result=CommandResult.CONTINUE,
                message="No chat history available for regeneration"
            )

        self.logger.info(colored("# cli-agent: Regenerating last response.", "green"))

        # Remove the last AI response to trigger regeneration
        context_chat.messages.pop()

        return CommandResponse(
            result=CommandResult.REGENERATE,
            message="Regenerating last response",
            user_input="",  # Empty input triggers regeneration
            should_regenerate=True
        )