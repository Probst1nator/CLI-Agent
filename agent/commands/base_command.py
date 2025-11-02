"""
Base command interface for CLI-Agent commands.

This module defines the abstract base class and protocols for all
CLI-Agent commands, ensuring consistent interfaces and proper typing.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.core.app_context import AppContext

class CommandResult(Enum):
    """Command execution result types."""
    SUCCESS = "success"
    ERROR = "error"
    CONTINUE = "continue"  # Continue to next input
    REGENERATE = "regenerate"  # Trigger regeneration
    EXIT = "exit"  # Exit the application

@dataclass
class CommandResponse:
    """Standardized command response with proper typing."""
    result: CommandResult
    message: str = ""
    base64_images: list[str] | None = None
    user_input: str = ""
    should_regenerate: bool = False

    def __post_init__(self) -> None:
        if self.base64_images is None:
            self.base64_images = []

class BaseCommand(abc.ABC):
    """
    Abstract base class for all CLI-Agent commands.

    All commands must inherit from this class and implement the execute method.
    This ensures consistent interfaces and enables dependency injection.
    """

    def __init__(self, context: AppContext) -> None:
        self.context = context
        self.logger = context.logger

    @property
    @abc.abstractmethod
    def aliases(self) -> list[str]:
        """List of command aliases (e.g., ['/r', '/regenerate'])."""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Command description for help text."""
        pass

    @property
    def sub_commands(self) -> dict[str, str] | None:
        """Optional sub-commands with descriptions."""
        return None

    @property
    def usage(self) -> str | None:
        """Optional usage information for help text."""
        return None

    @abc.abstractmethod
    async def execute(
        self,
        parts: list[str],
        **kwargs: Any
    ) -> CommandResponse:
        """
        Execute the command.

        Args:
            parts: Command parts (e.g., ['/perms', 'list'])
            **kwargs: Additional command-specific arguments

        Returns:
            CommandResponse with result and any data
        """
        pass

    def validate_args(self, parts: list[str]) -> bool:
        """Validate command arguments. Override for custom validation."""
        return True

    def get_help_text(self) -> str:
        """Get formatted help text for this command with colors."""
        from termcolor import colored

        # Check if command has usage information
        usage_info = ""
        if self.usage:
            usage_info = f" {colored(self.usage, 'yellow')}"

        # Color the aliases and description differently
        aliases_text = colored(', '.join(self.aliases), 'green', attrs=['bold'])
        description_text = colored(self.description, 'white')
        help_text = f"{aliases_text}{usage_info}: {description_text}"

        if self.sub_commands:
            help_text += f"\n      {colored('Sub-commands:', 'yellow')}:"
            for sub_cmd, desc in self.sub_commands.items():
                sub_cmd_colored = colored(sub_cmd, 'green')
                sub_desc_colored = colored(desc, 'white')
                help_text += f"\n        {sub_cmd_colored}: {sub_desc_colored}"

        return help_text



# Utility functions for command registration
CommandRegistry = dict[str, BaseCommand]

def register_command_aliases(
    registry: CommandRegistry,
    command: BaseCommand
) -> None:
    """Register all aliases for a command in the registry."""
    for alias in command.aliases:
        registry[alias] = command