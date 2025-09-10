"""
Base command interface for CLI-Agent commands.

This module defines the abstract base class and protocols for all
CLI-Agent commands, ensuring consistent interfaces and proper typing.
"""

from __future__ import annotations
import abc
from typing import Protocol, List, Optional, Any, Dict, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from agent.core.app_context import AppContext
    import argparse

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
    base64_images: List[str] = None
    user_input: str = ""
    
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
    def aliases(self) -> List[str]:
        """List of command aliases (e.g., ['/r', '/regenerate'])."""
        pass
        
    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Command description for help text."""
        pass
        
    @property
    def sub_commands(self) -> Optional[Dict[str, str]]:
        """Optional sub-commands with descriptions."""
        return None
        
    @abc.abstractmethod
    async def execute(
        self, 
        args: argparse.Namespace,
        parts: List[str],
        **kwargs: Any
    ) -> CommandResponse:
        """
        Execute the command.
        
        Args:
            args: Parsed command line arguments
            parts: Command parts (e.g., ['/perms', 'list'])
            **kwargs: Additional command-specific arguments
            
        Returns:
            CommandResponse with result and any data
        """
        pass
        
    def validate_args(self, parts: List[str]) -> bool:
        """Validate command arguments. Override for custom validation."""
        return True
        
    def get_help_text(self) -> str:
        """Get formatted help text for this command."""
        help_text = f"{', '.join(self.aliases)}: {self.description}"
        
        if self.sub_commands:
            help_text += "\n  Sub-commands:"
            for sub_cmd, desc in self.sub_commands.items():
                help_text += f"\n    {sub_cmd}: {desc}"
                
        return help_text

class AsyncCommand(Protocol):
    """Protocol for async command execution."""
    
    async def execute(
        self, 
        context: AppContext,
        args: argparse.Namespace,
        parts: List[str],
        **kwargs: Any
    ) -> CommandResponse:
        """Execute async command."""
        ...

class SyncCommand(Protocol):
    """Protocol for sync command execution."""
    
    def execute(
        self, 
        context: AppContext,
        args: argparse.Namespace,
        parts: List[str],
        **kwargs: Any
    ) -> CommandResponse:
        """Execute sync command."""
        ...

# Utility functions for command registration
CommandRegistry = Dict[str, BaseCommand]

def register_command_aliases(
    registry: CommandRegistry, 
    command: BaseCommand
) -> None:
    """Register all aliases for a command in the registry."""
    for alias in command.aliases:
        registry[alias] = command