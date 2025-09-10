"""
Command Handler for CLI-Agent.

This module extracts the massive command dispatching logic from
get_user_input_with_bindings into a clean, modular system.
"""

from __future__ import annotations
from typing import Dict, List, Any, TYPE_CHECKING

from agent.commands.base_command import (
    BaseCommand, CommandResponse, CommandResult, CommandRegistry,
    register_command_aliases
)

if TYPE_CHECKING:
    from agent.core.app_context import AppContext
    import argparse

class CommandHandler:
    """
    Central command dispatcher for CLI-Agent.
    
    This class manages all command registration, validation, and execution,
    replacing the massive switch-case logic in get_user_input_with_bindings.
    """
    
    def __init__(self, context: AppContext) -> None:
        self.context = context
        self.logger = context.logger
        self.registry: CommandRegistry = {}
        self._register_built_in_commands()
        
    def _register_built_in_commands(self) -> None:
        """Register all built-in commands."""
        # Import and register commands here to avoid circular imports
        from agent.commands.screenshot_command import ScreenshotCommand
        from agent.commands.regenerate_command import RegenerateCommand
        from agent.commands.help_command import HelpCommand
        from agent.commands.simple_commands import (
            MultilineCommand, ExitCommand, ClearCommand, UnsayCommand, 
            PrintCommand, AutoCommand, VoiceCommand, LocalCommand,
            OnlineCommand, FullOutputCommand, LlmCommand, 
            PermissionCommand, SaveCommand, CompactCommand
        )
        
        # Register all commands
        commands = [
            ScreenshotCommand(self.context),
            MultilineCommand(self.context),
            RegenerateCommand(self.context),
            LlmCommand(self.context),
            PermissionCommand(self.context),
            HelpCommand(self.context),
            ExitCommand(self.context),
            SaveCommand(self.context),
            ClearCommand(self.context),
            VoiceCommand(self.context),
            AutoCommand(self.context),
            LocalCommand(self.context),
            OnlineCommand(self.context),
            PrintCommand(self.context),
            UnsayCommand(self.context),
            FullOutputCommand(self.context),
            CompactCommand(self.context),
        ]
        
        for command in commands:
            register_command_aliases(self.registry, command)
            
    def register_command(self, command: BaseCommand) -> None:
        """Register a new command."""
        register_command_aliases(self.registry, command)
        
    def get_command_definitions(self) -> Dict[str, Any]:
        """Get command definitions for auto-completion."""
        defs = {}
        processed_commands = set()
        
        for alias, command in self.registry.items():
            if command not in processed_commands:
                if command.sub_commands:
                    defs[alias] = {sub: None for sub in command.sub_commands}
                else:
                    defs[alias] = None
                processed_commands.add(command)
                
        return defs
        
    def get_command_descriptions(self) -> Dict[str, str]:
        """Get command descriptions for auto-completion tooltips."""
        descs = {}
        processed_commands = set()
        
        for alias, command in self.registry.items():
            if command not in processed_commands:
                descs[alias] = command.description
                
                if command.sub_commands:
                    for sub_cmd, sub_desc in command.sub_commands.items():
                        descs[f"{alias} {sub_cmd}"] = sub_desc
                        
                processed_commands.add(command)
                
        return descs
        
    def is_command(self, user_input: str) -> bool:
        """Check if user input is a command."""
        return user_input.startswith('/')
        
    async def handle_command(
        self, 
        user_input: str, 
        args: argparse.Namespace,
        **kwargs: Any
    ) -> CommandResponse:
        """
        Handle command execution.
        
        Args:
            user_input: Raw user input (e.g., '/help' or '/perms list')
            args: Parsed command line arguments
            **kwargs: Additional context from the calling function
            
        Returns:
            CommandResponse with execution result
        """
        parts = user_input.split()
        if not parts:
            return CommandResponse(
                result=CommandResult.ERROR,
                message="Empty command"
            )
            
        command_alias = parts[0]
        
        # Check if command exists
        if command_alias not in self.registry:
            return CommandResponse(
                result=CommandResult.ERROR,
                message=f"Unknown command '{command_alias}'. Type '/help' for available commands."
            )
            
        command = self.registry[command_alias]
        
        # Validate arguments
        if not command.validate_args(parts):
            return CommandResponse(
                result=CommandResult.ERROR,
                message=f"Invalid arguments for {command_alias}. Use '/help' for usage."
            )
            
        try:
            # Execute command
            self.logger.debug(f"Executing command: {command_alias} with args: {parts[1:]}")
            response = await command.execute(args, parts, **kwargs)
            
            # Log result
            if response.result == CommandResult.ERROR:
                self.logger.error(f"Command {command_alias} failed: {response.message}")
            else:
                self.logger.debug(f"Command {command_alias} completed with result: {response.result}")
                
            return response
            
        except Exception as e:
            self.logger.exception(f"Error executing command {command_alias}")
            return CommandResponse(
                result=CommandResult.ERROR,
                message=f"Command execution failed: {str(e)}"
            )
            
    def get_help_text(self) -> str:
        """Get formatted help text for all commands."""
        help_lines = ["Available commands:"]
        
        # Group commands by their base command object to avoid duplicates
        unique_commands = {}
        for alias, command in self.registry.items():
            if command not in unique_commands.values():
                unique_commands[alias] = command
                
        for alias, command in sorted(unique_commands.items()):
            help_lines.append(f"  {command.get_help_text()}")
            
        return "\n".join(help_lines)
        
    def list_commands(self) -> List[str]:
        """Get list of all command aliases."""
        return list(self.registry.keys())