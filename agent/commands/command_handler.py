"""
Command Handler for CLI-Agent.

This module extracts the massive command dispatching logic from
get_user_input_with_bindings into a clean, modular system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.commands.base_command import (
    BaseCommand,
    CommandRegistry,
    CommandResponse,
    CommandResult,
    register_command_aliases,
)

if TYPE_CHECKING:
    from agent.core.app_context import AppContext

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
        from agent.commands.help_command import HelpCommand
        from agent.commands.profile_commands import (
            ProfileDeleteCommand,
            ProfileListCommand,
            ProfileLoadCommand,
            ProfileSaveCommand,
        )
        from agent.commands.regenerate_command import RegenerateCommand
        from agent.commands.screenshot_command import ScreenshotCommand
        from agent.commands.simple_commands import (
            CompactCommand,
            ConfigCommand,
            ExitCommand,
            FallbacksCommand,
            FullOutputCommand,
            LlmCommand,
            LocalCommand,
            MultilineCommand,
            OnlineCommand,
            OverrideLlmCommand,
            PermissionCommand,
            PlannerLlmCommand,
            PrintCommand,
            ResetCommand,
            RouterLlmCommand,
            SaveCommand,
            ToolsCommand,
            UnsayCommand,
            VoiceCommand,
            YoloCommand,
        )
        from agent.commands.tool_commands import DisableToolCommand, EnableToolCommand

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
            ResetCommand(self.context),
            VoiceCommand(self.context),
            YoloCommand(self.context),
            LocalCommand(self.context),
            OnlineCommand(self.context),
            PrintCommand(self.context),
            UnsayCommand(self.context),
            FullOutputCommand(self.context),
            CompactCommand(self.context),
            PlannerLlmCommand(self.context),
            RouterLlmCommand(self.context),
            OverrideLlmCommand(self.context),
            FallbacksCommand(self.context),
            ConfigCommand(self.context),
            ToolsCommand(self.context),
            ProfileSaveCommand(self.context),
            ProfileLoadCommand(self.context),
            ProfileListCommand(self.context),
            ProfileDeleteCommand(self.context),
            EnableToolCommand(self.context),
            DisableToolCommand(self.context),
        ]

        for command in commands:
            register_command_aliases(self.registry, command)

            # Also register with the global command registry
            from agent.core.command_registry import register_command_implementation
            register_command_implementation(command)

    def register_command(self, command: BaseCommand) -> None:
        """Register a new command."""
        register_command_aliases(self.registry, command)

        # Also register with the global command registry
        from agent.core.command_registry import register_command_implementation
        register_command_implementation(command)

    def get_command_definitions(self) -> dict[str, Any]:
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

    def get_command_descriptions(self) -> dict[str, str]:
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
        **kwargs: Any
    ) -> CommandResponse:
        """
        Handle command execution.

        Args:
            user_input: Raw user input (e.g., '/help' or '/perms list')
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
            response = await command.execute(parts, **kwargs)

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
        """Get formatted help text for all commands with enhanced coloring."""
        from termcolor import colored

        # Categorize commands for better organization
        categories = {
            "Core Commands": ["/h", "/help", "/e", "/exit"],
            "Input & Media": ["/m", "/message", "/img", "/screenshot", "/v", "/voice"],
            "Chat Management": ["/r", "/regenerate", "/p", "/print", "/clear", "/new_chat", "/new", "/compact", "/unsay"],
            "Configuration": ["/l", "/local", "/o", "/online", "/yolo", "/y", "/fo", "/full_output", "/config", "/c"],
            "LLM Management": ["/llm", "/planner_llm", "/router_llm", "/fallbacks", "/s", "/save"],
            "Permissions": ["/perms", "/permissions"]
        }

        # Group commands by their base command object to avoid duplicates
        unique_commands = {}
        for alias, command in self.registry.items():
            if command not in unique_commands.values():
                unique_commands[alias] = command

        help_lines = [colored("Available Commands:", "yellow", attrs=["bold"])]

        # Organize commands by category
        categorized_commands = {}
        uncategorized_commands = {}

        for alias, command in unique_commands.items():
            found_category = None
            for category, category_aliases in categories.items():
                if alias in category_aliases:
                    if category not in categorized_commands:
                        categorized_commands[category] = []
                    categorized_commands[category].append((alias, command))
                    found_category = category
                    break

            if not found_category:
                uncategorized_commands[alias] = command

        # Display categorized commands
        for category in categories:
            if category in categorized_commands:
                help_lines.append("")
                help_lines.append(colored(f"  {category}:", "cyan", attrs=["bold"]))
                for alias, command in sorted(categorized_commands[category]):
                    help_lines.append(f"    {command.get_help_text()}")

        # Display any uncategorized commands
        if uncategorized_commands:
            help_lines.append("")
            help_lines.append(colored("  Other Commands:", "cyan", attrs=["bold"]))
            for alias, command in sorted(uncategorized_commands.items()):
                help_lines.append(f"    {command.get_help_text()}")

        return "\n".join(help_lines)

    def list_commands(self) -> list[str]:
        """Get list of all command aliases."""
        return list(self.registry.keys())