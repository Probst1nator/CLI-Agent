"""
Single source of truth for command definitions in CLI-Agent.

This module provides a centralized registry for all commands, replacing
the scattered command definitions across multiple files and ensuring
consistency between static definitions and dynamic command registration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from agent.commands.base_command import BaseCommand


@dataclass
class CommandDefinition:
    """Canonical definition of a command."""
    canonical: str
    aliases: list[str]
    description: str
    sub_commands: Optional[dict[str, str]] = None
    implementation: Optional[BaseCommand] = None
    category: str = "general"
    deprecated: bool = False
    hidden: bool = False

    def __post_init__(self):
        if self.sub_commands is None:
            self.sub_commands = {}


class CommandRegistry:
    """
    Centralized registry for all CLI-Agent commands.

    This class serves as the single source of truth for command definitions,
    bridging static definitions and dynamic command implementations.
    """

    def __init__(self):
        self.logger = logging.getLogger("command_registry")
        self._commands: dict[str, CommandDefinition] = {}
        self._alias_map: dict[str, str] = {}
        self._initialized = False

    def initialize_builtin_commands(self) -> None:
        """Initialize built-in command definitions."""
        if self._initialized:
            return

        # Define all built-in commands
        builtin_commands = [
            # Core interaction commands
            CommandDefinition(
                canonical="screenshot",
                aliases=["/img", "/screenshot"],
                description="Take a screenshot for visual analysis",
                category="interaction"
            ),
            CommandDefinition(
                canonical="multiline",
                aliases=["/m", "/message"],
                description="Enter multiline input mode",
                category="interaction"
            ),
            CommandDefinition(
                canonical="regenerate",
                aliases=["/r", "/regenerate"],
                description="Regenerate the last AI response",
                category="interaction"
            ),

            # Configuration commands
            CommandDefinition(
                canonical="local",
                aliases=["/l", "/local"],
                description="Toggle local-only LLM mode",
                category="configuration"
            ),
            CommandDefinition(
                canonical="llm",
                aliases=["/llm"],
                description="Open LLM selection interface",
                category="configuration"
            ),
            CommandDefinition(
                canonical="yolo_mode",
                aliases=["/yolo", "/y"],
                description="Toggle YOLO mode (automatic code execution)",
                category="configuration"
            ),
            CommandDefinition(
                canonical="voice_mode",
                aliases=["/v", "/voice"],
                description="Toggle voice input/output mode",
                category="configuration"
            ),
            CommandDefinition(
                canonical="full_output",
                aliases=["/fo", "/full_output"],
                description="Show full model output instead of summary",
                category="configuration"
            ),
            CommandDefinition(
                canonical="online",
                aliases=["/o", "/online"],
                description="Toggle online mode for web-enabled features",
                category="configuration"
            ),
            CommandDefinition(
                canonical="planner_llm",
                aliases=["/planner_llm"],
                description="Set the planner LLM model",
                category="configuration"
            ),
            CommandDefinition(
                canonical="router_llm",
                aliases=["/router_llm"],
                description="Set the router LLM model",
                category="configuration"
            ),
            CommandDefinition(
                canonical="fallbacks",
                aliases=["/fallbacks"],
                description="Set the fallback LLM models",
                category="configuration"
            ),
            CommandDefinition(
                canonical="override_llm",
                aliases=["/override_llm"],
                description="Override the LLM model selection",
                category="configuration"
            ),
            CommandDefinition(
                canonical="config",
                aliases=["/config", "/c"],
                description="Display current agent and model configuration",
                category="configuration"
            ),

            # Profile management
            CommandDefinition(
                canonical="profile_save",
                aliases=["/profile_save", "/psave"],
                description="Save current LLM configuration as a profile",
                category="configuration"
            ),
            CommandDefinition(
                canonical="profile_load",
                aliases=["/profile_load", "/pload"],
                description="Load a saved LLM configuration profile",
                category="configuration"
            ),
            CommandDefinition(
                canonical="profile_list",
                aliases=["/profile_list", "/plist", "/profiles"],
                description="List all saved LLM configuration profiles",
                category="configuration"
            ),
            CommandDefinition(
                canonical="profile_delete",
                aliases=["/profile_delete", "/pdelete"],
                description="Delete a saved LLM configuration profile",
                category="configuration"
            ),

            # Session management commands
            CommandDefinition(
                canonical="save",
                aliases=["/s", "/save"],
                description="Save current LLM selection as default",
                category="session"
            ),
            CommandDefinition(
                canonical="clear",
                aliases=["/clear"],
                description="Clear chat history and start fresh",
                category="session"
            ),
            CommandDefinition(
                canonical="reset",
                aliases=["/reset"],
                description="Complete agent reset (chat, tools, images, context)",
                category="session"
            ),
            CommandDefinition(
                canonical="compact",
                aliases=["/compact"],
                description="Compact conversation (create summary)",
                category="session"
            ),
            CommandDefinition(
                canonical="unsay",
                aliases=["/unsay"],
                description="Remove last user message and AI response",
                category="session"
            ),

            # Information commands
            CommandDefinition(
                canonical="help",
                aliases=["/h", "/help"],
                description="Show help and available commands",
                category="information"
            ),
            CommandDefinition(
                canonical="print",
                aliases=["/p", "/print"],
                description="Print complete chat history",
                category="information"
            ),
            CommandDefinition(
                canonical="tools",
                aliases=["/tools", "/tool", "/t", "/health"],
                description="Display detailed tool health status",
                category="information"
            ),

            # System commands
            CommandDefinition(
                canonical="exit",
                aliases=["/e", "/exit"],
                description="Exit CLI-Agent",
                category="system"
            ),

            # Tool management
            CommandDefinition(
                canonical="enable_tool",
                aliases=["/enable", "/enable-tool"],
                description="Enable one or more disabled tools",
                category="configuration"
            ),
            CommandDefinition(
                canonical="disable_tool",
                aliases=["/disable", "/disable-tool"],
                description="Disable one or more tools",
                category="configuration"
            ),

            # Permission management
            CommandDefinition(
                canonical="permissions",
                aliases=["/perms", "/permissions"],
                description="Manage tool execution permissions",
                category="security",
                sub_commands={
                    "list": "List all saved permissions",
                    "remove": "Remove specific permission by ID",
                    "clear": "Clear all saved permissions",
                    "interactive": "Open interactive permission manager",
                    "manage": "Open interactive permission manager",
                }
            ),
        ]

        # Register all built-in commands
        for cmd_def in builtin_commands:
            self.register_command_definition(cmd_def)

        self._initialized = True
        self.logger.info(f"Initialized {len(builtin_commands)} built-in commands")

    def register_command_definition(self, command_def: CommandDefinition) -> None:
        """Register a command definition."""
        # Store the command definition
        self._commands[command_def.canonical] = command_def

        # Update alias mappings
        for alias in command_def.aliases:
            if alias in self._alias_map and self._alias_map[alias] != command_def.canonical:
                self.logger.warning(f"Alias '{alias}' conflicts with existing command")
            self._alias_map[alias] = command_def.canonical

        self.logger.debug(f"Registered command: {command_def.canonical} with aliases {command_def.aliases}")

    def register_command_implementation(self, implementation: BaseCommand) -> None:
        """Register a command implementation with an existing definition."""
        # Find matching command definition by aliases
        matching_canonical = None

        for alias in implementation.aliases:
            if alias in self._alias_map:
                matching_canonical = self._alias_map[alias]
                break

        if matching_canonical and matching_canonical in self._commands:
            self._commands[matching_canonical].implementation = implementation
            self.logger.debug(f"Registered implementation for: {matching_canonical}")
        else:
            # Create a new definition for this implementation
            self.logger.warning(f"No definition found for implementation with aliases {implementation.aliases}")
            new_def = CommandDefinition(
                canonical=implementation.aliases[0][1:] if implementation.aliases[0].startswith('/') else implementation.aliases[0],
                aliases=implementation.aliases,
                description=implementation.description,
                implementation=implementation,
                category="dynamic"
            )
            self.register_command_definition(new_def)

    def get_command_by_alias(self, alias: str) -> Optional[CommandDefinition]:
        """Get command definition by alias."""
        canonical = self._alias_map.get(alias)
        return self._commands.get(canonical) if canonical else None

    def get_command_by_canonical(self, canonical: str) -> Optional[CommandDefinition]:
        """Get command definition by canonical name."""
        return self._commands.get(canonical)

    def get_all_commands(self) -> dict[str, CommandDefinition]:
        """Get all registered commands."""
        return self._commands.copy()

    def get_commands_by_category(self, category: str) -> list[CommandDefinition]:
        """Get all commands in a specific category."""
        return [cmd for cmd in self._commands.values() if cmd.category == category]

    def get_all_aliases(self) -> dict[str, str]:
        """Get mapping of all aliases to canonical names."""
        return self._alias_map.copy()

    def list_categories(self) -> set[str]:
        """Get all command categories."""
        return {cmd.category for cmd in self._commands.values()}

    def get_command_definitions_for_completion(self) -> dict[str, Any]:
        """Get command definitions in format expected by completion system."""
        definitions = {}
        for alias, canonical in self._alias_map.items():
            command_def = self._commands[canonical]
            if command_def.sub_commands:
                definitions[alias] = {sub: None for sub in command_def.sub_commands}
            else:
                definitions[alias] = None
        return definitions

    def get_command_descriptions_for_completion(self) -> dict[str, str]:
        """Get command descriptions in format expected by completion system."""
        descriptions = {}

        # Add main command descriptions
        for alias, canonical in self._alias_map.items():
            command_def = self._commands[canonical]
            descriptions[alias] = command_def.description

            # Add sub-command descriptions
            if command_def.sub_commands:
                for sub_cmd, sub_desc in command_def.sub_commands.items():
                    descriptions[f"{alias} {sub_cmd}"] = sub_desc

        return descriptions

    def validate_registry(self) -> list[str]:
        """Validate the command registry and return any issues found."""
        issues = []

        # Check for commands without implementations
        for canonical, command_def in self._commands.items():
            if command_def.implementation is None and not command_def.deprecated:
                issues.append(f"Command '{canonical}' has no implementation")

        # Check for duplicate aliases
        alias_count = {}
        for alias in self._alias_map:
            alias_count[alias] = alias_count.get(alias, 0) + 1

        for alias, count in alias_count.items():
            if count > 1:
                issues.append(f"Alias '{alias}' is used {count} times")

        return issues

    def get_registry_stats(self) -> dict[str, Any]:
        """Get statistics about the command registry."""
        implemented_count = sum(1 for cmd in self._commands.values() if cmd.implementation is not None)
        deprecated_count = sum(1 for cmd in self._commands.values() if cmd.deprecated)

        return {
            "total_commands": len(self._commands),
            "total_aliases": len(self._alias_map),
            "implemented_commands": implemented_count,
            "deprecated_commands": deprecated_count,
            "categories": len(self.list_categories()),
            "category_breakdown": {
                category: len(self.get_commands_by_category(category))
                for category in self.list_categories()
            }
        }

    def export_command_map(self) -> dict[str, str]:
        """Export command map in legacy format for backwards compatibility."""
        return self._alias_map.copy()


# Global command registry instance
_global_registry: Optional[CommandRegistry] = None


def get_global_command_registry() -> CommandRegistry:
    """Get the global command registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CommandRegistry()
        _global_registry.initialize_builtin_commands()
    return _global_registry


def register_command_definition(command_def: CommandDefinition) -> None:
    """Register a command definition in the global registry."""
    registry = get_global_command_registry()
    registry.register_command_definition(command_def)


def register_command_implementation(implementation: BaseCommand) -> None:
    """Register a command implementation in the global registry."""
    registry = get_global_command_registry()
    registry.register_command_implementation(implementation)


def get_command_by_alias(alias: str) -> Optional[CommandDefinition]:
    """Get command definition by alias from global registry."""
    registry = get_global_command_registry()
    return registry.get_command_by_alias(alias)