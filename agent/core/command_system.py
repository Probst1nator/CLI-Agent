"""
Command system for CLI-Agent.

This module contains the unified command definition system and command completion
functionality extracted from main.py to provide reusable command management.
"""

import os
from typing import Dict, Any, List
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory


# --- LEGACY COMMAND DEFINITIONS ---
# These definitions are deprecated and maintained only for backwards compatibility.
# The new single source of truth is in agent.core.command_registry.

from agent.core.command_registry import get_global_command_registry

def _get_legacy_command_definitions() -> List[Dict[str, Any]]:
    """Get command definitions in legacy format for backwards compatibility."""
    registry = get_global_command_registry()
    legacy_definitions = []

    for canonical, command_def in registry.get_all_commands().items():
        legacy_def = {
            'canonical': command_def.canonical,
            'aliases': command_def.aliases,
            'description': command_def.description
        }
        if command_def.sub_commands:
            legacy_def['sub_commands'] = command_def.sub_commands
        legacy_definitions.append(legacy_def)

    return legacy_definitions

# Legacy global for backwards compatibility
_COMMAND_DEFINITIONS = _get_legacy_command_definitions()


def get_command_definitions() -> Dict[str, Any]:
    """
    Dynamically generates command definitions for the completer.

    Returns:
        Dictionary mapping command aliases to their sub-commands (if any)
    """
    registry = get_global_command_registry()
    return registry.get_command_definitions_for_completion()


def get_command_descriptions() -> Dict[str, str]:
    """
    Dynamically generates command descriptions for the completer tooltips.

    Returns:
        Dictionary mapping command aliases to their descriptions
    """
    registry = get_global_command_registry()
    return registry.get_command_descriptions_for_completion()


def _build_command_map() -> Dict[str, str]:
    """
    Builds a map from every alias to its canonical command name for the dispatcher.

    Returns:
        Dictionary mapping aliases to canonical names (e.g., {'/r': 'regenerate'})
    """
    registry = get_global_command_registry()
    return registry.export_command_map()


class CommandCompleterWithDescriptions(Completer):
    """
    LEGACY: Custom completer that shows command descriptions in completion tooltips.

    This class is maintained for backwards compatibility but now delegates to
    the unified completion system.
    """

    def __init__(self):
        self.commands = get_command_definitions()
        self.descriptions = get_command_descriptions()
        self._unified_completer = None

    def _get_unified_completer(self):
        """Get the unified completer, creating it if necessary."""
        if self._unified_completer is None:
            try:
                from agent.core.unified_completion import UnifiedCommandCompleter
                from agent.commands.command_handler import CommandHandler

                # Create a minimal context for the command handler
                # This is a temporary solution for backwards compatibility
                class MinimalContext:
                    def __init__(self):
                        import logging
                        self.logger = logging.getLogger("command_system")

                context = MinimalContext()
                command_handler = CommandHandler(context)
                self._unified_completer = UnifiedCommandCompleter(command_handler)
            except ImportError:
                # Fallback to legacy implementation
                self._unified_completer = None

        return self._unified_completer

    def get_completions(self, document, complete_event):
        """Generate completions with descriptions for the current input."""
        unified_completer = self._get_unified_completer()
        if unified_completer:
            # Use unified completion system
            try:
                return unified_completer.get_completions(document, complete_event)
            except Exception:
                # Fall back to legacy implementation
                pass

        # Legacy implementation as fallback
        text = document.text_before_cursor

        if not text.startswith('/'):
            return

        words = text.split()

        if len(words) <= 1:
            prefix = words[0] if words else ''
            for command in self.commands.keys():
                if command.startswith(prefix):
                    description = self.descriptions.get(command, 'No description available')
                    yield Completion(
                        command,
                        start_position=-len(prefix),
                        display_meta=description
                    )

        elif len(words) == 2:
            main_command = words[0]
            subcommand_prefix = words[1]

            if main_command in self.commands and isinstance(self.commands[main_command], dict):
                for subcommand in self.commands[main_command].keys():
                    if subcommand.startswith(subcommand_prefix):
                        full_command = f"{main_command} {subcommand}"
                        description = self.descriptions.get(full_command, 'No description available')
                        yield Completion(
                            subcommand,
                            start_position=-len(subcommand_prefix),
                            display_meta=description
                        )


# --- Global Prompt Session for Command Completion ---
# Initialize this once to maintain command history and completion across calls
_prompt_session = None


def get_prompt_session() -> PromptSession:
    """
    Get the global prompt session for command completion and history.
    
    Returns:
        Initialized PromptSession with command completion and history
    """
    global _prompt_session
    if _prompt_session is None:
        command_completer = CommandCompleterWithDescriptions()
        history = FileHistory(os.path.expanduser('~/.cli_agent_history'))
        
        _prompt_session = PromptSession(
            completer=command_completer,
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            complete_while_typing=True
        )
    return _prompt_session


# Expose the command map for use by the main dispatcher
COMMAND_MAP = _build_command_map()


