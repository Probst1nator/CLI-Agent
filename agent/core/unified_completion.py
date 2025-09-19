"""
Unified command completion system for CLI-Agent.

This module provides a single command completion system that works
across all interfaces (CLI, GUI, Web) with consistent behavior and
real-time command discovery.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

if TYPE_CHECKING:
    from agent.commands.command_handler import CommandHandler


class UnifiedCommandCompleter(Completer):
    """
    Unified command completer that works across all interfaces.

    This completer dynamically discovers commands from the command handler
    and provides consistent completion behavior regardless of the interface.
    """

    def __init__(self, command_handler: CommandHandler):
        self.command_handler = command_handler
        self._cached_commands = {}
        self._cached_descriptions = {}
        self._cache_valid = False

    def _refresh_cache(self) -> None:
        """Refresh the command cache from the command handler."""
        try:
            self._cached_commands = self.command_handler.get_command_definitions()
            self._cached_descriptions = self.command_handler.get_command_descriptions()
            self._cache_valid = True
        except Exception:
            # Fallback to empty cache if command handler is not available
            self._cached_commands = {}
            self._cached_descriptions = {}
            self._cache_valid = False

    def get_completions(self, document: Document, complete_event) -> List[Completion]:
        """Generate completions for the current input."""
        if not self._cache_valid:
            self._refresh_cache()

        text = document.text_before_cursor

        # Only provide completions for commands (starting with /)
        if not text.startswith('/'):
            return []

        words = text.split()

        if len(words) <= 1:
            # Complete main command
            prefix = words[0] if words else ''
            return self._complete_main_command(prefix)

        elif len(words) == 2:
            # Complete sub-command
            main_command = words[0]
            sub_prefix = words[1]
            return self._complete_sub_command(main_command, sub_prefix)

        # No completions for more than 2 words
        return []

    def _complete_main_command(self, prefix: str) -> List[Completion]:
        """Complete main command based on prefix."""
        completions = []

        for command in self._cached_commands.keys():
            if command.startswith(prefix):
                description = self._cached_descriptions.get(command, 'No description available')
                completion = Completion(
                    command,
                    start_position=-len(prefix),
                    display_meta=description
                )
                completions.append(completion)

        return sorted(completions, key=lambda x: x.text)

    def _complete_sub_command(self, main_command: str, sub_prefix: str) -> List[Completion]:
        """Complete sub-command based on main command and sub-prefix."""
        completions = []

        # Check if main command has sub-commands
        if main_command in self._cached_commands:
            sub_commands = self._cached_commands[main_command]
            if isinstance(sub_commands, dict):
                for sub_command in sub_commands.keys():
                    if sub_command.startswith(sub_prefix):
                        full_command = f"{main_command} {sub_command}"
                        description = self._cached_descriptions.get(full_command, 'No description available')
                        completion = Completion(
                            sub_command,
                            start_position=-len(sub_prefix),
                            display_meta=description
                        )
                        completions.append(completion)

        return sorted(completions, key=lambda x: x.text)

    def get_command_help(self, command: str) -> Optional[str]:
        """Get help text for a specific command."""
        if not self._cache_valid:
            self._refresh_cache()

        return self._cached_descriptions.get(command)

    def list_all_commands(self) -> List[str]:
        """Get a list of all available commands."""
        if not self._cache_valid:
            self._refresh_cache()

        return list(self._cached_commands.keys())

    def invalidate_cache(self) -> None:
        """Invalidate the command cache to force refresh."""
        self._cache_valid = False


class WebCommandCompleter:
    """
    Command completer specifically for web interfaces.

    This provides command completion functionality that can be serialized
    and sent to web clients via JSON.
    """

    def __init__(self, command_handler: CommandHandler):
        self.unified_completer = UnifiedCommandCompleter(command_handler)

    def get_completions_json(self, text: str, cursor_position: int = None) -> Dict[str, Any]:
        """
        Get command completions in JSON format for web interfaces.

        Args:
            text: The current input text
            cursor_position: Position of cursor (defaults to end of text)

        Returns:
            Dict containing completions and metadata
        """
        if cursor_position is None:
            cursor_position = len(text)

        # Create a document for the completer
        document = Document(text, cursor_position)

        # Get completions
        completions = self.unified_completer.get_completions(document, None)

        # Convert to JSON-serializable format
        json_completions = []
        for completion in completions:
            json_completions.append({
                'text': completion.text,
                'start_position': completion.start_position,
                'description': completion.display_meta or ''
            })

        return {
            'completions': json_completions,
            'text': text,
            'cursor_position': cursor_position,
            'total_count': len(json_completions)
        }

    def get_command_list_json(self) -> Dict[str, Any]:
        """Get a complete list of commands in JSON format."""
        commands = []

        for command in self.unified_completer.list_all_commands():
            description = self.unified_completer.get_command_help(command)
            commands.append({
                'command': command,
                'description': description or 'No description available'
            })

        return {
            'commands': commands,
            'total_count': len(commands)
        }


class GUICommandCompleter:
    """
    Command completer for GUI interfaces using prompt_toolkit.

    This provides integration between the unified completion system
    and prompt_toolkit's completion framework.
    """

    def __init__(self, command_handler: CommandHandler):
        self.unified_completer = UnifiedCommandCompleter(command_handler)

    def get_prompt_toolkit_completer(self) -> Completer:
        """Get a prompt_toolkit compatible completer."""
        return self.unified_completer

    def get_command_suggestions(self, partial_text: str) -> List[Tuple[str, str]]:
        """
        Get command suggestions for GUI display.

        Args:
            partial_text: Partial command text

        Returns:
            List of (command, description) tuples
        """
        document = Document(partial_text, len(partial_text))
        completions = self.unified_completer.get_completions(document, None)

        suggestions = []
        for completion in completions:
            # Reconstruct full command text
            full_text = partial_text[:document.cursor_position + completion.start_position] + completion.text
            description = completion.display_meta or 'No description available'
            suggestions.append((full_text, description))

        return suggestions


def create_unified_completer_system(command_handler: CommandHandler) -> Dict[str, Any]:
    """
    Create a complete unified completion system for all interfaces.

    Args:
        command_handler: The command handler instance

    Returns:
        Dict containing completers for different interfaces
    """
    unified_completer = UnifiedCommandCompleter(command_handler)

    return {
        'unified': unified_completer,
        'web': WebCommandCompleter(command_handler),
        'gui': GUICommandCompleter(command_handler),
        'cli': unified_completer  # CLI uses the unified completer directly
    }


class CompletionManager:
    """
    Manager class for the unified completion system.

    This class coordinates completion across all interfaces and handles
    cache invalidation when commands are dynamically registered.
    """

    def __init__(self, command_handler: CommandHandler):
        self.command_handler = command_handler
        self.completers = create_unified_completer_system(command_handler)

    def get_completer_for_interface(self, interface: str):
        """Get the appropriate completer for a specific interface."""
        return self.completers.get(interface)

    def refresh_all_completers(self) -> None:
        """Refresh all completers when commands change."""
        for completer_name, completer in self.completers.items():
            if hasattr(completer, 'invalidate_cache'):
                completer.invalidate_cache()
            elif hasattr(completer, 'unified_completer'):
                completer.unified_completer.invalidate_cache()

    def register_command_change_callback(self) -> None:
        """Register a callback to refresh completers when commands change."""
        # This would be called whenever the command handler registers new commands
        self.refresh_all_completers()

    def get_completion_stats(self) -> Dict[str, Any]:
        """Get statistics about the completion system."""
        unified_completer = self.completers['unified']
        if not unified_completer._cache_valid:
            unified_completer._refresh_cache()

        return {
            'total_commands': len(unified_completer._cached_commands),
            'total_descriptions': len(unified_completer._cached_descriptions),
            'cache_valid': unified_completer._cache_valid,
            'available_interfaces': list(self.completers.keys())
        }