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


# --- UNIFIED COMMAND DEFINITION SYSTEM ---

# The single source of truth for all CLI commands.
# Each entry defines:
#   - canonical: The unique, internal name for the command.
#   - aliases: A list of all user-facing strings that trigger the command (e.g., ['/r', '/regenerate']).
#   - description: The help text for the command.
#   - sub_commands: (Optional) A dictionary for commands with sub-actions.
_COMMAND_DEFINITIONS: List[Dict[str, Any]] = [
    # Basic commands
    {'canonical': 'screenshot', 'aliases': ['/img', '/screenshot'], 'description': 'Take a screenshot for visual analysis'},
    {'canonical': 'multiline', 'aliases': ['/m', '/message'], 'description': 'Enter multiline input mode'},
    {'canonical': 'regenerate', 'aliases': ['/r', '/regenerate'], 'description': 'Regenerate the last AI response'},
    {'canonical': 'local', 'aliases': ['/l', '/local'], 'description': 'Toggle local-only LLM mode'},
    {'canonical': 'llm', 'aliases': ['/llm'], 'description': 'Open LLM selection interface'},
    {'canonical': 'unsay', 'aliases': ['/unsay'], 'description': 'Remove last user message and AI response'},
    {'canonical': 'auto_execute', 'aliases': ['/a', '/auto'], 'description': 'Toggle automatic code execution mode'},
    {'canonical': 'voice_mode', 'aliases': ['/v', '/voice'], 'description': 'Toggle voice input/output mode'},
    {'canonical': 'full_output', 'aliases': ['/fo', '/full_output'], 'description': 'Show full model output instead of summary'},
    {'canonical': 'print', 'aliases': ['/p', '/print'], 'description': 'Print complete chat history'},

    # Navigation & session commands
    {'canonical': 'online', 'aliases': ['/o', '/online'], 'description': 'Toggle online mode for web-enabled features'},
    {'canonical': 'exit', 'aliases': ['/e', '/exit'], 'description': 'Exit CLI-Agent'},
    {'canonical': 'help', 'aliases': ['/h', '/help'], 'description': 'Show help and available commands'},
    {'canonical': 'save', 'aliases': ['/s', '/save'], 'description': 'Save current LLM selection as default'},
    {'canonical': 'clear', 'aliases': ['/clear'], 'description': 'Clear chat history and start fresh'},
    {'canonical': 'reset', 'aliases': ['/reset'], 'description': 'Complete agent reset (chat, tools, images, context)'},
    {'canonical': 'compact', 'aliases': ['/compact'], 'description': 'Compact conversation (create summary)'},

    # Permission management commands
    {
        'canonical': 'permissions',
        'aliases': ['/perms', '/permissions'],
        'description': 'Manage tool execution permissions',
        'sub_commands': {
            'list': 'List all saved permissions',
            'remove': 'Remove specific permission by ID',
            'clear': 'Clear all saved permissions',
            'interactive': 'Open interactive permission manager',
            'manage': 'Open interactive permission manager',
        }
    }
]


def get_command_definitions() -> Dict[str, Any]:
    """
    Dynamically generates command definitions for the completer.
    
    Returns:
        Dictionary mapping command aliases to their sub-commands (if any)
    """
    defs = {}
    for cmd_info in _COMMAND_DEFINITIONS:
        for alias in cmd_info['aliases']:
            if 'sub_commands' in cmd_info:
                # Create a simple structure for the completer
                defs[alias] = {sub: None for sub in cmd_info['sub_commands']}
            else:
                defs[alias] = None
    return defs


def get_command_descriptions() -> Dict[str, str]:
    """
    Dynamically generates command descriptions for the completer tooltips.
    
    Returns:
        Dictionary mapping command aliases to their descriptions
    """
    descs = {}
    for cmd_info in _COMMAND_DEFINITIONS:
        for alias in cmd_info['aliases']:
            descs[alias] = cmd_info['description']
        if 'sub_commands' in cmd_info:
            for sub_cmd, sub_desc in cmd_info['sub_commands'].items():
                # Provide descriptions for sub-commands as well
                for alias in cmd_info['aliases']:
                    descs[f"{alias} {sub_cmd}"] = sub_desc
    return descs


def _build_command_map() -> Dict[str, str]:
    """
    Builds a map from every alias to its canonical command name for the dispatcher.
    
    Returns:
        Dictionary mapping aliases to canonical names (e.g., {'/r': 'regenerate'})
    """
    command_map = {}
    for cmd_info in _COMMAND_DEFINITIONS:
        canonical_name = cmd_info['canonical']
        for alias in cmd_info['aliases']:
            command_map[alias] = canonical_name
    return command_map


class CommandCompleterWithDescriptions(Completer):
    """Custom completer that shows command descriptions in completion tooltips."""
    
    def __init__(self):
        self.commands = get_command_definitions()
        self.descriptions = get_command_descriptions()
    
    def get_completions(self, document, complete_event):
        """Generate completions with descriptions for the current input."""
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


