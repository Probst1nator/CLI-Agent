# core.permissions package
"""
Simplified permission system for CLI-Agent.
Claude Code-style config-based permissions with allow/deny/ask lists.
"""

from .dev_tool_hints import get_suggested_rules, get_tool_hint
from .permission_prompt import permission_prompt
from .settings_manager import (
    SettingsManager,
    extract_command_from_xml,
    extract_path_from_xml,
)

__all__ = [
    'SettingsManager',
    'extract_path_from_xml',
    'extract_command_from_xml',
    'permission_prompt',
    'get_tool_hint',
    'get_suggested_rules',
]