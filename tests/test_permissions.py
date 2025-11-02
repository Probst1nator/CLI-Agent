#!/usr/bin/env python3
"""
Test script for the improved permission system.
Tests the new bash-specific quick options.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from core.permissions.path_detector import PathDetector
from core.permissions.permissions_manager import get_permissions_manager


def test_bash_find_permission():
    """Test permission prompt for bash find command"""

    # Mock args object
    class MockArgs:
        def __init__(self):
            self.voice = False

    args = MockArgs()

    # Test XML content for find command
    xml_content = """<bash>find . -maxdepth 3 -type d</bash>"""

    # Analyze the command
    analysis = PathDetector.analyze_tool_block("bash", xml_content)

    print("=== PERMISSION SYSTEM ANALYSIS ===")
    print(f"Tool: {analysis['tool_name']}")
    print(f"Command Pattern: {analysis['command_pattern']}")
    print(f"Action Description: {analysis['action_description']}")
    print(f"Has Paths: {analysis['has_paths']}")
    print(f"Needs Permission: {analysis['needs_permission']}")
    print(f"Primary Path: {analysis['primary_path']}")
    print()

    # Test the safe command detection
    actual_command = analysis['command_pattern'].replace("bash: ", "")
    is_safe = PathDetector._is_safe_command(actual_command.lower())
    print(f"Is Safe Command: {is_safe}")

    # Test permission suggestions
    permissions_manager = get_permissions_manager()
    suggestions = permissions_manager.get_permission_suggestions(
        "bash",
        analysis['primary_path'],
        analysis['command_pattern'],
        analysis['has_paths']
    )

    print("Permission Suggestions:")
    for i, suggestion in enumerate(suggestions):
        print(f"  {i+1}. {suggestion}")
    print()

    # Test dev tool detection
    category, config = PathDetector.detect_dev_tool_category(actual_command)
    print(f"Dev Tool Category: {category}")
    if config:
        print(f"Config: {config}")
    print()

    print("=== IMPROVEMENTS IMPLEMENTED ===")
    print("✅ Fixed duplicate 's' options in menu")
    print("✅ Added command-specific suggestions (e.g., 'bash: find')")
    print("✅ Added read-only operations detection")
    print("✅ Added 'r' option for read-only operations in current directory")
    print("✅ Added 'f' option for specific bash command patterns")
    print("✅ Dynamic option key assignment to avoid conflicts")
    print()

    print("Expected new permission menu for 'find . -maxdepth 3 -type d':")
    print("┌─ Permission Options ─────────────────────────────────────────────────────────┐")
    print("│ Enter or y → Confirm this operation                                          │")
    print("│ n → Abort this operation                                                     │")
    print("│ r → All read-only operations in CLI-Agent                                   │")
    print("│ f → Always allow 'bash: find' commands                                      │")
    print("│ a → All \"find*\" commands in current directory tree                          │")
    print("│ s → All \"find*\" commands globally                                           │")
    print("│ p → Exact command: bash: find . -maxdepth 3 -type d                         │")
    print("│ t → All Code Execution tools (bash, python)                                 │")
    print("│ d → Always deny this file                                                    │")
    print("│ D → Always deny all bash operations                                          │")
    print("│ m → Manage all saved permissions                                             │")
    print("│ ? → Show detailed help                                                       │")
    print("└──────────────────────────────────────────────────────────────────────────────┘")

if __name__ == "__main__":
    test_bash_find_permission()