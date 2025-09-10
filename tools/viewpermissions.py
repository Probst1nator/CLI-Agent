# tools/viewpermissions.py
"""
This file implements the 'viewpermissions' tool. It allows the agent to inspect
the saved permission rules that govern automatic tool execution.
"""
from core.permissions.permissions_manager import get_permissions_manager

class viewpermissions:
    """
    A tool to view the currently saved permission rules that allow tools
    to execute without requiring user confirmation.
    """
    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool, used for parsing agent output."""
        return 'viewpermissions'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool for the agent's system prompt."""
        return {
            "name": "viewpermissions",
            "description": "Displays the currently saved permission rules that allow tools to execute automatically without user confirmation. This includes tool-specific, path-specific, and command-specific rules.",
            "example": "<viewpermissions></viewpermissions>"
        }

    @staticmethod
    def run(content: str = "") -> str:
        """
        Loads and formats the saved permission rules from the PermissionsManager.
        This tool takes no arguments inside its tags.
        """
        try:
            permissions_manager = get_permissions_manager()
            # Directly access the loaded permissions dictionary
            permissions = permissions_manager.permissions

            output_lines = ["🔓 PERMISSIONS - Actions I Can Execute Without Asking"]
            output_lines.append("=" * 60)
            has_rules = False

            # 1. Development Tools (most user-friendly section first)
            allow_dev_tools = permissions.get("always_allow_dev_tools", {})
            if allow_dev_tools:
                has_rules = True
                output_lines.append("\n🛠️  DEVELOPMENT TOOLS:")
                for category, patterns in sorted(allow_dev_tools.items()):
                    if patterns:
                        category_display = {
                            'linting': 'Code Linting (ruff, black, flake8, etc.)',
                            'testing': 'Running Tests (pytest, unittest, etc.)',
                            'building': 'Build Tools (npm, pip, make, etc.)',
                            'formatting': 'Code Formatters (black, prettier, etc.)',
                            'git': 'Git Commands (add, commit, status, etc.)',
                            'grepsearch': 'Search Tools (grep, find, ripgrep, etc.)'
                        }.get(category, f'{category.title()} Tools')
                        
                        output_lines.append(f"   ✓ {category_display}")
                        for pattern in sorted(patterns):
                            if pattern.endswith('/*'):
                                location = pattern[:-2].split('/')[-1] if '/' in pattern else pattern[:-2]
                                output_lines.append(f"     → In project: {location}")
                            elif pattern == category:
                                output_lines.append("     → Globally allowed")
                            else:
                                output_lines.append(f"     → Pattern: {pattern}")

            # 2. File Operations (simplified)
            allow_tools = permissions.get("always_allow_tools", {})
            allow_groups = permissions.get("always_allow_groups", {})
            
            if allow_tools or allow_groups:
                has_rules = True
                output_lines.append("\n📁 FILE OPERATIONS:")
                
                # Group similar tools together
                file_ops = {}
                
                # Process individual tools
                for tool, paths in sorted(allow_tools.items()):
                    if paths:
                        tool_display = {
                            'editfile': 'Edit Files',
                            'readfile': 'Read Files', 
                            'writefile': 'Write Files',
                            'bash': 'Execute Commands',
                            'python': 'Run Python Code'
                        }.get(tool, tool.title())
                        
                        if tool_display not in file_ops:
                            file_ops[tool_display] = []
                        
                        for path in paths:
                            if path.endswith('/*'):
                                location = path[:-2].split('/')[-1] if '/' in path else path[:-2]
                                file_ops[tool_display].append(f"In directory: {location}")
                            elif '/' in path:
                                filename = path.split('/')[-1]
                                file_ops[tool_display].append(f"File: {filename}")
                            else:
                                file_ops[tool_display].append(f"Pattern: {path}")
                
                # Process tool groups
                for group, paths in sorted(allow_groups.items()):
                    if paths:
                        group_display = {
                            'file_editing': 'Edit Files',
                            'file_reading': 'Read Files',
                            'code_execution': 'Execute Code'
                        }.get(group, group.replace('_', ' ').title())
                        
                        if group_display not in file_ops:
                            file_ops[group_display] = []
                        
                        for path in paths:
                            if path == group:  # Global permission
                                file_ops[group_display].append("Anywhere")
                            elif path.endswith('/*'):
                                location = path[:-2].split('/')[-1] if '/' in path else path[:-2]
                                file_ops[group_display].append(f"In directory: {location}")
                            elif '/' in path:
                                filename = path.split('/')[-1]
                                file_ops[group_display].append(f"File: {filename}")
                            else:
                                file_ops[group_display].append(f"Pattern: {path}")
                
                # Display grouped file operations
                for operation, locations in sorted(file_ops.items()):
                    output_lines.append(f"   ✓ {operation}")
                    for location in sorted(set(locations)):  # Remove duplicates
                        output_lines.append(f"     → {location}")

            # 3. Saved Commands (simplified, only if there are many)
            allow_commands = permissions.get("always_allow_commands", {})
            if allow_commands:
                has_rules = True
                total_commands = sum(len(hashes) for hashes in allow_commands.values())
                output_lines.append("\n⚡ SAVED COMMANDS:")
                output_lines.append(f"   ✓ {total_commands} specific command(s) pre-approved")
                output_lines.append("     → These run instantly without confirmation")

            if not has_rules:
                return "🔒 No automatic permissions set - I'll ask for confirmation before every action."

            return "\n".join(output_lines)

        except Exception as e:
            return f"Error retrieving permissions: {str(e)}"