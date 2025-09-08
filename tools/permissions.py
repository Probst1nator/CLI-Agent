# tools/permissions.py
"""
This file implements the 'permissions' tool for managing CLI-Agent permissions.
Allows users to view, edit, and remove specific saved permissions.
"""
import os
import json
import re

try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, attrs=None):
        return str(text)

class permissions:
    @staticmethod
    def get_delim() -> str:
        return 'permissions'

    @staticmethod
    def get_tool_info() -> dict:
        return {
            "name": "permissions",
            "description": "Manage CLI-Agent saved permissions. Prefer using command flags like -permissions list for better UX. View all permissions, remove specific permissions by ID, or clear all permissions. Supports listing permissions and interactive management.",
            "example": "Use flags: -permissions list, -permissions remove 3, -permissions interactive. Legacy XML: <permissions>list</permissions>"
        }

    @staticmethod
    def run(content: str) -> str:
        """
        Manage CLI-Agent permissions.
        Supports: list, remove <id>, clear
        """
        try:
            
            # Parse content for action
            action_match = re.search(r'<action>(.*?)</action>', content, re.DOTALL)
            id_match = re.search(r'<id>(.*?)</id>', content, re.DOTALL)
            
            if action_match:
                action = action_match.group(1).strip().lower()
                if action == "remove" and id_match:
                    permission_id = id_match.group(1).strip()
                    return permissions._remove_permission(permission_id)
                else:
                    return f"Error: Invalid action '{action}' or missing ID for remove action."
            else:
                # Simple format
                action = content.strip().lower()
            
            if action == "list":
                return permissions._list_permissions()
            elif action == "clear":
                return permissions._clear_all_permissions()
            elif action.startswith("remove"):
                # Try to extract ID from "remove 3" format
                parts = action.split()
                if len(parts) == 2:
                    return permissions._remove_permission(parts[1])
                else:
                    return "Error: Remove command requires permission ID. Use: <permissions><action>remove</action><id>3</id></permissions>"
            else:
                return f"Error: Unknown action '{action}'. Supported actions: list, remove <id>, clear"

        except Exception as e:
            return f"Error managing permissions: {str(e)}"

    @staticmethod
    def _list_permissions() -> str:
        """List all saved permissions with IDs."""
        try:
            from core.permissions.permissions_manager import get_permissions_manager
            pm = get_permissions_manager()
            
            permissions_file = os.path.expanduser("~/.cli_agent_permissions.json")
            
            if not os.path.exists(permissions_file):
                return colored("📋 No saved permissions found.", "yellow")
            
            with open(permissions_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                return colored("📋 No saved permissions found.", "yellow")
            
            result = []
            result.append(colored("┌─ SAVED PERMISSIONS ──────────────────────────────────────────────┐", "cyan"))
            result.append(colored("│ ID │ Tool      │ Type      │ Path/Pattern                    │", "cyan"))
            result.append(colored("├────┼───────────┼───────────┼─────────────────────────────────┤", "cyan"))
            
            permission_id = 1
            
            # List tool permissions
            for tool_name, tool_data in data.get('tools', {}).items():
                for perm_type, perm_value in tool_data.items():
                    if perm_type == 'allowed' and perm_value:
                        result.append(colored("│", "cyan") + 
                                    f" {permission_id:2d} │ {tool_name:9s} │ {'allowed':9s} │ {('all operations')[:31]:31s} │")
                        permission_id += 1
                    elif perm_type == 'denied' and perm_value:
                        result.append(colored("│", "cyan") + 
                                    f" {permission_id:2d} │ {tool_name:9s} │ {'denied':9s} │ {('all operations')[:31]:31s} │")
                        permission_id += 1
            
            # List file permissions
            for file_path, file_data in data.get('files', {}).items():
                display_path = file_path if len(file_path) <= 31 else "..." + file_path[-28:]
                for tool_name, perm_value in file_data.items():
                    if perm_value == 'allowed':
                        result.append(colored("│", "cyan") + 
                                    f" {permission_id:2d} │ {tool_name:9s} │ {'allowed':9s} │ {display_path:31s} │")
                        permission_id += 1
                    elif perm_value == 'denied':
                        result.append(colored("│", "cyan") + 
                                    f" {permission_id:2d} │ {tool_name:9s} │ {'denied':9s} │ {display_path:31s} │")
                        permission_id += 1
            
            # List command permissions
            for cmd_hash, cmd_data in data.get('commands', {}).items():
                if cmd_data.get('allowed'):
                    pattern = cmd_data.get('pattern', 'Unknown command')[:31]
                    result.append(colored("│", "cyan") + 
                                f" {permission_id:2d} │ {'command':9s} │ {'allowed':9s} │ {pattern:31s} │")
                    permission_id += 1
            
            result.append(colored("└────┴───────────┴───────────┴─────────────────────────────────────┘", "cyan"))
            result.append("")
            result.append(colored("💡 Usage (recommended flags):", "yellow"))
            result.append(colored("  • Remove specific: -permissions remove 3", "white"))
            result.append(colored("  • Clear all: -permissions clear", "white"))
            result.append(colored("  • Interactive: -permissions interactive", "white"))
            result.append("")
            result.append(colored("💡 Legacy XML usage:", "yellow"))
            result.append(colored("  • Remove specific: <permissions><action>remove</action><id>3</id></permissions>", "white"))
            result.append(colored("  • Clear all: <permissions>clear</permissions>", "white"))
            
            if permission_id == 1:
                return colored("📋 No saved permissions found.", "yellow")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error listing permissions: {str(e)}"

    @staticmethod
    def _remove_permission(permission_id_str: str) -> str:
        """Remove a specific permission by ID."""
        try:
            permission_id = int(permission_id_str)
            
            permissions_file = os.path.expanduser("~/.cli_agent_permissions.json")
            if not os.path.exists(permissions_file):
                return colored("❌ No permissions file found.", "red")
            
            with open(permissions_file, 'r') as f:
                data = json.load(f)
            
            # Find and remove the permission by ID
            current_id = 1
            removed = False
            removed_info = ""
            
            # Check tool permissions
            for tool_name in list(data.get('tools', {}).keys()):
                tool_data = data['tools'][tool_name]
                for perm_type in ['allowed', 'denied']:
                    if tool_data.get(perm_type):
                        if current_id == permission_id:
                            tool_data[perm_type] = False
                            removed_info = f"tool '{tool_name}' {perm_type} permission"
                            removed = True
                            break
                        current_id += 1
                if removed:
                    break
            
            # Check file permissions
            if not removed:
                for file_path in list(data.get('files', {}).keys()):
                    file_data = data['files'][file_path]
                    for tool_name in list(file_data.keys()):
                        if file_data[tool_name] in ['allowed', 'denied']:
                            if current_id == permission_id:
                                del file_data[tool_name]
                                if not file_data:  # Remove empty file entry
                                    del data['files'][file_path]
                                removed_info = f"file '{os.path.basename(file_path)}' {tool_name} permission"
                                removed = True
                                break
                            current_id += 1
                    if removed:
                        break
            
            # Check command permissions
            if not removed:
                for cmd_hash in list(data.get('commands', {}).keys()):
                    cmd_data = data['commands'][cmd_hash]
                    if cmd_data.get('allowed'):
                        if current_id == permission_id:
                            del data['commands'][cmd_hash]
                            pattern = cmd_data.get('pattern', 'Unknown command')
                            removed_info = f"command '{pattern[:30]}'"
                            removed = True
                            break
                        current_id += 1
            
            if not removed:
                return colored(f"❌ Permission ID {permission_id} not found.", "red")
            
            # Save updated permissions
            with open(permissions_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return colored(f"✅ Successfully removed {removed_info} (ID: {permission_id})", "green")
            
        except ValueError:
            return colored(f"❌ Invalid permission ID: '{permission_id_str}'. Must be a number.", "red")
        except Exception as e:
            return f"Error removing permission: {str(e)}"

    @staticmethod
    def _clear_all_permissions() -> str:
        """Clear all saved permissions."""
        try:
            permissions_file = os.path.expanduser("~/.cli_agent_permissions.json")
            
            if not os.path.exists(permissions_file):
                return colored("📋 No permissions file found to clear.", "yellow")
            
            # Backup the file first
            backup_file = permissions_file + ".backup"
            import shutil
            shutil.copy2(permissions_file, backup_file)
            
            # Clear the permissions file
            os.remove(permissions_file)
            
            result = []
            result.append(colored("✅ All permissions cleared successfully!", "green"))
            result.append(colored(f"💾 Backup saved as: {backup_file}", "yellow"))
            result.append(colored("💡 You can restore permissions by renaming the backup file.", "white"))
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error clearing permissions: {str(e)}"