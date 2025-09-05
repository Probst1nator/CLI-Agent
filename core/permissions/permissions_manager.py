# permissions_manager.py
"""
Permission management system for CLI-Agent tools with persistent "always allow" functionality.
Handles tool-specific and path-specific permission storage and retrieval.
"""
import json
import os
from typing import Dict, Optional, List

class PermissionsManager:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize permissions manager with config file path."""
        if config_path is None:
            # Default to user's home directory
            config_path = os.path.expanduser("~/.cli_agent_permissions.json")
        
        self.config_path = config_path
        self.permissions = self._load_permissions()
    
    def _load_permissions(self) -> Dict:
        """Load permissions from config file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return {
            "always_allow_tools": {},  # tool_name -> set of paths/patterns
            "always_deny_tools": {},   # tool_name -> set of paths/patterns
            "always_allow_commands": {},  # tool_name -> set of command_hashes
            "always_deny_commands": {},   # tool_name -> set of command_hashes
            "always_allow_groups": {},  # group_name -> set of paths/patterns
            "always_deny_groups": {},   # group_name -> set of paths/patterns
            "global_always_allow": False
        }
    
    def _save_permissions(self):
        """Save permissions to config file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.permissions, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save permissions: {e}")
    
    def is_always_allowed(self, tool_name: str, file_path: str = None, command_hash: str = None) -> bool:
        """Check if a tool/path/command combination is always allowed."""
        # Check global permission
        if self.permissions.get("global_always_allow", False):
            return True
        
        # Check command-specific permissions (for exact command matching)
        if command_hash:
            command_permissions = self.permissions.get("always_allow_commands", {}).get(tool_name, [])
            if command_hash in command_permissions:
                return True
        
        # Check tool-specific permissions (for path-based matching)
        tool_permissions = self.permissions.get("always_allow_tools", {}).get(tool_name, [])
        
        # Check group-based permissions
        from path_detector import PathDetector
        tool_groups = PathDetector.get_tool_groups(tool_name)
        group_permissions = []
        for group in tool_groups:
            group_perms = self.permissions.get("always_allow_groups", {}).get(group, [])
            group_permissions.extend(group_perms)
        
        # Combine tool and group permissions
        all_permissions = tool_permissions + group_permissions
        
        if not file_path:
            # If no path specified, check if tool has any always-allow permissions
            return len(all_permissions) > 0
        
        file_path = os.path.abspath(file_path)
        
        # Check exact path matches
        if file_path in all_permissions:
            return True
        
        # Check pattern matches (directories)
        for allowed_pattern in all_permissions:
            if allowed_pattern.endswith("/*"):
                # Directory pattern
                dir_pattern = allowed_pattern[:-2]
                if file_path.startswith(dir_pattern):
                    return True
            elif allowed_pattern.endswith("/"):
                # Directory pattern without wildcard
                if file_path.startswith(allowed_pattern):
                    return True
        
        return False
    
    def is_always_denied(self, tool_name: str, file_path: str = None) -> bool:
        """Check if a tool/path combination is always denied."""
        tool_permissions = self.permissions.get("always_deny_tools", {}).get(tool_name, [])
        
        if not file_path:
            return False
        
        file_path = os.path.abspath(file_path)
        
        # Check exact path matches
        if file_path in tool_permissions:
            return True
        
        # Check pattern matches
        for denied_pattern in tool_permissions:
            if denied_pattern.endswith("/*"):
                dir_pattern = denied_pattern[:-2]
                if file_path.startswith(dir_pattern):
                    return True
        
        return False
    
    def add_always_allow(self, tool_name: str, file_path: str = None, pattern: str = None, command_hash: str = None, group_name: str = None):
        """Add an always-allow permission for a tool/path/command/group combination."""
        if command_hash:
            # Add command-specific permission
            if "always_allow_commands" not in self.permissions:
                self.permissions["always_allow_commands"] = {}
            
            if tool_name not in self.permissions["always_allow_commands"]:
                self.permissions["always_allow_commands"][tool_name] = []
            
            if command_hash not in self.permissions["always_allow_commands"][tool_name]:
                self.permissions["always_allow_commands"][tool_name].append(command_hash)
                self._save_permissions()
        elif group_name:
            # Add group-specific permission
            if "always_allow_groups" not in self.permissions:
                self.permissions["always_allow_groups"] = {}
            
            if group_name not in self.permissions["always_allow_groups"]:
                self.permissions["always_allow_groups"][group_name] = []
            
            permission_target = pattern if pattern else (os.path.abspath(file_path) if file_path else group_name)
            
            if permission_target not in self.permissions["always_allow_groups"][group_name]:
                self.permissions["always_allow_groups"][group_name].append(permission_target)
                self._save_permissions()
        else:
            # Add path/pattern-specific permission
            if "always_allow_tools" not in self.permissions:
                self.permissions["always_allow_tools"] = {}
            
            if tool_name not in self.permissions["always_allow_tools"]:
                self.permissions["always_allow_tools"][tool_name] = []
            
            permission_target = pattern if pattern else (os.path.abspath(file_path) if file_path else tool_name)
            
            if permission_target not in self.permissions["always_allow_tools"][tool_name]:
                self.permissions["always_allow_tools"][tool_name].append(permission_target)
                self._save_permissions()
    
    def add_always_deny(self, tool_name: str, file_path: str = None):
        """Add an always-deny permission for a tool/path combination."""
        if "always_deny_tools" not in self.permissions:
            self.permissions["always_deny_tools"] = {}
        
        if tool_name not in self.permissions["always_deny_tools"]:
            self.permissions["always_deny_tools"][tool_name] = []
        
        permission_target = os.path.abspath(file_path) if file_path else tool_name
        
        if permission_target not in self.permissions["always_deny_tools"][tool_name]:
            self.permissions["always_deny_tools"][tool_name].append(permission_target)
            self._save_permissions()
    
    def remove_permission(self, tool_name: str, file_path: str = None, permission_type: str = "allow"):
        """Remove a specific permission."""
        permission_key = f"always_{permission_type}_tools"
        if permission_key not in self.permissions:
            return
        
        if tool_name not in self.permissions[permission_key]:
            return
        
        permission_target = os.path.abspath(file_path) if file_path else tool_name
        
        if permission_target in self.permissions[permission_key][tool_name]:
            self.permissions[permission_key][tool_name].remove(permission_target)
            self._save_permissions()
    
    def get_permission_suggestions(self, tool_name: str, file_path: str = None, command_pattern: str = None, has_paths: bool = False) -> List[str]:
        """Get suggested permission patterns based on file path or command pattern."""
        suggestions = []
        
        # Import here to avoid circular dependency
        from path_detector import PathDetector
        
        if has_paths and file_path:
            # Path-based suggestions for tools with file operations
            abs_path = os.path.abspath(file_path)
            suggestions.append(f"Exact file: {abs_path}")
            
            # Directory suggestions
            parent_dir = os.path.dirname(abs_path)
            suggestions.append(f"Directory: {parent_dir}/*")
            
            # Project root suggestion
            if "CLI-Agent" in abs_path:
                project_root = abs_path.split("CLI-Agent")[0] + "CLI-Agent"
                suggestions.append(f"Project: {project_root}/*")
            
            # Group-based suggestions
            tool_groups = PathDetector.get_tool_groups(tool_name)
            for group in tool_groups:
                group_tools = PathDetector.get_group_tools(group)
                if len(group_tools) > 1:  # Only show group if it contains multiple tools
                    group_display = group.replace('_', ' ').title()
                    suggestions.append(f"All {group_display} tools ({', '.join(group_tools)})")
            
            suggestions.append(f"All {tool_name} operations")
        else:
            # Command-pattern based suggestions for tools without paths
            if command_pattern:
                suggestions.append(f"Exact command: {command_pattern}")
            
            # Group-based suggestions for non-path commands
            tool_groups = PathDetector.get_tool_groups(tool_name)
            for group in tool_groups:
                group_tools = PathDetector.get_group_tools(group)
                if len(group_tools) > 1:  # Only show group if it contains multiple tools
                    group_display = group.replace('_', ' ').title()
                    suggestions.append(f"All {group_display} tools ({', '.join(group_tools)})")
            
            suggestions.append(f"All {tool_name} operations")
        
        return suggestions
    
    def clear_all_permissions(self):
        """Clear all stored permissions."""
        self.permissions = {
            "always_allow_tools": {},
            "always_deny_tools": {},
            "global_always_allow": False
        }
        self._save_permissions()

# Global instance
_permissions_manager = None

def get_permissions_manager() -> PermissionsManager:
    """Get the global permissions manager instance."""
    global _permissions_manager
    if _permissions_manager is None:
        _permissions_manager = PermissionsManager()
    return _permissions_manager