# permissions_manager.py
"""
Permission management system for CLI-Agent tools with persistent "always allow" functionality.
Handles tool-specific and path-specific permission storage and retrieval.
"""
import json
import os
from typing import Dict, Optional, List, Tuple, Any

class PermissionsManager:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize permissions manager with config file path."""
        if config_path is None:
            # Use XDG-compliant config directory
            try:
                from core.globals import g
                config_path = os.path.join(g.CLIAGENT_CONFIG_DIR, "permissions.json")
            except ImportError:
                # Fallback to XDG-compliant path
                xdg_config_home = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
                config_path = os.path.join(xdg_config_home, "cli-agent", "permissions.json")

        self.config_path = config_path
        self.permissions = self._load_permissions()

    def _load_permissions(self) -> Dict[str, Any]:
        """Load permissions from config file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Initialize default permissions structure with safe search tools pre-approved
        default_permissions = {
            "always_allow_tools": {},  # tool_name -> set of paths/patterns
            "always_deny_tools": {},   # tool_name -> set of paths/patterns
            "always_allow_commands": {},  # tool_name -> set of command_hashes
            "always_deny_commands": {},   # tool_name -> set of command_hashes
            "always_allow_groups": {},  # group_name -> set of paths/patterns
            "always_deny_groups": {},   # group_name -> set of paths/patterns
            "always_allow_dev_tools": {
                "grepsearch": ["grepsearch"]  # Global permission for search tools like grep
            },  # dev_tool_category -> set of paths/patterns
            "pattern_permissions": {},  # pattern -> scope_path (e.g. "bash:ls*" -> "/current/path/*")
            "global_always_allow": False
        }
        return default_permissions

    def _save_permissions(self):
        """Save permissions to config file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.permissions, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save permissions: {e}")

    def has_sensitive_content(self, content: str) -> Tuple[bool, List[str]]:
        """
        Check if content contains sensitive keywords that should block execution.

        Args:
            content: Content to check (XML, commands, paths, etc.)

        Returns:
            Tuple of (has_sensitive_content, list_of_found_keywords)
        """
        from .path_detector import PathDetector
        return PathDetector.contains_sensitive_keywords(content)

    def is_sensitive_content_blocked(self, tool_name: str, xml_content: str) -> Tuple[bool, str]:
        """
        Check if tool execution should be blocked due to sensitive content.

        Args:
            tool_name: Name of the tool being executed
            xml_content: XML content of the tool call

        Returns:
            Tuple of (should_block, reason_message)
        """
        from .path_detector import PathDetector

        # Check for sensitive keywords in the content
        has_sensitive, found_keywords = PathDetector.check_sensitive_content_in_xml(xml_content)

        if has_sensitive:
            # Only block for tools that can actually access or modify files
            risky_tools = ['bash', 'readfile', 'editfile', 'writefile', 'viewfiles']

            if tool_name in risky_tools:
                # Additional context-based filtering for .env specifically
                if '.env' in found_keywords:
                    # Always block .env access
                    keywords_str = ", ".join(found_keywords)
                    reason = f"🔒 SECURITY: Access to {keywords_str} files is automatically blocked"
                    return True, reason

                # For other sensitive keywords, check context
                other_keywords = [k for k in found_keywords if k != '.env']
                if other_keywords:
                    keywords_str = ", ".join(other_keywords)
                    reason = f"🔒 SECURITY: Sensitive content detected: {keywords_str}"
                    return True, reason

        return False, ""

    def is_always_allowed(self, tool_name: str, file_path: Optional[str] = None, command_hash: Optional[str] = None, xml_content: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if a tool/path/command combination is always allowed.

        Returns:
            Tuple of (is_allowed, rule_description) where rule_description explains which rule allowed the execution
        """
        # SECURITY CHECK: Block execution if sensitive content is detected
        if xml_content:
            should_block, reason = self.is_sensitive_content_blocked(tool_name, xml_content)
            if should_block:
                print(f"\n⚠️ Additional confirmation required, reason: {reason}")
                return False, None

        # Check global permission
        if self.permissions.get("global_always_allow", False):
            return True, "Global permission (all tools allowed)"

        # Check command-specific permissions (for exact command matching)
        if command_hash:
            command_permissions = self.permissions.get("always_allow_commands", {}).get(tool_name, [])
            if command_hash in command_permissions:
                return True, f"Exact command permission ({tool_name})"

        # Check tool-specific permissions (for path-based matching)
        tool_permissions = self.permissions.get("always_allow_tools", {}).get(tool_name, [])

        # Check group-based permissions
        from .path_detector import PathDetector
        tool_groups = PathDetector.get_tool_groups(tool_name)
        group_permissions = []
        for group in tool_groups:
            group_perms = self.permissions.get("always_allow_groups", {}).get(group, [])
            group_permissions.extend(group_perms)

        # Check development tool category permissions based on command content
        if xml_content and tool_name == 'bash':
            command_pattern = PathDetector.get_command_pattern(tool_name, xml_content)
            dev_tool_result = self._is_dev_tool_allowed(command_pattern, file_path)
            if isinstance(dev_tool_result, tuple):
                is_allowed, rule_desc = dev_tool_result
                if is_allowed:
                    return True, rule_desc
            elif dev_tool_result:  # For backwards compatibility
                return True, "Development tool permission"

        # Check pattern-based permissions (e.g., "bash:ls*")
        if xml_content:
            command_pattern = PathDetector.get_command_pattern(tool_name, xml_content)
            pattern_result = self._is_pattern_allowed(tool_name, command_pattern, file_path)
            if isinstance(pattern_result, tuple):
                is_allowed, rule_desc = pattern_result
                if is_allowed:
                    return True, rule_desc
            elif pattern_result:  # For backwards compatibility
                return True, f"Pattern permission ({command_pattern[:50]}...)" if len(command_pattern) > 50 else f"Pattern permission ({command_pattern})"

        # Combine tool and group permissions
        all_permissions = tool_permissions + group_permissions

        if not file_path:
            # If no path specified, check if tool has any always-allow permissions
            if tool_name in all_permissions:
                # Determine if this came from tool permissions or group permissions
                if tool_name in tool_permissions:
                    return True, f"Tool-wide permission ({tool_name})"
                else:
                    # Find which group this came from
                    tool_groups = PathDetector.get_tool_groups(tool_name)
                    for group in tool_groups:
                        group_perms = self.permissions.get("always_allow_groups", {}).get(group, [])
                        if tool_name in group_perms:
                            group_display = group.replace('_', ' ').title()
                            return True, f"Group permission ({group_display})"
                    return True, f"Tool permission ({tool_name})"
            return False, None

        file_path = os.path.abspath(file_path)

        # Check exact path matches
        if file_path in all_permissions:
            # Determine source of permission
            if file_path in tool_permissions:
                return True, f"Exact file permission ({os.path.basename(file_path)})"
            else:
                # Must be from group permissions
                tool_groups = PathDetector.get_tool_groups(tool_name)
                for group in tool_groups:
                    group_perms = self.permissions.get("always_allow_groups", {}).get(group, [])
                    if file_path in group_perms:
                        group_display = group.replace('_', ' ').title()
                        return True, f"Group permission ({group_display} - {os.path.basename(file_path)})"
                return True, f"File permission ({os.path.basename(file_path)})"

        # Check pattern matches (directories)
        for allowed_pattern in all_permissions:
            if allowed_pattern.endswith("/*"):
                # Directory pattern
                dir_pattern = allowed_pattern[:-2]
                if file_path.startswith(dir_pattern):
                    # Determine source of permission
                    if allowed_pattern in tool_permissions:
                        return True, f"Directory permission ({os.path.basename(dir_pattern)}/*)"
                    else:
                        # Must be from group permissions
                        tool_groups = PathDetector.get_tool_groups(tool_name)
                        for group in tool_groups:
                            group_perms = self.permissions.get("always_allow_groups", {}).get(group, [])
                            if allowed_pattern in group_perms:
                                group_display = group.replace('_', ' ').title()
                                return True, f"Group permission ({group_display} - {os.path.basename(dir_pattern)}/*)"
                        return True, f"Directory permission ({os.path.basename(dir_pattern)}/*)"
            elif allowed_pattern.endswith("/"):
                # Directory pattern without wildcard
                if file_path.startswith(allowed_pattern):
                    # Determine source of permission
                    if allowed_pattern in tool_permissions:
                        return True, f"Directory permission ({os.path.basename(allowed_pattern.rstrip('/'))})"
                    else:
                        # Must be from group permissions
                        tool_groups = PathDetector.get_tool_groups(tool_name)
                        for group in tool_groups:
                            group_perms = self.permissions.get("always_allow_groups", {}).get(group, [])
                            if allowed_pattern in group_perms:
                                group_display = group.replace('_', ' ').title()
                                return True, f"Group permission ({group_display} - {os.path.basename(allowed_pattern.rstrip('/'))})"
                        return True, f"Directory permission ({os.path.basename(allowed_pattern.rstrip('/'))})"

        return False, None

    def is_always_denied(self, tool_name: str, file_path: Optional[str] = None) -> bool:
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

    def add_always_allow(self, tool_name: str, file_path: Optional[str] = None, pattern: Optional[str] = None, command_hash: Optional[str] = None, group_name: Optional[str] = None, dev_tool_category: Optional[str] = None, command_pattern: Optional[str] = None, scope_path: Optional[str] = None):
        """Add an always-allow permission for a tool/path/command/group/dev_tool/pattern combination."""
        if command_pattern:
            # Add pattern-based permission (e.g., "bash:ls*" with scope "/current/path/*")
            if "pattern_permissions" not in self.permissions:
                self.permissions["pattern_permissions"] = {}

            # Create the pattern key (tool:command_pattern)
            pattern_key = f"{tool_name}:{command_pattern}"

            # Use provided scope_path or default to global
            permission_scope = scope_path if scope_path else "*"  # "*" means global

            self.permissions["pattern_permissions"][pattern_key] = permission_scope
            self._save_permissions()

        elif command_hash:
            # Add command-specific permission
            if "always_allow_commands" not in self.permissions:
                self.permissions["always_allow_commands"] = {}

            if tool_name not in self.permissions["always_allow_commands"]:
                self.permissions["always_allow_commands"][tool_name] = []

            if command_hash not in self.permissions["always_allow_commands"][tool_name]:
                self.permissions["always_allow_commands"][tool_name].append(command_hash)
                self._save_permissions()
        elif dev_tool_category:
            # Add dev tool category permission
            if "always_allow_dev_tools" not in self.permissions:
                self.permissions["always_allow_dev_tools"] = {}

            if dev_tool_category not in self.permissions["always_allow_dev_tools"]:
                self.permissions["always_allow_dev_tools"][dev_tool_category] = []

            permission_target = pattern if pattern else (os.path.abspath(file_path) if file_path else dev_tool_category)

            if permission_target not in self.permissions["always_allow_dev_tools"][dev_tool_category]:
                self.permissions["always_allow_dev_tools"][dev_tool_category].append(permission_target)
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

    def add_always_deny(self, tool_name: str, file_path: Optional[str] = None):
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

    def get_permission_suggestions(self, tool_name: str, file_path: Optional[str] = None, command_pattern: Optional[str] = None, has_paths: bool = False) -> List[str]:
        """Get suggested permission patterns based on file path or command pattern."""
        suggestions = []

        # Import here to avoid circular dependency
        from .path_detector import PathDetector

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
            if command_pattern and tool_name == 'bash':
                # Add pattern-based suggestions for bash commands
                base_command = command_pattern.split()[0] if command_pattern.split() else ""
                if base_command:
                    # Pattern suggestions for current directory tree
                    suggestions.append(f"All \"{base_command}*\" commands in current directory tree")
                    suggestions.append(f"All \"{base_command}*\" commands globally")
                    suggestions.append(f"Exact command: {command_pattern}")
                else:
                    suggestions.append(f"Exact command: {command_pattern}")
            elif command_pattern:
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

    def is_smart_action_already_permitted(self, smart_action: dict, tool_name: str, file_path: Optional[str] = None) -> bool:
        """
        Check if a smart action category is already covered by existing permissions.

        Args:
            smart_action: Smart action dictionary with category, scope_path, etc.
            tool_name: Name of the tool being executed
            file_path: Optional file path being operated on

        Returns:
            True if the smart action is already covered by existing permissions
        """
        if not smart_action:
            return False

        category = smart_action.get('category')
        if not category:
            return False

        # Check if this dev tool category is already allowed
        dev_tool_permissions = self.permissions.get("always_allow_dev_tools", {})
        if category in dev_tool_permissions:
            category_perms = dev_tool_permissions[category]

            # Use file path context if provided, otherwise fall back to CWD
            context_path = file_path if file_path else os.getcwd()

            # For file paths, use the directory containing the file
            if file_path and os.path.isfile(context_path):
                context_path = os.path.dirname(context_path)

            for perm_pattern in category_perms:
                if perm_pattern.endswith("/*"):
                    # Directory pattern - check if current context is covered
                    dir_pattern = perm_pattern[:-2]
                    if context_path.startswith(dir_pattern):
                        return True
                elif perm_pattern == category:
                    # Global category permission
                    return True
                elif os.path.abspath(perm_pattern) == context_path:
                    # Exact directory match
                    return True

        return False

    def _is_dev_tool_allowed(self, command_pattern: str, file_path: str = None) -> Tuple[bool, Optional[str]]:
        """
        Check if a development tool command pattern is allowed based on category permissions.

        Args:
            command_pattern: The command pattern to check
            file_path: Optional file path being operated on (for context-aware checking)

        Returns:
            Tuple of (is_allowed, rule_description) where rule_description explains which dev tool rule matched
        """
        from .path_detector import PathDetector

        # Detect what category this command belongs to
        category, config = PathDetector.detect_dev_tool_category(command_pattern)
        if not category:
            return False, None

        # Check if we have permissions for this dev tool category
        dev_tool_permissions = self.permissions.get("always_allow_dev_tools", {})

        # Check for exact category permissions
        if category in dev_tool_permissions:
            category_perms = dev_tool_permissions[category]

            # Use file path context if provided, otherwise fall back to CWD
            context_path = file_path if file_path else os.getcwd()

            # For file paths, use the directory containing the file
            if file_path and os.path.isfile(context_path):
                context_path = os.path.dirname(context_path)

            for perm_pattern in category_perms:
                if perm_pattern.endswith("/*"):
                    # Directory pattern
                    dir_pattern = perm_pattern[:-2]
                    if context_path.startswith(dir_pattern):
                        category_display = category.replace('_', ' ').title()
                        return True, f"Dev tool permission ({category_display} - {os.path.basename(dir_pattern)}/*)"
                elif perm_pattern == category:
                    # Global category permission
                    category_display = category.replace('_', ' ').title()
                    return True, f"Dev tool permission ({category_display} - global)"
                elif os.path.abspath(perm_pattern) == context_path:
                    # Exact directory match
                    category_display = category.replace('_', ' ').title()
                    return True, f"Dev tool permission ({category_display} - {os.path.basename(context_path)})"

        return False, None

    def _is_pattern_allowed(self, tool_name: str, command_pattern: str, file_path: str = None) -> Tuple[bool, Optional[str]]:
        """
        Check if a command pattern is allowed based on stored pattern permissions.

        Args:
            tool_name: Name of the tool (e.g., 'bash')
            command_pattern: The actual command being executed (e.g., 'bash: ls -l')
            file_path: Optional file path being operated on (for context-aware checking)

        Returns:
            Tuple of (is_allowed, rule_description) where rule_description explains which pattern rule matched
        """
        pattern_permissions = self.permissions.get("pattern_permissions", {})

        # Extract the actual command from the pattern (remove "tool: " prefix)
        actual_command = command_pattern or ""
        if command_pattern and ":" in command_pattern:
            actual_command = command_pattern.split(":", 1)[1].strip()

        # Extract the base command (first word) from the actual command
        base_command = actual_command.split()[0] if actual_command and actual_command.split() else ""

        if not base_command:
            return False, None

        # Check for exact pattern match (e.g., "bash:ls -l")
        exact_pattern_key = f"{tool_name}:{actual_command}"
        if exact_pattern_key in pattern_permissions:
            scope = pattern_permissions[exact_pattern_key]
            if self._check_pattern_scope(scope, file_path):
                scope_desc = "global" if scope == "*" else f"{os.path.basename(scope.rstrip('/*'))}" if scope.endswith("/*") else scope
                return True, f"Pattern permission ({base_command} - {scope_desc})"

        # Check for wildcard pattern matches (e.g., "bash:ls*")
        for pattern_key, scope in pattern_permissions.items():
            if not pattern_key.startswith(f"{tool_name}:"):
                continue

            stored_pattern = pattern_key.split(":", 1)[1]  # Get the command part

            # Handle wildcard patterns
            if stored_pattern.endswith("*"):
                pattern_prefix = stored_pattern[:-1]  # Remove the *
                if base_command.startswith(pattern_prefix):
                    if self._check_pattern_scope(scope, file_path):
                        scope_desc = "global" if scope == "*" else f"{os.path.basename(scope.rstrip('/*'))}" if scope.endswith("/*") else scope
                        return True, f"Pattern permission ({stored_pattern} - {scope_desc})"
            elif stored_pattern == base_command:
                if self._check_pattern_scope(scope, file_path):
                    scope_desc = "global" if scope == "*" else f"{os.path.basename(scope.rstrip('/*'))}" if scope.endswith("/*") else scope
                    return True, f"Pattern permission ({stored_pattern} - {scope_desc})"

        return False, None

    def _check_pattern_scope(self, scope: str, file_path: str = None) -> bool:
        """
        Check if the current context matches the pattern scope.

        Args:
            scope: The scope string (e.g., "/path/*", "*" for global)
            file_path: Optional file path being operated on (for context-aware checking)

        Returns:
            True if the current context matches the scope
        """
        if scope == "*":
            # Global permission
            return True

        # Use file path context if provided, otherwise fall back to CWD
        context_path = file_path if file_path else os.getcwd()

        # For file paths, use the directory containing the file
        if file_path and os.path.isfile(context_path):
            context_path = os.path.dirname(context_path)

        if scope.endswith("/*"):
            # Directory scope permission
            allowed_dir = scope[:-2]
            return context_path.startswith(allowed_dir)
        else:
            # Exact directory match
            return context_path == scope

    def clear_all_permissions(self):
        """Clear all stored permissions."""
        self.permissions = {
            "always_allow_tools": {},
            "always_deny_tools": {},
            "always_allow_dev_tools": {},  # Add dev tools category
            "global_always_allow": False
        }
        self._save_permissions()

    def save_permissions(self):
        """Public method to save permissions to file."""
        self._save_permissions()

# Global instance
_permissions_manager = None

def get_permissions_manager() -> PermissionsManager:
    """Get the global permissions manager instance."""
    global _permissions_manager
    if _permissions_manager is None:
        _permissions_manager = PermissionsManager()
    return _permissions_manager