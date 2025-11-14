from typing import Optional

"""
Simplified permission settings manager for CLI-Agent.
Claude Code-style config-based permission system with pattern matching.
"""
import fnmatch
import json
import os
import re


# Tool classifications for auto-allow policies
READ_ONLY_FILE_TOOLS = {
    'readfile',       # Read file contents (text, PDF, images, Office docs, etc.)
    'readdirectory',  # List directory structure as tree
    'searchfile',     # BM25-based search in files/directories
    'grepsearch'      # Regex search in file contents and names
}

# Agent memory directory - safe for read-only tool access
# Specifically targets the memory subdirectory where agent stores working context
AGENT_MEMORY_DIR = os.path.expanduser("~/.cli-agent/memory/")


class SettingsManager:
    """
    Manages permission settings with simple allow/deny/ask lists.

    Settings file format (~/.cli-agent/settings.json):
    {
        "permissions": {
            "allow": ["bash(npm:*)", "readfile(./src/**)"],
            "deny": ["readfile(.env)", "bash(sudo:*)"],
            "ask": ["bash(git push:*)"]
        }
    }

    Pattern syntax:
    - tool(pattern) - e.g., "bash(ls:*)" matches bash tool with ls command
    - tool(path) - e.g., "readfile(./src/**)" matches readfile in src directory
    - Glob patterns supported: *, **, ?, [abc]
    - No aliases - use raw tool names (readfile, bash, editfile, etc.)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings manager.

        Args:
            config_path: Path to settings.json. If None, uses ~/.cli-agent/settings.json
        """
        if config_path is None:
            config_path = os.path.expanduser("~/.cli-agent/settings.json")

        self.config_path = config_path
        self.settings = self._load_settings()

    def _load_settings(self) -> dict:
        """Load settings from JSON file or create default structure."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load settings from {self.config_path}: {e}")

        # Default empty settings
        return {
            "permissions": {
                "allow": [],
                "deny": [],
                "ask": []
            }
        }

    def save_settings(self) -> bool:
        """
        Save settings to JSON file.

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except OSError as e:
            print(f"Warning: Could not save settings: {e}")
            return False

    def _extract_tool_and_spec(self, rule: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract tool name and specifier from rule.

        Args:
            rule: Permission rule like "bash(npm:*)" or "readfile(./src/**)"

        Returns:
            Tuple of (tool_name, specifier) or (None, None) if invalid

        Examples:
            "bash(npm:*)" → ("bash", "npm:*")
            "readfile(./src/**)" → ("readfile", "./src/**")
            "bash" → ("bash", None)
        """
        # Check if rule has specifier
        if '(' not in rule:
            return rule.strip().lower(), None

        # Extract: "tool(spec)" → tool, spec
        match = re.match(r'^([a-z_]+)\((.*)\)$', rule.strip(), re.IGNORECASE)
        if not match:
            return None, None

        tool_name = match.group(1).lower()
        spec = match.group(2).strip()

        return tool_name, spec if spec else None

    def _matches_pattern(self, rule: str, tool_name: str,
                        file_path: Optional[str] = None,
                        command: Optional[str] = None) -> bool:
        """
        Check if a permission rule matches the tool being used.

        Args:
            rule: Permission rule from settings
            tool_name: Tool being executed (e.g., "bash", "readfile")
            file_path: File path if applicable
            command: Command string if tool is bash

        Returns:
            True if rule matches this tool invocation

        Examples:
            rule="bash(npm:*)", tool_name="bash", command="npm install" → True
            rule="readfile(./src/**)", tool_name="readfile", file_path="./src/main.py" → True
            rule="bash", tool_name="bash" → True (matches all bash commands)
        """
        rule_tool, rule_spec = self._extract_tool_and_spec(rule)

        if not rule_tool:
            return False

        # Tool name must match
        if rule_tool != tool_name.lower():
            return False

        # If no specifier, matches all uses of this tool
        if not rule_spec:
            return True

        # Command pattern matching (for bash tool)
        if ':' in rule_spec and command:
            # Format: "command:*" or "command:args"
            cmd_pattern = rule_spec.split(':', 1)[0]
            # Use word boundaries to prevent bypass attacks
            # e.g., "git push:*" should NOT match "git pushx malicious.sh"
            import re
            # Create regex with word boundary after command
            pattern = re.compile(rf'^{re.escape(cmd_pattern)}\b', re.IGNORECASE)
            return pattern.match(command.strip()) is not None

        # File path pattern matching
        if file_path and rule_spec:
            # Expand ~ and resolve relative paths for comparison
            # SECURITY: Normalize paths to prevent ../ bypass attacks
            try:
                expanded_pattern = os.path.expanduser(rule_spec)
                expanded_path = os.path.expanduser(file_path)

                # Normalize paths to prevent ../ traversal bypass
                normalized_path = os.path.normpath(expanded_path)
                normalized_pattern = os.path.normpath(expanded_pattern)

                # Convert glob patterns (** → *)
                glob_pattern = normalized_pattern.replace('**/', '**/').replace('**', '*')

                # Use fnmatch for pattern matching with normalized paths
                if fnmatch.fnmatch(normalized_path, glob_pattern):
                    return True

                # Also try relative matching with original paths
                if fnmatch.fnmatch(file_path, rule_spec):
                    return True

            except Exception:
                pass

        return False

    def check_permission(self, tool_name: str,
                        file_path: Optional[str] = None,
                        command: Optional[str] = None) -> tuple[str, str]:
        """
        Check permission for a tool execution.

        Precedence order: deny → auto-allow (read-only memory) → ask → allow → default

        Args:
            tool_name: Name of tool being executed
            file_path: File path if applicable
            command: Command string if tool is bash

        Returns:
            Tuple of (decision, reason)
            decision: "deny", "ask", "allow", or "default"
            reason: Human-readable explanation

        Examples:
            >>> mgr.check_permission("bash", command="sudo apt-get")
            ("deny", "Blocked by permission rule: bash(sudo:*)")

            >>> mgr.check_permission("readfile", file_path="./src/main.py")
            ("allow", "Permitted by rule: readfile(./src/**)")

            >>> mgr.check_permission("readfile", file_path="~/.cli-agent/memory/context.txt")
            ("allow", "Auto-allowed: read-only access to agent memory (readfile)")
        """
        permissions = self.settings.get("permissions", {})

        # Layer 1: Check deny list (always blocks - highest priority)
        for deny_rule in permissions.get("deny", []):
            if self._matches_pattern(deny_rule, tool_name, file_path, command):
                return "deny", f"Blocked by permission rule: {deny_rule}"

        # Layer 2: Auto-allow read-only file tools when accessing agent memory directory
        if tool_name.lower() in READ_ONLY_FILE_TOOLS and file_path:
            try:
                # Normalize paths to prevent ../ bypass attacks
                normalized_path = os.path.normpath(os.path.abspath(os.path.expanduser(file_path)))
                normalized_memory = os.path.normpath(os.path.abspath(AGENT_MEMORY_DIR))

                # Check if path is within agent memory directory
                if normalized_path.startswith(normalized_memory):
                    return "allow", f"Auto-allowed: read-only access to agent memory ({tool_name})"
            except Exception:
                # Fall through to normal permission checking if path resolution fails
                pass

        # Layer 3: Check ask list (requires confirmation)
        for ask_rule in permissions.get("ask", []):
            if self._matches_pattern(ask_rule, tool_name, file_path, command):
                return "ask", f"Requires confirmation: {ask_rule}"

        # Layer 4: Check allow list (always permits)
        for allow_rule in permissions.get("allow", []):
            if self._matches_pattern(allow_rule, tool_name, file_path, command):
                return "allow", f"Permitted by rule: {allow_rule}"

        # Default: no explicit permission set
        return "default", "No matching permission rule"

    def add_permission(self, permission_list: str, rule: str) -> bool:
        """
        Add a permission rule to allow/deny/ask list.

        Args:
            permission_list: "allow", "deny", or "ask"
            rule: Permission rule to add

        Returns:
            True if added successfully, False if invalid
        """
        if permission_list not in ["allow", "deny", "ask"]:
            return False

        # Validate rule format
        tool, spec = self._extract_tool_and_spec(rule)
        if not tool:
            return False

        # Add to list if not already present
        if rule not in self.settings["permissions"][permission_list]:
            self.settings["permissions"][permission_list].append(rule)
            return True

        return False

    def remove_permission(self, rule: str) -> bool:
        """
        Remove a permission rule from all lists.

        Args:
            rule: Permission rule to remove

        Returns:
            True if removed from at least one list, False otherwise
        """
        removed = False

        for list_name in ["allow", "deny", "ask"]:
            if rule in self.settings["permissions"][list_name]:
                self.settings["permissions"][list_name].remove(rule)
                removed = True

        return removed

    def get_all_permissions(self) -> dict[str, list[str]]:
        """
        Get all permission rules organized by type.

        Returns:
            Dictionary with keys: "allow", "deny", "ask"
        """
        return self.settings.get("permissions", {
            "allow": [],
            "deny": [],
            "ask": []
        })

    def clear_all_permissions(self):
        """Clear all permission rules."""
        self.settings["permissions"] = {
            "allow": [],
            "deny": [],
            "ask": []
        }


def extract_path_from_xml(xml_content: str, tool_name: str) -> Optional[str]:
    """
    Extract file path from XML tool block.

    Args:
        xml_content: XML content of tool call
        tool_name: Name of the tool

    Returns:
        Extracted file path or None

    Examples:
        <readfile><path>./src/main.py</path></readfile> → "./src/main.py"
        <bash><command>ls ./src</command></bash> → None
    """
    # Common path tag names
    path_tags = ['path', 'file_path', 'filepath', 'filename', 'file']

    for tag in path_tags:
        pattern = f'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, xml_content, re.DOTALL)
        if match:
            path = match.group(1).strip()
            # Expand user paths
            return os.path.expanduser(path) if path else None

    return None


def extract_command_from_xml(xml_content: str) -> Optional[str]:
    """
    Extract command from bash tool XML block.

    Args:
        xml_content: XML content of bash tool call

    Returns:
        Extracted command or None

    Examples:
        <bash><command>npm install</command></bash> → "npm install"
    """
    # Look for command tag
    match = re.search(r'<command>(.*?)</command>', xml_content, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None
