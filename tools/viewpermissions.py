# tools/viewpermissions.py
"""
This file implements the 'viewpermissions' tool. It allows the agent to inspect
the saved permission rules using the simplified settings.json format.
"""
from core.permissions.settings_manager import SettingsManager


class viewpermissions:
    """
    A tool to view the currently saved permission rules (simplified system).
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
            "description": "Displays the currently saved permission rules (allow/deny/ask lists). Shows which tools and paths are auto-approved, blocked, or require confirmation.",
            "example": "<viewpermissions></viewpermissions>"
        }

    @staticmethod
    def run(content: str = "") -> str:
        """
        Loads and formats the saved permission rules from SettingsManager.
        This tool takes no arguments inside its tags.
        """
        try:
            settings_mgr = SettingsManager()
            permissions = settings_mgr.get_all_permissions()

            output_lines = ["🔓 PERMISSION SETTINGS"]
            output_lines.append("=" * 60)

            has_rules = False

            # 1. ALLOW list (auto-approved operations)
            allow_list = permissions.get("allow", [])
            if allow_list:
                has_rules = True
                output_lines.append("\n✅ ALLOW (execute without asking):")
                for rule in sorted(allow_list):
                    output_lines.append(f"   • {rule}")

            # 2. ASK list (require confirmation)
            ask_list = permissions.get("ask", [])
            if ask_list:
                has_rules = True
                output_lines.append("\n❓ ASK (require confirmation):")
                for rule in sorted(ask_list):
                    output_lines.append(f"   • {rule}")

            # 3. DENY list (blocked operations)
            deny_list = permissions.get("deny", [])
            if deny_list:
                has_rules = True
                output_lines.append("\n❌ DENY (blocked):")
                for rule in sorted(deny_list):
                    output_lines.append(f"   • {rule}")

            # Footer information
            output_lines.append(f"\n📄 Config file: {settings_mgr.config_path}")

            if not has_rules:
                return "🔒 No permissions configured - all operations will require confirmation."

            output_lines.append("\n" + "=" * 60)
            output_lines.append("\nPermission Rule Syntax:")
            output_lines.append("  • tool_name(pattern) - e.g., bash(npm:*), readfile(./src/**)")
            output_lines.append("  • tool_name - allow/deny all uses of tool")
            output_lines.append("  • Glob patterns: *, **, ?, [abc]")
            output_lines.append("\nEdit settings manually: nano ~/.cli-agent/settings.json")
            output_lines.append("Or use: /perms add allow <rule>")

            return "\n".join(output_lines)

        except Exception as e:
            return f"Error retrieving permissions: {str(e)}"
