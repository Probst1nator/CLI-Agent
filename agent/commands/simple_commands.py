"""Simple command implementations that don't require complex logic."""

from __future__ import annotations
from typing import List, Any, Dict, Optional, TYPE_CHECKING
from termcolor import colored

from agent.commands.base_command import BaseCommand, CommandResponse, CommandResult

if TYPE_CHECKING:
    pass

class MultilineCommand(BaseCommand):
    """Command for multiline input mode."""

    @property
    def aliases(self) -> List[str]:
        return ['/m', '/message']

    @property
    def description(self) -> str:
        return 'Enter multiline input mode'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        multiline_input = self._handle_multiline_input()
        return CommandResponse(
            result=CommandResult.SUCCESS,
            user_input=multiline_input
        )

    def _handle_multiline_input(self) -> str:
        """Handle multiline input (extracted from main.py)."""
        print(colored("# cli-agent: Entering multiline mode. Type 'END' on a new line when finished.", "cyan"))
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            except KeyboardInterrupt:
                print(colored("\n# cli-agent: Multiline input cancelled.", "yellow"))
                return ""
        return "\n".join(lines)

class ExitCommand(BaseCommand):
    """Command to exit CLI-Agent."""

    @property
    def aliases(self) -> List[str]:
        return ['/e', '/exit']

    @property
    def description(self) -> str:
        return 'Exit CLI-Agent'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        print(colored("# cli-agent: Goodbye!", "cyan"))
        return CommandResponse(result=CommandResult.EXIT)


class UnsayCommand(BaseCommand):
    """Command to remove last user message and AI response."""

    @property
    def aliases(self) -> List[str]:
        return ['/unsay']

    @property
    def description(self) -> str:
        return 'Remove last user message and AI response'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        context_chat = kwargs.get('context_chat')
        if context_chat and len(context_chat.messages) >= 2:
            context_chat.messages.pop()  # Remove AI response
            context_chat.messages.pop()  # Remove user message
            print(colored("# cli-agent: Last exchange removed.", "green"))
        else:
            print(colored("# cli-agent: No messages to remove.", "red"))
        return CommandResponse(result=CommandResult.CONTINUE)

class PrintCommand(BaseCommand):
    """Command to print complete chat history."""

    @property
    def aliases(self) -> List[str]:
        return ['/p', '/print']

    @property
    def description(self) -> str:
        return 'Print complete chat history'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        context_chat = kwargs.get('context_chat')
        if context_chat:
            print(colored("=== Chat History ===", "blue", attrs=["bold"]))
            if hasattr(context_chat, 'messages') and context_chat.messages:
                context_chat.print_chat()
            else:
                print(colored("# cli-agent: Chat object exists but has no messages.", "yellow"))
        else:
            print(colored("# cli-agent: No chat history available.", "yellow"))
        return CommandResponse(result=CommandResult.CONTINUE)

# Toggle commands for various modes
class YoloCommand(BaseCommand):
    """Toggle YOLO mode (automatic code execution)."""

    @property
    def aliases(self) -> List[str]:
        return ['/yolo', '/y']

    @property
    def description(self) -> str:
        return 'Toggle YOLO mode (automatic code execution)'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        from core.globals import g

        self.context.settings.auto_execute = not self.context.settings.auto_execute
        # Sync with global state for compatibility
        g.auto_execute = self.context.settings.auto_execute

        status = "enabled" if self.context.settings.auto_execute else "disabled"
        print(colored(f"# cli-agent: YOLO mode {status}.", "green"))

        # Broadcast state change to frontend
        try:
            from api.server import get_server
            import asyncio
            server = get_server()
            if server:
                asyncio.create_task(server.broadcast_state_change())
        except Exception:
            pass  # Ignore if server not available

        return CommandResponse(result=CommandResult.CONTINUE)

class VoiceCommand(BaseCommand):
    """Toggle voice input/output mode."""

    @property
    def aliases(self) -> List[str]:
        return ['/v', '/voice']

    @property
    def description(self) -> str:
        return 'Toggle voice input/output mode'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        from core.globals import g

        self.context.settings.voice_mode = not self.context.settings.voice_mode
        # Sync with global state for compatibility
        g.voice_mode = self.context.settings.voice_mode

        status = "enabled" if self.context.settings.voice_mode else "disabled"
        print(colored(f"# cli-agent: Voice mode {status}.", "green"))

        # Broadcast state change to frontend
        try:
            from api.server import get_server
            import asyncio
            server = get_server()
            if server:
                asyncio.create_task(server.broadcast_state_change())
        except Exception:
            pass  # Ignore if server not available

        return CommandResponse(result=CommandResult.CONTINUE)



class LocalCommand(BaseCommand):
    """Toggle local-only LLM mode."""

    @property
    def aliases(self) -> List[str]:
        return ['/l', '/local']

    @property
    def description(self) -> str:
        return 'Toggle local-only LLM mode'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        from core.globals import g

        self.context.settings.local_mode = not self.context.settings.local_mode
        # Sync with global state for compatibility
        g.local_mode = self.context.settings.local_mode

        status = "enabled" if self.context.settings.local_mode else "disabled"
        print(colored(f"# cli-agent: Local mode {status}.", "green"))

        # Broadcast state change to frontend
        try:
            from api.server import get_server
            import asyncio
            server = get_server()
            if server:
                asyncio.create_task(server.broadcast_state_change())
        except Exception:
            pass  # Ignore if server not available

        return CommandResponse(result=CommandResult.CONTINUE)

class OnlineCommand(BaseCommand):
    """Toggle online mode for web-enabled features."""

    @property
    def aliases(self) -> List[str]:
        return ['/o', '/online']

    @property
    def description(self) -> str:
        return 'Toggle online mode for web-enabled features'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        from core.globals import g

        self.context.settings.online_mode = not self.context.settings.online_mode
        # Sync with global state for compatibility
        g.online_mode = self.context.settings.online_mode

        status = "enabled" if self.context.settings.online_mode else "disabled"
        print(colored(f"# cli-agent: Online mode {status}.", "green"))
        return CommandResponse(result=CommandResult.CONTINUE)

class FullOutputCommand(BaseCommand):
    """Toggle full model output mode."""

    @property
    def aliases(self) -> List[str]:
        return ['/fo', '/full_output']

    @property
    def description(self) -> str:
        return 'Show full model output instead of summary'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        self.context.settings.full_output_mode = not self.context.settings.full_output_mode
        status = "enabled" if self.context.settings.full_output_mode else "disabled"
        print(colored(f"# cli-agent: Full output mode {status}.", "green"))
        return CommandResponse(result=CommandResult.CONTINUE)

# Stub classes for complex commands that need full implementation
class LlmCommand(BaseCommand):
    """LLM selection command implementation."""

    @property
    def aliases(self) -> List[str]:
        return ['/llm']

    @property
    def description(self) -> str:
        return 'Open LLM selection interface'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        """Execute LLM selection using the enhanced LlmSelector UI."""
        try:
            from agent.llm_selection import LlmSelector
            from core.globals import g
            from main import get_provider_emoji_and_name

            self.logger.info(colored("# cli-agent: Opening LLM selection interface.", "green"))

            selector = LlmSelector()

            # Use the selector to get user selection
            # Get args from kwargs if available
            cli_args = kwargs.get('args')
            result = await selector.get_selection(
                preselected_llms=getattr(cli_args, 'llm', None) if cli_args else None,
                force_local=getattr(cli_args, 'local', False) if cli_args else False,
                save_selection=True
            )

            # Process the structured result from the selector
            if result.get("status") != "Success":
                message = result.get('message', 'LLM selection cancelled or failed.')
                self.logger.warning(colored(f"# cli-agent: {message}", "yellow"))
                return CommandResponse(result=CommandResult.CONTINUE, message=message)

            # Handle the special 'any_local' case
            if result.get("selection_type") == "any_local":
                g.FORCE_LOCAL = True
                self.context.settings.local_mode = True
                # Note: LLM selection handled separately in context
                self.context.settings.local_mode = True
                self.logger.info(colored("# cli-agent: 'Any but local' option selected. Local mode enabled.", "green"))
                return CommandResponse(result=CommandResult.CONTINUE, message="Local mode enabled")

            # Handle specific model selections
            selected_configs = result.get("selected_configs", [])

            # No evaluator models needed in simplified system

            selected_llms = [config['model_key'] for config in selected_configs]

            # Update context and globals with the latest selection
            # Note: LLM selection stored in context
            g.SELECTED_LLMS = selected_llms
            self.context.llm_config.model_key = selected_llms[0] if selected_llms else ""

            # Reload the config in `g` so the main loop gets the fresh settings
            g.load_llm_config()

            if not selected_llms:
                message = "LLM set to auto, type (-h) for info"
                self.logger.info(colored(f"# cli-agent: {message}", "green"))
            else:
                # Count unique models by provider and configuration
                unique_models = []
                seen = set()
                for model in selected_llms:
                    if model not in seen:
                        unique_models.append(model)
                        seen.add(model)

                provider_counts = {}

                for model in unique_models:
                    emoji, provider_name = get_provider_emoji_and_name(model)
                    provider_key = f"{emoji} {provider_name}"
                    provider_counts[provider_key] = provider_counts.get(provider_key, 0) + 1

                # Display summary
                summary_lines = ["LLM Configuration:"]
                for provider, count in provider_counts.items():
                    summary_lines.append(f"  {provider}: {count} model{'s' if count > 1 else ''}")

                message = "\n".join(summary_lines)
                print(colored(message, "cyan"))

            return CommandResponse(result=CommandResult.CONTINUE, message="LLM selection complete")

        except Exception as e:
            error_msg = f"LLM selection failed: {str(e)}"
            self.logger.error(error_msg)
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)

class PermissionCommand(BaseCommand):
    """Permission management command implementation."""

    @property
    def aliases(self) -> List[str]:
        return ['/perms', '/permissions']

    @property
    def description(self) -> str:
        return 'Manage tool execution permissions'

    @property
    def sub_commands(self) -> Optional[Dict[str, str]]:
        return {
            'list': 'List all permissions with IDs for removal (default)',
            'view': 'View permissions in a user-friendly format',
            'add': 'Add new permission with guided prompts',
            'add dev <category>': 'Add development tool permission',
            'add tool <name> <target>': 'Add tool permission',
            'add group <name> <target>': 'Add group permission',
            'add command <pattern> <scope>': 'Add command permission',
            'remove': 'Remove specific permission by ID',
            'clear': 'Clear all saved permissions',
        }

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        """Execute permission management command."""
        try:
            # Import the new tool for the 'view' command
            from tools.viewpermissions import viewpermissions

            action = parts[1].lower() if len(parts) > 1 else "list"

            if action == "list":
                # List permissions with numbers for management
                result = self._list_permissions_for_management()
                print(result)
                return CommandResponse(result=CommandResult.CONTINUE, message="Permissions listed")

            elif action == "view":
                # Use the new tool for a pretty, readable view
                result = viewpermissions.run("")
                print(result)
                return CommandResponse(result=CommandResult.CONTINUE, message="Permissions viewed")

            elif action == "add":
                # Add new permission with guided wizard
                await self._add_permission_wizard(parts)
                return CommandResponse(result=CommandResult.CONTINUE, message="Add permission completed")

            elif action == "remove" and len(parts) > 2:
                perm_id = parts[2]
                result = self._remove_permission(perm_id)
                print(result)
                return CommandResponse(result=CommandResult.CONTINUE, message=f"Permission {perm_id} processed")

            elif action == "clear":
                result = self._clear_all_permissions()
                print(result)
                return CommandResponse(result=CommandResult.CONTINUE, message="Permissions cleared")

            else:
                error_msg = f"Unknown permission action: {action}. Valid actions: list, view, remove <id>, clear, interactive"
                print(colored(f"❓ {error_msg}", "red"))
                return CommandResponse(result=CommandResult.ERROR, message=error_msg)

        except Exception as e:
            error_msg = f"Error handling permission command: {str(e)}"
            print(colored(f"❌ {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)

    async def _add_permission_wizard(self, parts: List[str]) -> None:
        """Simple add permission wizard."""
        try:
            print()
            print(colored("✨ Add Permission Wizard", "cyan", attrs=['bold']))
            print(colored("=" * 40, "cyan"))

            # Check if specific type was provided
            if len(parts) >= 3:
                perm_type = parts[2].lower()
                if perm_type == "dev":
                    await self._add_dev_permission(parts[3:])
                elif perm_type == "tool":
                    await self._add_tool_permission(parts[3:])
                elif perm_type == "group":
                    await self._add_group_permission(parts[3:])
                elif perm_type == "command":
                    await self._add_command_permission(parts[3:])
                else:
                    print(colored(f"❌ Unknown permission type: {perm_type}", "red"))
                    print(colored("Valid types: dev, tool, group, command", "yellow"))
                return

            # Interactive type selection
            print(colored("\n📂 Choose permission type:", "yellow"))
            types = [
                ("1", "dev", "Development tools (grep, find, etc.)"),
                ("2", "tool", "Specific tools on files/directories"),
                ("3", "group", "Groups of related tools"),
                ("4", "command", "Command patterns"),
            ]

            for key, ptype, desc in types:
                print(f"  {colored(key, 'green', attrs=['bold'])}: {colored(ptype, 'white')} - {desc}")

            try:
                choice = input(colored("\nEnter choice (1-4): ", "cyan")).strip()

                if choice == '1':
                    await self._add_dev_permission([])
                elif choice == '2':
                    await self._add_tool_permission([])
                elif choice == '3':
                    await self._add_group_permission([])
                elif choice == '4':
                    await self._add_command_permission([])
                else:
                    print(colored("❌ Invalid choice.", "red"))

            except KeyboardInterrupt:
                print(colored("\n❌ Wizard cancelled.", "yellow"))

        except Exception as e:
            print(colored(f"❌ Error in add wizard: {e}", "red"))

    async def _add_dev_permission(self, args: List[str]) -> None:
        """Add development tool permission."""
        try:
            from core.permissions.permissions_manager import get_permissions_manager
            pm = get_permissions_manager()

            print(colored("\n🛠️  Development Tool Permission", "cyan"))

            if args and len(args) >= 1:
                category = args[0]
                scope = args[1] if len(args) >= 2 else "*"
            else:
                print(colored("Common categories: grepsearch, filesystems, network, development", "yellow"))
                category = input(colored("Enter category name: ", "cyan")).strip()
                if not category:
                    print(colored("❌ Category required.", "red"))
                    return

                scope = input(colored("Enter scope (path or '*' for global): ", "cyan")).strip()
                if not scope:
                    scope = "*"

            pm.add_always_allow(tool_name="bash", dev_tool_category=category, pattern=scope)
            print(colored(f"✅ Added dev tool permission: {category} -> {scope}", "green"))

        except Exception as e:
            print(colored(f"❌ Error adding dev permission: {e}", "red"))

    async def _add_tool_permission(self, args: List[str]) -> None:
        """Add tool-specific permission."""
        try:
            from core.permissions.permissions_manager import get_permissions_manager
            pm = get_permissions_manager()

            print(colored("\n🔧 Tool Permission", "cyan"))

            if args and len(args) >= 2:
                tool_name = args[0]
                target = args[1]
            else:
                tool_name = input(colored("Enter tool name (e.g., 'bash', 'readfile'): ", "cyan")).strip()
                if not tool_name:
                    print(colored("❌ Tool name required.", "red"))
                    return

                target = input(colored("Enter target (file path, directory, or tool name): ", "cyan")).strip()
                if not target:
                    target = tool_name

            pm.add_always_allow(
                tool_name=tool_name,
                file_path=target if target and target != tool_name else None,
                pattern=target if target and target != tool_name else None
            )
            print(colored(f"✅ Added tool permission: {tool_name} -> {target}", "green"))

        except Exception as e:
            print(colored(f"❌ Error adding tool permission: {e}", "red"))

    async def _add_group_permission(self, args: List[str]) -> None:
        """Add group permission."""
        try:
            from core.permissions.permissions_manager import get_permissions_manager
            pm = get_permissions_manager()

            print(colored("\n📁 Group Permission", "cyan"))

            if args and len(args) >= 2:
                group_name = args[0]
                target = args[1]
            else:
                print(colored("Common groups: file_tools, system_tools, development_tools, search_tools", "yellow"))
                group_name = input(colored("Enter group name: ", "cyan")).strip()
                if not group_name:
                    print(colored("❌ Group name required.", "red"))
                    return

                target = input(colored("Enter target (path or group name for global): ", "cyan")).strip()
                if not target:
                    target = group_name

            pm.add_always_allow(
                tool_name="",
                group_name=group_name,
                pattern=target if target and target != group_name else None
            )
            print(colored(f"✅ Added group permission: {group_name} -> {target}", "green"))

        except Exception as e:
            print(colored(f"❌ Error adding group permission: {e}", "red"))

    async def _add_command_permission(self, args: List[str]) -> None:
        """Add command pattern permission."""
        try:
            from core.permissions.permissions_manager import get_permissions_manager
            pm = get_permissions_manager()

            print(colored("\n⚡ Command Permission", "cyan"))
            print(colored("Examples: 'ls*', 'git status', 'npm*'", "yellow"))

            if args and len(args) >= 2:
                command_pattern = args[0]
                scope = args[1]
            else:
                command_pattern = input(colored("Enter command pattern: ", "cyan")).strip()
                if not command_pattern:
                    print(colored("❌ Command pattern required.", "red"))
                    return

                scope = input(colored("Enter scope path (or '*' for global): ", "cyan")).strip()
                if not scope:
                    scope = "*"

            pm.add_always_allow(
                tool_name="bash",
                command_pattern=command_pattern,
                scope_path=scope
            )
            print(colored(f"✅ Added command permission: bash:{command_pattern} -> {scope}", "green"))

        except Exception as e:
            print(colored(f"❌ Error adding command permission: {e}", "red"))
    def _list_permissions_for_management(self) -> str:
        """
        Generates a numbered list of permissions suitable for management (e.g., removal).
        This logic is ported from the old `tools/permissions.py` tool.
        """
        from core.permissions.permissions_manager import get_permissions_manager

        try:
            permissions_manager = get_permissions_manager()
            perms = permissions_manager.permissions

            if not any(perms.values()):
                # Use the user-friendly message from viewpermissions
                from tools.viewpermissions import viewpermissions
                return viewpermissions.run("")

            output_lines = ["# 🔓 SAVED PERMISSIONS - (Use IDs with `remove`)"]

            permission_count = 0

            # Development Tools
            dev_tools = perms.get("always_allow_dev_tools", {})
            if dev_tools:
                output_lines.append("\n🛠️  DEVELOPMENT TOOLS:")
                for category, patterns in sorted(dev_tools.items()):
                    if patterns:
                        category_display = f'{category.title()} Tools'
                        for pattern in patterns:
                            permission_count += 1
                            output_lines.append(f"   [{permission_count:2d}] {category_display}: {pattern}")

            # Tool Permissions
            tool_perms = perms.get("always_allow_tools", {})
            if tool_perms:
                output_lines.append("\n🔧 TOOL PERMISSIONS:")
                for tool, paths in sorted(tool_perms.items()):
                    for path in paths:
                        permission_count += 1
                        output_lines.append(f"   [{permission_count:2d}] {tool.title()}: {path}")

            # Group Permissions
            group_perms = perms.get("always_allow_groups", {})
            if group_perms:
                output_lines.append("\n📁 GROUP PERMISSIONS:")
                for group, paths in sorted(group_perms.items()):
                    for path in paths:
                        permission_count += 1
                        output_lines.append(f"   [{permission_count:2d}] {group.replace('_', ' ').title()}: {path}")

            # Command Permissions
            cmd_perms = perms.get("always_allow_commands", {})
            if cmd_perms:
                total_commands = sum(len(hashes) for hashes in cmd_perms.values())
                output_lines.append(f"\n⚡ COMMAND PERMISSIONS: {total_commands} saved commands")
                for tool, hashes in sorted(cmd_perms.items()):
                    if hashes:
                        # Don't list individual command hashes for brevity, just show counts
                        permission_count += len(hashes)
                        output_lines.append(f"   [{permission_count-len(hashes)+1}-{permission_count}] {tool}: {len(hashes)} command(s)")

            if permission_count > 0:
                output_lines.append(f"\n📊 Total: {permission_count} active permissions")

                # Add usage examples
                output_lines.append("\n💡 USAGE EXAMPLES:")
                output_lines.append("   /perms add dev grepsearch        # Add dev tool permission")
                output_lines.append("   /perms add tool bash /home/user  # Add tool permission")
                output_lines.append("   /perms remove 1                  # Remove permission #1")
                output_lines.append("   /perms view                      # Pretty view")
                output_lines.append("   /perms clear                     # Clear all")

            return "\n".join(output_lines)

        except Exception as e:
            return f"Error listing permissions for management: {str(e)}"

    def _remove_permission(self, perm_id: str) -> str:
        """Remove a specific permission by ID."""
        try:
            from core.permissions.permissions_manager import get_permissions_manager

            try:
                perm_num = int(perm_id)
            except ValueError:
                return f"❌ Invalid permission ID: {perm_id}. Please use a number from the `/perms list`."

            permissions_manager = get_permissions_manager()
            perms = permissions_manager.permissions

            # Build a list of all permissions with their IDs (same logic as original)
            all_permissions = []

            # Development Tools
            dev_tools = perms.get("always_allow_dev_tools", {})
            for category, patterns in sorted(dev_tools.items()):
                for pattern in patterns:
                    all_permissions.append(('dev_tools', category, pattern))

            # Tool Permissions
            tool_perms = perms.get("always_allow_tools", {})
            for tool, paths in sorted(tool_perms.items()):
                for path in paths:
                    all_permissions.append(('tools', tool, path))

            # Group Permissions
            group_perms = perms.get("always_allow_groups", {})
            for group, paths in sorted(group_perms.items()):
                for path in paths:
                    all_permissions.append(('groups', group, path))

            # Command Permissions
            cmd_perms = perms.get("always_allow_commands", {})
            for tool, hashes in sorted(cmd_perms.items()):
                for hash_val in hashes:
                    all_permissions.append(('commands', tool, hash_val))

            if perm_num < 1 or perm_num > len(all_permissions):
                return f"❌ Permission ID {perm_num} not found. Valid range: 1-{len(all_permissions)}"

            # Remove the permission
            perm_type, key, value = all_permissions[perm_num - 1]

            if perm_type == 'dev_tools':
                perms["always_allow_dev_tools"][key].remove(value)
                if not perms["always_allow_dev_tools"][key]:
                    del perms["always_allow_dev_tools"][key]
            elif perm_type == 'tools':
                perms["always_allow_tools"][key].remove(value)
                if not perms["always_allow_tools"][key]:
                    del perms["always_allow_tools"][key]
            elif perm_type == 'groups':
                perms["always_allow_groups"][key].remove(value)
                if not perms["always_allow_groups"][key]:
                    del perms["always_allow_groups"][key]
            elif perm_type == 'commands':
                perms["always_allow_commands"][key].remove(value)
                if not perms["always_allow_commands"][key]:
                    del perms["always_allow_commands"][key]

            permissions_manager.save_permissions()
            return f"✅ Removed permission #{perm_num}: {key} - {value}"

        except Exception as e:
            return f"❌ Error removing permission: {str(e)}"

    def _clear_all_permissions(self) -> str:
        """Clear all saved permissions."""
        try:
            from core.permissions.permissions_manager import get_permissions_manager

            permissions_manager = get_permissions_manager()

            # Count current permissions
            perms = permissions_manager.permissions
            total_count = 0
            total_count += sum(len(patterns) for patterns in perms.get("always_allow_dev_tools", {}).values())
            total_count += sum(len(paths) for paths in perms.get("always_allow_tools", {}).values())
            total_count += sum(len(paths) for paths in perms.get("always_allow_groups", {}).values())
            total_count += sum(len(hashes) for hashes in perms.get("always_allow_commands", {}).values())

            if total_count == 0:
                return "ℹ️  No permissions to clear - already empty."

            # Clear all permissions
            permissions_manager.permissions = {
                "always_allow_dev_tools": {},
                "always_allow_tools": {},
                "always_allow_groups": {},
                "always_allow_commands": {}
            }
            permissions_manager.save_permissions()

            return f"✅ Cleared all {total_count} saved permissions. I will now ask for confirmation before every action."

        except Exception as e:
            return f"❌ Error clearing permissions: {str(e)}"

class SaveCommand(BaseCommand):
    """Save current LLM config command implementation."""

    @property
    def aliases(self) -> List[str]:
        return ['/s', '/save']

    @property
    def description(self) -> str:
        return 'Save current LLM selection as default'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        """Save the current LLM selection as the new default."""
        try:
            from core.globals import g

            self.logger.info(colored("💾 Saving current LLM selection as the new default...", "magenta"))

            # Get the complete, most recent configuration data for ALL models
            # force_reread ensures we have the latest base config before modifying
            full_config = g.get_llm_config(force_reread=True)

            # Get the set of models active in THIS session
            currently_selected_models = set(g.SELECTED_LLMS)

            if not currently_selected_models:
                message = "No LLM models are currently selected to save."
                self.logger.warning(colored(f"⚠️ {message}", "yellow"))
                return CommandResponse(result=CommandResult.CONTINUE, message=message)

            # Update the 'selected' status in the full config based on the current session
            for model_key, data in full_config.items():
                if isinstance(data, dict):
                    data['selected'] = model_key in currently_selected_models

            # Save the modified configuration back to the persistent file
            g.save_llm_config(full_config)

            # Update the context as well
            if currently_selected_models:
                self.context.llm_config.model_key = list(currently_selected_models)[0]

            success_message = f"✅ Default configuration updated with {len(currently_selected_models)} selected models."
            self.logger.info(colored(success_message, "green"))

            # Show which models were saved
            saved_models = list(currently_selected_models)
            if len(saved_models) <= 3:
                models_display = ", ".join(saved_models)
            else:
                models_display = f"{', '.join(saved_models[:3])}, and {len(saved_models) - 3} more"

            print(colored(f"💾 Saved as default: {models_display}", "green"))

            return CommandResponse(result=CommandResult.CONTINUE, message="LLM configuration saved as default")

        except Exception as e:
            error_msg = f"Failed to save LLM configuration: {str(e)}"
            self.logger.error(error_msg)
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)

class CompactCommand(BaseCommand):
    """Compact conversation command implementation."""

    @property
    def aliases(self) -> List[str]:
        return ['/compact']

    @property
    def description(self) -> str:
        return 'Compact conversation using LLM compression'

    @property
    def usage(self) -> str:
        return '<prompt> [level] [model]'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        """Compact/summarize the current conversation using Chat.compress()."""
        try:
            context_chat = kwargs.get('context_chat')

            if not context_chat or not context_chat.messages:
                message = "No conversation to compact."
                self.logger.warning(colored(f"# cli-agent: {message}", "yellow"))
                return CommandResponse(result=CommandResult.CONTINUE, message=message)

            # Parse command arguments
            if len(parts) < 2:
                message = "Usage: /compact <prompt> [level] [model]\nExample: /compact 'Summarize our Python discussion' detailed"
                print(colored(f"# cli-agent: {message}", "yellow"))
                return CommandResponse(result=CommandResult.CONTINUE, message=message)

            prompt = parts[1]
            level = parts[2] if len(parts) > 2 else 'detailed'
            model = parts[3] if len(parts) > 3 else ''

            # Validate level
            if level not in ['dense', 'detailed']:
                message = f"Invalid level '{level}'. Use 'dense' or 'detailed'."
                print(colored(f"# cli-agent: {message}", "yellow"))
                return CommandResponse(result=CommandResult.CONTINUE, message=message)

            self.logger.info(colored(f"# cli-agent: Compacting conversation with {level} level...", "green"))

            # Use the Chat.compact() method
            compressed_chat = context_chat.compact(prompt=prompt, level=level, model=model)

            # Replace the current context chat with the compressed version
            context_chat.messages = compressed_chat.messages
            context_chat.metadata = compressed_chat.metadata

            # Log compression results
            if 'compression' in compressed_chat.metadata:
                compression_info = compressed_chat.metadata['compression']
                original_length = compression_info.get('original_length', 0)
                compressed_length = compression_info.get('compressed_length', 0)
                reduction = ((original_length - compressed_length) / original_length * 100) if original_length > 0 else 0

                print(colored(f"✅ Conversation compressed: {original_length} → {compressed_length} chars ({reduction:.1f}% reduction)", "green"))
            else:
                print(colored("✅ Conversation compacted successfully.", "green"))

            return CommandResponse(
                result=CommandResult.SUCCESS,
                message="Conversation compacted using LLM compression"
            )

        except Exception as e:
            error_msg = f"Conversation compacting failed: {str(e)}"
            self.logger.error(error_msg)
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


class ResetCommand(BaseCommand):
    """Start a new chat (archives current conversation and resets context)."""

    @property
    def aliases(self) -> List[str]:
        return ['/new_chat', '/new', '/clear', '/reset']

    @property
    def description(self) -> str:
        return 'Start a new chat (archives current conversation and resets context)'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        context_chat = kwargs.get('context_chat')

        # Clear chat history
        if context_chat:
            print(f"🔧 Debug: Chat before clear has {len(context_chat.messages)} messages")
            print(f"🔧 Debug: Chat object ID: {id(context_chat)}")
            context_chat.messages.clear()
            print(f"🔧 Debug: Chat after clear has {len(context_chat.messages)} messages")
            print(f"🔧 Debug: Chat object after clear - type: {type(context_chat)}, bool: {bool(context_chat)}")

        # Clear any cached images or context
        from core.globals import g
        if hasattr(g, 'current_images'):
            g.current_images = []

        # Reset to default state
        try:
            print("🔧 Debug: Starting engineered prompt reconstruction")
            # Re-initialize chat with engineered prompt
            from main import construct_engineered_prompt
            print("🔧 Debug: Successfully imported construct_engineered_prompt")
            print(f"🔧 Debug: context_chat type: {type(context_chat)}, bool: {bool(context_chat)}")
            if context_chat is not None:
                print("🔧 Debug: Attempting to construct engineered prompt")
                # Create new chat with engineered prompt
                gui_mode = hasattr(g, 'GUI_MODE') and g.GUI_MODE
                print(f"🔧 Debug: GUI mode is {gui_mode}")
                new_chat = await construct_engineered_prompt("", gui_mode)
                print(f"🔧 Debug: New chat has {len(new_chat.messages)} messages")
                # Copy messages from new chat to existing chat object
                context_chat.messages.extend(new_chat.messages)
                print(f"🔧 Debug: Context chat now has {len(context_chat.messages)} messages")
            else:
                print("🔧 Debug: context_chat is None for restoration")
        except Exception as e:
            # Fallback if not available - add a basic instruction message
            print(f"⚠️  Warning: Could not construct engineered prompt: {e}")
            import traceback
            traceback.print_exc()
            if context_chat:
                from core.chat import Role
                context_chat.add_message(Role.SYSTEM, "You are a helpful AI assistant.")
                print("🔧 Debug: Added fallback system message")

        print(colored("# cli-agent: New chat started. Previous conversation archived.", "green"))
        return CommandResponse(result=CommandResult.CONTINUE)


class PlannerLlmCommand(BaseCommand):
    """Command to set the planner LLM with optional fallback chain."""

    @property
    def aliases(self) -> List[str]:
        return ['/planner_llm']

    @property
    def description(self) -> str:
        return 'Set the planner LLM with optional fallback chain'

    @property
    def usage(self) -> str:
        return '<model1> [model2] [model3] ... (use "None" to disable)'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        if len(parts) < 2:
            return CommandResponse(
                result=CommandResult.ERROR,
                message="Usage: /planner_llm <model1> [model2] [model3] ... (use 'None' to disable)"
            )

        # Support multi-model fallback chain
        model_keys = [m for m in parts[1:] if m.lower() != "none"]

        try:
            from core.globals import g

            if not model_keys:
                # Disable planner
                g.update_llm_role('planner', model_keys=None)
                message = "Planner LLM disabled"
            else:
                # Set planner with fallback chain
                g.update_llm_role('planner', model_keys=model_keys)
                if len(model_keys) == 1:
                    message = f"Planner LLM set to: {model_keys[0]}"
                else:
                    message = f"Planner LLM set to: {model_keys[0]} (with {len(model_keys)-1} fallback(s))"

            print(colored(f"# cli-agent: {message}", "green"))
            return CommandResponse(result=CommandResult.CONTINUE, message=message)

        except Exception as e:
            error_msg = f"Failed to set planner LLM: {str(e)}"
            print(colored(f"# cli-agent: {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


class RouterLlmCommand(BaseCommand):
    """Command to set the router LLM with optional fallback chain."""

    @property
    def aliases(self) -> List[str]:
        return ['/router_llm']

    @property
    def description(self) -> str:
        return 'Set the router LLM with optional fallback chain'

    @property
    def usage(self) -> str:
        return '<model1> [model2] [model3] ... (use "None" to disable)'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        if len(parts) < 2:
            return CommandResponse(
                result=CommandResult.ERROR,
                message="Usage: /router_llm <model1> [model2] [model3] ... (use 'None' to disable)"
            )

        # Support multi-model fallback chain
        model_keys = [m for m in parts[1:] if m.lower() != "none"]

        try:
            from core.globals import g

            if not model_keys:
                # Disable router
                g.update_llm_role('router', model_keys=None)
                message = "Router LLM disabled"
            else:
                # Set router with fallback chain
                g.update_llm_role('router', model_keys=model_keys)
                if len(model_keys) == 1:
                    message = f"Router LLM set to: {model_keys[0]}"
                else:
                    message = f"Router LLM set to: {model_keys[0]} (with {len(model_keys)-1} fallback(s))"

            print(colored(f"# cli-agent: {message}", "green"))
            return CommandResponse(result=CommandResult.CONTINUE, message=message)

        except Exception as e:
            error_msg = f"Failed to set router LLM: {str(e)}"
            print(colored(f"# cli-agent: {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


class OverrideLlmCommand(BaseCommand):
    """Command to set the override LLM with optional fallback chain."""

    @property
    def aliases(self) -> List[str]:
        return ['/override_llm']

    @property
    def description(self) -> str:
        return 'Set the override LLM with optional fallback chain (bypasses router/planner selection)'

    @property
    def usage(self) -> str:
        return '<model1> [model2] [model3] ... (use "None" to disable)'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        if len(parts) < 2:
            return CommandResponse(
                result=CommandResult.ERROR,
                message="Usage: /override_llm <model1> [model2] [model3] ... (use 'None' to disable)"
            )

        # Support multi-model fallback chain
        model_keys = [m for m in parts[1:] if m.lower() != "none"]

        try:
            from core.globals import g

            if not model_keys:
                # Disable override
                g.update_llm_role('override', model_keys=None)
                message = "Override LLM disabled"
            else:
                # Set override with fallback chain
                g.update_llm_role('override', model_keys=model_keys)
                if len(model_keys) == 1:
                    message = f"Override LLM set to: {model_keys[0]}"
                else:
                    message = f"Override LLM set to: {model_keys[0]} (with {len(model_keys)-1} fallback(s))"

            print(colored(f"# cli-agent: {message}", "green"))
            return CommandResponse(result=CommandResult.CONTINUE, message=message)

        except Exception as e:
            error_msg = f"Failed to set override LLM: {str(e)}"
            print(colored(f"# cli-agent: {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


class FallbacksCommand(BaseCommand):
    """Command to set the fallback LLMs."""

    @property
    def aliases(self) -> List[str]:
        return ['/fallbacks']

    @property
    def description(self) -> str:
        return 'Set fallback LLMs'

    @property
    def usage(self) -> str:
        return '<model1> [model2] [model3]'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        if len(parts) < 2:
            return CommandResponse(
                result=CommandResult.ERROR,
                message="Usage: /fallbacks <model1> [model2] [model3] (use 'None' to clear a slot)"
            )

        # Get up to 3 fallback models, filter out "None" entries
        model_keys = []
        for i in range(1, min(len(parts), 4)):  # parts[1:4]
            model = parts[i]
            if model.lower() != "none":
                model_keys.append(model)

        try:
            from core.globals import g
            g.FALLBACK_MODELS = model_keys

            if model_keys:
                message = f"Fallback LLMs set to: {', '.join(model_keys)}"
            else:
                message = "All fallback LLMs cleared"

            print(colored(f"# cli-agent: {message}", "green"))
            return CommandResponse(result=CommandResult.CONTINUE, message=message)

        except Exception as e:
            error_msg = f"Failed to set fallback LLMs: {str(e)}"
            print(colored(f"# cli-agent: {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


class ConfigCommand(BaseCommand):
    """Command to display current agent and model configuration."""

    @property
    def aliases(self) -> List[str]:
        return ['/config', '/c']

    @property
    def description(self) -> str:
        return 'Display current agent and model configuration'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        try:
            from shared.utils.display_utils import colorize_model_list
            from core.globals import g

            # Display configuration without triggering potential side effects
            print(colored("--- Agent Configuration ---", "yellow"))

            # Auto-execution status
            auto_status = 'ON (manual confirmation)' if g.auto_execute else 'OFF'
            auto_color = 'green' if g.auto_execute else 'red'
            print("Auto-execution:".ljust(24) + " " + colored(auto_status, auto_color))

            # Force Local mode status
            force_local = getattr(g, 'FORCE_LOCAL', False) or self.context.settings.local_mode
            local_status = colored('ON', 'green', attrs=['bold']) if force_local else colored('OFF', 'red')
            print("Force Local:".ljust(24) + f" {local_status}")

            # Voice mode status
            voice_status = colored('ON', 'green', attrs=['bold']) if g.voice_mode else colored('OFF', 'red')
            print("Voice Mode:".ljust(24) + f" {voice_status}")

            # TTS mode status
            tts_status = colored('Cloud TTS', 'green', attrs=['bold']) if g.online_mode else colored('Local TTS', 'green', attrs=['bold'])
            print("TTS Mode:".ljust(24) + f" {tts_status}")

            # Display model configuration
            selected_models = g.SELECTED_LLMS if g.SELECTED_LLMS else []
            if selected_models:
                print(colored("--- Model Configuration ---", "yellow"))
                count = len(selected_models)
                label = f"  🤖 {count} Selected Models:".ljust(28)
                models_str = colorize_model_list(selected_models, tui_mode=False)
                print(colored(label, "light_green") + " " + models_str)

            print(colored("--- Ready for User Input ---", "yellow"))

            return CommandResponse(result=CommandResult.CONTINUE, message="Configuration displayed")

        except Exception as e:
            error_msg = f"Failed to display configuration: {str(e)}"
            print(colored(f"# cli-agent: {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


class ToolsCommand(BaseCommand):
    """Command to display detailed tool health status."""

    @property
    def aliases(self) -> List[str]:
        return ['/tools', '/tool', '/t', '/health']

    @property
    def description(self) -> str:
        return 'Display detailed tool health status'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        try:
            from core.globals import g

            print(colored("--- Tool Health Status ---", "yellow"))

            # Check if we have tool info
            if not hasattr(g, 'loaded_tool_info') or not g.loaded_tool_info:
                print(colored("Tool information not available. Tools may not have been initialized yet.", "red"))
                return CommandResponse(result=CommandResult.CONTINUE, message="Tool info not available")

            tool_info = g.loaded_tool_info

            # Display enabled tools
            if tool_info.get('loaded_tools'):
                enabled_count = len(tool_info['loaded_tools'])
                print(colored(f"Enabled Tools ({enabled_count}):", "green"))
                for i, tool in enumerate(tool_info['loaded_tools'], 1):
                    status = "OK"
                    # Check for specific tool status (like Home Assistant)
                    if tool == "homeassistant":
                        try:
                            from tools.optional.homeassistant import homeassistant
                            if hasattr(homeassistant, '_status_message'):
                                status = homeassistant._status_message
                        except ImportError:
                            status = "Import error"
                    print(f"  {i:2d}. {tool.ljust(20)} - {colored(status, 'green')}")

            # Display failed tools
            if tool_info.get('failed_tools'):
                failed_count = len(tool_info['failed_tools'])
                print(f"\n{colored(f'Failed Tools ({failed_count}):', 'red')}")
                for i, tool in enumerate(tool_info['failed_tools'], 1):
                    reason = "Unknown error"
                    # Try to get more specific error info if available
                    if isinstance(tool, dict):
                        tool_name = tool.get('name', 'Unknown')
                        reason = tool.get('error', reason)
                    elif isinstance(tool, str):
                        tool_name = tool
                        # Common failure reasons based on tool name
                        if 'class not found' in tool:
                            tool_name = tool.split('(')[0].strip()
                            reason = "Class not found"
                        elif 'missing required methods' in tool:
                            tool_name = tool.split('(')[0].strip()
                            reason = "Missing required methods"
                    else:
                        tool_name = str(tool)

                    print(f"  {i:2d}. {tool_name.ljust(20)} - {colored(reason, 'red')}")

            # Display disabled/skipped tools
            if tool_info.get('skipped_tools'):
                skipped_count = len(tool_info['skipped_tools'])
                print(f"\n{colored(f'Disabled Tools ({skipped_count}):', 'yellow')}")
                for i, tool in enumerate(tool_info['skipped_tools'], 1):
                    reason = "Disabled"
                    if isinstance(tool, str) and '(' in tool:
                        tool_name = tool.split('(')[0].strip()
                        reason = tool.split('(')[1].rstrip(')')
                    else:
                        tool_name = str(tool)
                    print(f"  {i:2d}. {tool_name.ljust(20)} - {colored(reason, 'yellow')}")

            # Summary
            total_tools = len(tool_info.get('loaded_tools', []))
            failed_tools = len(tool_info.get('failed_tools', []))
            skipped_tools = len(tool_info.get('skipped_tools', []))

            print(f"\n{colored('Summary:', 'yellow')}")
            print(f"  Total: {total_tools + failed_tools + skipped_tools} tools")
            print(f"  {colored('Enabled:', 'green')} {total_tools}")
            if failed_tools > 0:
                print(f"  {colored('Failed:', 'red')} {failed_tools}")
            if skipped_tools > 0:
                print(f"  {colored('Disabled:', 'yellow')} {skipped_tools}")

            return CommandResponse(result=CommandResult.CONTINUE, message="Tool status displayed")

        except Exception as e:
            error_msg = f"Failed to display tool status: {str(e)}"
            print(colored(f"# cli-agent: {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)