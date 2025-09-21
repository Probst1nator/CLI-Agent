"""Simple command implementations that don't require complex logic."""

from __future__ import annotations
from typing import List, Any, Dict, Optional, TYPE_CHECKING
from termcolor import colored

from agent.commands.base_command import BaseCommand, CommandResponse, CommandResult

if TYPE_CHECKING:
    import argparse

class MultilineCommand(BaseCommand):
    """Command for multiline input mode."""

    @property
    def aliases(self) -> List[str]:
        return ['/m', '/message']

    @property
    def description(self) -> str:
        return 'Enter multiline input mode'

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
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

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
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

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
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

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
        context_chat = kwargs.get('context_chat')
        if context_chat:
            print(colored("=== Chat History ===", "blue", attrs=["bold"]))
            context_chat.print_chat()
        else:
            print(colored("# cli-agent: No chat history available.", "yellow"))
        return CommandResponse(result=CommandResult.CONTINUE)

# Toggle commands for various modes
class AutoCommand(BaseCommand):
    """Toggle automatic code execution mode."""

    @property
    def aliases(self) -> List[str]:
        return ['/a', '/auto']

    @property
    def description(self) -> str:
        return 'Toggle automatic code execution mode'

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
        self.context.settings.auto_execute = not self.context.settings.auto_execute
        status = "enabled" if self.context.settings.auto_execute else "disabled"
        print(colored(f"# cli-agent: Auto-execute mode {status}.", "green"))
        return CommandResponse(result=CommandResult.CONTINUE)

class VoiceCommand(BaseCommand):
    """Toggle voice input/output mode."""

    @property
    def aliases(self) -> List[str]:
        return ['/v', '/voice']

    @property
    def description(self) -> str:
        return 'Toggle voice input/output mode'

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
        self.context.settings.voice_mode = not self.context.settings.voice_mode
        status = "enabled" if self.context.settings.voice_mode else "disabled"
        print(colored(f"# cli-agent: Voice mode {status}.", "green"))
        return CommandResponse(result=CommandResult.CONTINUE)


class FastModeCommand(BaseCommand):
    """Toggle fast mode (deprecated feature)."""

    @property
    def aliases(self) -> List[str]:
        return ['/f', '/fast']

    @property
    def description(self) -> str:
        return 'Toggle fast mode (deprecated)'

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
        args.fast = not getattr(args, 'fast', False)
        status = "enabled" if args.fast else "disabled"
        print(colored(f"# cli-agent: Fast mode {status} (deprecated).", "green"))
        return CommandResponse(result=CommandResult.CONTINUE)

class LocalCommand(BaseCommand):
    """Toggle local-only LLM mode."""

    @property
    def aliases(self) -> List[str]:
        return ['/l', '/local']

    @property
    def description(self) -> str:
        return 'Toggle local-only LLM mode'

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
        self.context.settings.local_mode = not self.context.settings.local_mode
        status = "enabled" if self.context.settings.local_mode else "disabled"
        print(colored(f"# cli-agent: Local mode {status}.", "green"))
        return CommandResponse(result=CommandResult.CONTINUE)

class OnlineCommand(BaseCommand):
    """Toggle online mode for web-enabled features."""

    @property
    def aliases(self) -> List[str]:
        return ['/o', '/online']

    @property
    def description(self) -> str:
        return 'Toggle online mode for web-enabled features'

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
        self.context.settings.online_mode = not self.context.settings.online_mode
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

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
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

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
        """Execute LLM selection using the enhanced LlmSelector UI."""
        try:
            from agent.llm_selection import LlmSelector
            from core.globals import g
            from main import get_provider_emoji_and_name

            self.logger.info(colored("# cli-agent: Opening LLM selection interface.", "green"))

            selector = LlmSelector()

            # Use the selector to get user selection
            result = await selector.get_selection(
                preselected_llms=getattr(args, 'llm', None),
                force_local=getattr(args, 'local', False),
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
                args.local = True
                args.llm = None
                self.context.settings.local_mode = True
                self.logger.info(colored("# cli-agent: 'Any but local' option selected. Local mode enabled.", "green"))
                return CommandResponse(result=CommandResult.CONTINUE, message="Local mode enabled")

            # Handle specific model selections
            selected_configs = result.get("selected_configs", [])

            # No evaluator models needed in simplified system

            selected_llms = [config['model_key'] for config in selected_configs]

            # Update context and globals with the latest selection
            args.llm = selected_llms
            g.SELECTED_LLMS = selected_llms
            self.context.llm_config.model_key = selected_llms[0] if selected_llms else ""

            # Reload the config in `g` so the main loop gets the fresh settings
            g.load_llm_config()

            if not args.llm:
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
            'list': 'List all permissions with IDs for removal',
            'view': 'View permissions in a user-friendly format',
            'remove': 'Remove specific permission by ID',
            'clear': 'Clear all saved permissions',
            'interactive': 'Open interactive permission manager',
            'manage': 'Open interactive permission manager',
        }

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
        """Execute permission management command."""
        try:
            # Import the new tool for the 'view' command
            from tools.viewpermissions import viewpermissions

            action = parts[1].lower() if len(parts) > 1 else "list"

            if action == "list" or action == "":
                # Default action is to list permissions with numbers for management
                result = self._list_permissions_for_management()
                print(result)
                return CommandResponse(result=CommandResult.CONTINUE, message="Permissions listed")

            elif action == "view":
                # Use the new tool for a pretty, readable view
                result = viewpermissions.run("")
                print(result)
                return CommandResponse(result=CommandResult.CONTINUE, message="Permissions viewed")

            elif action in ["interactive", "manage"]:
                # Open interactive manager
                await self._interactive_permission_manager(args, kwargs)
                return CommandResponse(result=CommandResult.CONTINUE, message="Interactive session completed")

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

    async def _interactive_permission_manager(self, args: argparse.Namespace, kwargs: Dict[str, Any]) -> None:
        """Interactive permission management interface."""
        print()
        print(colored("🔧 INTERACTIVE PERMISSION MANAGER", "yellow", attrs=['bold']))
        print(colored("═" * 50, "yellow"))

        while True:
            # Show current permissions with numbers for management
            result = self._list_permissions_for_management()
            print(result)

            print()
            print(colored("Commands:", "cyan"))
            print(colored("  list                    # Refresh permission list", "white"))
            print(colored("  remove <id>             # Remove permission by ID", "white"))
            print(colored("  clear                   # Clear all permissions", "white"))
            print(colored("  exit, quit, back        # Return to main CLI", "white"))
            print()

            try:
                # Get user input (simplified for refactored version)
                user_cmd = input(colored("Permissions> ", "cyan"))
                user_cmd = user_cmd.strip().lower()
            except KeyboardInterrupt:
                print(colored("\n↩️  Returning to main CLI (Ctrl+C)...", "yellow"))
                break

            if user_cmd in ['exit', 'quit', 'back', '']:
                print(colored("↩️  Returning to main CLI...", "green"))
                break

            elif user_cmd == 'list':
                continue

            elif user_cmd.startswith('remove ') and len(user_cmd.split()) >= 2:
                perm_id = user_cmd.split()[1]
                result = self._remove_permission(perm_id)
                print(result)

            elif user_cmd == 'clear':
                print(colored("\n⚠️  WARNING: This will clear ALL saved permissions!", "red", attrs=['bold']))
                print(colored("Type 'yes' to confirm or anything else to cancel:", "yellow"))

                try:
                    confirmation = input(colored("> ", "cyan")).strip().lower()
                    if confirmation == 'yes':
                        result = self._clear_all_permissions()
                        print(result)
                    else:
                        print(colored("❌ Clear operation cancelled.", "yellow"))
                except KeyboardInterrupt:
                    print(colored("\n❌ Clear operation cancelled (Ctrl+C).", "yellow"))

            else:
                print(colored(f"❓ Unknown command: {user_cmd}", "red"))
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

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
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
        return 'Compact conversation (create summary)'

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
        """Compact/summarize the current conversation."""
        try:
            import time
            from core.globals import g

            context_chat = kwargs.get('context_chat')

            if not context_chat or not context_chat.messages:
                message = "No conversation to compact."
                self.logger.warning(colored(f"# cli-agent: {message}", "yellow"))
                return CommandResponse(result=CommandResult.CONTINUE, message=message)

            self.logger.info(colored("# cli-agent: Compacting conversation...", "green"))

            # Set the compacting flag
            g.AGENT_IS_COMPACTING = True

            # Create the automated message for conversation summarization
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            storage_path = getattr(g, 'CLIAGENT_PERSISTENT_STORAGE_PATH', './storage')

            automated_message = f"Let's pause. Please summarize our conversation and progress into a markdown file at: {storage_path}/MEMORY/summary_{timestamp}.md"

            # Try to use the memory tool for compaction
            try:
                from tools.disabled.memory import memory
                await memory.compact_and_store(context_chat)
                self.logger.info(colored("✅ Conversation compacted and stored in memory.", "green"))
            except Exception as e:
                self.logger.warning(f"Memory storage failed, using fallback: {e}")

            return CommandResponse(
                result=CommandResult.SUCCESS,
                message="Conversation compacting initiated",
                user_input=automated_message
            )

        except Exception as e:
            error_msg = f"Conversation compacting failed: {str(e)}"
            self.logger.error(error_msg)
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


class ResetCommand(BaseCommand):
    """Complete agent reset (chat, tools, images, context)."""

    @property
    def aliases(self) -> List[str]:
        return ['/reset']

    @property
    def description(self) -> str:
        return 'Complete agent reset (chat, tools, images, context)'

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
        context_chat = kwargs.get('context_chat')

        # Clear chat history
        if context_chat:
            context_chat.messages.clear()

        # Clear any cached images or context
        from core.globals import g
        if hasattr(g, 'current_images'):
            g.current_images = []

        # Reset to default state
        try:
            # Re-initialize chat with engineered prompt
            from main import construct_engineered_prompt
            if context_chat:
                context_chat = await construct_engineered_prompt("voice" if getattr(args, 'voice', False) else "")
        except ImportError:
            # Fallback if not available
            pass

        print(colored("# cli-agent: Complete reset performed.", "green"))
        return CommandResponse(result=CommandResult.CONTINUE)


class PlannerLlmCommand(BaseCommand):
    """Command to set the planner LLM."""

    @property
    def aliases(self) -> List[str]:
        return ['/planner_llm']

    @property
    def description(self) -> str:
        return 'Set the planner LLM (use "None" to disable)'

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
        if len(parts) < 2:
            return CommandResponse(
                result=CommandResult.ERROR,
                message="Usage: /planner_llm <model_key> (use 'None' to disable)"
            )

        model_key = parts[1]
        if model_key.lower() == "none":
            model_key = None

        try:
            from core.globals import g
            g.update_planner_llm(model_key)

            if model_key:
                message = f"Planner LLM set to: {model_key}"
            else:
                message = "Planner LLM disabled"

            print(colored(f"# cli-agent: {message}", "green"))
            return CommandResponse(result=CommandResult.CONTINUE, message=message)

        except Exception as e:
            error_msg = f"Failed to set planner LLM: {str(e)}"
            print(colored(f"# cli-agent: {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


class RouterLlmCommand(BaseCommand):
    """Command to set the router LLM."""

    @property
    def aliases(self) -> List[str]:
        return ['/router_llm']

    @property
    def description(self) -> str:
        return 'Set the router LLM (use "None" to disable)'

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
        if len(parts) < 2:
            return CommandResponse(
                result=CommandResult.ERROR,
                message="Usage: /router_llm <model_key> (use 'None' to disable)"
            )

        model_key = parts[1]
        if model_key.lower() == "none":
            model_key = None

        try:
            from core.globals import g
            g.update_router_llm(model_key)

            if model_key:
                message = f"Router LLM set to: {model_key}"
            else:
                message = "Router LLM disabled"

            print(colored(f"# cli-agent: {message}", "green"))
            return CommandResponse(result=CommandResult.CONTINUE, message=message)

        except Exception as e:
            error_msg = f"Failed to set router LLM: {str(e)}"
            print(colored(f"# cli-agent: {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


class FallbacksCommand(BaseCommand):
    """Command to set the fallback LLMs."""

    @property
    def aliases(self) -> List[str]:
        return ['/fallbacks']

    @property
    def description(self) -> str:
        return 'Set fallback LLMs (up to 3 models)'

    async def execute(self, args: argparse.Namespace, parts: List[str], **kwargs: Any) -> CommandResponse:
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