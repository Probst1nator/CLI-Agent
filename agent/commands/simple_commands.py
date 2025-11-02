"""Simple command implementations that don't require complex logic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from termcolor import colored

from agent.commands.base_command import BaseCommand, CommandResponse, CommandResult

if TYPE_CHECKING:
    pass

class MultilineCommand(BaseCommand):
    """Command for multiline input mode."""

    @property
    def aliases(self) -> list[str]:
        return ['/m', '/message']

    @property
    def description(self) -> str:
        return 'Enter multiline input mode'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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
    def aliases(self) -> list[str]:
        return ['/e', '/exit']

    @property
    def description(self) -> str:
        return 'Exit CLI-Agent'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
        print(colored("# cli-agent: Goodbye!", "cyan"))
        return CommandResponse(result=CommandResult.EXIT)


class UnsayCommand(BaseCommand):
    """Command to remove last user message and AI response."""

    @property
    def aliases(self) -> list[str]:
        return ['/unsay']

    @property
    def description(self) -> str:
        return 'Remove last user message and AI response'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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
    def aliases(self) -> list[str]:
        return ['/p', '/print']

    @property
    def description(self) -> str:
        return 'Print complete chat history'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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
    def aliases(self) -> list[str]:
        return ['/yolo', '/y']

    @property
    def description(self) -> str:
        return 'Toggle YOLO mode (automatic code execution)'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
        from core.globals import g

        self.context.settings.auto_execute = not self.context.settings.auto_execute
        # Sync with global state for compatibility
        g.auto_execute = self.context.settings.auto_execute

        status = "enabled" if self.context.settings.auto_execute else "disabled"
        print(colored(f"# cli-agent: YOLO mode {status}.", "green"))

        # Broadcast state change to frontend
        try:
            import asyncio

            from api.server import get_server
            server = get_server()
            if server:
                asyncio.create_task(server.broadcast_state_change())
        except Exception:
            pass  # Ignore if server not available

        return CommandResponse(result=CommandResult.CONTINUE)

class VoiceCommand(BaseCommand):
    """Toggle voice input/output mode."""

    @property
    def aliases(self) -> list[str]:
        return ['/v', '/voice']

    @property
    def description(self) -> str:
        return 'Toggle voice input/output mode'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
        import logging

        from core.globals import g

        self.context.settings.voice_mode = not self.context.settings.voice_mode
        # Sync with global state for compatibility
        g.voice_mode = self.context.settings.voice_mode

        # Update system prompt with new voice_mode setting
        try:
            from main import update_system_prompt
            update_system_prompt()
        except Exception as e:
            logging.warning(f"Failed to update system prompt: {e}")

        status = "enabled" if self.context.settings.voice_mode else "disabled"
        print(colored(f"# cli-agent: Voice mode {status}.", "green"))

        # Broadcast state change to frontend
        try:
            import asyncio

            from api.server import get_server
            server = get_server()
            if server:
                asyncio.create_task(server.broadcast_state_change())
        except Exception:
            pass  # Ignore if server not available

        return CommandResponse(result=CommandResult.CONTINUE)



class LocalCommand(BaseCommand):
    """Toggle local-only LLM mode."""

    @property
    def aliases(self) -> list[str]:
        return ['/l', '/local']

    @property
    def description(self) -> str:
        return 'Toggle local-only LLM mode'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
        from core.globals import g

        self.context.settings.local_mode = not self.context.settings.local_mode
        # Sync with global state for compatibility
        g.local_mode = self.context.settings.local_mode

        status = "enabled" if self.context.settings.local_mode else "disabled"
        print(colored(f"# cli-agent: Local mode {status}.", "green"))

        # Broadcast state change to frontend
        try:
            import asyncio

            from api.server import get_server
            server = get_server()
            if server:
                asyncio.create_task(server.broadcast_state_change())
        except Exception:
            pass  # Ignore if server not available

        return CommandResponse(result=CommandResult.CONTINUE)

class OnlineCommand(BaseCommand):
    """Toggle online mode for web-enabled features."""

    @property
    def aliases(self) -> list[str]:
        return ['/o', '/online']

    @property
    def description(self) -> str:
        return 'Toggle online mode for web-enabled features'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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
    def aliases(self) -> list[str]:
        return ['/fo', '/full_output']

    @property
    def description(self) -> str:
        return 'Show full model output instead of summary'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
        self.context.settings.full_output_mode = not self.context.settings.full_output_mode
        status = "enabled" if self.context.settings.full_output_mode else "disabled"
        print(colored(f"# cli-agent: Full output mode {status}.", "green"))
        return CommandResponse(result=CommandResult.CONTINUE)

# Stub classes for complex commands that need full implementation
class LlmCommand(BaseCommand):
    """LLM selection command implementation."""

    @property
    def aliases(self) -> list[str]:
        return ['/llm']

    @property
    def description(self) -> str:
        return 'Open LLM selection interface'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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
    """Simplified permission management command."""

    @property
    def aliases(self) -> list[str]:
        return ['/perms', '/permissions', '/settings']

    @property
    def description(self) -> str:
        return 'Manage tool execution permissions (simplified system)'

    @property
    def sub_commands(self) -> dict[str, str] | None:
        return {
            'view': 'View current permission settings (default)',
            'add allow <rule>': 'Add permission to allow list',
            'add deny <rule>': 'Add permission to deny list',
            'add ask <rule>': 'Add permission to ask list',
            'remove <rule>': 'Remove a permission rule',
            'reset': 'Reset to empty settings',
            'template': 'Copy default template to ~/.cli-agent/settings.json',
        }

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
        """Execute simplified permission management command."""
        from core.permissions.settings_manager import SettingsManager

        try:
            mgr = SettingsManager()
            action = parts[1].lower() if len(parts) > 1 else "view"

            if action == "view":
                self._view_settings(mgr)
                return CommandResponse(result=CommandResult.CONTINUE, message="Settings viewed")

            elif action == "add" and len(parts) >= 4:
                list_name = parts[2].lower()  # allow, deny, ask
                rule = " ".join(parts[3:])  # Join remaining parts as the rule
                return self._add_rule(mgr, list_name, rule)

            elif action == "remove" and len(parts) >= 3:
                rule = " ".join(parts[2:])
                return self._remove_rule(mgr, rule)

            elif action == "reset":
                return self._reset_settings(mgr)

            elif action == "template":
                return self._copy_template(mgr)

            else:
                error_msg = "Usage: /perms [view|add <allow|deny|ask> <rule>|remove <rule>|reset|template]"
                print(colored(error_msg, "yellow"))
                return CommandResponse(result=CommandResult.ERROR, message=error_msg)

        except Exception as e:
            error_msg = f"Error handling permission command: {str(e)}"
            print(colored(f"❌ {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)

    def _view_settings(self, mgr) -> None:
        """Display current permission settings."""
        perms = mgr.get_all_permissions()

        print("\n" + "=" * 60)
        print(colored("📋 Permission Settings", "cyan", attrs=['bold']))
        print("=" * 60)

        print(colored("\n✅ ALLOW (execute without asking):", "green", attrs=['bold']))
        if perms.get("allow"):
            for rule in perms["allow"]:
                print(f"  • {rule}")
        else:
            print(colored("  (none)", "yellow"))

        print(colored("\n❓ ASK (require confirmation):", "yellow", attrs=['bold']))
        if perms.get("ask"):
            for rule in perms["ask"]:
                print(f"  • {rule}")
        else:
            print(colored("  (none)", "yellow"))

        print(colored("\n❌ DENY (blocked):", "red", attrs=['bold']))
        if perms.get("deny"):
            for rule in perms["deny"]:
                print(f"  • {rule}")
        else:
            print(colored("  (none)", "yellow"))

        print(f"\n📄 Config file: {mgr.config_path}")
        print("=" * 60 + "\n")

    def _add_rule(self, mgr, list_name: str, rule: str) -> CommandResponse:
        """Add a new permission rule."""
        if list_name not in ["allow", "deny", "ask"]:
            error_msg = f"Invalid list '{list_name}'. Use: allow, deny, or ask"
            print(colored(f"❌ {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)

        if mgr.add_permission(list_name, rule):
            mgr.save_settings()
            print(colored(f"✅ Added '{rule}' to {list_name} list", "green"))
            return CommandResponse(result=CommandResult.CONTINUE, message=f"Added to {list_name}")
        else:
            error_msg = "Failed to add rule (may already exist or invalid format)"
            print(colored(f"❌ {error_msg}", "yellow"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)

    def _remove_rule(self, mgr, rule: str) -> CommandResponse:
        """Remove a permission rule."""
        if mgr.remove_permission(rule):
            mgr.save_settings()
            print(colored(f"✅ Removed '{rule}' from all lists", "green"))
            return CommandResponse(result=CommandResult.CONTINUE, message="Rule removed")
        else:
            error_msg = f"Rule '{rule}' not found in any list"
            print(colored(f"❌ {error_msg}", "yellow"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)

    def _reset_settings(self, mgr) -> CommandResponse:
        """Reset permissions to empty state."""
        confirm = input(colored("Are you sure you want to clear all permissions? (yes/no): ", "yellow"))
        if confirm.lower() != 'yes':
            print(colored("Reset cancelled", "yellow"))
            return CommandResponse(result=CommandResult.CONTINUE, message="Reset cancelled")

        mgr.clear_all_permissions()
        mgr.save_settings()
        print(colored("✅ All permissions cleared", "green"))
        return CommandResponse(result=CommandResult.CONTINUE, message="Settings reset")

    def _copy_template(self, mgr) -> CommandResponse:
        """Copy default template to settings.json."""
        import os
        import shutil

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "templates/default_settings.json"
        )

        if not os.path.exists(template_path):
            error_msg = f"Template not found at {template_path}"
            print(colored(f"❌ {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)

        if os.path.exists(mgr.config_path):
            confirm = input(colored(f"Overwrite existing {mgr.config_path}? (yes/no): ", "yellow"))
            if confirm.lower() != 'yes':
                print(colored("Template copy cancelled", "yellow"))
                return CommandResponse(result=CommandResult.CONTINUE, message="Cancelled")

        try:
            shutil.copy(template_path, mgr.config_path)
            print(colored(f"✅ Copied template to {mgr.config_path}", "green"))
            print(colored("Edit this file to customize your permissions", "cyan"))
            return CommandResponse(result=CommandResult.CONTINUE, message="Template copied")
        except Exception as e:
            error_msg = f"Failed to copy template: {str(e)}"
            print(colored(f"❌ {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


# Old complex permission system methods removed (350+ lines deleted)
# Now using simplified settings.json approach (~100 lines above)

class SaveCommand(BaseCommand):
    """Save current LLM config command implementation."""

    @property
    def aliases(self) -> list[str]:
        return ['/s', '/save']

    @property
    def description(self) -> str:
        return 'Save current LLM selection as default'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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
    def aliases(self) -> list[str]:
        return ['/compact']

    @property
    def description(self) -> str:
        return 'Compact conversation using LLM compression'

    @property
    def usage(self) -> str:
        return '<prompt> [level] [model]'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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
    def aliases(self) -> list[str]:
        return ['/new_chat', '/new', '/clear', '/reset']

    @property
    def description(self) -> str:
        return 'Start a new chat (archives current conversation and resets context)'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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
    def aliases(self) -> list[str]:
        return ['/planner_llm']

    @property
    def description(self) -> str:
        return 'Set the planner LLM with optional fallback chain'

    @property
    def usage(self) -> str:
        return '<model1> [model2] [model3] ... (use "None" to disable)'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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
    def aliases(self) -> list[str]:
        return ['/router_llm']

    @property
    def description(self) -> str:
        return 'Set the router LLM with optional fallback chain'

    @property
    def usage(self) -> str:
        return '<model1> [model2] [model3] ... (use "None" to disable)'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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
    def aliases(self) -> list[str]:
        return ['/override_llm']

    @property
    def description(self) -> str:
        return 'Set the override LLM with optional fallback chain (bypasses router/planner selection)'

    @property
    def usage(self) -> str:
        return '<model1> [model2] [model3] ... (use "None" to disable)'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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
    def aliases(self) -> list[str]:
        return ['/fallbacks']

    @property
    def description(self) -> str:
        return 'Set fallback LLMs'

    @property
    def usage(self) -> str:
        return '<model1> [model2] [model3]'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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
    def aliases(self) -> list[str]:
        return ['/config', '/c']

    @property
    def description(self) -> str:
        return 'Display current agent and model configuration'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
        try:
            from core.globals import g
            from shared.utils.display_utils import colorize_model_list

            # Display configuration without triggering potential side effects
            print(colored("--- Agent Configuration ---", "yellow"))

            # Auto-execution status
            auto_status = 'ON' if g.auto_execute else 'OFF'
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
    def aliases(self) -> list[str]:
        return ['/tools', '/tool', '/t', '/health']

    @property
    def description(self) -> str:
        return 'Display detailed tool health status'

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
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