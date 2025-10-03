"""
Commands for LLM profile management.
"""

from typing import List, Any
from termcolor import colored
from agent.commands.base_command import BaseCommand, CommandResult, CommandResponse


class ProfileSaveCommand(BaseCommand):
    """Save current LLM configuration as a profile."""

    @property
    def aliases(self) -> List[str]:
        return ['/profile_save', '/psave']

    @property
    def description(self) -> str:
        return 'Save current LLM configuration as a profile'

    @property
    def usage(self) -> str:
        return '<profile_id>'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        if len(parts) < 2:
            return CommandResponse(
                result=CommandResult.ERROR,
                message="Usage: /profile_save <profile_id>"
            )

        profile_id = parts[1]

        try:
            from core.llm_profiles import get_profile_manager

            manager = get_profile_manager()

            if manager.get_profile(profile_id):
                profile = manager.update_profile(profile_id, from_current=True)
                message = f"Updated profile '{profile_id}'"
            else:
                profile = manager.create_profile(profile_id, from_current=True)
                message = f"Created profile '{profile_id}'"

            details = []
            if profile.override:
                details.append(f"Override: {profile.override}")
            if profile.router:
                details.append(f"Router: {profile.router}")
            if profile.planner:
                details.append(f"Planner: {profile.planner}")
            if profile.fallbacks:
                details.append(f"Fallbacks: {', '.join(profile.fallbacks[:3])}{'...' if len(profile.fallbacks) > 3 else ''}")

            full_message = f"{message}\n" + "\n".join(f"  • {d}" for d in details)
            print(colored(f"# cli-agent: {full_message}", "green"))

            return CommandResponse(result=CommandResult.CONTINUE, message=message)

        except Exception as e:
            error_msg = f"Failed to save profile: {str(e)}"
            print(colored(f"# cli-agent: {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


class ProfileLoadCommand(BaseCommand):
    """Load a saved LLM configuration profile."""

    @property
    def aliases(self) -> List[str]:
        return ['/profile_load', '/pload']

    @property
    def description(self) -> str:
        return 'Load a saved LLM configuration profile'

    @property
    def usage(self) -> str:
        return '<profile_id>'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        if len(parts) < 2:
            return CommandResponse(
                result=CommandResult.ERROR,
                message="Usage: /profile_load <profile_id>"
            )

        profile_id = parts[1]

        try:
            from core.llm_profiles import get_profile_manager

            manager = get_profile_manager()
            manager.apply_profile(profile_id, set_active=True)

            profile = manager.get_profile(profile_id)
            if not profile:
                raise ValueError(f"Profile '{profile_id}' not found")

            details = []
            if profile.override:
                details.append(f"Override: {profile.override}")
            if profile.router:
                details.append(f"Router: {profile.router}")
            if profile.planner:
                details.append(f"Planner: {profile.planner}")
            if profile.fallbacks:
                details.append(f"Fallbacks: {', '.join(profile.fallbacks[:3])}{'...' if len(profile.fallbacks) > 3 else ''}")

            message = f"Loaded profile '{profile_id}'\n" + "\n".join(f"  • {d}" for d in details)
            print(colored(f"# cli-agent: {message}", "green"))

            return CommandResponse(result=CommandResult.CONTINUE, message=message)

        except Exception as e:
            error_msg = f"Failed to load profile: {str(e)}"
            print(colored(f"# cli-agent: {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


class ProfileListCommand(BaseCommand):
    """List all saved LLM configuration profiles."""

    @property
    def aliases(self) -> List[str]:
        return ['/profile_list', '/plist', '/profiles']

    @property
    def description(self) -> str:
        return 'List all saved LLM configuration profiles'

    @property
    def usage(self) -> str:
        return ''

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        try:
            from core.llm_profiles import get_profile_manager

            manager = get_profile_manager()
            profiles = manager.list_profiles()
            active_profile = manager.get_active_profile()

            if not profiles:
                message = "No saved profiles found. Use /profile_save to create one."
                print(colored(f"# cli-agent: {message}", "yellow"))
                return CommandResponse(result=CommandResult.CONTINUE, message=message)

            print(colored("\n📋 Saved LLM Profiles:", "cyan", attrs=["bold"]))
            print(colored("─" * 70, "cyan"))

            for profile in sorted(profiles, key=lambda p: p.id):
                is_active = profile.id == active_profile
                prefix = "▶ " if is_active else "  "
                id_display = colored(profile.id, "green" if is_active else "white", attrs=["bold"])

                print(f"{prefix}{id_display}")

                config_parts = []
                if profile.override:
                    config_parts.append(f"Override: {colored(profile.override, 'red')}")
                if profile.router:
                    config_parts.append(f"Router: {colored(profile.router, 'yellow')}")
                if profile.planner:
                    config_parts.append(f"Planner: {colored(profile.planner, 'cyan')}")
                if profile.fallbacks:
                    fallback_str = ", ".join(profile.fallbacks[:2])
                    if len(profile.fallbacks) > 2:
                        fallback_str += f" +{len(profile.fallbacks) - 2} more"
                    config_parts.append(f"Fallbacks: {fallback_str}")

                if config_parts:
                    print(f"    {' | '.join(config_parts)}")

                print()

            print(colored("─" * 70, "cyan"))
            print(colored(f"Total: {len(profiles)} profiles", "cyan"))
            print()

            return CommandResponse(result=CommandResult.CONTINUE, message=f"Listed {len(profiles)} profiles")

        except Exception as e:
            error_msg = f"Failed to list profiles: {str(e)}"
            print(colored(f"# cli-agent: {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)


class ProfileDeleteCommand(BaseCommand):
    """Delete a saved LLM configuration profile."""

    @property
    def aliases(self) -> List[str]:
        return ['/profile_delete', '/pdelete']

    @property
    def description(self) -> str:
        return 'Delete a saved LLM configuration profile'

    @property
    def usage(self) -> str:
        return '<profile_id>'

    async def execute(self, parts: List[str], **kwargs: Any) -> CommandResponse:
        if len(parts) < 2:
            return CommandResponse(
                result=CommandResult.ERROR,
                message="Usage: /profile_delete <profile_id>"
            )

        profile_id = parts[1]

        try:
            from core.llm_profiles import get_profile_manager

            manager = get_profile_manager()
            manager.delete_profile(profile_id)

            message = f"Deleted profile '{profile_id}'"
            print(colored(f"# cli-agent: {message}", "green"))

            return CommandResponse(result=CommandResult.CONTINUE, message=message)

        except Exception as e:
            error_msg = f"Failed to delete profile: {str(e)}"
            print(colored(f"# cli-agent: {error_msg}", "red"))
            return CommandResponse(result=CommandResult.ERROR, message=error_msg)
