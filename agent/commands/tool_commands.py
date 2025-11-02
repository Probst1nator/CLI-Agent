"""Tool management commands for enabling and disabling tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from termcolor import colored

from agent.commands.base_command import BaseCommand, CommandResponse, CommandResult

if TYPE_CHECKING:
    pass


class EnableToolCommand(BaseCommand):
    """Command for enabling specific tools."""

    @property
    def aliases(self) -> list[str]:
        return ['/enable', '/enable-tool']

    @property
    def description(self) -> str:
        return 'Enable one or more disabled tools'

    @property
    def usage(self) -> str:
        return """
Usage: /enable <tool1> <tool2> ...

Enable previously disabled tools and update the system prompt.

Examples:
  /enable bash              Enable the bash tool
  /enable bash readfile     Enable multiple tools at once
        """

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
        """Execute enable command."""
        tool_names = parts[1:] if len(parts) > 1 else []

        if not tool_names:
            return CommandResponse(
                result=CommandResult.ERROR,
                message="No tool names specified. Usage: /enable <tool1> <tool2> ..."
            )

        from core.tool_config import get_tool_config_manager

        tool_config = get_tool_config_manager()

        # Enable the tools
        tool_config.enable_tools(tool_names)

        # Update global state
        from core.globals import g
        g.disabled_tools = tool_config.get_disabled_tools()

        # Update system prompt to include newly enabled tools
        try:
            from main import update_system_prompt
            update_system_prompt()
            print(colored(f"✓ Enabled tools: {', '.join(tool_names)}", "green"))
            print(colored("  System prompt updated with new tool availability", "green"))
        except Exception as e:
            print(colored(f"✓ Enabled tools: {', '.join(tool_names)}", "green"))
            print(colored(f"  Warning: Could not update system prompt: {e}", "yellow"))

        return CommandResponse(result=CommandResult.CONTINUE)


class DisableToolCommand(BaseCommand):
    """Command for disabling specific tools."""

    @property
    def aliases(self) -> list[str]:
        return ['/disable', '/disable-tool']

    @property
    def description(self) -> str:
        return 'Disable one or more tools'

    @property
    def usage(self) -> str:
        return """
Usage: /disable <tool1> <tool2> ...

Disable tools to prevent the LLM from using them, and update the system prompt.

Examples:
  /disable bash              Disable the bash tool
  /disable bash writefile    Disable multiple tools at once
        """

    async def execute(self, parts: list[str], **kwargs: Any) -> CommandResponse:
        """Execute disable command."""
        tool_names = parts[1:] if len(parts) > 1 else []

        if not tool_names:
            return CommandResponse(
                result=CommandResult.ERROR,
                message="No tool names specified. Usage: /disable <tool1> <tool2> ..."
            )

        from core.tool_config import get_tool_config_manager

        tool_config = get_tool_config_manager()

        # Disable the tools
        tool_config.disable_tools(tool_names)

        # Update global state
        from core.globals import g
        g.disabled_tools = tool_config.get_disabled_tools()

        # Update system prompt to remove disabled tools
        try:
            from main import update_system_prompt
            update_system_prompt()
            print(colored(f"✓ Disabled tools: {', '.join(tool_names)}", "red"))
            print(colored("  System prompt updated to exclude disabled tools", "green"))
        except Exception as e:
            print(colored(f"✓ Disabled tools: {', '.join(tool_names)}", "red"))
            print(colored(f"  Warning: Could not update system prompt: {e}", "yellow"))

        return CommandResponse(result=CommandResult.CONTINUE)
