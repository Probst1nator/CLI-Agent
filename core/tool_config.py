"""
Tool configuration and persistence management for CLI-Agent.

Manages the list of disabled tools, providing persistence across sessions
and a centralized interface for modifying tool availability.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ToolConfigManager:
    """
    Manages tool configuration state, including disabled tools.

    Provides persistence to ~/.cli-agent/disabled_tools.json and
    thread-safe access to tool state.
    """

    def __init__(self, config_dir: Path = None):
        """
        Initialize the tool configuration manager.

        Args:
            config_dir: Directory for config storage (defaults to ~/.cli-agent)
        """
        if config_dir is None:
            config_dir = Path.home() / ".cli-agent"

        self.config_dir = config_dir
        self.config_file = config_dir / "disabled_tools.json"
        self._disabled_tools: set[str] = set()

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load existing configuration
        self.load()

    def load(self) -> None:
        """Load disabled tools from config file."""
        if not self.config_file.exists():
            logger.debug("No disabled_tools.json found, starting with empty set")
            self._disabled_tools = set()
            return

        try:
            with open(self.config_file) as f:
                data = json.load(f)
                self._disabled_tools = set(data.get('disabled_tools', []))
                logger.debug(f"Loaded {len(self._disabled_tools)} disabled tools from config")
        except Exception as e:
            logger.warning(f"Failed to load disabled tools config: {e}")
            self._disabled_tools = set()

    def save(self) -> None:
        """Save disabled tools to config file."""
        try:
            data = {
                'disabled_tools': sorted(list(self._disabled_tools))
            }
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self._disabled_tools)} disabled tools to config")
        except Exception as e:
            logger.error(f"Failed to save disabled tools config: {e}")

    def get_disabled_tools(self) -> list[str]:
        """
        Get the current list of disabled tools.

        Returns:
            List of disabled tool names
        """
        return sorted(list(self._disabled_tools))

    def disable_tools(self, tool_names: list[str]) -> None:
        """
        Add tools to the disabled list.

        Args:
            tool_names: List of tool names to disable
        """
        self._disabled_tools.update(tool_names)
        self.save()
        logger.info(f"Disabled tools: {tool_names}")

    def enable_tools(self, tool_names: list[str]) -> None:
        """
        Remove tools from the disabled list.

        Args:
            tool_names: List of tool names to enable
        """
        self._disabled_tools.difference_update(tool_names)
        self.save()
        logger.info(f"Enabled tools: {tool_names}")

    def is_disabled(self, tool_name: str) -> bool:
        """
        Check if a tool is disabled.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool is disabled, False otherwise
        """
        return tool_name in self._disabled_tools

    def clear(self) -> None:
        """Clear all disabled tools (enable all tools)."""
        self._disabled_tools.clear()
        self.save()
        logger.info("Cleared all disabled tools (all tools enabled)")

    def set_disabled_tools(self, tool_names: list[str]) -> None:
        """
        Replace the entire disabled tools list.

        Args:
            tool_names: List of tool names to set as disabled
        """
        self._disabled_tools = set(tool_names)
        self.save()
        logger.info(f"Set disabled tools: {tool_names}")


# Global singleton instance
_tool_config_manager: Optional[ToolConfigManager] = None


def get_tool_config_manager() -> ToolConfigManager:
    """
    Get the global tool configuration manager instance.

    Returns:
        The global ToolConfigManager instance
    """
    global _tool_config_manager
    if _tool_config_manager is None:
        _tool_config_manager = ToolConfigManager()
    return _tool_config_manager
