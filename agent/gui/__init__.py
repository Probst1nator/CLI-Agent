"""
Agent GUI Components

This package contains GUI components for the CLI-Agent using prompt_toolkit.
Provides layouts, widgets, and display components for the enhanced GUI interface.
"""

__version__ = "1.0.0"

from .layouts import CLIAgentGUILayout
from .widgets import ChatDisplay, StatusPanel

__all__ = [
    "CLIAgentGUILayout",
    "ChatDisplay", 
    "StatusPanel"
]