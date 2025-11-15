# tool_context.py
"""
Tool context manager for handling cross-tool communication, particularly for
passing base64 images from tools to the chat context.
"""
import threading
from typing import Any


class ToolContext:
    """
    Global context for tools to register data that needs to be passed to the chat.
    This allows tools like readfile to register base64 images that can be consumed
    by vision-capable models in the chat context.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.reset()
            return cls._instance

    def reset(self):
        """Reset all context data."""
        self.base64_images: list[str] = []
        self.image_descriptions: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.current_viz_id: str = None
        self.end_turn: bool = False  # Default: tools do NOT end the agent's turn
        self.consecutive_tool_count: int = 0
        self.last_tool_used: str | None = None

    def add_base64_image(self, base64_data: str, description: str = ""):
        """
        Add a base64 encoded image to the context.

        Args:
            base64_data: Base64 encoded image data (without data: prefix)
            description: Optional description of the image
        """
        self.base64_images.append(base64_data)
        self.image_descriptions.append(description)

    def has_images(self) -> bool:
        """Check if there are any registered images."""
        return len(self.base64_images) > 0

    def get_images(self) -> list[str]:
        """Get all registered base64 images."""
        return self.base64_images.copy()

    def get_image_descriptions(self) -> list[str]:
        """Get descriptions of all registered images."""
        return self.image_descriptions.copy()

    def clear_images(self):
        """Clear all registered images."""
        self.base64_images.clear()
        self.image_descriptions.clear()

    def set_metadata(self, key: str, value: Any):
        """Set metadata value."""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)

    def set_current_viz_id(self, viz_id: str):
        """Set the current visualresponse message ID for tool updates."""
        self.current_viz_id = viz_id

    def get_current_viz_id(self) -> str:
        """Get the current visualresponse message ID."""
        return self.current_viz_id

    def set_end_turn(self, value: bool):
        """
        Set whether the current tool execution should end the agent's turn.
        Used by response/visualresponse tools to control turn-ending behavior.

        Args:
            value: True to end the turn, False to continue executing
        """
        self.end_turn = value

    def get_end_turn(self) -> bool:
        """
        Get whether the current tool execution should end the agent's turn.

        Returns:
            bool: True if turn should end, False to continue (default)
        """
        return self.end_turn

    def clear_end_turn(self):
        """Reset the end_turn flag to its default value (False)."""
        self.end_turn = False

    def track_tool_usage(self, tool_name: str) -> int:
        """
        Track consecutive usage of a tool and return the current count.

        If a different tool is used, the counter resets to 1.
        If the same tool is used again, the counter increments.

        Args:
            tool_name: Name of the tool being executed

        Returns:
            int: Current consecutive usage count for this tool
        """
        if self.last_tool_used != tool_name:
            # Different tool - reset counter
            self.consecutive_tool_count = 1
            self.last_tool_used = tool_name
        else:
            # Same tool - increment counter
            self.consecutive_tool_count += 1

        return self.consecutive_tool_count

    def reset_tool_tracking(self):
        """Reset tool usage tracking (called when user provides input or turn ends)."""
        self.consecutive_tool_count = 0
        self.last_tool_used = None

    def get_consecutive_count(self) -> int:
        """Get the current consecutive tool usage count."""
        return self.consecutive_tool_count

# Global instance
tool_context = ToolContext()