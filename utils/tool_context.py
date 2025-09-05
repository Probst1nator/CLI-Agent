# tool_context.py
"""
Tool context manager for handling cross-tool communication, particularly for
passing base64 images from tools to the chat context.
"""
from typing import List, Dict, Any
import threading

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
        self.base64_images: List[str] = []
        self.image_descriptions: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
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
    
    def get_images(self) -> List[str]:
        """Get all registered base64 images."""
        return self.base64_images.copy()
    
    def get_image_descriptions(self) -> List[str]:
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

# Global instance
tool_context = ToolContext()