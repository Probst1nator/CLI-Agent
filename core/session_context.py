"""
Session Context for Multi-Tab Support

Provides session-scoped state management to enable multiple browser tabs
to connect to the same CLI-Agent container while maintaining independent
chat histories and settings.

Each WebSocket connection gets a unique session_id and corresponding
SessionContext instance. This replaces the global state pattern and
enables true multi-user/multi-tab support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from core.chat import Chat


@dataclass
class SessionContext:
    """
    Per-session isolated state for multi-tab support within a container

    Attributes:
        session_id: Unique identifier for this session
        chat: Chat instance for this session
        current_images: Base64-encoded images for next LLM request
        voice_mode: Whether voice input/output is enabled
        tts_only_mode: Whether to only use TTS (no transcription)
        auto_execute: Whether to auto-execute tool calls
        created_at: When this session was created
        last_activity: Last activity timestamp for cleanup
        user_id: Future: Link to user account system
        retention_days: Container retention period (future: per-user override)
    """
    session_id: str
    chat: Chat  # Forward reference handled by __future__ annotations
    current_images: list[str] = field(default_factory=list)
    voice_mode: bool = False
    tts_only_mode: bool = False
    auto_execute: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    # Future: user account integration
    user_id: Optional[str] = None
    retention_days: int = 7  # Default, will be overridden by user account settings

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()

    def is_expired(self, timeout_seconds: int = 3600) -> bool:
        """
        Check if session has expired (inactive for too long)

        Args:
            timeout_seconds: Inactivity timeout in seconds (default: 1 hour)

        Returns:
            True if session should be cleaned up
        """
        elapsed = (datetime.utcnow() - self.last_activity).total_seconds()
        return elapsed > timeout_seconds

    def clear_images(self):
        """Clear current images (after they've been sent to LLM)"""
        self.current_images.clear()

    def add_image(self, base64_image: str):
        """Add image to current session"""
        self.current_images.append(base64_image)

    def get_state_dict(self) -> dict:
        """Get session state as dictionary for frontend"""
        return {
            'session_id': self.session_id,
            'voice_mode': self.voice_mode,
            'tts_only_mode': self.tts_only_mode,
            'auto_execute': self.auto_execute,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'message_count': len(self.chat.messages) if self.chat else 0,
            'image_count': len(self.current_images)
        }
