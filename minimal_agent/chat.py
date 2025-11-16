# chat.py
"""
Simplified chat message management.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Role(Enum):
    """Message roles in the conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in the chat."""
    role: Role
    content: str

    def to_dict(self) -> dict:
        """Convert message to dictionary format."""
        return {
            "role": self.role.value,
            "content": self.content
        }


class Chat:
    """Manages a conversation as a list of messages."""

    def __init__(self):
        self.messages: list[Message] = []

    def add_system_message(self, content: str):
        """Add a system message."""
        self.messages.append(Message(Role.SYSTEM, content))

    def add_user_message(self, content: str):
        """Add a user message."""
        self.messages.append(Message(Role.USER, content))

    def add_assistant_message(self, content: str):
        """Add an assistant message."""
        self.messages.append(Message(Role.ASSISTANT, content))

    def to_dict_list(self) -> list[dict]:
        """Convert all messages to dictionary format."""
        return [msg.to_dict() for msg in self.messages]

    def get_last_message(self) -> Optional[Message]:
        """Get the most recent message."""
        return self.messages[-1] if self.messages else None

    def clear(self):
        """Clear all messages."""
        self.messages.clear()
