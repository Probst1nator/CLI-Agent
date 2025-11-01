import copy
import json
import math
import os
import logging
import time
import tkinter as tk
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any, TypedDict
from dataclasses import dataclass, field
import threading
import queue
from termcolor import colored

# This import is no longer strictly necessary for the modified save/load functions,
# but might be used elsewhere in the project. We'll leave it for now.
from core.globals import g
from ollama._types import Message as OllamaMessage, Image as OllamaImage
from shared.utils.project_storage import get_project_storage_path, ensure_project_directory

logger = logging.getLogger(__name__)

# LlmRouter will be imported lazily in compress() method to avoid circular imports

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# Metadata type definitions
class ToolInfo(TypedDict, total=False):
    """Tool execution information stored in message metadata"""
    name: str
    model: Optional[str]
    execution_time: Optional[float]
    status: Optional[str]  # 'success', 'error', 'pending'
    args: Optional[str]
    context_returned: Optional[str]


class MessageMetadata(TypedDict, total=False):
    """
    Message-level metadata structure.

    All fields are optional (total=False) to allow partial metadata.
    """
    is_bootstrap: bool  # Whether this is a bootstrap training message
    timestamp: str  # ISO 8601 timestamp
    model: str  # LLM model used to generate this message
    tool_info: List[ToolInfo]  # Information about tools executed in this message
    message_id: int  # Unique message identifier


class ChatMetadata(TypedDict, total=False):
    """
    Chat-level metadata structure.

    All fields are optional (total=False) to allow partial metadata.
    """
    created_at: str  # ISO 8601 creation timestamp
    updated_at: str  # ISO 8601 last update timestamp

@dataclass
class Message:
    """
    Represents a single message in a conversation.

    This is a provider-agnostic message format that can be converted
    to Ollama, Gemini, Anthropic, or OpenAI formats.

    Attributes:
        role: The role of the message sender (system, user, assistant, ipython)
        content: The text content of the message
        images: Optional list of base64-encoded images associated with this message
        metadata: Message metadata (timestamps, tool_info, etc.) - conforms to MessageMetadata type
    """
    role: Role
    content: str
    images: List[str] = field(default_factory=list)
    metadata: MessageMetadata = field(default_factory=dict)  # Runtime: dict, Type hint: MessageMetadata

class Chat:
    # Role-to-color mapping for terminal output
    ROLE_COLORS = {
        Role.ASSISTANT: ('blue', 'cyan'),
        Role.SYSTEM: ('blue', 'cyan'),
        Role.USER: ('light_green', 'green'),
    }
    DEFAULT_COLORS = ('light_yellow', 'yellow')  # Fallback

    def __init__(self, instruction_message: str = "", debug_title: Optional[str] = None):
        """
        Initializes a new Chat instance.

        :param instruction_message: The initial system instruction message.
        :param debug_title: Optional title for the debug window.
        """
        self.messages: List[Message] = []
        self.metadata: ChatMetadata = {}  # Chat-level metadata - conforms to ChatMetadata type
        self.base64_images: List[str] = []  # DEPRECATED: Use Message.images instead
        self._window: Optional[tk.Tk] = None
        self._text_widget: Optional[tk.Text] = None
        self.debug_title: str = debug_title or instruction_message[:50].split("\n")[0] or "Unnamed Context"
        self._update_queue: Optional[queue.Queue] = None
        self._window_thread: Optional[threading.Thread] = None
        self.is_main_context: bool = False  # Flag to mark this as the main context chat for broadcasting
        if instruction_message:
            self.add_message(Role.SYSTEM, instruction_message)

    def get_debug_title_prefix(self) -> str:
        """
        Get a formatted prefix string for debug messages that includes the chat's debug title if available.

        Returns:
            str: The formatted prefix string
        """
        return f"[Tokens: {math.ceil(len(self.__str__())*3/4)} | {self.debug_title}] " if self.debug_title else f"[Tokens: {math.ceil(len(self.__str__())*3/4)} | Messages: {len(self.messages)}] "

    def __len__(self) -> int:
        """
        Returns the total length of all string messages in the chat.

        :return: The total number of characters across all messages.
        """
        return sum(len(message.content) for message in self.messages)

    def _create_debug_window(self):
        """Creates and runs the debug window in a separate thread."""
        def window_thread():
            self._window = tk.Tk()
            self._window.title(self.debug_title)

            frame = tk.Frame(self._window)
            frame.pack(expand=True, fill='both')

            y_scrollbar = tk.Scrollbar(frame)
            y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            x_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
            x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

            self._text_widget = tk.Text(frame, wrap=tk.NONE,
                                      yscrollcommand=y_scrollbar.set,
                                      xscrollcommand=x_scrollbar.set)
            self._text_widget.pack(expand=True, fill='both')

            y_scrollbar.config(command=self._text_widget.yview)
            x_scrollbar.config(command=self._text_widget.xview)

            self._window.geometry("800x600")
            self._window.grid_rowconfigure(0, weight=1)
            self._window.grid_columnconfigure(0, weight=1)

            def check_queue():
                try:
                    while True:  # Process all pending updates
                        if self._update_queue:
                            messages = self._update_queue.get_nowait()
                            if self._text_widget:
                                self._text_widget.delete('1.0', tk.END)
                                for message in messages:
                                    role_str = f"{message.role.name}:\n"
                                    self._text_widget.insert(tk.END, role_str, 'bold')
                                    self._text_widget.insert(tk.END, f"{message.content}\n\n")
                                self._text_widget.tag_configure('bold', font=('TkDefaultFont', 10, 'bold'))
                except queue.Empty:
                    pass
                finally:
                    if self._window:  # Check if window still exists
                        self._window.after(100, check_queue)  # Schedule next check

            check_queue()
            self._window.protocol("WM_DELETE_WINDOW", self._on_window_close)
            self._window.mainloop()

        self._update_queue = queue.Queue()
        self._window_thread = threading.Thread(target=window_thread, daemon=True)
        self._window_thread.start()

    def _on_window_close(self):
        """Handle window closing."""
        if self._window:
            self._window.destroy()
            self._window = None
            self._text_widget = None

    def _update_window_display(self):
        """Updates the window display with current chat messages if debug mode is enabled."""
        # This check now relies solely on explicitly set global flags.
        if not g.DEBUG_CHATS:
            return

        if self._window is None and self._window_thread is None:
            self._create_debug_window()

        if self._update_queue is not None:
            try:
                self._update_queue.put_nowait(self.messages)
            except queue.Full:
                pass  # Skip update if queue is full

    def set_instruction_message(self, instruction_message: str) -> "Chat":
        """
        Sets the instruction message for the chat.
        If a system message already exists, it will be replaced.
        If not, the system message will be inserted at the beginning of the chat.

        :param instruction_message: The system instruction message.
        :return: The updated Chat instance.
        """
        # Check if first message is a system message
        if self.messages and self.messages[0].role == Role.SYSTEM:
            self.messages[0].content = instruction_message
        else:
            self.messages.insert(0, Message(role=Role.SYSTEM, content=instruction_message))

        self._update_window_display()
        return self

    def add_message(self, role: Role, content: str, images: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> "Chat":
        """
        Adds a message to the chat.

        :param role: The role of the message sender.
        :param content: The content of the message.
        :param images: Optional list of base64-encoded images to attach to this message.
        :param metadata: Optional metadata dictionary (e.g., tool_info, model used, timestamps).
        :return: The updated Chat instance.
        """
        if not (content and role):
            return self

        # Ensure content is a string for consistent merging
        content = str(content)
        images = images or []
        metadata = metadata or {}

        if self.messages and self.messages[-1].role == role and not images:
            # Append to the last message's content
            self.messages[-1].content += content

            # Merge metadata if both have it
            if metadata:
                if not self.messages[-1].metadata:
                    self.messages[-1].metadata = {}

                # Merge tool_info arrays if both exist
                if 'tool_info' in metadata and 'tool_info' in self.messages[-1].metadata:
                    self.messages[-1].metadata['tool_info'].extend(metadata['tool_info'])
                elif 'tool_info' in metadata:
                    self.messages[-1].metadata['tool_info'] = metadata['tool_info']

                # For other metadata keys, prefer new values
                for key, value in metadata.items():
                    if key != 'tool_info':
                        self.messages[-1].metadata[key] = value
        else:
            # Add a new message
            self.messages.append(Message(role=role, content=content, images=images, metadata=metadata))

        self._update_window_display()
        return self

    def update_message(self, index: int, new_content: str) -> bool:
        """
        Update the content of a message at the specified index.

        :param index: The index of the message to update (0-based).
        :param new_content: The new content for the message.
        :return: True if successful, False if index invalid or system message.
        """
        if not new_content or not new_content.strip():
            return False

        if index < 0 or index >= len(self.messages):
            return False

        if index == 0 and self.messages[0].role == Role.SYSTEM:
            return False

        # Update the message content while keeping role and images
        self.messages[index].content = new_content
        self._update_window_display()
        return True

    def get_messages_as_string(self, start_index: int, end_index: Optional[int] = None) -> str:
        """
        Get a string representation of messages from start_index to end_index.
        Args:
        start_index (int): The starting index of messages to include. Negative indices count from the end.
        end_index (Optional[int]): The ending index of messages to include (exclusive).
                                If None, includes all messages from start_index to the end.
                                Negative indices count from the end.
        Returns:
        str: A string representation of the selected messages.
        """
        # Normalize indices
        normalized_start = start_index if start_index >= 0 else len(self.messages) + start_index
        normalized_end = end_index if end_index is None else (
            end_index if end_index >= 0 else len(self.messages) + end_index
        )
        # Clamp indices to valid range
        normalized_start = max(0, min(normalized_start, len(self.messages)))
        if normalized_end is not None:
            normalized_end = max(normalized_start, min(normalized_end, len(self.messages)))
        else:
            normalized_end = len(self.messages)
        selected_messages = self.messages[normalized_start:normalized_end]
        # Build the string representation
        message_strings = []
        for message in selected_messages:
            if isinstance(message, Message):
                sender_name = message.role.name if hasattr(message.role, 'name') else str(message.role)
                message_strings.append(f"{sender_name}: {message.content}")
            else:
                # Handle potential malformed messages
                message_strings.append(str(message))
        return "\n".join(message_strings)

    def __getitem__(self, key: Union[int, slice, Tuple[int, ...]]) -> "Chat":
        """
        Retrieves a subset of the chat messages.

        :param key: The index, slice, or tuple of indices.
        :return: A new Chat instance with the specified messages.
        """
        if isinstance(key, (int, slice)):
            sliced_messages = self.messages[key]
            if isinstance(sliced_messages, Message):
                sliced_messages = [sliced_messages]
            new_chat = Chat()
            if isinstance(sliced_messages, list):
                for msg in sliced_messages:
                    new_chat.messages.append(msg)
            return new_chat
        elif isinstance(key, tuple):
            new_chat = Chat()
            for index in key:
                if isinstance(index, int):
                    new_chat.messages.append(self.messages[index])
                else:
                    raise TypeError("Invalid index type inside tuple.")
            return new_chat
        else:
            raise TypeError("Invalid argument type.")

    def __str__(self) -> str:
        """
        Returns a string representation of the chat messages.

        :return: A JSON string of the chat messages and metadata.
        """
        return self.to_json()

    def print_chat(self, start_index: Optional[int] = None, end_index: Optional[int] = None, return_only: bool = False) -> str:
        """
        Prints the chat messages with colored and bold roles, and similarly colored content using termcolor.

        Args:
            start_index (Optional[int]): The starting index of messages to include. Negative indices count from the end.
                                         If None, starts from the first message.
            end_index (Optional[int]): The ending index of messages to include (exclusive).
                                       If None, includes all messages from start_index to the end.
                                       Negative indices count from the end.
        """
        # Set default values for start_index and end_index if they are None
        if start_index is None:
            start_index = 0

        # Normalize indices
        normalized_start = start_index if start_index >= 0 else len(self.messages) + start_index
        normalized_end = end_index if end_index is None else (
            end_index if end_index >= 0 else len(self.messages) + end_index
        )

        # Clamp indices to valid range
        normalized_start = max(0, min(normalized_start, len(self.messages)))
        if normalized_end is not None:
            normalized_end = max(normalized_start, min(normalized_end, len(self.messages)))
        else:
            normalized_end = len(self.messages)

        # Get selected messages
        selected_messages = self.messages[normalized_start:normalized_end]

        joined_string = ""

        # Print the selected messages with appropriate formatting
        for message in selected_messages:
            role_value = message.role.value if isinstance(message.role, Role) else message.role

            # Get colors from the mapping
            role_color, content_color = self.ROLE_COLORS.get(message.role, self.DEFAULT_COLORS)

            formatted_role = colored(f"{role_value.upper()}:\n", role_color, attrs=['bold', "underline"])
            formatted_content = colored(message.content, content_color)

            if not return_only:
                print(f"{formatted_role} {formatted_content}")
            joined_string += f"{role_value.upper()}:\n {message.content}\n"
        return joined_string.strip()

    def save_to_json(self, file_name: str = "recent_chat.json", merge: bool = False, also_archive: bool = True):
        """
        Saves the chat instance to a JSON file in centralized ~/.cli-agent/projects/ storage.

        :param file_name: The name of the file to save to.
        :param merge: Whether to merge with existing file content (read-modify-write operation).
                     If True, loads existing file, merges messages, and overwrites the entire file.
                     For true append operations, consider using save_to_jsonl instead.
        :param also_archive: Whether to also save a timestamped copy to the archive.
        """
        # Get centralized storage path for current project
        storage_path = str(ensure_project_directory())
        file_path = os.path.join(storage_path, file_name)

        if merge:
            try:
                # Load from the same CWD-based path
                few_shot_prompts = Chat.load_from_json(file_name)
                few_shot_prompts.add_message(self.messages[0].role, self.messages[0].content)
                few_shot_prompts.add_message(self.messages[1].role, self.messages[1].content)
            except FileNotFoundError:
                # If the file doesn't exist when merging, start a new one
                few_shot_prompts = self
        else:
            few_shot_prompts = self

        with open(file_path, "w") as file:
            json.dump(few_shot_prompts._to_dict(), file, indent=4)

        # Also save to archive if requested
        if also_archive and len(self.messages) > 2:
            try:
                self.save_to_archive()
            except Exception as e:
                logger.warning(f"Failed to save chat to archive: {e}")

    @classmethod
    def load_from_json(cls, file_name: str = "recent_chat.json") -> "Chat":
        """
        Loads a Chat instance from a JSON file in centralized ~/.cli-agent/projects/ storage.

        :param file_name: The name of the file to load from.
        :return: The loaded Chat instance.
        """
        # Get centralized storage path for current project
        storage_path = str(ensure_project_directory())
        file_path = os.path.join(storage_path, file_name)

        with open(file_path, "r") as file:
            data_dict = json.load(file)
        return cls.from_dict(data_dict)

    def _to_dict(self) -> Dict[str, Any]:
        """
        Converts the chat instance to a dictionary.

        :return: A dictionary representing the chat instance including messages, metadata, and images.
        """
        return {
            "messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                    "metadata": message.metadata or {}
                }
                for message in self.messages
            ],
            "metadata": self.metadata,
            "base64_images": self.base64_images
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "Chat":
        """
        Creates a Chat instance from a dictionary.

        :param data_dict: A dictionary containing chat messages, metadata, and images.
        :return: The created Chat instance.
        """
        chat_instance = cls()

        # Add messages
        if "messages" in data_dict:
            messages = data_dict["messages"]
            for message in messages:
                role = Role[message["role"].upper()]
                content = message["content"]
                msg_metadata = message.get("metadata", {})
                chat_instance.add_message(role, content, metadata=msg_metadata)

        # Add metadata if present
        if "metadata" in data_dict:
            chat_instance.metadata = data_dict["metadata"]

        # Add base64 images if present
        if "base64_images" in data_dict:
            chat_instance.base64_images = data_dict["base64_images"]

        return chat_instance

    def to_json(self) -> str:
        """
        Converts the chat instance to a JSON string.

        :return: The JSON string representing the chat instance.
        """
        return json.dumps(self._to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Chat":
        """
        Creates a Chat instance from a JSON string.

        :param json_str: The JSON string containing chat messages.
        :return: The created Chat instance.
        """
        data_dict = json.loads(json_str)
        return cls.from_dict(data_dict)

    @staticmethod
    def save_to_jsonl(chats: List["Chat"], file_path: str = "saved_chat.jsonl") -> None:
        """
        Saves a list of Chat instances to a JSONL file.

        :param chats: The list of Chat instances to save.
        :param file_path: The path to the JSONL file.
        """
        if chats:
            with open(file_path, 'a') as file:
                for chat in chats:
                    chat_json = chat.to_json()
                    file.write(chat_json + '\n')

    @staticmethod
    def load_from_jsonl(file_path: str) -> List["Chat"]:
        """
        Loads a list of Chat instances from a JSONL file.

        :param file_path: The path to the JSONL file.
        :return: A list of loaded Chat instances.
        """
        chats = []
        with open(file_path, 'r') as file:
            for line in file:
                chat_data = json.loads(line)
                chat = Chat.from_dict(chat_data)
                chats.append(chat)
        return chats

    def save_to_archive(self) -> str:
        """
        Saves the chat to a timestamped archive file and updates the index.

        :return: The chat ID (hash) of the saved chat.
        """
        import hashlib
        from datetime import datetime as dt

        # Generate timestamp and hash
        timestamp = dt.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

        # Create hash from timestamp + first user message
        hash_input = timestamp_str
        for message in self.messages:
            if message.role == Role.USER:
                hash_input += message.content[:100]
                break
        chat_hash = hashlib.md5(hash_input.encode()).hexdigest()[:4]

        # Define paths in centralized storage
        storage_path = str(ensure_project_directory())
        archive_path = os.path.join(storage_path, "chat_history")

        # Save chat file
        filename = f"chat_{timestamp_str}_{chat_hash}.json"
        file_path = os.path.join(archive_path, filename)

        with open(file_path, "w") as file:
            json.dump(self._to_dict(), file, indent=4)

        # Update index
        self._update_archive_index(chat_hash, timestamp.timestamp(), filename)

        logger.debug(f"Chat archived: {filename}")
        return chat_hash

    @staticmethod
    def _update_archive_index(chat_id: str, timestamp: float, filename: str):
        """Updates the chat history index with new chat metadata."""
        storage_path = str(get_project_storage_path()) + "/chat_history"
        index_path = os.path.join(storage_path, "index.json")

        # Load existing index
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    index = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted index.json detected (line {e.lineno}), rebuilding from scratch: {e}")
                # Backup corrupted index
                import shutil
                backup_path = index_path + ".corrupted"
                shutil.copy(index_path, backup_path)
                logger.info(f"Backed up corrupted index to: {backup_path}")
                # Start fresh
                index = {"chats": []}
        else:
            index = {"chats": []}

        # Load the chat file to get metadata
        chat_path = os.path.join(storage_path, filename)
        with open(chat_path, "r") as f:
            chat_data = json.load(f)

        # Extract preview from the last message in the chat
        preview = ""
        message_count = len(chat_data.get("messages", []))

        # Get last non-system message
        for msg in reversed(chat_data.get("messages", [])):
            if msg.get("role") != "system":
                content = msg.get("content", "")
                # Take last 150 characters for preview
                if len(content) > 150:
                    preview = "..." + content[-150:]
                else:
                    preview = content
                break

        # Add to index
        index["chats"].append({
            "id": chat_id,
            "timestamp": timestamp,
            "filename": filename,
            "preview": preview,
            "message_count": message_count
        })

        # Keep only last 100 chats
        index["chats"] = sorted(index["chats"], key=lambda x: x["timestamp"], reverse=True)[:100]

        # Save index atomically (write to temp file, then rename)
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(dir=storage_path, prefix=".index_tmp_", suffix=".json")
        try:
            with os.fdopen(temp_fd, "w") as f:
                json.dump(index, f, indent=2)
            # Atomic rename (replaces old file only if write succeeded)
            os.replace(temp_path, index_path)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    @staticmethod
    def list_archived_chats() -> List[Dict[str, Any]]:
        """
        Lists all archived chats from the index.

        :return: List of chat metadata dictionaries.
        """
        storage_path = str(get_project_storage_path()) + "/chat_history"
        index_path = os.path.join(storage_path, "index.json")

        if not os.path.exists(index_path):
            return []

        with open(index_path, "r") as f:
            index = json.load(f)

        return index.get("chats", [])

    @classmethod
    def load_from_archive(cls, chat_id: str) -> "Chat":
        """
        Loads a chat from the archive by its ID.

        :param chat_id: The chat ID (hash).
        :return: The loaded Chat instance.
        """
        storage_path = str(get_project_storage_path()) + "/chat_history"
        index_path = os.path.join(storage_path, "index.json")

        # Find the filename from index
        with open(index_path, "r") as f:
            index = json.load(f)

        filename = None
        for chat in index.get("chats", []):
            if chat["id"] == chat_id:
                filename = chat["filename"]
                break

        if not filename:
            raise FileNotFoundError(f"Chat with ID {chat_id} not found in archive")

        # Load the chat file
        file_path = os.path.join(storage_path, filename)
        with open(file_path, "r") as f:
            data_dict = json.load(f)

        return cls.from_dict(data_dict)

    @staticmethod
    def delete_from_archive(chat_id: str) -> bool:
        """
        Deletes a chat from the archive.

        :param chat_id: The chat ID (hash).
        :return: True if deleted successfully.
        """
        storage_path = str(get_project_storage_path()) + "/chat_history"
        index_path = os.path.join(storage_path, "index.json")

        # Load index
        with open(index_path, "r") as f:
            index = json.load(f)

        # Find and remove from index
        filename = None
        index["chats"] = [chat for chat in index["chats"] if chat["id"] != chat_id or (filename := chat["filename"], False)[1]]

        # Find the filename
        for chat in index.get("chats", []):
            if chat["id"] == chat_id:
                filename = chat["filename"]
                break

        if filename:
            # Delete the file
            file_path = os.path.join(storage_path, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

            # Update index
            index["chats"] = [chat for chat in index["chats"] if chat["id"] != chat_id]
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)

            return True

        return False

    @staticmethod
    def search_chats(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Searches through all archived chats using BM25 ranking algorithm.

        :param query: The search query string.
        :param top_k: Maximum number of results to return.
        :return: List of chat metadata with scores, sorted by relevance.
        """
        import re
        from collections import Counter

        storage_path = str(get_project_storage_path()) + "/chat_history"
        index_path = os.path.join(storage_path, "index.json")

        if not os.path.exists(index_path):
            return []

        # Load index
        with open(index_path, "r") as f:
            index = json.load(f)

        chats = index.get("chats", [])
        if not chats:
            return []

        # Tokenize query
        def tokenize(text: str) -> List[str]:
            return re.findall(r'\w+', text.lower())

        def generate_ngrams(token: str, n: int = 3) -> List[str]:
            """Generate character n-grams for fuzzy matching"""
            if len(token) < n:
                return [token]
            return [token[i:i+n] for i in range(len(token) - n + 1)]

        def simple_stem(word: str) -> str:
            """Simple stemming - removes common suffixes"""
            suffixes = ['ung', 'heit', 'keit', 'isch', 'lich', 'en', 'er', 'es', 'e', 's']
            word_lower = word.lower()
            for suffix in suffixes:
                if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                    return word_lower[:-len(suffix)]
            return word_lower

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Create stemmed versions and n-grams for fuzzy matching
        query_stems = [simple_stem(token) for token in query_tokens]
        query_ngrams = {}
        for token in query_tokens:
            if len(token) >= 3:  # Only use n-grams for words with 3+ chars
                query_ngrams[token] = set(generate_ngrams(token))

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        # Calculate document stats
        doc_contents = []
        doc_lengths = []

        for chat in chats:
            chat_path = os.path.join(storage_path, chat['filename'])
            try:
                with open(chat_path, 'r') as f:
                    chat_data = json.load(f)

                # Extract all message content
                content_parts = []
                for msg in chat_data.get('messages', []):
                    if msg.get('content'):
                        content_parts.append(str(msg['content']))

                full_content = ' '.join(content_parts)
                tokens = tokenize(full_content)
                doc_contents.append(tokens)
                doc_lengths.append(len(tokens))
            except Exception as e:
                logger.warning(f"Failed to load chat {chat['id']} for search: {e}")
                doc_contents.append([])
                doc_lengths.append(0)

        if not doc_lengths or sum(doc_lengths) == 0:
            return []

        avgdl = sum(doc_lengths) / len(doc_lengths)
        N = len(chats)

        # Calculate document frequencies (including stems)
        df = Counter()
        df_stem = Counter()
        for doc_tokens in doc_contents:
            unique_tokens = set(doc_tokens)
            unique_stems = set(simple_stem(t) for t in doc_tokens)
            for token in unique_tokens:
                df[token] += 1
            for stem in unique_stems:
                df_stem[stem] += 1

        # Calculate BM25 scores with fuzzy matching and stemming
        scores = []
        for i, (chat, doc_tokens) in enumerate(zip(chats, doc_contents)):
            if doc_lengths[i] == 0:
                scores.append(0.0)
                continue

            doc_freq = Counter(doc_tokens)
            doc_stems = Counter(simple_stem(t) for t in doc_tokens)
            score = 0.0

            for j, term in enumerate(query_tokens):
                term_score = 0.0

                # 1. Exact match (highest weight)
                if term in doc_freq:
                    idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
                    tf = doc_freq[term]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_lengths[i] / avgdl)
                    term_score += idf * (numerator / denominator)

                # 2. Stem match (medium weight - 0.7x)
                stem = query_stems[j]
                if stem in doc_stems and term not in doc_freq:
                    idf = math.log((N - df_stem[stem] + 0.5) / (df_stem[stem] + 0.5) + 1.0)
                    tf = doc_stems[stem]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_lengths[i] / avgdl)
                    term_score += 0.7 * idf * (numerator / denominator)

                # 3. Fuzzy n-gram match (lower weight - 0.4x)
                if term in query_ngrams and term not in doc_freq and stem not in doc_stems:
                    query_grams = query_ngrams[term]
                    for doc_token in set(doc_tokens):
                        if len(doc_token) >= 3:
                            doc_grams = set(generate_ngrams(doc_token))
                            overlap = len(query_grams & doc_grams)
                            if overlap > 0:
                                # Jaccard similarity
                                similarity = overlap / len(query_grams | doc_grams)
                                if similarity > 0.5:  # At least 50% overlap
                                    tf = doc_freq[doc_token]
                                    # Use a default IDF for fuzzy matches
                                    idf = 2.0
                                    numerator = tf * (k1 + 1)
                                    denominator = tf + k1 * (1 - b + b * doc_lengths[i] / avgdl)
                                    term_score += 0.4 * similarity * idf * (numerator / denominator)
                                    break  # Only count best fuzzy match per query term

                score += term_score

            scores.append(score)

        # Create results with scores and matching snippets
        results = []
        for i, (chat, score) in enumerate(zip(chats, scores)):
            if score > 0:
                result = chat.copy()
                result['score'] = score

                # Find best matching snippet (with fuzzy/stem matching)
                snippet = _find_best_snippet(doc_contents[i], query_tokens, query_stems, query_ngrams, chat['filename'], storage_path)
                result['search_snippet'] = snippet

                results.append(result)

        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k]

    def joined_messages(self) -> str:
        """
        Returns all messages joined with "\n".
        :return: The joined messages as a single string.
        """
        return "\n".join(str(message.content) for message in self.messages)

    def count_tokens(self, encoding_name: str = "cl100k_base") -> int:
        """
        Counts the number of tokens in the chat messages.

        :param encoding_name: The name of the encoding to use.
        :return: The number of tokens.
        """
        return math.floor(len(self.joined_messages())/4)

    def to_ollama(self) -> Sequence[OllamaMessage]:
        """
        Converts chat messages to Ollama format.
        :return: The chat messages in Ollama format.
        """
        message_sequence = []
        for message in self.messages:
            images = None
            if message.images:
                images = [OllamaImage(value=img) for img in message.images]

            ollama_msg = OllamaMessage(
                role=message.role.value,
                content=message.content,
                images=images
            )
            message_sequence.append(ollama_msg)

        # DEPRECATED: Support old base64_images for backward compatibility
        if self.base64_images:
            if message_sequence:
                last_msg = message_sequence[-1]
                existing_images = list(last_msg.images) if last_msg.images else []
                existing_images.extend([OllamaImage(value=img) for img in self.base64_images])
                message_sequence[-1] = OllamaMessage(
                    role=last_msg.role,
                    content=last_msg.content,
                    images=existing_images
                )
            self.base64_images = []

        return message_sequence

    def to_openai(self) -> List[Dict[str, str]]:
        """
        Converts chat messages to OpenAI chat format.

        :return: The chat messages in OpenAI chat format.
        """
        result = []
        for i, message in enumerate(self.messages):
            if message.role.value == "ipython" and i > 0:
                result[-1]["content"] += f"\n\n{message.content}"
            else:
                result.append({"role": message.role.value, "content": message.content})
        return result

    def to_groq(self) -> List[Dict[str, str]]:
        """
        Converts the chat to a list of dictionaries for use with the Groq API.

        Returns:
            List[Dict[str, str]]: The chat messages formatted for Groq's API.
        """
        groq_messages = []

        for message in self.messages:
            if message.role == Role.SYSTEM:
                groq_messages.append({"role": "system", "content": message.content})
            elif message.role == Role.USER:
                groq_messages.append({"role": "user", "content": message.content})
            elif message.role == Role.ASSISTANT:
                groq_messages.append({"role": "assistant", "content": message.content})
            else:
                groq_messages.append({"role": "user", "content": message.content})

        return groq_messages

    def to_gemini(self, base64_images: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Converts the chat to a list of dictionaries for use with the Google Gemini API.

        Args:
            base64_images: Optional list of base64-encoded images to include in the message.

        Returns:
            List[Dict[str, Any]]: The chat messages formatted for Gemini's API.
        """
        gemini_messages: List[Dict[str, Any]] = []

        system_prompt = None
        for message in self.messages:
            if message.role == Role.SYSTEM:
                system_prompt = message.content
                break

        for message in self.messages:
            if message.role == Role.USER:
                parts: List[Any] = [str(message.content)]

                if message.images:
                    for image in message.images:
                        mime_type = "image/jpeg"
                        try:
                            import base64
                            image_data = base64.b64decode(image[:100])
                            if image_data.startswith(b'\x89PNG'):
                                mime_type = "image/png"
                            elif image_data.startswith(b'GIF8'):
                                mime_type = "image/gif"
                            elif image_data.startswith(b'\xff\xd8\xff'):
                                mime_type = "image/jpeg"
                            elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:20]:
                                mime_type = "image/webp"
                        except Exception:
                            pass

                        parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image
                            }
                        })

                gemini_messages.append({"role": "user", "parts": parts})

            elif message.role == Role.ASSISTANT:
                gemini_messages.append({"role": "model", "parts": [str(message.content)]})

        if system_prompt and gemini_messages and gemini_messages[0]["role"] == "user":
            first_message = gemini_messages[0]
            first_message["parts"].insert(0, f"System instruction: {system_prompt}\n\n")

        return gemini_messages

    def to_gemma(self) -> str:
        """
        Converts chat messages to Gemma format.

        Format specification:
        - Each turn begins with <start_of_turn> followed by the role (user or model)
        - Each turn ends with <end_of_turn>
        - The entire conversation begins with <bos>
        - Images are embedded with <img> tags

        :return: The chat messages in Gemma format.
        """
        gemma_format = "<bos>"

        for message in self.messages:
            if message.role == Role.SYSTEM:
                continue

            if message.role == Role.USER:
                gemma_role = "user"
            elif message.role == Role.ASSISTANT:
                gemma_role = "model"
            else:
                continue

            gemma_format += f"<start_of_turn>{gemma_role}\n"

            processed_content = message.content
            if message.role == Role.USER and self.base64_images and gemma_role == "user":
                for i, image in enumerate(self.base64_images):
                    image_placeholder = f"Image {chr(65+i)}: "
                    if image_placeholder in processed_content:
                        processed_content = processed_content.replace(
                            image_placeholder,
                            f"Image {chr(65+i)}: <img>\n"
                        )

            gemma_format += f"{processed_content}<end_of_turn>\n"

        return gemma_format

    def deep_copy(self) -> 'Chat':
        """
        Creates a deep copy of the Chat instance.

        :return: A new Chat instance that is a deep copy of the current instance.
        """
        new_chat = Chat("Copy of " + self.debug_title)
        new_chat.messages = copy.deepcopy(self.messages)
        new_chat.base64_images = copy.deepcopy(self.base64_images)
        new_chat.metadata = copy.deepcopy(self.metadata)
        return new_chat

    def join(self, chat: 'Chat') -> 'Chat':
        """
        Joins the messages of another Chat instance to the current instance.

        :param chat: The Chat instance to join.
        :return: The updated Chat instance.
        """
        messages_to_add = []
        if chat.messages[0].role == Role.SYSTEM:
            messages_to_add = chat.messages[1:]
        else:
            messages_to_add = chat.messages

        for message_to_add in messages_to_add:
            self.add_message(message_to_add.role, message_to_add.content)

        self.base64_images.extend(chat.base64_images)

        if hasattr(chat, 'metadata') and chat.metadata:
            for key, value in chat.metadata.items():
                if key not in self.metadata:
                    self.metadata[key] = value

        return self

    def replace_latest_user_message(self, new_content: str) -> "Chat":
        """
        Replaces the content of the most recent user message with new content.

        :param new_content: The new content to replace the latest user message with
        :return: The updated Chat instance
        """
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == Role.USER:
                self.messages[i].content = new_content
                self._update_window_display()
                break
        return self

    def compact(self, prompt: str, level: str = 'detailed', model: str = '') -> 'Chat':
        """
        Compacts the chat conversation using an LLM to generate a summary.

        :param prompt: Custom compression instruction/context
        :param level: Compression level - 'dense' (brief) or 'detailed' (comprehensive)
        :param model: Optional specific model to use (defaults to system configuration)
        :return: A new Chat instance with compacted conversation
        """
        try:
            from core.llm_router import LlmRouter
        except ImportError:
            logger.warning("LlmRouter not available for compression, returning copy of original chat")
            return self.deep_copy()

        if level not in ['dense', 'detailed']:
            logger.warning(f"Invalid compression level '{level}', defaulting to 'detailed'")
            level = 'detailed'

        if level == 'dense':
            compression_instruction = (
                "Create a brief, essential summary of this conversation. "
                "Focus on key decisions, important facts, and critical context needed to continue the discussion. "
                "Be concise but preserve the most important information."
            )
        else:
            compression_instruction = (
                "Create a comprehensive summary of this conversation that maintains all important details, "
                "decisions, technical context, and key points. The summary should provide full context "
                "for someone to continue the conversation seamlessly."
            )

        chat = Chat(compression_instruction)
        chat.add_message(Role.USER, f"{prompt}\n\nConversation:\n{self.print_chat()}")

        try:
            import asyncio
            preferred_models = [model] if model else []
            response = asyncio.run(LlmRouter.generate_completion(chat, preferred_models=preferred_models))
        except Exception as e:
            logger.error(f"LLM compression failed: {e}")
            return self.deep_copy()

        system_msg = ""
        if self.messages and self.messages[0].role == Role.SYSTEM:
            system_msg = self.messages[0].content

        memory_delimiter_prefix: str = "\n<context>"
        memory_delimiter_suffix: str = "</context>"

        if (memory_delimiter_prefix in system_msg):
            system_msg = system_msg.split(memory_delimiter_prefix)[0]

        system_msg += memory_delimiter_prefix + response + memory_delimiter_suffix

        compressed_chat = Chat()
        compressed_chat.set_instruction_message(system_msg)

        compressed_chat.metadata['compression'] = {
            'original_length': len(self),
            'compressed_length': len(compressed_chat),
            'level': level,
            'model': model,
            'prompt': prompt,
            'timestamp': time.time()
        }

        return compressed_chat

def _find_best_snippet(doc_tokens: List[str], query_tokens: List[str], query_stems: List[str], query_ngrams: Dict[str, set], filename: str, storage_path: str, context_words: int = 10) -> str:
    """
    Finds the best matching snippet from the document that contains query terms.
    Now supports fuzzy matching and stemming.

    :param doc_tokens: Tokenized document
    :param query_tokens: Tokenized query
    :param query_stems: Stemmed query tokens
    :param query_ngrams: N-grams for each query token
    :param filename: Chat filename for loading original text
    :param storage_path: Path to chat storage
    :param context_words: Number of words to show around matches
    :return: Text snippet with matches
    """
    if not doc_tokens or not query_tokens:
        return ""

    def simple_stem(word: str) -> str:
        """Simple stemming - removes common suffixes"""
        suffixes = ['ung', 'heit', 'keit', 'isch', 'lich', 'en', 'er', 'es', 'e', 's']
        word_lower = word.lower()
        for suffix in suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                return word_lower[:-len(suffix)]
        return word_lower

    def generate_ngrams(token: str, n: int = 3) -> set:
        """Generate character n-grams for fuzzy matching"""
        if len(token) < n:
            return {token}
        return set(token[i:i+n] for i in range(len(token) - n + 1))

    def matches_query(doc_token: str) -> bool:
        """Check if doc token matches any query term (exact, stem, or fuzzy)"""
        # Exact match
        if doc_token in query_tokens:
            return True

        # Stem match
        doc_stem = simple_stem(doc_token)
        if doc_stem in query_stems:
            return True

        # Fuzzy match
        if len(doc_token) >= 3:
            doc_grams = generate_ngrams(doc_token)
            for term, term_grams in query_ngrams.items():
                if term_grams:
                    overlap = len(term_grams & doc_grams)
                    similarity = overlap / len(term_grams | doc_grams) if term_grams | doc_grams else 0
                    if similarity > 0.5:
                        return True

        return False

    # Find positions of query terms in document (with fuzzy/stem matching)
    match_positions = []

    for idx, token in enumerate(doc_tokens):
        if matches_query(token):
            match_positions.append(idx)

    if not match_positions:
        return ""

    # Find the region with most matches (clustering)
    best_start = match_positions[0]
    best_count = 1

    for i in range(len(match_positions)):
        window_start = match_positions[i]
        count = 1

        for j in range(i + 1, len(match_positions)):
            if match_positions[j] - window_start < context_words * 2:
                count += 1
            else:
                break

        if count > best_count:
            best_count = count
            best_start = window_start

    # Extract snippet around best match region
    snippet_start = max(0, best_start - context_words)
    snippet_end = min(len(doc_tokens), best_start + context_words)

    snippet_tokens = doc_tokens[snippet_start:snippet_end]

    # Reconstruct original text (approximation by joining tokens)
    snippet = ' '.join(snippet_tokens)

    # Add ellipsis if truncated
    if snippet_start > 0:
        snippet = '...' + snippet
    if snippet_end < len(doc_tokens):
        snippet = snippet + '...'

    # Limit length
    if len(snippet) > 200:
        snippet = snippet[:200] + '...'

    return snippet


