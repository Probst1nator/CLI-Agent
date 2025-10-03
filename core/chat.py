import copy
import json
import math
import os
import logging
import time
import tkinter as tk
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import threading
import queue
from termcolor import colored

# This import is no longer strictly necessary for the modified save/load functions,
# but might be used elsewhere in the project. We'll leave it for now.
from core.globals import g
from ollama._types import Message

logger = logging.getLogger(__name__)

# LlmRouter will be imported lazily in compress() method to avoid circular imports

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    IPYTHON = "ipython"
    ASSISTANT = "assistant"

class Chat:
    # Role-to-color mapping for terminal output
    ROLE_COLORS = {
        Role.ASSISTANT: ('blue', 'cyan'),
        Role.SYSTEM: ('blue', 'cyan'),
        Role.USER: ('light_green', 'green'),
        Role.IPYTHON: ('light_yellow', 'yellow'),
    }
    DEFAULT_COLORS = ('light_yellow', 'yellow')  # Fallback

    def __init__(self, instruction_message: str = "", debug_title: Optional[str] = None):
        """
        Initializes a new Chat instance.

        :param instruction_message: The initial system instruction message.
        :param debug_title: Optional title for the debug window.
        """
        self.messages: List[Tuple[Role, str]] = []
        self.base64_images: List[str] = []
        self.metadata: Dict[str, Any] = {}  # Dictionary to store additional metadata
        self._window: Optional[tk.Tk] = None
        self._text_widget: Optional[tk.Text] = None
        self.debug_title: str = debug_title or instruction_message[:50].split("\n")[0] or "Unnamed Context"
        self._update_queue: Optional[queue.Queue] = None
        self._window_thread: Optional[threading.Thread] = None
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
        return sum(len(content) for _, content in self.messages)

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
                        messages = self._update_queue.get_nowait()
                        self._text_widget.delete('1.0', tk.END)
                        for role, content in messages:
                            role_str = f"{role.name}:\n"
                            self._text_widget.insert(tk.END, role_str, 'bold')
                            self._text_widget.insert(tk.END, f"{content}\n\n")
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
        if self.messages and self.messages[0][0] == Role.SYSTEM:
            self.messages[0] = (Role.SYSTEM, instruction_message)
        else:
            self.messages.insert(0, (Role.SYSTEM, instruction_message))

        self._update_window_display()
        return self

    def add_message(self, role: Role, content: str) -> "Chat":
        """
        Adds a message to the chat.

        :param role: The role of the message sender.
        :param content: The content of the message.
        :return: The updated Chat instance.
        """
        if not (content and role):
            return self

        # Ensure content is a string for consistent merging
        content = str(content)

        if self.messages and self.messages[-1][0] == role:
            # Append to the last message's content
            self.messages[-1] = (role, self.messages[-1][1] + content)
        else:
            # Add a new message tuple
            self.messages.append((role, content))

        self._update_window_display()
        return self

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
            if isinstance(message, (list, tuple)) and len(message) >= 2:
                sender = message[0]
                content = message[1]
                sender_name = sender.name if hasattr(sender, 'name') else str(sender)
                message_strings.append(f"{sender_name}: {content}")
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
            if isinstance(sliced_messages, tuple):
                sliced_messages = [sliced_messages]
            new_chat = Chat()
            new_chat.messages = sliced_messages
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
        for role, content in selected_messages:
            role_value = role.value if isinstance(role, Role) else role

            # Get colors from the mapping
            role_color, content_color = self.ROLE_COLORS.get(role, self.DEFAULT_COLORS)

            formatted_role = colored(f"{role_value.upper()}:\n", role_color, attrs=['bold', "underline"])
            formatted_content = colored(content, content_color)

            if not return_only:
                print(f"{formatted_role} {formatted_content}")
            joined_string += f"{role_value.upper()}:\n {content}\n"
        return joined_string.strip()

    def save_to_json(self, file_name: str = "recent_chat.json", merge: bool = False, also_archive: bool = True):
        """
        Saves the chat instance to a JSON file in a .cli-agent folder in the current working directory.

        :param file_name: The name of the file to save to.
        :param merge: Whether to merge with existing file content (read-modify-write operation).
                     If True, loads existing file, merges messages, and overwrites the entire file.
                     For true append operations, consider using save_to_jsonl instead.
        :param also_archive: Whether to also save a timestamped copy to the archive.
        """
        # Define the storage path in the current working directory
        storage_path = os.path.join(os.getcwd(), ".cli-agent")
        file_path = os.path.join(storage_path, file_name)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if merge:
            try:
                # Load from the same CWD-based path
                few_shot_prompts = Chat.load_from_json(file_name)
                few_shot_prompts.add_message(self.messages[0][0], self.messages[0][1])
                few_shot_prompts.add_message(self.messages[1][0], self.messages[1][1])
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
        Loads a Chat instance from a JSON file in a .cli-agent folder in the current working directory.

        :param file_name: The name of the file to load from.
        :return: The loaded Chat instance.
        """
        # Define the storage path in the current working directory
        storage_path = os.path.join(os.getcwd(), ".cli-agent")
        file_path = os.path.join(storage_path, file_name)

        with open(file_path, "r") as file:
            data_dict = json.load(file)
        return cls.from_dict(data_dict)

    def _to_dict(self) -> Dict[str, Any]:
        """
        Converts the chat instance to a dictionary.

        :return: A dictionary representing the chat instance including messages and metadata.
        """
        return {
            "messages": [
                {"role": role.value, "content": content}
                for role, content in self.messages
            ],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "Chat":
        """
        Creates a Chat instance from a dictionary.

        :param data_dict: A dictionary containing chat messages and metadata.
        :return: The created Chat instance.
        """
        chat_instance = cls()

        # Add messages
        if "messages" in data_dict:
            messages = data_dict["messages"]
            for message in messages:
                role = Role[message["role"].upper()]
                content = message["content"]
                chat_instance.add_message(role, content)

        # Add metadata if present
        if "metadata" in data_dict:
            chat_instance.metadata = data_dict["metadata"]

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
        for role, content in self.messages:
            if role == Role.USER:
                hash_input += content[:100]
                break
        chat_hash = hashlib.md5(hash_input.encode()).hexdigest()[:4]

        # Define paths
        storage_path = os.path.join(os.getcwd(), ".cli-agent")
        archive_path = os.path.join(storage_path, "chat_history")
        os.makedirs(archive_path, exist_ok=True)

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
        storage_path = os.path.join(os.getcwd(), ".cli-agent", "chat_history")
        index_path = os.path.join(storage_path, "index.json")

        # Load existing index
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                index = json.load(f)
        else:
            index = {"chats": []}

        # Load the chat file to get metadata
        chat_path = os.path.join(storage_path, filename)
        with open(chat_path, "r") as f:
            chat_data = json.load(f)

        # Extract preview from first user message
        preview = ""
        message_count = len(chat_data.get("messages", []))
        for msg in chat_data.get("messages", []):
            if msg.get("role") == "user":
                preview = msg.get("content", "")[:100]
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

        # Save index
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    @staticmethod
    def list_archived_chats() -> List[Dict[str, Any]]:
        """
        Lists all archived chats from the index.

        :return: List of chat metadata dictionaries.
        """
        storage_path = os.path.join(os.getcwd(), ".cli-agent", "chat_history")
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
        storage_path = os.path.join(os.getcwd(), ".cli-agent", "chat_history")
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
        storage_path = os.path.join(os.getcwd(), ".cli-agent", "chat_history")
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




    def joined_messages(self) -> str:
        """
        Returns all messages joined with "\n".
        :return: The joined messages as a single string.
        """
        # Join only the content (second element) from each message tuple
        # Each message is a tuple of (Role, content)
        return "\n".join(str(message[1]) for message in self.messages)

    def count_tokens(self, encoding_name: str = "cl100k_base") -> int:
        """
        Counts the number of tokens in the chat messages.

        :param encoding_name: The name of the encoding to use.
        :return: The number of tokens.
        """
        return math.floor(len(self.joined_messages())/4)

    def to_ollama(self) -> Sequence[Message]:
        """
        Converts chat messages to Ollama format.
        :return: The chat messages in Ollama format.
        """
        message_sequence = [
            Message(role=message[0].value, content=message[1])
            for message in self.messages
        ]

        # Make sure there are base64_images to decode and assign
        if self.base64_images:
            message = message_sequence[-1]
            message["images"] = [image for image in self.base64_images]
            message_sequence[-1] = message
            self.base64_images = [] # Reset base64_images

        return message_sequence

    def to_openai(self) -> List[Dict[str, str]]:
        """
        Converts chat messages to OpenAI chat format.

        :return: The chat messages in OpenAI chat format.
        """
        result = []
        for i, message in enumerate(self.messages):
            if message[0].value == "ipython" and i > 0:
                result[-1]["content"] += f"\n\n{message[1]}"
            else:
                result.append({"role": message[0].value, "content": message[1]})
        return result

    def to_groq(self) -> List[Dict[str, str]]:
        """
        Converts the chat to a list of dictionaries for use with the Groq API.

        Returns:
            List[Dict[str, str]]: The chat messages formatted for Groq's API.
        """
        groq_messages = []

        for role, content in self.messages:
            if role == Role.SYSTEM:
                groq_messages.append({"role": "system", "content": content})
            elif role == Role.USER:
                groq_messages.append({"role": "user", "content": content})
            elif role == Role.ASSISTANT:
                groq_messages.append({"role": "assistant", "content": content})
            else:
                # For any other role, treat as a user message
                groq_messages.append({"role": "user", "content": content})

        return groq_messages

    def to_gemini(self, base64_images: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Converts the chat to a list of dictionaries for use with the Google Gemini API.

        Args:
            base64_images: Optional list of base64-encoded images to include in the message.

        Returns:
            List[Dict[str, Any]]: The chat messages formatted for Gemini's API.
        """
        gemini_messages = []

        # Use local base64_images if provided, otherwise use self.base64_images
        images_to_include = base64_images if base64_images is not None else self.base64_images

        # Handle system prompt separately
        system_prompt = None
        for role, content in self.messages:
            if role == Role.SYSTEM:
                system_prompt = content
                break

        for role, content in self.messages:
            if role == Role.USER:
                # Handle user messages, which might contain images
                if isinstance(content, list):
                    # Multimodal content
                    content_parts = []

                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                content_parts.append(item.get("text", ""))
                            elif item.get("type") == "image":
                                image_url = item.get("image_url", {}).get("url", "")
                                if image_url.startswith("data:image"):
                                    # Handle base64 encoded images
                                    try:
                                        import base64
                                        import google.generativeai as genai
                                        image_data = image_url.split(",")[1]
                                        mime_type = image_url.split(";")[0].split(":")[1]
                                        decoded_image = base64.b64decode(image_data)
                                        content_parts.append(genai.Part.from_data(decoded_image, mime_type=mime_type))
                                    except Exception:
                                        # Fallback for when genai isn't available
                                        content_parts.append("[Image data]")
                                elif image_url.startswith("http"):
                                    # Handle URL images
                                    try:
                                        import google.generativeai as genai
                                        content_parts.append(genai.Part.from_uri(image_url))
                                    except Exception:
                                        # Fallback for when genai isn't available
                                        content_parts.append(f"[Image URL: {image_url}]")
                        else:
                            # Simple text message
                            content_parts.append(str(item))

                    gemini_messages.append({"role": "user", "parts": content_parts})
                else:
                    # Simple text message
                    gemini_messages.append({"role": "user", "parts": [str(content)]})

                    # Add any images attached to the message
                    if images_to_include:
                        current_message = gemini_messages[-1]
                        for image in images_to_include:
                            # Detect MIME type from base64 header or default to jpeg
                            mime_type = "image/jpeg"  # default
                            try:
                                import base64
                                # Try to detect format from magic bytes
                                image_data = base64.b64decode(image[:100])  # Just first few bytes for detection
                                if image_data.startswith(b'\x89PNG'):
                                    mime_type = "image/png"
                                elif image_data.startswith(b'GIF8'):
                                    mime_type = "image/gif"
                                elif image_data.startswith(b'\xff\xd8\xff'):
                                    mime_type = "image/jpeg"
                                elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:20]:
                                    mime_type = "image/webp"
                            except Exception:
                                pass  # Keep default

                            current_message["parts"].append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": image
                                }
                            })
                        images_to_include = []  # Clear images after adding them

            elif role == Role.ASSISTANT:
                # Assistant messages are always text
                gemini_messages.append({"role": "model", "parts": [str(content)]})

            # System messages are handled separately (above)

        # Add system prompt if present (prepend to the first user message)
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
        # Start with the <bos> tag
        gemma_format = "<bos>"

        for role, content in self.messages:
            # Skip system messages as they don't have a direct equivalent in Gemma format
            if role == Role.SYSTEM:
                continue

            # Map the role to Gemma's expected format
            if role == Role.USER:
                gemma_role = "user"
            elif role == Role.ASSISTANT:
                gemma_role = "model"
            elif role == Role.IPYTHON:
                # Handle IPython as part of the user message
                # This might need adjustment based on your specific needs
                gemma_role = "ipython"
            else:
                # Skip any unrecognized roles
                continue

            # Start a new turn
            gemma_format += f"<start_of_turn>{gemma_role}\n"

            # Process content, handling any possible images
            processed_content = content
            if role == Role.USER and self.base64_images and gemma_role == "user":
                # If there are base64_images and this is a user message,
                # we need to embed them using <img> tags
                for i, image in enumerate(self.base64_images):
                    # Look for image placeholders in the pattern "Image X: "
                    image_placeholder = f"Image {chr(65+i)}: "
                    if image_placeholder in processed_content:
                        # Replace the placeholder with the image tag
                        processed_content = processed_content.replace(
                            image_placeholder,
                            f"Image {chr(65+i)}: <img>\n"
                        )

            # Add the processed content and end the turn
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
        if chat.messages[0][0] == Role.SYSTEM:
            messages_to_add = chat.messages[1:]
        else:
            messages_to_add = chat.messages
        # Doing it like this will take care of duplicate roles
        for message_to_add in messages_to_add:
            self.add_message(message_to_add[0], message_to_add[1])

        self.base64_images.extend(chat.base64_images)

        # Merge metadata, prioritizing values from the current chat instance
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
        # Find the last user message index
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i][0] == Role.USER:
                self.messages[i] = (Role.USER, new_content)
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
        # Lazy import to avoid circular imports
        try:
            from core.llm_router import LlmRouter
        except ImportError:
            logger.warning("LlmRouter not available for compression, returning copy of original chat")
            return self.deep_copy()

        # Validate level parameter
        if level not in ['dense', 'detailed']:
            logger.warning(f"Invalid compression level '{level}', defaulting to 'detailed'")
            level = 'detailed'

        # Create compression prompt based on level
        if level == 'dense':
            compression_instruction = (
                "Create a brief, essential summary of this conversation. "
                "Focus on key decisions, important facts, and critical context needed to continue the discussion. "
                "Be concise but preserve the most important information."
            )
        else:  # detailed
            compression_instruction = (
                "Create a comprehensive summary of this conversation that maintains all important details, "
                "decisions, technical context, and key points. The summary should provide full context "
                "for someone to continue the conversation seamlessly."
            )

        chat = Chat(compression_instruction)
        chat.add_message(Role.USER, f"{prompt}\n\nConversation:\n{self.print_chat()}")

        try:
            router = LlmRouter.get_instance()
            response = router.generate_response(chat, model_name=model if model else None, stream=False)
        except Exception as e:
            logger.error(f"LLM compression failed: {e}")
            return self.deep_copy()

        system_msg = ""
        if self.messages and self.messages[0][0] == Role.SYSTEM:
            system_msg = self.messages[0][1]

        memory_delimiter_prefix: str = "\n<context>"
        memory_delimiter_suffix: str = "</context>"

        if (memory_delimiter_prefix in system_msg): # cleanup previous chat memory in sys_msg
            system_msg = system_msg.split(memory_delimiter_prefix)[0]

        system_msg += memory_delimiter_prefix + response + memory_delimiter_suffix

        # Create new compressed chat
        compressed_chat = Chat()
        compressed_chat.set_instruction_message(system_msg)

        # Add compression metadata
        compressed_chat.metadata['compression'] = {
            'original_length': len(self),
            'compressed_length': len(compressed_chat),
            'level': level,
            'model': model,
            'prompt': prompt,
            'timestamp': time.time()
        }

        return compressed_chat