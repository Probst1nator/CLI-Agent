import copy
import json
import math
import os
import logging
import tkinter as tk
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union
import threading
import queue
from termcolor import colored

from py_classes.globals import g
from ollama._types import Message

logger = logging.getLogger(__name__)

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    IPYTHON = "ipython"
    ASSISTANT = "assistant"

class Chat:
    def __init__(self, instruction_message: str = "", debug_title: Optional[str] = None):
        """
        Initializes a new Chat instance.
        
        :param instruction_message: The initial system instruction message.
        :param debug_title: Optional title for the debug window.
        """
        self.messages: List[Tuple[Role, str]] = []
        self.base64_images: List[str] = []
        self._window: Optional[tk.Tk] = None
        self._text_widget: Optional[tk.Text] = None
        self.debug_title: str = debug_title or instruction_message or "Unnamed Context"
        self._update_queue: Optional[queue.Queue] = None
        self._window_thread: Optional[threading.Thread] = None
        if self.debug_title == "Unnamed Context":
            pass
        if instruction_message:
            self.add_message(Role.SYSTEM, instruction_message)
    
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
        if not (logger.isEnabledFor(logging.DEBUG) or (hasattr(g, 'args') and g.args and g.args.debug_chats)):
            return

        if self._window is None and self._window_thread is None:
            self._create_debug_window()
            
        if self._update_queue is not None:
            try:
                self._update_queue.put_nowait(self.messages)
            except queue.Full:
                pass  # Skip update if queue is full
    
    def add_message(self, role: Role, content: str) -> "Chat":
        """
        Adds a message to the chat.
        
        :param role: The role of the message sender.
        :param content: The content of the message.
        :return: The updated Chat instance.
        """
        if content and role:
            if self.messages and self.messages[-1][0] == role:
                # If the last message has the same role, append the new content
                self.messages[-1] = (role, self.messages[-1][1] + '\n' + content)
            else:
                # Otherwise, add a new message
                self.messages.append((role, content))
            
            # Update window display if debug mode is enabled
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
    
    def __str__(self):
        """
        Returns a string representation of the chat messages.
        
        :return: A JSON string of the chat messages.
        """
        return json.dumps(
            [
                {"role": message[0].value, "content": message[1]}
                for message in self.messages
            ]
        )

    def print_chat(self):
        """
        Prints the chat messages with colored and bold roles, and similarly colored content using termcolor.
        """
        for role, content in self.messages:
            role_value = role.value if isinstance(role, Role) else role
            
            if role in {Role.ASSISTANT, Role.SYSTEM}:
                role_color = 'blue'
                content_color = 'cyan'
            elif role == Role.USER:
                role_color = 'light_green'
                content_color = 'green'
            else:
                role_color = 'light_yellow'
                content_color = 'yellow'

            formatted_role = colored(f"{role_value.upper()}:\n", role_color, attrs=['bold', "underline"])
            formatted_content = colored(content, content_color)
            
            print(f"{formatted_role} {formatted_content}")

    def save_to_json(self, file_name: str = "recent_chat.json", append: bool = False):
        """
        Saves the chat instance to a JSON file.
        
        :param file_name: The name of the file to save to.
        :param append: Whether to append to the file.
        """
        file_path = os.path.join(g.PROJ_PERSISTENT_STORAGE_PATH,file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if append:
            few_shot_prompts = Chat.load_from_json(file_name)
            few_shot_prompts.add_message(self.messages[0][0], self.messages[0][1])
            few_shot_prompts.add_message(self.messages[1][0], self.messages[1][1])
        else:
            few_shot_prompts = self

        with open(file_path, "w") as file:
            json.dump(few_shot_prompts._to_dict(), file, indent=4)

    @classmethod
    def load_from_json(cls, file_name: str = "recent_chat.json") -> "Chat":
        """
        Loads a Chat instance from a JSON file.
        
        :param file_name: The name of the file to load from.
        :return: The loaded Chat instance.
        """
        file_path = os.path.join(g.PROJ_PERSISTENT_STORAGE_PATH, file_name)
        with open(file_path, "r") as file:
            data_dict = json.load(file)
        return cls.from_dict(data_dict)

    def _to_dict(self) -> List[Dict[str, str]]:
        """
        Converts the chat instance to a dictionary.
        
        :return: A list of dictionaries representing the chat messages.
        """
        return [
                {"role": role.value, "content": content}
                for role, content in self.messages
            ]

    @classmethod
    def from_dict(cls, data_dict: List[Dict[str, str]]) -> "Chat":
        """
        Creates a Chat instance from a dictionary.
        
        :param data_dict: A dictionary containing chat messages.
        :return: The created Chat instance.
        """
        chat_instance = cls()
        for message in data_dict:
            role = Role[message["role"].upper()]
            content = message["content"]
            chat_instance.add_message(role, content)
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
    

    def length(self) -> int:
        """
        Returns the total length of all string messages in the chat.
        
        :return: The total length of all messages.
        """
        return sum(len(content) for _, content in self.messages)

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
        # try:
        #     encoding = tiktoken.get_encoding(encoding_name)
        #     num_tokens = len(encoding.encode(self.joined_messages()))
        # except Exception as e:
        #     print(colored(f"Error: tiktoken threw an error: {e}", "red"))
        #     return math.floor(len(self.joined_messages())/4)
        # return math.floor(num_tokens*1.05) # 0.05 added as grace heuristic because we're likely using the incorrect embedding

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
        Formats the chat messages for Groq API consumption.
        
        :return: The formatted messages as a list of dictionaries.
        """
        result = []
        for i, message in enumerate(self.messages):
            if message[0].value == "ipython" and i > 0:
                result[-1]["content"] += f"\n\n{message[1]}"
            else:
                result.append({"role": message[0].value, "content": message[1]})
        return result

    def deep_copy(self) -> 'Chat':
        """
        Creates a deep copy of the Chat instance.
        
        :return: A new Chat instance that is a deep copy of the current instance.
        """
        new_chat = Chat("Copy of " + self.debug_title)
        new_chat.messages = copy.deepcopy(self.messages)
        new_chat.base64_images = copy.deepcopy(self.base64_images)
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