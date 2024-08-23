import json
import math
import os
from enum import Enum
from typing import Dict, List, Sequence, Tuple, Union

from termcolor import colored

from globals import g
from ollama._types import Message

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Chat:
    def __init__(self, instruction_message: str = ""):
        """
        Initializes a new Chat instance.
        
        :param instruction_message: The initial system instruction message.
        """
        self.messages: List[Tuple[Role, str]] = []
        self.base64_images: List[str] = []
        if instruction_message:
            self.add_message(Role.SYSTEM, instruction_message)

    def add_message(self, role: Role, content: str) -> "Chat":
        """
        Adds a message to the chat.
        
        :param role: The role of the message sender.
        :param content: The content of the message.
        :param used_model: The model used to generate the message.
        :return: The updated Chat instance.
        """
        if content and role:
            if (len(self.messages)>0):
                if self.messages[-1][0] == role:
                    self.messages[-1] = (role, self.messages[-1][1] + content)
                    return self
            self.messages.append((role, content))
        return self
    
    def get_messages_as_string(self, start_index: int, end_index: int = -1) -> str:
        selected_messages = self.messages[start_index:end_index]
        return "\n".join([f"{message[0].name}: {message[1]}" for message in selected_messages])
        
    
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
        Prints the chat messages with colored roles.
        """
        for role, content in self.messages:
            role_value = role.value if isinstance(role, Role) else role
            if role in {Role.ASSISTANT, Role.SYSTEM}:
                formatted_content = colored(content, 'blue')
                print(f"{formatted_content} :{role_value}")
            else:
                formatted_role = colored(role_value, 'green')
                print(f"{formatted_role}: {content}")

    def save_to_json(self, file_name: str = "recent_chat.json", append: bool = False):
        """
        Saves the chat instance to a JSON file.
        
        :param file_name: The name of the file to save to.
        :param append: Whether to append to the file.
        """
        file_path = os.path.join(g.PROJ_VSCODE_DIR_PATH,file_name)
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
        file_path = os.path.join(g.PROJ_VSCODE_DIR_PATH, file_name)
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
        return "\n".join(content for _, content in self.messages)
    
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
        return [
            {"role": message[0].value, "content": message[1]}
            for message in self.messages
        ]
    def to_groq(self) -> List[Dict[str, str]]:
        """
        Formats the chat messages for Groq API consumption.
        
        :return: The formatted messages as a list of dictionaries.
        """
        return [
            {"role": message[0].value, "content": message[1]}
            for message in self.messages
        ]