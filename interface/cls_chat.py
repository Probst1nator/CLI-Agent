import json
import math
import os
from enum import Enum
from typing import Dict, List, Tuple, Union

from termcolor import colored
from jinja2 import Template


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Chat:
    user_cli_agent_dir = os.path.expanduser('~/.local/share') + "/cli-agent"
    os.makedirs(user_cli_agent_dir, exist_ok=True)
    def __init__(self, instruction_message: str = ""):
        self.messages: List[Tuple[Role, str]] = []
        if instruction_message:
            self.add_message(Role.SYSTEM, instruction_message)

    def add_message(self, role: Role, content: str) -> "Chat":
        self.messages.append((role, content))
        return self

    def __getitem__(self, key: Union[int, slice, Tuple[int, ...]]) -> "Chat":
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
        return json.dumps(
            [
                {"role": message[0].value, "content": message[1]}
                for message in self.messages
            ]
        )
    
    def to_format_Mistral(self):
            prompt:str = ""
            for msg in self.messages:
                if msg[0] == Role.USER:
                    prompt += f"<s>[INST] {msg[1]}[/INST] "
                if msg[0] == Role.ASSISTANT:
                    prompt += f"{msg[1]}</s>"
            return prompt
        
    def to_format_ChatML(self):
        for msg in self.messages:
            if msg[0] == Role.SYSTEM:
                prompt = f"system\n{msg[1]}\n"
            if msg[0] == Role.USER:
                prompt += f"user\n{msg[1]}\n"
            if msg[0] == Role.ASSISTANT:
                prompt += f"assistant\n{msg[1]}\n"
        return prompt.strip()

    def to_openai_chat(self) -> List[Dict[str, str]]:
        return [
            {"role": message[0].value, "content": message[1]}
            for message in self.messages
        ]

    def to_oobabooga_history(self) -> Tuple[Dict[str, List[List[str]]], str]:
        internal_arr: List[List[str]] = []
        visible_arr: List[List[str]] = []
        instruction: str = ""

        for i, (role, content) in enumerate(self.messages):
            if role == Role.SYSTEM:
                instruction = content if i == 0 else instruction
            elif role in [Role.USER, Role.ASSISTANT]:
                pair = [content, ""] if role == Role.USER else ["", content]
                internal_arr.append(pair)
                visible_arr.append(pair)

        return {"internal": internal_arr, "visible": visible_arr}, instruction

    def to_jinja2(self, template_str: str) -> str:
        # Create a template from the corrected string
        template = Template(template_str)
        formatted_message = ""
        # {"system": self.message[0].value, "prompt": self.message[1].value}
        for i in range(math.ceil(len(self.messages) / 2)):
            # role: str = i % 2 == 0 if "human" else "assistant"
            formatted_message += template.render(
                {
                    "system": self.messages[i * 2][1],
                    "prompt": self.messages[i * 2 + 1][1],
                }
            )

        return formatted_message

    def print_chat(self):
        """
        Print the chat messages with colored roles.
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
        """Save the Chat instance to a JSON file."""
        file_path = os.path.join(self.user_cli_agent_dir,file_name)
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if append:
            few_shot_prompts = Chat.load_from_json(file_name)
            few_shot_prompts.add_message(self.messages[0][0], self.messages[0][1])
            few_shot_prompts.add_message(self.messages[1][0], self.messages[1][1])
        else:
            few_shot_prompts = self

        # Convert the Chat object to a dictionary and save it as JSON
        with open(file_path, "w") as file:
            json.dump(few_shot_prompts._to_dict(), file, indent=4)


    @classmethod
    def load_from_json(cls, file_name: str = "recent_chat.json"):
        """Load a Chat instance from a JSON file."""
        file_path = os.path.join(cls.user_cli_agent_dir,file_name)
        with open(file_path, "r") as file:
            data_dict = json.load(file)
        return cls.from_dict(data_dict)

    def _to_dict(self):
        return [
                {"role": role.value, "content": content}
                for role, content in self.messages
            ]

    @classmethod
    def from_dict(cls, data_dict):
        chat_instance = cls()
        for message in data_dict:
            role = Role[message["role"].upper()]
            content = message["content"]
            chat_instance.add_message(role, content)
        return chat_instance
    
    def to_json(self) -> str:
        return json.dumps(self._to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        data_dict = json.loads(json_str)
        return cls.from_dict(data_dict)
    
    
    @staticmethod
    def save_to_jsonl(chats: List["Chat"], file_path: str = "saved_chat.jsonl") -> None:
        """
        Save a list of Chat instances to a JSONL file.

        Args:
            chats (List[Chat]): The list of Chat instances to save.
            file_path (str): The path to the JSONL file where chats will be saved.
        """
        if chats:
            with open(file_path, 'a') as file:
                for chat in chats:
                    # Convert each Chat instance to a dictionary and then to a JSON string
                    chat_json = chat.to_json()
                    file.write(chat_json + '\n')

    @staticmethod
    def load_from_jsonl(file_path: str) -> List["Chat"]:
        """
        Load a list of Chat instances from a JSONL file.

        Args:
            file_path (str): The path to the JSONL file to load chats from.

        Returns:
            List[Chat]: A list of Chat instances loaded from the JSONL file.
        """
        chats = []
        with open(file_path, 'r') as file:
            for line in file:
                chat_data = json.loads(line)
                chat = Chat.from_dict(chat_data)
                chats.append(chat)
        return chats
    
    
    def to_groq_format(self) -> List[Dict[str, str]]:
        """
        Format the chat messages for Groq API consumption.

        Returns:
            List[Dict[str, str]]: The formatted messages as a list of dictionaries.
        """
        formatted_messages = [
            {"role": message[0].value, "content": message[1]}
            for message in self.messages
        ]
        return formatted_messages
