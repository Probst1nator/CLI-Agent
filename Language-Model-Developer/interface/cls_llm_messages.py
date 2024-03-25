import json
import math
import re
from enum import Enum
from typing import Dict, List, Tuple

from jinja2 import Template


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Chat:
    def __init__(self, instruction_message: str = ""):
        self.messages: List[Tuple[Role, str]] = []
        if instruction_message:
            self.add_message(Role.SYSTEM, instruction_message)

    def add_message(self, role: Role, content: str):
        if len(self.messages) == 0 and role == Role.USER:
            self.messages.append((Role.ASSISTANT, ""))
        if len(self.messages) > 0 and self.messages[-1][0] == role:
            raise Exception("Two messages of the same role cannot be sent in sequence")
        self.messages.append((role, content))
        return self

    def update_last_message(self, content: str):
        if self.messages:
            last_message_role, _ = self.messages[-1]
            self.messages[-1] = (last_message_role, content)
        else:
            raise Exception("No messages to update")

    def __getitem__(self, key):
        return self.messages[key]

    def __str__(self):
        return json.dumps(
            [
                {"role": message[0].value, "content": message[1]}
                for message in self.messages
            ]
        )

    def to_coloured_string(self) -> str:
        COLORED_TEXT = {
            "assistant": "\033[94m",  # Blue
            "system": "\033[91m",  # Red
            "user": "\033[92m",  # Green
            "reset": "\033[0m",  # Reset color
        }

        colored_messages = "\n\n".join(
            [
                COLORED_TEXT[message[0].value]
                + message[0].value.upper()
                + ": "
                + message[1]
                + COLORED_TEXT["reset"]
                for message in self.messages
            ]
        )

        return colored_messages

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
        corrected_template_str = re.sub(
            r"\{\{\s*if\s*(.*?)\s*\}\}", r"{% if \1 %}", template_str
        )
        corrected_template_str = re.sub(
            r"\{\{\s*endif\s*\}\}", r"{% endif %}", corrected_template_str
        )
        corrected_template_str = re.sub(
            r"\{\{\s*else\s*\}\}", r"{% else %}", corrected_template_str
        )

        # Create the template with the corrected string
        template = Template(corrected_template_str)
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
