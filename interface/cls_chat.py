import json
import math
import os
from enum import Enum
from typing import Dict, List, Tuple, Union

from termcolor import colored
from jinja2 import Template
import tiktoken

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Chat:
    user_cli_agent_dir = os.path.expanduser('~/.local/share') + "/cli-agent"
    os.makedirs(user_cli_agent_dir, exist_ok=True)
    
    def __init__(self, instruction_message: str = ""):
        """
        Initializes a new Chat instance.
        
        :param instruction_message: The initial system instruction message.
        """
        self.messages: List[Tuple[Role, str]] = []
        if instruction_message:
            self.add_message(Role.SYSTEM, instruction_message)

    def add_message(self, role: Role, content: str) -> "Chat":
        """
        Adds a message to the chat.
        
        :param role: The role of the message sender.
        :param content: The content of the message.
        :return: The updated Chat instance.
        """
        if content and role:
            if (len(self.messages)>0):
                if self.messages[-1][0] == role:
                    self.messages[-1] = (role, self.messages[-1][1] + content)
                    return self
            self.messages.append((role, content))
        return self
    
    # def optimize(self, model: str = None, local: bool = None, iterations: int = 1, **kwargs):
    #     """
    #     Optimizes the chat by refining assistant responses based on suggestions.
        
    #     :param session: The session to use for generating completions.
    #     :param model: The model to use for generating completions.
    #     :param local: Whether to use local completions.
    #     :param iterations: The number of optimization iterations.
    #     """
    #     for iteration in range(iterations):
    #         for i, msg in enumerate(self.messages):
    #             if msg[0] == Role.SYSTEM:
    #                 continue
    #             if msg[0] == Role.USER:
    #                 prompt = msg[1]
    #             if msg[0] == Role.ASSISTANT:
    #                 response = msg[1]
                    
    #                 optimization_instructions: List[str]
    #                 if "```" in response:
    #                     optimization_instructions = [
    #                         "Suggested commands are required to run fully without requiring any user interaction. (Examples: `apt-get install -y python3` or `yes | command`)", 
    #                         "The commands are executed autonomously and in isolation from each other. Never rely on the state of the system from previous commands.", 
    #                         "The response should be as concise as possible while still being clear, do not include duplicate suggestions.",
    #                         "Commands need to be in a block starting with ```bash and ending with ```.",
    #                         "Only bash blocks are supported, other languages must always be contained within a bash command.",
    #                         "Multiline bash commands (e.g., when writing to files) must use 'EOF' or 'echo -e' to indicate multiline input.",
    #                         "Relative paths like './' are not supported (except for '~/'), ensure only absolute paths (or '~/') are used.",
    #                     ]
    #                     optimization_instruction = "\n".join([f"{i}. {inst}" for i,inst in enumerate(optimization_instructions)])
    #                 else:
    #                     optimization_instructions = [
    #                         "The response should be friendly and engaging.",
    #                         "The response should be concise and to the point.",
    #                         "The response should be informative and helpful.",
    #                         "The response should include emphatic emojis.",
    #                     ]
    #                     optimization_instruction = "\n".join([f"{i}. {inst}" for i,inst in enumerate(optimization_instructions)])
                    
    #                 chat = Chat("The system examines the request and rates the suggested response, it also provides feedback on how to improve the response.")
    #                 optimization_prompt = f"{optimization_instruction}\n```REQUEST\n{prompt}\n```\n```RESPONSE\n{response}\n```"
    #                 chat.add_message(Role.USER, optimization_prompt)
    #                 optimization_suggestions = LlmRouter.generate_completion(chat, silent=True, model=model, force_free=True, local=local, kwargs=kwargs)
    #                 chat.add_message(Role.ASSISTANT, optimization_suggestions)
                    
    #                 chat.messages[0] = (Role.SYSTEM, "The system incorporates the suggestions and provides a improved response.")
                    
    #                 chat.add_message(Role.USER, "I am going to repeat the request, please act on the suggestions and provide a enhanced response.")
    #                 chat.add_message(Role.ASSISTANT, "Sure! Please repeat the request and I will provide an improved response.")
    #                 chat.add_message(Role.USER, prompt)
    #                 improved_response = LlmRouter.generate_completion(chat, silent=True, model=model, force_free=True, local=local, kwargs=kwargs)
    #                 self.messages[i] = (Role.ASSISTANT, improved_response)
                
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
    
    def to_format_Mistral(self) -> str:
        """
        Converts chat messages to Mistral format.
        
        :return: The chat messages in Mistral format.
        """
        prompt:str = ""
        for msg in self.messages:
            if msg[0] == Role.USER:
                prompt += f"<s>[INST] {msg[1]}[/INST] "
            if msg[0] == Role.ASSISTANT:
                prompt += f"{msg[1]}</s>"
        return prompt
        
    def to_format_ChatML(self) -> str:
        """
        Converts chat messages to ChatML format.
        
        :return: The chat messages in ChatML format.
        """
        for msg in self.messages:
            if msg[0] == Role.SYSTEM:
                prompt = f"system\n{msg[1]}\n"
            if msg[0] == Role.USER:
                prompt += f"user\n{msg[1]}\n"
            if msg[0] == Role.ASSISTANT:
                prompt += f"assistant\n{msg[1]}\n"
        return prompt.strip()

    def to_openai_chat(self) -> List[Dict[str, str]]:
        """
        Converts chat messages to OpenAI chat format.
        
        :return: The chat messages in OpenAI chat format.
        """
        return [
            {"role": message[0].value, "content": message[1]}
            for message in self.messages
        ]

    def to_oobabooga_history(self) -> Tuple[Dict[str, List[List[str]]], str]:
        """
        Converts chat messages to Oobabooga history format.
        
        :return: A tuple containing internal and visible history arrays and the instruction.
        """
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
        """
        Renders chat messages using a Jinja2 template.
        
        :param template_str: The Jinja2 template string.
        :return: The rendered string.
        """
        template = Template(template_str)
        formatted_message = ""
        for i in range(math.ceil(len(self.messages) / 2)):
            formatted_message += template.render(
                {
                    "system": self.messages[i * 2][1],
                    "prompt": self.messages[i * 2 + 1][1],
                }
            )
        return formatted_message

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
        file_path = os.path.join(self.user_cli_agent_dir,file_name)
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
        file_path = os.path.join(cls.user_cli_agent_dir,file_name)
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
        return json.dumps(self._to_dict())

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
    
    def to_groq_format(self) -> List[Dict[str, str]]:
        """
        Formats the chat messages for Groq API consumption.
        
        :return: The formatted messages as a list of dictionaries.
        """
        return [
            {"role": message[0].value, "content": message[1]}
            for message in self.messages
        ]

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
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(self.joined_messages()))
        return math.floor(num_tokens*1.05) # 0.05 added as grace heuristic because we're likely using the incorrect embedding
