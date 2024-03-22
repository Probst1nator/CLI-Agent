import asyncio
import json
import os

import openai
from dotenv import load_dotenv

from interface.cls_chat import Chat, Role
from websocket_client.oobabooga_client import OobaboogaClient

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class LlmHostInterface:
    def __init__(self, model: str):
        self.model = model
        self.cache_file = "./cache/text_generations_cache.json"
        self.cache = self._load_cache()

    def _load_cache(self):
        if not os.path.exists(self.cache_file):
            with open(self.cache_file, "w") as cache_file:
                json.dump([], cache_file)
        with open(self.cache_file, "r") as json_file:
            return json.load(json_file)

    def _get_cached_text_generation(self, chat: Chat):
        for entry in self.cache:
            if entry.get("messages") == str(chat) and entry.get("model") == self.model:
                return entry.get("generated_text")
        return None

    def _add_to_cache(self, chat: Chat, generated_text: str):
        self.cache.append(
            {
                "model": self.model,
                "messages": str(chat),
                "generated_text": generated_text,
            }
        )
        with open(self.cache_file, "w") as json_file:
            json.dump(self.cache, json_file, indent=4)

    async def _send_prompt_async(self, chat: Chat, max_new_tokens: int) -> str:
        cached_text = self._get_cached_text_generation(chat)
        if cached_text:
            return cached_text

        if self.model != "gpt-3.5-turbo":
            return asyncio.run(
                OobaboogaClient().prompt_model(str(chat), max_new_tokens)
            )
        response = openai.ChatCompletion.create(
            model=self.model, messages=chat.to_openai_chat(), max_tokens=max_new_tokens
        )
        generated_text = response.choices[0].message.content
        self._add_to_cache(chat, generated_text)
        return generated_text

    def prompt(
        self,
        user_message: str,
        instruction_message: str = "",
        condition_assistant_response: str = "",
        condition_assistant_response_end: str = "",
        max_new_tokens: int = 1024,
    ) -> str:
        chat = Chat(user_message, instruction_message)
        chat.add_message(Role.ASSISTANT, condition_assistant_response)

        response = self._send_prompt(chat, max_new_tokens)
        response = self._post_process_response(response)

        if condition_assistant_response_end:
            response = f"{response.strip()} {condition_assistant_response_end}"
            response = self._send_prompt(response, max_new_tokens)

        return response

    def _post_process_response(self, response: str) -> str:
        if any(
            phrase in response.lower()
            for phrase in ["i hope", "have any questions", "let me know"]
        ):
            last_newline_index = response.rfind("\n")
            response = (
                response[:last_newline_index].strip()
                if last_newline_index != -1
                else response.strip()
            )
        return response
