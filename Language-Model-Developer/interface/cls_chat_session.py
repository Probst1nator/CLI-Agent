
from interface.cls_llm_messages import Chat, Role
from interface.cls_ollama_client import OllamaClient


class ChatSession:
    def __init__(
        self,
        instruction: str = "Anticipate user needs and conversation directions, responding in a manner that is both informative and practical.",
    ):
        self.client: OllamaClient = OllamaClient()
        self.chat = Chat(instruction)

    def generate_completion(
        self,
        prompt: str,
        start_response_with: str = "",
        model="openchat",
        temperature=0.8,
        # stream=False,
        **kwargs,
    ) -> str:
        self.chat.add_message(Role.USER, prompt)

        # Call the generate_completion method of the OllamaClient
        response = self.client.generate_completion(
            model=model,
            prompt=self.chat,
            start_response_with=start_response_with,
            temperature=temperature,
            # stream=stream,
            **kwargs,
        )
        # if (isinstance(response, str)):
        #     for msg in response:
        #         msg
        self.chat.add_message(Role.ASSISTANT, response)
        return response
