import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class AnthropicChat:
    @staticmethod
    def generate_response(chat, model="claude-3-5-sonnet-20240620", temperature=0.7, silent=False):
        """
        Generates a response using the Anthropic API based on the provided model and messages, with error handling and retries.

        :param model: The model string to use for generating the response.
        :param chat: An instance of a chat class containing the messages.
        :param temperature: Controls the randomness of the response.
        :param silent: If True, suppresses print statements.
        :return: A string containing the generated response.
        """
        try:
            client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            
            if not silent:
                print("Anthropic API is generating response... using model: " + model)

            response = client.completions.create(
                model=model,
                messages=chat.to_anthropic_format(),
                temperature=temperature,
                stream=True,
            )

            full_response = ""
            for chunk in response:
                token = chunk.choices[0].delta.content
                if token:
                    if not silent:
                        print(token, end="")
                    full_response += token
            if not silent:
                print()
            return full_response
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return None
