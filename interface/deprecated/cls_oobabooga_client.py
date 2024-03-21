import asyncio
import html
import json
import re
import sys
from typing import Dict, List

import requests
import websockets

# Assuming interface.cls_llm_messages and other dependencies are correctly set up
from interface.cls_chat import Chat, Role

# Constants for WebSocket and REST URIs
WEBSOCKET_HOST = "localhost:5005"
WEBSOCKET_PROMPT_URI = f"ws://{WEBSOCKET_HOST}/api/v1/stream"
WEBSOCKET_CHAT_URI = f'ws://{WEBSOCKET_HOST}/api/v1/chat-stream'
REST_HOST = "localhost:5000"
REST_URI = f"http://{REST_HOST}/api/v1/model"

class OobaboogaClient:
    def __init__(self):
        # Initialize and list available models
        print("\n".join(self._websocket_model_list()))
        loaded_model = self._websocket_loaded_model()
        if loaded_model == "None":
            # Load the first available model if none is loaded
            self._websocket_model_load(self._websocket_model_list()[0])
            print(f"LOADING MODEL:\t{self._websocket_model_list()[0]}")

    # Text generation using WebSocket
    async def _websocket_generate_text_stream(self, prompt: str, max_new_tokens: int) -> str:
        # Configuration for text generation
        action_data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "auto_max_new_tokens": False,
            "max_tokens_second": 0,
            "preset": "None",
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.1,
            "typical_p": 1,
            "epsilon_cutoff": 0,
            "eta_cutoff": 0,
            "tfs": 1,
            "top_a": 0,
            "repetition_penalty": 1.18,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "repetition_penalty_range": 0,
            "top_k": 40,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "penalty_alpha": 0,
            "length_penalty": 1,
            "early_stopping": False,
            "mirostat_mode": 0,
            "mirostat_tau": 5,
            "mirostat_eta": 0.1,
            "grammar_string": "",
            "guidance_scale": 1,
            "negative_prompt": "",
            "seed": -1,
            "add_bos_token": True,
            "truncation_length": 2048,
            "ban_eos_token": False,
            "custom_token_bans": "",
            "skip_special_tokens": True,
            "stopping_strings": [],
        }

        full_response = ""
        async for response in self._websocket_api(action_data):
            full_response += response
            print(response, end="")
            sys.stdout.flush()
        return full_response

    # Chat generation using WebSocket
    async def _websocket_generate_chat_stream(self, user_input: str, history: Dict[str, List], instruction: str = "") -> str:
        # Configuration for chat generation
        request = {
            'user_input': user_input,
            'max_new_tokens': 250,
            'auto_max_new_tokens': False,
            'max_tokens_second': 0,
            'history': history,
            'mode': 'instruct',  # Valid options: 'chat', 'chat-instruct', 'instruct'
            'character': 'Example',
            'instruction_template': 'Vicuna-v1.1',  # Will get autodetected if unset
            'your_name': 'You',
            # 'name1': 'name of user', # Optional
            # 'name2': 'name of character', # Optional
            # 'context': 'character context', # Optional
            # 'greeting': 'greeting', # Optional
            # 'name1_instruct': 'You', # Optional
            # 'name2_instruct': 'Assistant', # Optional
            # 'context_instruct': 'context_instruct', # Optional
            # 'turn_template': 'turn_template', # Optional
            'regenerate': False,
            '_continue': True,
            'chat_instruct_command':  instruction if instruction else 'Continue the chat dialogue below. Write a single reply for the character "<|character|>".\n\n<|prompt|>' ,

            # Generation params. If 'preset' is set to different than 'None', the values
            # in presets/preset-name.yaml are used instead of the individual numbers.
            'preset': 'None',
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.1,
            'typical_p': 1,
            'epsilon_cutoff': 0,  # In units of 1e-4
            'eta_cutoff': 0,  # In units of 1e-4
            'tfs': 1,
            'top_a': 0,
            'repetition_penalty': 1.18,
            'presence_penalty': 0,
            'frequency_penalty': 0,
            'repetition_penalty_range': 0,
            'top_k': 40,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mirostat_eta': 0.1,
            'grammar_string': '',
            'guidance_scale': 1,
            'negative_prompt': '',

            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 2048,
            'ban_eos_token': False,
            'custom_token_bans': '',
            'skip_special_tokens': True,
            'stopping_strings': []
        }

        full_response = ""
        cur_len = 0
        async for response in self._websocket_api(request, True):
            cur_message = html.unescape(response['visible'][-1][1][cur_len:])
            cur_len += len(cur_message)
            full_response += cur_message
            print(cur_message, end="")
            sys.stdout.flush()
        return full_response

    # Decode custom escape sequences
    def decode_custom_escape_sequences(self, input_string: str):
        pattern = r'&#x([0-9A-Fa-f]+);'
        def replace(match):
            char_code = int(match.group(1), 16)
            return chr(char_code)
        return re.sub(pattern, replace, input_string)

    # WebSocket API communication
    async def _websocket_api(self, action_data: dict, chat_mode: bool = False):
        uri = WEBSOCKET_CHAT_URI if chat_mode else WEBSOCKET_PROMPT_URI
        async with websockets.connect(uri, ping_interval=None) as websocket:
            await websocket.send(json.dumps(action_data))
            while True:
                incoming_data_json = await websocket.recv()
                incoming_data = json.loads(incoming_data_json)
                match incoming_data['event']:
                    case 'text_stream':
                        yield incoming_data['history' if chat_mode else 'text']
                    case 'stream_end':
                        return

    # REST API communication
    def _rest_api(self, action_data: dict) -> dict:
        response = requests.post(REST_URI, json=action_data)
        if response.status_code == 200:
            return response.json()
        else:
            print(json.dumps(response))

    # Load a model using WebSocket
    def _websocket_model_load(self, model_name: str) -> dict:
        action_data = {"action": "load", "model_name": model_name}
        return self._rest_api(action_data)

    # List available models
    def _websocket_model_list(self) -> List[str]:
        action_data = {"action": "list"}
        models = self._rest_api(action_data)["result"]
        models.remove("None")
        return models

    # Get information about the loaded model
    def websocket_model_info(self) -> dict:
        action_data = {"action": "info"}
        return self._rest_api(action_data)

    # Get the name of the currently loaded model
    def _websocket_loaded_model(self) -> str:
        return self.websocket_model_info()["result"]["model_name"]

    # Unload the current model
    def _websocket_model_unload(self) -> dict:
        action_data = {"action": "unload"}
        return self._rest_api(action_data)

    # Stop text generation
    def _websocket_stop_text_generation(self) -> dict:
        action_data = {"action": "stop-stream"}
        return self._rest_api(action_data)

    # Count tokens in a prompt
    def _websocket_count_tokens_in_prompt(self, prompt: str) -> dict:
        action_data = {"action": "token-count", "prompt": prompt}
        return self._rest_api(action_data)

    # Mimic the prompt_model method of ModelPrompterClient
    async def prompt_model(self, prompt: str, max_new_tokens: int = 512) -> str:
        return await self._websocket_generate_text_stream(prompt, max_new_tokens)

    # Mimic the list_available_models method of ModelPrompterClient
    def list_available_models(self) -> List[str]:
        return self._websocket_model_list()

    # High-level method to handle chat
    def websocket_chat(self, chat: Chat) -> Chat:
        history, instruction = chat.to_oobabooga_history()
        generation = asyncio.run(self._websocket_generate_chat_stream("", history, instruction))
        chat.add_message(Role.ASSISTANT, generation)
        return chat
