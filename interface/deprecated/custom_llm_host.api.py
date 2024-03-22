import asyncio
import json
import os
from typing import Dict, List

import torch
import websockets
from dotenv import load_dotenv
from transformers import AutoTokenizer, GPTQConfig, pipeline

load_dotenv()
model_directory = os.getenv("LOCAL_LLM_DIRECTORY")


class LLMHost:
    def __init__(self, model_path: str) -> None:
        self.device = "gpu"
        self.active_model_path: str = ""
        self.context_length: str = ""
        self.load_model(model_path)

    def load_model(self, modelName: str) -> None:
        try:
            model_path = model_directory + modelName
            if (
                self.active_model_path != model_path
                and modelName
                and os.path.exists(model_path)
            ):
                gptq_config = GPTQConfig(bits=4, disable_exllama=True)
                self.pipe = pipeline(
                    "text-generation",
                    model=model_path,
                    device=self.device,
                    quantization_config=gptq_config,
                ).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path).to(
                    self.device
                )
                self.active_model_path = model_path
                print("INFO: Switched to model: " + model_path)
        except Exception as e:
            print(f"Failed to load model: {str(e)}")

    def prompt(self, messages_json: str, max_new_tokens: int) -> str:
        try:
            with open("./cache/text_generations_cache.json", "r") as json_file:
                self.prompt_responses = json.load(json_file)
        except FileNotFoundError:
            self.prompt_responses = []

        for entry in self.prompt_responses:
            if entry.get("messages") == messages_json and entry.get(
                "model"
            ) == os.path.dirname(self.active_model_path):
                print(
                    self.tokenizer.apply_chat_template(
                        json.loads(messages_json),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                response = entry.get("response")
                print(
                    f"\n####################################################################"
                )
                print(
                    f"### LOADED_GENERATED_TEXT ## using ## {os.path.basename(self.active_model_path)} ###"
                )
                print(
                    f"####################################################################\n"
                )
                print(response)
                return response

        response = self.generate_response(json.loads(messages_json), max_new_tokens)
        self.prompt_responses.append(
            {
                "model": os.path.basename(self.active_model_path),
                "messages": messages_json,
                "response": response,
            }
        )
        with open("./cache/text_generations_cache.json", "w") as json_file:
            json.dump(self.prompt_responses, json_file, indent=4)

        return response

    def generate_response(
        self, messages: List[Dict[str, str]], max_new_tokens: int
    ) -> str:
        with torch.no_grad():
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt = "".join(prompt.split(self.tokenizer.eos_token)[:-1])
            print(prompt)
            print(
                f"####################################################################"
            )
            print(
                f"### GENERATED_TEXT ## using ## {os.path.basename(self.active_model_path)} ###"
            )
            print(
                f"####################################################################"
            )
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
            generated_text: str = outputs[0]["generated_text"].replace(prompt, "")
            print(generated_text)
        return generated_text


async def websocket_handler(websocket, path):
    print("\n\n--- REQUEST RECEIVED ---\n")
    try:
        if path.startswith("/list_directory"):
            await list_available_models(websocket)
        else:
            await prompt_model(websocket)
    except websockets.ConnectionClosedError:
        print("Client connection closed.")
    except Exception as e:
        print(f"WebSocket handler error: {str(e)}")


async def prompt_model(websocket):
    try:
        async for message in websocket:
            dict_model_prompt = json.loads(message)
            llm_host.load_model(dict_model_prompt["model"])
            messages_json: str = dict_model_prompt["prompt"]
            max_new_tokens: int = dict_model_prompt["max_new_tokens"]
            print(f"\n################################################")
            print("### RECEIVED_PROMPT ###")
            print(f"################################################\n")
            response = llm_host.prompt(messages_json, max_new_tokens)
            await websocket.send(response)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {str(e)}")
        await websocket.send(f"Error decoding JSON: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.send(f"Error: {str(e)}")


async def list_available_models(websocket):
    try:
        file_names = os.listdir(model_directory)
        await websocket.send(json.dumps(file_names))
    except Exception as e:
        print(f"Error listing available models: {str(e)}")
        await websocket.send(f"Error listing available models: {str(e)}")


if __name__ == "__main__":
    try:
        llm_host = LLMHost("default_model")  # Initialize LLMHost instance
        start_server = websockets.serve(websocket_handler, "0.0.0.0", 8765)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        asyncio.get_event_loop().run_until_complete(start_server.wait_closed())
