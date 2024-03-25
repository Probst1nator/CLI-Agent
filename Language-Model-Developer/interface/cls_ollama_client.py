import json
import logging
import os
import subprocess
import time
from enum import Enum
from typing import Any, Dict, Iterator, Optional

import requests
from jinja2 import Template

from interface.cls_llm_messages import Chat, Role
from interface.enum_available_models import AvailableModels

# Configurations
BASE_URL = "http://localhost:11434/api"
TIMEOUT = 960  # Timeout for API requests in seconds
OLLAMA_CONTAINER_NAME = "ollama"  # Name of the Ollama Docker container
OLLAMA_START_COMMAND = [
    "docker",
    "run",
    "-d",
    "--gpus=all",
    "-v",
    "ollama:/root/.ollama",
    "-p",
    "11434:11434",
    "--name",
    OLLAMA_CONTAINER_NAME,
    "ollama/ollama",
]

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ANSI escape codes for colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DARK_GREEN = "\033[32m"  # Darker green
DARK_YELLOW = "\033[33m"  # Darker yellow
DARK_RED = "\033[31m"  # Darker red
RESET = "\033[0m"  # Resets the color to default


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class OllamaClient(metaclass=SingletonMeta):
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self._ensure_container_running()
        self.cache_file = "./cache/ollama_cache.json"
        self.cache = self._load_cache()

    def _ensure_container_running(self):
        """Ensure that the Ollama Docker container is running."""
        if self._check_container_exists():
            if not self._check_container_status():
                logger.info("Restarting the existing Ollama Docker container...")
                self._restart_container()
        else:
            logger.info("Starting a new Ollama Docker container...")
            self._start_container()

    def _check_container_status(self):
        """Check if the Ollama Docker container is running."""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "inspect",
                    '--format="{{ .State.Running }}"',
                    OLLAMA_CONTAINER_NAME,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip().strip('"') == "true"
        except subprocess.CalledProcessError:
            return False

    def _check_container_exists(self):
        """Check if a Docker container with the Ollama name exists."""
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "--filter", f"name={OLLAMA_CONTAINER_NAME}"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() != ""

    def _restart_container(self):
        """Restart the existing Ollama Docker container."""
        subprocess.run(["docker", "restart", OLLAMA_CONTAINER_NAME], check=True)

    def _start_container(self):
        """Start the Ollama Docker container."""
        try:
            subprocess.run(OLLAMA_START_COMMAND, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                "Error starting the Ollama Docker container. Please check the Docker setup."
            )
            raise

    def _download_model(self, model_name: str):
        """Download the specified model if not available."""
        logger.info(f"Checking if model '{model_name}' is available...")
        if not self._is_model_available(model_name):
            logger.info(f"Model '{model_name}' not found. Downloading...")
            subprocess.run(
                ["docker", "exec", OLLAMA_CONTAINER_NAME, "ollama", "pull", model_name],
                check=True,
            )
            logger.info(f"Model '{model_name}' downloaded.")

    def _is_model_available(self, model_name: str) -> bool:
        """Check if a specified model is available in the Ollama container."""
        result = subprocess.run(
            ["docker", "exec", OLLAMA_CONTAINER_NAME, "ollama", "list"],
            capture_output=True,
            text=True,
        )
        return model_name in result.stdout

    def _load_cache(self):
        """Load cache from a file."""
        if not os.path.exists(self.cache_file):
            if not os.path.exists(os.path.dirname(self.cache_file)):
                os.makedirs(os.path.dirname(self.cache_file))
            with open(self.cache_file, "w") as cache_file:
                json.dump({}, cache_file)
        with open(self.cache_file, "r") as json_file:
            return json.load(json_file)

    def _get_cached_completion(self, model: str, temperature: str, prompt: str):
        """Retrieve cached completion if available."""
        cache_key = f"{model}:{temperature}:{prompt}"
        return self.cache.get(cache_key)

    def _update_cache(self, model: str, temperature: str, prompt: str, completion: str):
        """Update the cache with new completion."""
        cache_key = f"{model}:{temperature}:{prompt}"
        self.cache[cache_key] = completion
        with open(self.cache_file, "w") as json_file:
            json.dump(self.cache, json_file, indent=4)

    def _send_request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> requests.Response:
        """Send an HTTP request to the given endpoint."""
        url = f"{self.base_url}/{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, timeout=TIMEOUT)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=TIMEOUT)
            elif method == "DELETE":
                response = requests.delete(url, json=data, timeout=TIMEOUT)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            return response
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            raise

    def _get_template(self, model: str):
        data = {"name": model}
        response = self._send_request("POST", "show", data).json()
        if "error" in response:
            self._download_model(model)
            response = self._send_request("POST", "show", data).json()
        return response["template"]

    def test_response_time(self, model: str):
        start_time = time.time()
        response = self.generate_completion(
            "Please enumerate all numbers from 0 to 10.",
            model=model,
        )
        print("Time taken:", time.time() - start_time, "seconds")

    def list_models(self) -> Dict[str, Any]:
        """
        Lists all models available locally.

        Returns:
        Dict[str, Any]: A dictionary containing information about the local models.
        """
        response = self._send_request("GET", "tags")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(
                f"Error listing models: {response.status_code} - {response.text}"
            )
            return {}

    def generate_completion(
        self,
        prompt: str | Chat,
        model: AvailableModels = AvailableModels.EXPERT,
        start_response_with: str = "",
        instruction: str = "Provide expert assistance to fullfill user needs and context specific requirements, responding in an optimally precise and diverged exploratory mannner.",
        temperature: float = 0.8,
        include_start_response_with=True,
        # stream: bool = False,
    ) -> str:
        model_str: str = model.value
        template_str = self._get_template(model_str)
        template_str = template_str.replace(".Prompt", "prompt").replace(
            ".System", "system"
        )
        if isinstance(prompt, Chat):
            prompt_str = prompt.to_jinja2(template_str)
        else:
            template = Template(template_str)
            context = {"system": instruction, "prompt": prompt}
            prompt_str = template.render(context) + start_response_with

        prompt_str += start_response_with

        # Check cache first
        cached_completion = self._get_cached_completion(
            model_str, temperature, prompt_str
        )
        if cached_completion:
            if not isinstance(prompt, Chat):
                prompt_chat: Chat = Chat().add_message(Role.USER, prompt)
                print(
                    f"\n'''prompted to {model.value}\n"
                    + prompt_chat.to_coloured_string()
                    + "\n'''\n"
                )
            else:
                print(
                    f"\n'''prompted to {model.value}\n"
                    + prompt.to_coloured_string()
                    + "\n'''\n"
                )
            print(
                DARK_YELLOW
                + "'''start_response_with\n"
                + start_response_with
                + "\n'''\n"
                + RESET
            )
            print(
                DARK_RED
                + "'''cached_completion\n"
                + cached_completion["response"]
                + "\n'''\n"
                + RESET
            )
            if include_start_response_with:
                return start_response_with + cached_completion["response"]
            else:
                return cached_completion["response"]

        # If not cached, generate completion
        data = {
            "model": model_str,
            "prompt": prompt_str,
            "temperature": temperature,
            "raw": bool(instruction),
            "stream": False,
        }
        if not isinstance(prompt, Chat):
            prompt_chat: Chat = Chat().add_message(Role.USER, prompt)
            print(
                f"\n'''prompted to {model.value}\n"
                + prompt_chat.to_coloured_string()
                + "\n'''\n"
            )
        else:
            print(f"\n'''prompted to {model.value}\n" + prompt.to_coloured_string() + "\n'''\n")

        print(
            YELLOW
            + "'''start_response_with\n"
            + start_response_with
            + "\n'''\n"
            + RESET
        )
        response = self._send_request("POST", "generate", data)

        # Assuming response is a dictionary containing the completion
        # if stream:
        #     return response.iter_lines()
        # else:
        completion = response.json()
        if hasattr(completion, "error"):
            return completion["error"]

        # Update cache
        self._update_cache(model_str, temperature, prompt_str, completion)

        print(
            RED
            + "'''completion['response']\n"
            + completion["response"]
            + "\n'''\n"
            + RESET
        )
        if include_start_response_with:
            return start_response_with + completion["response"]
        else:
            return completion["response"]


ollama_client = OllamaClient()
