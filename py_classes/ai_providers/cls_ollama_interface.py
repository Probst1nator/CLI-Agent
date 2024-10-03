import ast
from enum import Enum
import json
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import chromadb
import ollama
from termcolor import colored
from py_classes.cls_chat import Chat
from py_classes.cls_custom_coloring import CustomColoring
import os
from py_methods.logger import logger
from py_classes.cls_ai_provider_interface import ChatClientInterface
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime
from py_classes.globals import g

class ToolType(Enum):
    """
    Enumeration of tool types supported by the Ollama API.
    
    Attributes:
        FUNCTION (str): Represents a function tool type.
    """
    FUNCTION = "function"

@dataclass
class FunctionParameters:
    """
    Represents parameters for a function tool.
    
    Attributes:
        type (str): The type of the parameters object (usually "object").
        properties (Dict[str, Dict[str, Any]]): A dictionary of parameter properties.
        required (List[str]): A list of required parameter names.
    """
    type: str
    properties: Dict[str, Dict[str, Any]]
    required: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class FunctionDefinition:
    """
    Defines a function tool with its name, description, and parameters.
    
    Attributes:
        name (str): The name of the function.
        description (str): A description of what the function does.
        parameters (FunctionParameters): The parameters of the function.
    """
    name: str
    description: str
    parameters: FunctionParameters

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_dict()
        }

@dataclass
class FunctionTool:
    """
    Represents a complete function tool with its type and definition.
    
    Attributes:
        type (ToolType): The type of the tool (e.g., ToolType.FUNCTION).
        function (FunctionDefinition): The definition of the function.
    """
    type: ToolType
    function: FunctionDefinition

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,  # Convert enum to string
            "function": self.function.to_dict()
        }

@dataclass
class FunctionCall:
    """
    Represents a function call made by the model.
    
    Attributes:
        name (str): The name of the function being called.
        arguments (Dict[str, Any]): The arguments passed to the function.
    """
    name: str
    arguments: Dict[str, Any]

@dataclass
class ToolCall:
    """
    Represents a processed tool call with its type and function call details.
    
    Attributes:
        type (ToolType): The type of the tool being called.
        function (FunctionCall): The details of the function call.
    """
    function: FunctionCall

def ollama_convert_method_to_tool(method_string: str) -> FunctionTool:
    """
    Converts a Python method string into a FunctionTool object.

    Args:
    - method_string (str): A string containing the Python method to be converted.

    Returns:
    - FunctionTool: A FunctionTool object representing the method in the specified format.
    """
    # Parse the method string into an AST
    parsed = ast.parse(method_string)
    # Extract the function definition
    func_def = parsed.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("The provided string does not contain a valid function definition.")
    
    # Extract function name
    func_name = func_def.name
    
    # Extract docstring
    docstring = ast.get_docstring(func_def)
    
    # Extract parameters
    params = {
        "type": "object",
        "properties": {},
        "required": []
    }
    for arg in func_def.args.args:
        arg_name = arg.arg
        params["properties"][arg_name] = {
            "type": "string",  # Default to string, as we can't infer type from AST easily
            "description": f"Parameter: {arg_name}"
        }
        params["required"].append(arg_name)
    
    # Try to extract parameter types and descriptions from docstring
    if docstring:
        docstring_lines = docstring.split('\n')
        for line in docstring_lines:
            if ':param' in line:
                parts = line.split(':')
                if len(parts) >= 3:
                    param_name = parts[1].split()[1]
                    param_description = ':'.join(parts[2:]).strip()
                    if param_name in params["properties"]:
                        params["properties"][param_name]["description"] = param_description
    
    # Create the FunctionTool object
    function_parameters = FunctionParameters(
        type="object",
        properties=params["properties"],
        required=params["required"]
    )
    
    function_definition = FunctionDefinition(
        name=func_name,
        description=docstring.split('\n')[0] if docstring else "No description provided.",
        parameters=function_parameters
    )
    
    return FunctionTool(
        type=ToolType.FUNCTION,
        function=function_definition
    )

class OllamaClient(ChatClientInterface):
    """
    Implementation of the ChatClientInterface for the Ollama API.
    """
    reached_hosts: List[str] = []
    unreachable_hosts: List[str] = []

    @staticmethod
    def check_host_reachability(host: str) -> bool:
        """
        Validates if a host is reachable using a socket connection.
        
        Args:
            host (str): The hostname to validate.
        
        Returns:
            bool: True if the host is reachable, False otherwise.
        """
        try:
            hostname, port = host.split(':') if ':' in host else (host, 11434)
            print(colored(f"Checking host {host}...", "yellow"))
            with socket.create_connection((hostname, int(port)), timeout=3):
                return True
        except (socket.timeout, socket.error):
            print(colored(f"Host {host} is unreachable", "red"))
            return False

    @staticmethod
    def get_valid_client(model_key: str) -> Tuple[ollama.Client|None, str]:
        """
        Returns a valid client for the given model, pulling the model if necessary on auto-download hosts.
        
        Args:
            model_key (str): The model to find a valid client for.
        
        Returns:
            Tuple[Optional[ollama.Client], str]: [A valid client or None, found model_key].
        """
        ollama_hosts = [os.getenv(env_var) for env_var in os.environ if env_var.startswith("OLLAMA_HOST_") and env_var.count('_') == 2]
        auto_download_hosts = set(os.getenv(env_var) for env_var in os.environ if env_var.startswith("OLLAMA_HOST_AUTO_DOWNLOAD_MODELS_"))
        
        for host in ollama_hosts:
            if host not in OllamaClient.reached_hosts and host not in OllamaClient.unreachable_hosts:
                if OllamaClient.check_host_reachability(host):
                    OllamaClient.reached_hosts.append(host)
                else:
                    OllamaClient.unreachable_hosts.append(host)
            
            if host in OllamaClient.reached_hosts and host not in OllamaClient.unreachable_hosts:
                client = ollama.Client(host=f'http://{host}:11434')
                try:
                    model_list = OllamaModelList.from_json(json.dumps(client.list()))
                    found_model_key = next((model.name for model in model_list.models if model_key in model.name), None)
                    if found_model_key:
                        return client, found_model_key 
                    elif host in auto_download_hosts:
                        print(colored(f"{host} is pulling {model_key}...", "yellow"))
                        try:
                            def bytes_to_mb(bytes_value):
                                return bytes_value / (1024 * 1024)

                            for response in client.pull(model_key, stream=True):
                                if "status" in response:
                                    if response["status"] == "pulling manifest":
                                        status = colored("Pulling manifest...", "yellow")
                                    elif response["status"].startswith("pulling"):
                                        digest = response.get("digest", "")
                                        total = bytes_to_mb(response.get("total", 0))
                                        completed = bytes_to_mb(response.get("completed", 0))
                                        status = colored(f"Pulling {digest}: {completed:.2f}/{total:.2f} MB", "yellow")
                                    else:
                                        continue
                                    
                                    sys.stdout.write('\r' + status)
                                    sys.stdout.flush()
                            print()
                            return client, model_key
                        except Exception as e:
                            print(f"Error pulling model {model_key} on host {host}: {e}")
                except Exception as e:
                    print(f"Error checking models on host {host}: {e}")
                    OllamaClient.unreachable_hosts.append(host)
        
        print(f"No valid client found for model {model_key}")
        return None, None

    @staticmethod
    def generate_response(
        chat: Chat,
        model_key: str = "phi3.5:3.8b",
        temperature: Optional[float] = 0.75,
        silent: bool = False,
        tools: Optional[List[FunctionTool]] = None
    ) -> Optional[Union[str, List[ToolCall]]]:
        """
        Generates a response using the Ollama API, with support for tool calling.

        Args:
            chat (Chat): The chat object containing messages.
            model (str): The model identifier (e.g., "phi3.5:3.8b", "llama3.1").
            temperature (float): The temperature setting for the model.
            silent (bool): Whether to suppress print statements.
            tools (List[FunctionTool], optional): A list of tool definitions for the model to use.

        Returns:
            Optional[Union[str, List[ToolCall]]]: The generated response, or None if an error occurs.
        """
        options = ollama.Options()
        if "hermes" in model_key.lower():
            options.update(stop=["<|end_of_text|>"])
        if temperature:
            options.update(temperature=temperature)
        options.update(max_tokens=1500)
        
        # ! This is a bad solution because a remote machine likely has a different core count, but will work for the Pepper project as needed
        # cpu_core_count = os.cpu_count()-1 # lets leave one core for other processes
        # options.update(num_thread=cpu_core_count)
        
        tooling = CustomColoring()
        logger.debug(json.dumps({"last_message": chat.messages[-1][1]}, indent=2))

        client: ollama.Client | None
        client, model_key = OllamaClient.get_valid_client(model_key)
        if not client:
            logger.error(f"No valid host found for model {model_key}")
            return None
        assert client is not None
        host: str = client._client.base_url.host

        try:
            if silent:
                print(f"Ollama-Api: <{colored(model_key, 'green')}> is {colored('silently', 'green')} generating response using <{colored(host, 'green')}>...")
            else:
                print(f"Ollama-Api: <{colored(model_key, 'green')}> is generating response using <{colored(host, 'green')}>...")
            
            if tools:
                response = client.chat(
                    model=model_key,
                    messages=chat.to_ollama(),
                    stream=False,
                    options=options,
                    keep_alive=1800,
                    tools=[tool.to_dict() for tool in tools]
                )
                
                if "tool_calls" in response["message"]:
                    return [ToolCall(**tool_call) for tool_call in response["message"]["tool_calls"]]
                else:
                    return response["message"]["content"]
            else:
                response_stream = client.chat(model=model_key, messages=chat.to_ollama(), stream=True, options=options, keep_alive=1800)
                full_response = ""
                for line in response_stream:
                    next_string = line["message"]["content"]
                    full_response += next_string
                    if not silent:
                        print(tooling.apply_color(next_string), end="")
                if not silent:
                    print()
                logger.debug(json.dumps({"full_response": full_response}, indent=2))
                if "instruction" in full_response.lower():
                    full_response = full_response.split("instruction")[0]
                return full_response

        except Exception as e:
            print(f"Ollama-Api: Failed to generate response using <{colored(host, 'red')}> with model <{colored(model_key, 'red')}>: {e}")
            OllamaClient.unreachable_hosts.append(f"{host}{model_key}")
            logger.error(f"Ollama-Api: Failed to generate response using <{host}> with model <{model_key}>: {e}")
            return None

    @staticmethod
    def generate_response_raw(prompt: str, model: str = "nuextract", host: str = None) -> Optional[Dict[str, Any]]:
        """
        Generates a raw response from the Ollama API.

        Args:
            prompt (str): The input prompt.
            model (str): The model to use.
            host (str, optional): The specific host to use. If None, uses the first validated host.

        Returns:
            Optional[Dict[str, Any]]: The raw response from the API, or None if an error occurs.
        """
        if not host:
            client, model_key = OllamaClient.get_valid_client(model)
            if not client:
                raise ValueError("No validated Ollama hosts available")
        else:
            client = ollama.Client(host=f'http://{host}:11434')

        try:
            response = client.generate(model=model, prompt=prompt, stream=False, keep_alive=1800)
            return response
        except Exception as e:
            logger.error(f"Ollama-Api: Failed to generate raw response using <{host}> with model <{model}>: {e}")
            return None

    @staticmethod
    def generate_embedding(text: str, model: str = "bge-m3") -> Optional[List[float]]:
        """
        Generates an embedding for the given text using the specified Ollama model.
        
        Args:
            text (str): The input text to generate an embedding for.
            model (str): The embedding model to use.
        
        Returns:
            Optional[List[float]]: The generated embedding as a list of floats, or None if an error occurs.
        """
        if len(text)<3:
            return None
        client, model_key = OllamaClient.get_valid_client(model)
        if not client:
            logger.error(f"No valid host found for model {model}")
            return None
        assert client is not None
        host: str = client._client.base_url.host

        try:
            print(f"Ollama-Api: <{colored(model, 'green')}> is generating embedding using <{colored(host, 'green')}>...")
            response = client.embeddings(model=model, prompt=text, keep_alive=7200)
            embedding = response["embedding"]
            # ! Test: Storing all embeddings as long term memories (storing->preloading->RAG)
            client = chromadb.PersistentClient(g.PROJ_VSCODE_DIR_PATH)
            collection = client.get_or_create_collection(name="long-term-memories")
            if not collection.get(text):
                collection.add(
                    ids=[text],
                    embeddings=[embedding],
                    documents=[text]
                )
            with open(os.path.join(g.PROJ_VSCODE_DIR_PATH, "long_term_memory_textView.txt"), "a") as f:
                f.write(f"{text}\n\n# # #\n\n")
            # ! Test
            return embedding
        except Exception as e:
            print(f"Ollama-Api: Failed to generate embedding using <{colored(host, 'red')}> with model <{colored(model, 'red')}>: {e}")
            OllamaClient.unreachable_hosts.append(f"{host}{model}")
            logger.error(f"Ollama-Api: Failed to generate embedding using <{host}> with model <{model}>: {e}")
            return None

@dataclass
class OllamaDetails:
    """
    Represents details of an Ollama model.
    
    Attributes:
        format (str): The format of the model (e.g., 'gguf').
        family (str): The family of the model (e.g., 'llama').
        parameter_size (str): The parameter size of the model (e.g., '7B').
        quantization_level (str): The quantization level of the model (e.g., 'Q4_0').
        parent_model (str): The parent model, if any.
        families (Optional[List[str]]): List of model families, if multiple.
    """
    format: str
    family: str
    parameter_size: str
    quantization_level: str
    parent_model: str
    families: Optional[List[str]] = None
    
    def to_dict(self) -> dict:
        """
        Converts OllamaDetails to a dictionary.
        
        Returns:
            dict: A dictionary representation of the OllamaDetails instance.
        """
        return {
            "format": self.format,
            "family": self.family,
            "families": self.families,
            "parent_model": self.parent_model,
            "parameter_size": self.parameter_size,
            "quantization_level": self.quantization_level
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'OllamaDetails':
        """
        Creates an OllamaDetails instance from a dictionary.
        
        Args:
            data (dict): A dictionary containing OllamaDetails attributes.
        
        Returns:
            OllamaDetails: An instance of OllamaDetails.
        """
        return cls(**data)

@dataclass
class OllamaModel:
    """
    Represents an Ollama model with its details and metadata.
    
    Attributes:
        name (str): The name of the model.
        modified_at (str): The last modification timestamp of the model.
        size (int): The size of the model in bytes.
        digest (str): The digest (hash) of the model.
        details (OllamaDetails): Detailed information about the model.
    """
    name: str
    modified_at: str
    size: int
    digest: str
    details: OllamaDetails
    
    def to_dict(self) -> dict:
        """
        Converts OllamaModel to a dictionary.
        
        Returns:
            dict: A dictionary representation of the OllamaModel instance.
        """
        return {
            "name": self.name,
            "modified_at": self.modified_at,
            "size": self.size,
            "digest": self.digest,
            "details": self.details.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'OllamaModel':
        """
        Creates an OllamaModel instance from a dictionary.
        
        Args:
            data (dict): A dictionary containing OllamaModel attributes.
        
        Returns:
            OllamaModel: An instance of OllamaModel.
        """
        details = OllamaDetails.from_dict(data['details'])
        return cls(
            name=data['name'],
            modified_at=data['modified_at'],
            size=data['size'],
            digest=data['digest'],
            details=details
        )

@dataclass
class OllamaModelList:
    """
    Represents a list of Ollama models.
    
    Attributes:
        models (List[OllamaModel]): A list of OllamaModel instances.
    """
    models: List[OllamaModel] = field(default_factory=list)
    
    def to_json(self) -> str:
        """
        Converts OllamaModelList to a JSON string.
        
        Returns:
            str: A JSON string representation of the OllamaModelList.
        """
        return json.dumps({"models": [model.to_dict() for model in self.models]}, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'OllamaModelList':
        """
        Creates an OllamaModelList instance from a JSON string.
        
        Args:
            json_str (str): A JSON string containing a list of models.
        
        Returns:
            OllamaModelList: An instance of OllamaModelList.
        """
        data = json.loads(json_str)
        return cls(models=[OllamaModel.from_dict(model_data) for model_data in data['models']])