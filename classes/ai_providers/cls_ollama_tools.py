from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class Ollama_Parameter:
    name: str
    type: str
    description: str
    required: bool = True

@dataclass
class Ollama_FunctionSpec:
    name: str
    description: str
    parameters: List[Ollama_Parameter]

    @property
    def required_parameters(self) -> List[str]:
        return [param.name for param in self.parameters if param.required]

@dataclass
class Ollama_Tool:
    function: Ollama_FunctionSpec
    type: str = field(default="function", init=False)

    @classmethod
    def create(cls, name: str, description: str, parameters: List[Dict[str, Any]]) -> 'Ollama_Tool':
        function_parameters = [
            Ollama_Parameter(
                name=param['name'],
                type=param['type'],
                description=param['description'],
                required=param.get('required', True)
            )
            for param in parameters
        ]
        return cls(
            function=Ollama_FunctionSpec(
                name=name,
                description=description,
                parameters=function_parameters
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "function": {
                "name": self.function.name,
                "description": self.function.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param.name: {"type": param.type, "description": param.description}
                        for param in self.function.parameters
                    },
                    "required": self.function.required_parameters
                }
            }
        }

# Example usage
weather_tool = Ollama_Tool.create(
    name="get_current_weather",
    description="Get the current weather for a city",
    parameters=[
        {
            "name": "city",
            "type": "string",
            "description": "The name of the city"
        }
    ]
)

print(weather_tool)
print(weather_tool.to_dict())

# Creating another arbitrary tool
calculator_tool = Ollama_Tool.create(
    name="perform_calculation",
    description="Perform a mathematical calculation",
    parameters=[
        {
            "name": "operation",
            "type": "string",
            "description": "The mathematical operation to perform (add, subtract, multiply, divide)"
        },
        {
            "name": "x",
            "type": "number",
            "description": "The first number"
        },
        {
            "name": "y",
            "type": "number",
            "description": "The second number"
        }
    ]
)

print(calculator_tool)
print(calculator_tool.to_dict())