from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

class ToolResponse(TypedDict, total=False):
    reasoning: str
    tool: str
    reply: str
    command: str
    web_query: str
    title: str
    requirements: str
    status: str
    error: Optional[str]

@dataclass
class ToolMetadata:
    name: str
    description: str
    parameters: Dict[str, Any]
    required_params: List[str]
    example_usage: str

class BaseTool(ABC):
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Return the tool's metadata"""
        pass
    
    @property
    @abstractmethod
    def prompt_template(self) -> str:
        """Return the prompt template for the tool"""
        pass

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """Execute the tool with the given parameters"""
        pass

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate the parameters against the tool's requirements"""
        # Check both root level and under 'params' key
        if 'params' in params:
            params = params['params']
        if not all(param in params for param in self.metadata.required_params):
            return False
        return True

    @property
    def validation_rules(self) -> Optional[Dict[str, Any]]:
        """Optional validation rules for the tool parameters"""
        return None

    def format_response(self, reasoning: str, **kwargs) -> ToolResponse:
        """Helper method to format tool responses consistently"""
        response: ToolResponse = {
            "reasoning": reasoning,
            "tool": self.metadata.name
        }
        response.update(kwargs)
        return response 