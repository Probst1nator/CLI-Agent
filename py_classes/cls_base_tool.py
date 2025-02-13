from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

class ToolResponse(TypedDict):
    status: str  # "success" | "error"
    summary: str  # A descriptive summary of what happened, including any error details if status is "error"
    tool: str    # Added automatically by format_response

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

    def format_response(self, status: str, summary: str) -> ToolResponse:
        """Format tool responses consistently with required status and summary.
        
        Args:
            status (str): Either "success" or "error"
            summary (str): A descriptive summary of what happened, including any error details if status is "error"
        
        Returns:
            ToolResponse: A consistent response format for all tools
        """
        if status not in ["success", "error"]:
            raise ValueError("status must be either 'success' or 'error'")
            
        return {
            "status": status,
            "summary": summary,
            "tool": self.metadata.name
        } 