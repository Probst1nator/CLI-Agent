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
    detailed_description: str
    constructor: str

class BaseTool(ABC):
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Return the tool's metadata"""
        pass
    
    @abstractmethod
    async def run(self, params: Dict[str, Any]) -> ToolResponse:
        """Execute the tool with the given parameters"""
        pass

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate the parameters against the tool's requirements"""
        # Check if parameters are in the correct structure
        if not isinstance(params, dict):
            return False
            
        # Get parameters from the parameters property
        parameters = params.get('parameters', {})
        if not isinstance(parameters, dict):
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