from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
import asyncio

from py_classes.cls_chat import Chat

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
    def default_timeout(self) -> float:
        """Default timeout in seconds for tool execution"""
        return 30.0
    
    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        """Return the tool's metadata"""
        pass
    
    @abstractmethod
    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        """The actual implementation of the tool execution logic"""
        pass
    
    async def run(self, params: Dict[str, Any], context_chat: Chat, timeout: Optional[float] = None) -> ToolResponse:
        """Execute the tool with the given parameters and a timeout
        
        Args:
            params: The parameters for tool execution
            timeout: Optional custom timeout in seconds. If None, uses the default_timeout.
                     Set to 0 to disable timeout.
        
        Returns:
            ToolResponse: The result of the tool execution or a timeout error
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        
        try:
            if effective_timeout <= 0:  # No timeout
                return await self._run(params, context_chat)
            else:
                return await asyncio.wait_for(self._run(params, context_chat), timeout=effective_timeout)
        except asyncio.TimeoutError:
            return self.format_response(
                "error", 
                f"Tool execution timed out after {effective_timeout} seconds"
            )

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