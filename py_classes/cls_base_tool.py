from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict
import asyncio

from py_classes.cls_chat import Chat

class ToolStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL_SUCCESS = "partial_success"

class ToolResponse(TypedDict):
    status: ToolStatus  # SUCCESS | ERROR
    summary: str  # A descriptive summary of what happened, including any error details if status is "error"
    tool: str    # Added automatically by format_response
    followup_tools: Optional[List[str]]  # Optional list of suggested follow-up tools

@dataclass
class ToolMetadata:
    name: str
    description: str
    detailed_description: str
    constructor: str
    default_followup_tools: List[str] = field(default_factory=list)
    is_followup_only: bool = False  # If True, this tool is only visible when referenced as a followup tool

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
                result = await self._run(params, context_chat)
            else:
                result = await asyncio.wait_for(self._run(params, context_chat), timeout=effective_timeout)
            result.update(followup_tools=self.metadata.default_followup_tools)
            return result
        
        except asyncio.TimeoutError:
            return self.format_response(
                ToolStatus.ERROR, 
                f"Tool execution timed out after {effective_timeout} seconds",
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

    def format_response(self, status: ToolStatus, summary: str, followup_tools: Optional[List[str]] = None) -> ToolResponse:
        """Format tool responses consistently with required status and summary.
        
        Args:
            status (ToolStatus): Either ToolStatus.SUCCESS or ToolStatus.ERROR
            summary (str): A descriptive summary of what happened, including any error details if status is "error"
            followup_tools (Optional[List[str]]): Optional list of tool names that would be appropriate to use next
        
        Returns:
            ToolResponse: A consistent response format for all tools
        """
            
        response: ToolResponse = {
            "status": status.value,
            "summary": summary,
            "tool": self.metadata.name,
            "followup_tools": list(set(self.metadata.default_followup_tools + (followup_tools if followup_tools else [])))
        }
            
        return response 