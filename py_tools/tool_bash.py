import subprocess
from typing import Any, Dict
import re
import sys
import time
from termcolor import colored

from py_tools.cls_base_tool import BaseTool, ToolMetadata, ToolResponse
from py_classes.cls_llm_router import LlmRouter
from py_classes.cls_chat import Chat
from py_methods.tooling import listen_microphone, text_to_speech
from py_classes.globals import g

class BashTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="bash",
            description="Execute bash commands safely",
            detailed_description="""Use this tool when you need to:
- Execute system commands
- Manage files and directories
- Get system information
- Run command-line utilities
- Install packages
Perfect for:
- File operations
- System queries
- Process management
- Package installation
- Running system utilities""",
            constructor="""
def run(command: str) -> None:
    \"\"\"Execute a bash command safely.
    
    Args:
        command: The bash command to execute
    \"\"\"
"""
        )

    def _check_command_safety(self, command: str, force_local: bool = False) -> bool:
        """Check command safety using LLM"""
        command_guard_prompt = (
            "The following command must follow these guidelines:\n"
            "1. After execution it must exit fully automatically.\n"
            "2. It must not modify the operating system in major ways, although it is allowed to install trusted apt packages and updated software.\n"
            "Respond only with 'Safe' or 'Unsafe'\n\n"
            f"Command: {command}"
        )
        
        safety_response = LlmRouter.generate_completion(
            command_guard_prompt, 
            ["llama-guard"], 
            force_local=force_local
        )
        
        # Consider command safe if response indicates safe or has acceptable security levels
        is_safe = (
            "safe" in safety_response.lower() or
            "S8" in safety_response or  # Ignore Intellectual Property
            "S7" in safety_response     # Ignore Privacy
        )
        
        return is_safe

    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: command"
            )

        command = params["parameters"]["command"]

        # LLM safety check
        is_safe = self._check_command_safety(command, force_local=g.FORCE_LOCAL)
        
        # Print command for visibility
        print(colored(command, 'magenta'))
        
        # Handle unsafe commands
        if not is_safe:
            return self.format_response(
                status="error",
                summary=f"Command '{command}' was flagged as potentially unsafe"
            )

        # Execute command
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True
            )
            
            # Format output for summary
            output_summary = f"Exit code: {result.returncode}\n"
            if result.stdout:
                output_summary += f"Output: {result.stdout}\n"
            if result.stderr:
                output_summary += f"Errors: {result.stderr}"
                
            return self.format_response(
                status="success" if result.returncode == 0 else "error",
                summary=output_summary.strip()
            )

        except Exception as e:
            return self.format_response(
                status="error",
                summary=f"Error executing command '{command}': {str(e)}"
            )