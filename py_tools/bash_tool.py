import subprocess
from typing import Any, Dict
import re
import sys
import time
from termcolor import colored

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse
from py_classes.cls_llm_router import LlmRouter
from py_methods.tooling import listen_microphone, text_to_speech

class BashTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="bash",
            description="Execute bash commands safely",
            parameters={
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                }
            },
            required_params=["command"],
            example_usage="""
            {
                "reasoning": "Need to list files in the current directory",
                "tool": "bash",
                "command": "ls -la"
            }
            """
        )

    @property
    def prompt_template(self) -> str:
        return """Use the bash tool to execute shell commands.
Always include clear reasoning for what the command will be doing.
You can use this tool to combine multiple commands, gather system information, and more...

Example:
{
    "reasoning": "I need to check the current date and time, and get the weather in my area",
    "tool": "bash",
    "command": "date "+%Y-%m-%d %H:%M:%S" && curl -s wttr.in/?format=3"
}"""


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

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status="error",
                summary="Missing required parameter: command"
            )

        command = params["command"]

        # Get global args from the agent context
        from py_classes.globals import g
        args = g.get_args()  # This ensures we have all args with proper defaults
        
        # LLM safety check
        is_safe = self._check_command_safety(command, force_local=args.local)
        
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