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
            force_local=force_local,
            silent_reasoning=True
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
                "Invalid parameters provided",
                status="error",
                error="Missing required parameter: command"
            )

        command = params["command"]

        # Basic pattern validation
        if not self._validate_command(command):
            return self.format_response(
                "Command validation failed",
                status="error",
                error="Command contains unsafe patterns or exceeds length limit"
            )

        # Get global args from the agent context
        from py_classes.globals import g
        args = g.get_args()  # This ensures we have all args with proper defaults
        
        # LLM safety check
        is_safe = self._check_command_safety(command, force_local=args.local)
        
        # Print command for visibility
        print(colored(command, 'magenta'))
        
        # Handle command execution based on safety and auto mode
        if not is_safe or args.auto is None:
            # For unsafe commands or when not in auto mode, always prompt
            if args.voice or args.speak:
                confirmation_prompt = "Do you want me to execute these steps? (Yes/no)"
                if args.speak:
                    text_to_speech(confirmation_prompt)
                if args.voice:
                    user_input, _, _ = listen_microphone(10)
                else:
                    user_input = input(colored(f"{confirmation_prompt} ", 'yellow')).lower()
            else:
                user_input = input(colored("Do you want me to execute these steps? (Y/n) ", 'yellow')).lower()
                
            if not (user_input == "" or user_input == "y" or "yes" in user_input or "sure" in user_input or "ja" in user_input):
                return self.format_response(
                    "Command execution cancelled by user",
                    status="cancelled",
                    command=command
                )
        
        # If we're in auto mode and command is safe, or user approved execution
        if args.auto is not None:
            print(colored(f"Command will be executed in {args.auto} seconds, press Ctrl+C to abort.", 'yellow'))
            try:
                for remaining in range(args.auto, 0, -1):
                    sys.stdout.write("\r" + colored(f"Executing in {remaining} seconds... ", 'yellow'))
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\n")
            except KeyboardInterrupt:
                return self.format_response(
                    "Command execution aborted by user",
                    status="cancelled",
                    command=command
                )

        # Execute command
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True
            )
            return self.format_response(
                reasoning=params.get("reasoning", "Command executed successfully"),
                status="completed",
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                command=command
            )

        except Exception as e:
            return self.format_response(
                reasoning="Error executing command",
                status="error",
                error=str(e),
                command=command
            )