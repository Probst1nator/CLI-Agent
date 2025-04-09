from typing import Any, Dict, Tuple, Optional
import tempfile
import os
import sys
import subprocess
import json
from termcolor import colored
import re
import time
import asyncio
import traceback
from io import StringIO

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse, ToolStatus
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import AIStrengths, LlmRouter
from py_methods.cmd_execution import select_and_execute_commands
from py_classes.globals import g
from py_methods.utils import extract_json

class PythonTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="python",
            description="Call this tool if you require to perform a calculation, implement visualizations or perform more complex tasks in a single python script. (NOT ALLOWED: Internet search, Api key requirements (use web_search tool instead))",
            detailed_description="""Use this tool when you need to:
- Create visualizations or plots
- Utilise advanced python libraries for data analysis, physics simulations, machine learning and mathematical computation or other""",
            constructor="""
def run(title: str, prompt: str) -> None:
    \"\"\"
    This tool is will implement and execute a python script.
    The script will be implemented automatically and the result will be returned.

    Args:
        title: A title for the python script.
        prompt: A high level, textual description of the python script to execute.
    \"\"\"
""",
            default_followup_tools=["python_edit"]
        )

    def _truncate_output(self, output: str, max_length: int = 1000, tail_length: int = 200) -> str:
        """
        Truncate long output to a reasonable length, keeping the last part.
        
        Args:
            output: The output string to truncate
            max_length: Maximum allowed length before truncation
            tail_length: Number of characters to keep from the end when truncating
            
        Returns:
            Truncated string with "..." prefix if it was truncated
        """
        if len(output) <= max_length:
            return output
            
        # Try to find the last newline within the tail section
        tail_start = len(output) - tail_length
        tail = output[tail_start:]
        last_newline = tail.rfind('\n')
        
        if last_newline != -1:
            # Found a newline in the tail, adjust tail_start to start at that newline
            tail_start = tail_start + last_newline + 1
            tail = output[tail_start:]
            
        return f"...{tail}"

    def execute_script(
        self,
        script_path: str
    ) -> Tuple[str, bool]:
        """
        Execute a Python script and handle the results.
        
        Args:
            script_path: Path to the script to execute
        
        Returns:
            Tuple of (execution_details: str, success: bool)
        """
        execution_details, execution_summary = select_and_execute_commands(
            [f"python3 {script_path}"],
            auto_execute=True,
            detached=False
        )
        
        truncated_execution_details = self._truncate_output(execution_details)
        
        if "error" in execution_summary.lower() or "error" in execution_details.lower():
            return  truncated_execution_details, False
        
        return truncated_execution_details, True


    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status=ToolStatus.ERROR,
                summary="Invalid parameters for python tool."
            )
        
        try:
            # Validate and process requirements
            title = params["parameters"]["title"]
            prompt = params["parameters"]["prompt"]
            # Create script directory
            script_dir = os.path.join(g.PROJ_PERSISTENT_STORAGE_PATH, "python_tool")
            os.makedirs(script_dir, exist_ok=True)
            
            # Generate unique script name using title and timestamp
            timestamp = int(time.time())
            base_name = os.path.splitext(title)[0]
            # Replace spaces and any other problematic characters with underscores
            safe_base_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
            script_name = f"{safe_base_name}_{timestamp}.py"
            script_path = os.path.join(script_dir, script_name)
            
            implement_prompt = f"""Please implement the following python script: {prompt}
Include a main guard, informative error handling and prints to display the execution states of the script. If the script needs to save anything to the disk, always use os.path.dirname(os.path.abspath(__file__)) as the root path."""
            if any(keyword in prompt.lower() for keyword in ["api", "key"]):
                implement_prompt += "\nThe description might indicate to use an API-key, please instead implement a solution that does not require an API-key and uses local ressources instead."
            
            implement_chat = Chat()
            implement_chat.add_message(Role.USER, implement_prompt)
            implementation_response = LlmRouter.generate_completion(implement_chat, strength=AIStrengths.CODE)
            
            # Clean response to ensure we only get code
            if "```python" in implementation_response or implementation_response.count("```") >= 2:
                if "```python" in implementation_response:
                    start_idx = implementation_response.find("```python") + 9
                else:
                    start_idx = implementation_response.find("```", start_idx) + 4
                end_idx = implementation_response.find("```", start_idx)
                if end_idx != -1:
                    implementation = implementation_response[start_idx:end_idx]
                else:
                    implementation = implementation_response[start_idx:]
            else:
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary=f"The tool failed to generate a python script. Please rephrase the prompt and try again.",
                )
            
            # Write implementation to file
            with open(script_path, "w") as f:
                f.write(implementation)
            
            # Count the number of python execution and pip install calls in the response
            python_execution_call_count = len(re.findall(r"```bash\s*python", implementation_response))
            pip_install_call_count = len(re.findall(r"```bash\s*pip", implementation_response))
            
            # If the response contains more bash commands than python execution commands, it is likely that the llm wants to execute an additional command
            if (implementation_response.count("```bash") > python_execution_call_count + pip_install_call_count):
                # Extract all bash commands from the implementation response
                bash_blocks = re.findall(r"```bash(.*?)```", implementation_response, re.DOTALL)
                additional_commands = "\n".join([block.strip() for block in bash_blocks])
                return self.format_response(
                    status=ToolStatus.PARTIAL_SUCCESS,
                    summary=f"{script_path} has been saved but not executed. The tool is asking to execute the following commands first:\n```bash\n{additional_commands}\n```",
                    followup_tools=["python_execute", "python_edit"]
                )
            
            # Execute script
            execution_details, success = self.execute_script(script_path)
            
            if success:
                return self.format_response(
                    status=ToolStatus.SUCCESS,
                    summary=f"{script_path} has been saved and executed successfully!\n{'Returned:' + execution_details if execution_details else ''}",
                    followup_tools=["python_edit"]
                )
            else:
                return self.format_response(
                    status=ToolStatus.PARTIAL_SUCCESS,
                    summary=f"{script_path} has been saved but executed unsuccessfully.\nYou can use the python_edit tool to edit the script and try again.\n{'Returned:' + execution_details if execution_details else ''}",
                    followup_tools=["python_edit"]
                )

        except Exception as e:
            error_traceback = traceback.format_exc()
            return self.format_response(
                status=ToolStatus.ERROR,
                summary=f"Error calling python tool, are the args valid?:\n{error_traceback}"
            )

# Test code when script is run directly
if __name__ == "__main__":
    async def test_python_tool():
        tool = PythonTool()
        chat = Chat()  # Create an empty chat for the context
        
        # Example parameters
        params = {
            "parameters": {
                "title": "Hello World",
                "prompt": "Create a simple script that prints 'Hello, World!' and the current date/time."
            }
        }
        
        # Execute the tool
        try:
            response = await tool.run(params, chat)
            print("Tool Response Status:", response["status"])
            print("Tool Response Summary:", response["summary"])
        except Exception as e:
            print(f"Error testing PythonTool: {str(e)}")

    # Run the async test function
    asyncio.run(test_python_tool())