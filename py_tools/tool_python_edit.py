from typing import Any, Dict, Tuple, Optional
import os
import sys
import traceback

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse, ToolStatus
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import AIStrengths, LlmRouter
from py_methods.cmd_execution import select_and_execute_commands
from py_classes.globals import g

class PythonEditTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="python_edit",
            description="Edit an existing Python script based on modification instructions.",
            detailed_description="""Use this tool when you need to:
- Modify a previously created Python script
- Fix errors in an existing Python script
- Add new functionality to an existing script
- Refine or optimize code in a script
Perfect for:
- Iterative development
- Responding to feedback on script output
- Fixing bugs or errors identified in script execution
- Adding new features requested by the user""",
            constructor="""
def run(script_path: str, modification_prompt: str) -> None:
    \"\"\"
    Edit an and run an existing Python script based on modification instructions.
    
    Args:
        script_path: The path to the existing Python script to edit.
        modification_prompt: A description of the changes to make to the script.
    \"\"\"
""",
            followup_tools=["python_edit", "file_read"],
            is_followup_only=True
        )

    def execute_script(self, script_path: str) -> Tuple[str, bool]:
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
        
        # Truncate output if needed
        if len(execution_details) > 1000:
            tail_length = 200
            tail_start = len(execution_details) - tail_length
            tail = execution_details[tail_start:]
            last_newline = tail.rfind('\n')
            
            if last_newline != -1:
                tail_start = tail_start + last_newline + 1
                tail = execution_details[tail_start:]
                
            execution_details = f"...{tail}"
        
        if "error" in execution_summary.lower() or "error" in execution_details.lower():
            return execution_details, False
        
        return execution_details, True

    async def _run(self, params: Dict[str, Any], context_chat: Chat) -> ToolResponse:
        if not self.validate_params(params):
            return self.format_response(
                status=ToolStatus.ERROR,
                summary="Invalid parameters for python edit tool."
            )
        
        try:
            # Get script path from parameters or from context metadata
            parameters = params.get("parameters", {})
            script_path = parameters.get("script_path")
            
            # If script_path is not provided, try to get it from context metadata
            if not script_path and hasattr(context_chat, "metadata") and context_chat.metadata:
                script_path = context_chat.metadata.get("last_python_script_path")
                
            if not script_path:
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary="No script_path provided and no previous Python script found in context."
                )
                
            modification_prompt = parameters.get("modification_prompt")
            if not modification_prompt:
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary="Missing required parameter: modification_prompt"
                )
            
            # Check if the script exists
            if not os.path.exists(script_path):
                return self.format_response(
                    status=ToolStatus.ERROR,
                    summary=f"The script at path '{script_path}' does not exist."
                )
            
            # Read the existing script
            with open(script_path, "r") as f:
                existing_code = f.read()
            
            # Create a prompt to modify the script
            implement_chat = Chat()
            implement_chat.add_message(
                Role.USER, 
                f"Please modify the following Python script according to these instructions: {modification_prompt}\n\nExisting script:\n\n```python\n{existing_code}\n```"
            )
            
            # Generate the modified code
            modified_code = LlmRouter.generate_completion(implement_chat, strength=AIStrengths.CODE)
            
            # Clean response to ensure we only get code
            if "```python" in modified_code:
                start_idx = modified_code.find("```python") + 9
                end_idx = modified_code.find("```", start_idx)
                if end_idx != -1:
                    modified_code = modified_code[start_idx:end_idx]
                else:
                    modified_code = modified_code[start_idx:]
            
            modified_code = modified_code.strip()
            
            # Write the modified code back to the file
            with open(script_path, "w") as f:
                f.write(modified_code)
            
            # Execute the modified script
            execution_details, success = self.execute_script(script_path)
            
            execution_message = "Returned: " + execution_details if execution_details else ""
            status_message = "successfully" if success else "with errors"
            
            return self.format_response(
                status=ToolStatus.SUCCESS,
                summary=f"Modified and executed {script_path} {status_message}!\n{execution_message}",
                followup_tools=self.metadata.followup_tools
            )
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            return self.format_response(
                status=ToolStatus.ERROR,
                summary=f"Error calling python edit tool: {error_traceback}"
            )

# Test code when script is run directly
if __name__ == "__main__":
    import asyncio
    
    async def test_python_edit_tool():
        tool = PythonEditTool()
        chat = Chat()  # Create an empty chat for the context
        
        # Example parameters
        params = {
            "parameters": {
                "script_path": "/path/to/your/script.py",
                "modification_prompt": "Add a function to calculate the factorial of a number"
            }
        }
        
        # Execute the tool
        try:
            response = await tool.run(params, chat)
            print("Tool Response Status:", response["status"])
            print("Tool Response Summary:", response["summary"])
        except Exception as e:
            print(f"Error testing PythonEditTool: {str(e)}")

    # Run the async test function
    asyncio.run(test_python_edit_tool()) 