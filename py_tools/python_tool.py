from typing import Any, Dict, Tuple, Optional
import tempfile
import os
import sys
import subprocess
import json
from termcolor import colored
import re
import time

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import AIStrengths, LlmRouter
from py_methods.cmd_execution import select_and_execute_commands
from py_classes.globals import g
from py_methods.tooling import extract_json

class PythonTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="python",
            description="Call this tool if you require to perform a calculation, implement visualizations or perform more complex tasks in a single python script. (NOT ALLOWED: Internet search, Api key requirements (use web_search tool instead))",
            detailed_description="""Use this tool when you need to:
- Create visualizations or plots
- Perform calculations
Perfect for tasks like:
- Requested visualizations
- Statistical data visualization
- Simple interactive applications (using matplotlib, pygame or pyglet)
NOT ALLOWED:
- Api key requirements
- Internet search (use web_search tool instead)""",
            constructor="""
def run(
    title: str,
    requirements: str,
    additional_data: str = None
) -> None:
    \"\"\"Create and run a Python script.
    
    Args:
        title: Name for the Python script (e.g., 'plot_data.py', 'calculate_stats.py')
        requirements: A string containing a numbered list or bullet points describing what the Python script should do
        additional_data: A string containing data for visualizations, calculations from previous steps or other relevant data
    \"\"\"
"""
        )

    def evaluate_existing_script(
        self,
        script_path: str, 
        context_chat: Chat, 
        requirements: str
    ) -> Dict:
        """
        Evaluate if an existing script meets current requirements.
        
        Args:
            script_path: Path to the existing script
            context_chat: Current conversation context
            requirements: New requirements for the script
        
        Returns:
            Dict containing evaluation results
        """
        with open(script_path, "r") as f:
            file_content = f.read()
        
        evaluation_chat = context_chat.deep_copy()
        evaluation_chat.add_message(
            Role.USER,
            f"""Evaluate if this existing Python script meets these requirements:

REQUIREMENTS:
{requirements}

EXISTING SCRIPT:
```python
{file_content}
```

Respond with ONLY a JSON object containing:
{{
    "reasoning": "detailed evaluation of script",
    "decision": "keep" | "modify" | "replace",
    "modifications_needed": ["list of needed changes"] | null
}}"""
        )
        
        response = LlmRouter.generate_completion(evaluation_chat, strength=AIStrengths.TOOLUSE)
        parsed_json = extract_json(response, required_keys=["reasoning", "decision"])
        if parsed_json:
            return parsed_json
        return {
            "reasoning": "Failed to parse evaluation response",
            "decision": "replace",
            "modifications_needed": None
        }

    def request_implementation(
        self,
        context_chat: Chat, 
        requirements: str
    ) -> str:
        """
        Request a new implementation with clear requirements.
        
        Args:
            context_chat: Current conversation context
            requirements: Requirements for the script
        
        Returns:
            str containing the Python implementation
        """
        implement_chat = context_chat.deep_copy()
        implement_chat.add_message(
            Role.USER,
            f"""Create a Python script that meets these requirements:
{requirements}

IMPORTANT RESPONSE FORMAT INSTRUCTIONS:
1. Start your response with a brief plan of implementation
2. ALWAYS wrap your code in a Python code block using:
   ```python
   # Your code here
   ```
3. DO NOT include any other code blocks or explanations after the code
4. DO NOT split the code into multiple blocks

The implementation must:
1. Use type hints for all functions and variables
2. Include comprehensive error handling with try/except blocks
3. Include docstrings for all functions and classes
4. Be completely self-contained:
   - Use Pyglet instead of pptx for presentations
5. Include a main guard: if __name__ == "__main__":
6. Include all necessary imports at the top

```python
# Complete implementation here
```"""
        )
        
        response = LlmRouter.generate_completion(implement_chat, strength=AIStrengths.CODE)
        
        # More robust code extraction
        def extract_python_code(text: str) -> str:
            # Try to find code between ```python and ``` markers first
            python_block_match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
            if python_block_match:
                return python_block_match.group(1).strip()
            
            # If no python block found, try generic code blocks
            code_block_match = re.search(r'```\n?(.*?)\n```', text, re.DOTALL)
            if code_block_match:
                return code_block_match.group(1).strip()
            
            # If no code blocks found, try to find Python-like content
            # Look for content that starts with imports or comments
            python_content_match = re.search(r'(?:(?:import|from|#).*?\n(?:.*?\n)*)', text, re.DOTALL)
            if python_content_match:
                return python_content_match.group(0).strip()
            
            # If all else fails, return the entire response
            # This will be validated later during execution
            return text.strip()

        extracted_code = extract_python_code(response)
        
        # Validate the extracted code has basic Python structure
        if not any(keyword in extracted_code for keyword in ['import ', 'def ', 'class ', '#']):
            # If code doesn't look like Python, try one more time with more explicit instructions
            implement_chat.add_message(
                Role.USER,
                """My system was unable to extract the Python code from the response. If your previouse response is incomplete, shorten your code. If it was complete, try again strictly following the instructions above."""
            )
            
            retry_response = LlmRouter.generate_completion(implement_chat, strength=AIStrengths.CODE)
            extracted_code = extract_python_code(retry_response)
        
        # Ensure the code has a main guard if it's not just a function/class definition
        if 'if __name__ == "__main__":' not in extracted_code and ('print' in extracted_code or 'input' in extracted_code):
            extracted_code += "\n\nif __name__ == \"__main__\":\n    main()"
        
        return extracted_code.strip()

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
        
        if "error" in execution_summary.lower() or "error" in execution_details.lower():
            return self._truncate_output(execution_details), False
        
        return self._truncate_output(execution_details), True


    async def run(self, params: Dict[str, Any]) -> ToolResponse:
        """
        Execute the Python tool with comprehensive error handling and script management.
        
        Args:
            params: Dictionary containing the tool parameters
            
        Returns:
            ToolResponse indicating success or failure with details
        """
        try:
            # Validate and process requirements
            parameters = params["parameters"]
            if not isinstance(parameters.get('requirements'), str):
                try:
                    if isinstance(parameters['requirements'], (dict, list)):
                        requirements_str = "\n".join(
                            f"{k}. {v}" if isinstance(parameters['requirements'], dict) 
                            else f"{i+1}. {v}" 
                            for i, (k, v) in enumerate(
                                parameters['requirements'].items() 
                                if isinstance(parameters['requirements'], dict) 
                                else enumerate(parameters['requirements'])
                            )
                        )
                        parameters['requirements'] = requirements_str
                    else:
                        return self.format_response(
                            status="error",
                            summary="Requirements must be a string containing a numbered list or bullet points"
                        )
                except Exception as e:
                    return self.format_response(
                        status="error",
                        summary=f"Could not process requirements: {str(e)}"
                    )
            if not isinstance(parameters.get('additional_data'), str):
                return self.format_response(
                    status="error",
                    summary="Additional data must be a string"
                )

            # Create script directory
            script_dir = os.path.join(g.PROJ_PERSISTENT_STORAGE_PATH, "python_tool")
            os.makedirs(script_dir, exist_ok=True)
            
            # Generate unique script name using title and timestamp
            timestamp = int(time.time())
            base_name = os.path.splitext(parameters['title'])[0]
            # Replace spaces and any other problematic characters with underscores
            safe_base_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
            script_name = f"{safe_base_name}_{timestamp}.py"
            script_path = os.path.join(script_dir, script_name)
            
            # Create implementation chat for generating the script
            implement_chat = Chat()
            implement_chat.add_message(
                Role.USER,
                f"""Create a Python script that meets these requirements:
{parameters['requirements']}

Additional data:
{parameters['additional_data']}

The implementation must:
1. Use type hints for all functions and variables
2. Include docstrings for all functions and classes
3. Include inline comments for complex logic
4. Be completely self-contained
5. Follow PEP 8 style guidelines
6. Include a main guard: if __name__ == "__main__":
7. Include all necessary imports at the top

Please start by providing a outline of an implementation, then provide the complete implementation as a python code block."""
            )
            
            # Generate the implementation
            implementation = LlmRouter.generate_completion(implement_chat, strength=AIStrengths.CODE)
            
            # Clean response to ensure we only get code
            if "```python" in implementation:
                implementation = implementation[implementation.find("```python") + 9:implementation.rfind("```")]
            implementation = implementation.strip()
            
            # Write implementation to file
            with open(script_path, "w") as f:
                f.write(implementation)
            
            # Execute script
            execution_details, success = self.execute_script(script_path)
            
            if not success:
                # Create a new chat for error fixing with analysis and implementation
                fix_chat = Chat()
                fix_chat.add_message(
                    Role.USER,
                    f"""Fix this Python script that failed with the following error:

ERROR OUTPUT:
{execution_details}

CURRENT SCRIPT:
```python
{implementation}
```

Requirements for the fix:
{parameters['requirements']}

First, analyze the error and explain your fix strategy. Then, provide the complete fixed implementation.

Your response should follow this format:
1. ERROR ANALYSIS: Brief explanation of what went wrong
2. FIX STRATEGY: How you plan to fix it
3. IMPLEMENTATION: The complete fixed code in a Python code block

The implementation must maintain all original requirements including type hints, error handling, and documentation."""
                )
                
                # Get the analysis and fixed implementation
                fix_response = LlmRouter.generate_completion(fix_chat, strength=AIStrengths.CODE)
                
                # Extract the analysis sections for potential error reporting
                error_analysis = ""
                
                # Try to extract the analysis sections
                if "ERROR ANALYSIS:" in fix_response:
                    error_parts = fix_response.split("FIX STRATEGY:")
                    if len(error_parts) > 1:
                        error_analysis = error_parts[0].split("ERROR ANALYSIS:")[1].strip()
                
                # Use the existing request_implementation method to extract and validate the code
                fixed_script = self.request_implementation(fix_chat, parameters['requirements'])
                
                # Write fixed script
                with open(script_path, "w") as f:
                    f.write(fixed_script)
                
                # Try executing the fixed script
                execution_details, success = self.execute_script(script_path)
                
                if not success:
                    return self.format_response(
                        status="error",
                        summary=f"""Task: {parameters['requirements']}
Execution failed: {execution_details}
Analysis: {error_analysis}"""
                    )
                
                return self.format_response(
                    status="success",
                    summary=f"""Task: {parameters['requirements']}
Output: {execution_details}"""
                )
            
            return self.format_response(
                status="success",
                summary=f"""Task: {parameters['requirements']}
Output: {execution_details}"""
            )

        except Exception as e:
            return self.format_response(
                status="error",
                summary=f"""Task: {parameters.get('requirements', 'Unknown')}
Error in Python tool execution: {str(e)}"""
            )