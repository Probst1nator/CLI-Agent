from typing import Any, Dict, Tuple, Optional
import tempfile
import os
import sys
import subprocess
import json
from termcolor import colored
import re

from py_classes.cls_base_tool import BaseTool, ToolMetadata, ToolResponse
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_methods.cmd_execution import select_and_execute_commands
from py_classes.globals import g

class PythonTool(BaseTool):
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="python",
            description="Execute Python code for visualization, computation, data analysis, or automation tasks. Use this tool when you need to create and run Python scripts for tasks like data processing, plotting graphs, mathematical calculations, or file operations.",
            parameters={
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of why this script is needed and how it helps solve the current task"
                },
                "title": {
                    "type": "string",
                    "description": "Name for the Python script (e.g., 'plot_data.py', 'calculate_stats.py')"
                },
                "requirements": {
                    "type": "string",
                    "description": "A string containing a numbered list or bullet points describing what the Python script should do"
                }
            },
            required_params=["reasoning", "title", "requirements"],
            example_usage="""
            {
                "tool": "python",
                "reasoning": "Need to create a visualization of data points in 3D space",
                "title": "plot_3d_scatter.py",
                "requirements": "1. Generate sample data points in 3D space\\n2. Create an interactive 3D scatter plot\\n3. Add proper axis labels and title\\n4. Save the plot as 'scatter_3d.png'\\n5. Display the plot in a window"
            }
            """
        )

    @property
    def prompt_template(self) -> str:
        """Return the prompt template for the tool"""
        return """Use the Python tool to implement and execute Python code.
To use the Python tool, provide in this exact order:
1. Reasoning for why Python is needed for this task
2. The tool name (python)
3. A descriptive title for the script (e.g., 'plot_data.py')
4. Clear requirements for what the script should do
5. The code will be implemented automatically by the tool call, so DO NOT IMPLEMENT THE CODE, if you do, it will be ignored.

Example:
{
    "reasoning": "Need to create an interactive simulation of Conway's Game of Life",
    "tool": "python",
    "title": "game_of_life.py", 
    "requirements": "Create a script that:
1. Implements a grid for Conway's Game of Life using numpy
2. Initializes with random starting cells
3. Applies the standard Conway's Game of Life rules:
   - Live cell survives with 2-3 neighbors
   - Dead cell revives with exactly 3 neighbors
4. Uses pygame to display the grid with basic controls:
   - Space to pause/resume
   - Click to toggle cells
   - 'r' to randomize
5. Updates continuously until quit"
}"""

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
        
        response = LlmRouter.generate_completion(
            evaluation_chat, 
            ["claude-3-5-sonnet-latest"],
            silent_reasoning=True
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
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
4. Include inline comments for complex logic
5. Be completely self-contained:
   - Use Pyglet instead of pptx for presentations
6. Follow PEP 8 style guidelines
7. Include a main guard: if __name__ == "__main__":
8. Include all necessary imports at the top

Example response format:
Implementation plan:
1. First, I'll...
2. Then, I'll...
3. Finally, I'll...

```python
# Complete implementation here
```"""
        )
        
        response = LlmRouter.generate_completion(
            implement_chat,
            ["claude-3-5-sonnet-latest", "qwen2.5-coder:7b-instruct"],
            silent_reasoning=True
        )
        
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
                """The previous response did not contain valid Python code. Please provide ONLY the Python implementation wrapped in code blocks. Example:
```python
import sys
# Rest of the implementation
```"""
            )
            
            retry_response = LlmRouter.generate_completion(
                implement_chat,
                ["claude-3-5-sonnet-latest", "qwen2.5-coder:7b-instruct"],
                silent_reasoning=True
            )
            extracted_code = extract_python_code(retry_response)
        
        # Ensure the code has a main guard if it's not just a function/class definition
        if 'if __name__ == "__main__":' not in extracted_code and ('print' in extracted_code or 'input' in extracted_code):
            extracted_code += "\n\nif __name__ == \"__main__\":\n    main()"
        
        return extracted_code.strip()

    def handle_execution_error(
        self,
        error_details: str, 
        context_chat: Chat
    ) -> Dict:
        """
        Analyze and handle script execution errors.
        
        Args:
            error_details: Error output from script execution
            context_chat: Current conversation context
        
        Returns:
            Dict containing error analysis
        """
        error_chat = context_chat.deep_copy()
        error_chat.add_message(
            Role.USER,
            f"""Analyze this Python execution error and suggest fixes:
{error_details}

Respond with ONLY a JSON object containing:
{{
    "error_type": "classification of error",
    "analysis": "what went wrong",
    "fix_strategy": "how to fix it",
    "requires_rewrite": boolean
}}"""
        )
        
        response = LlmRouter.generate_completion(
            error_chat,
            ["qwen2.5-coder:7b-instruct", "claude-3-5-sonnet-latest"],
            silent_reasoning=True
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "error_type": "unknown",
                "analysis": "Failed to analyze error",
                "fix_strategy": "Complete rewrite recommended",
                "requires_rewrite": True
            }

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
            return execution_details, False
        
        return execution_details, True

    def _get_missing_params(self, params: Dict[str, Any]) -> list[str]:
        """
        Check which required parameters are missing from the input.
        
        Args:
            params: Dictionary of provided parameters
            
        Returns:
            List of missing parameter names
        """
        required_params = self.metadata.required_params
        return [param for param in required_params if param not in params or not params[param]]

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """Execute the Python tool with comprehensive error handling and script management."""
        missing_params = self._get_missing_params(params)
        if missing_params:
            return self.format_response(
                "Missing required parameters",
                status="error",
                error=f"The following required parameters are missing or empty: {', '.join(missing_params)}"
            )

        # Validate requirements is a string
        if not isinstance(params.get('requirements'), str):
            try:
                # Try to convert requirements to string if it's a dict or list
                if isinstance(params['requirements'], (dict, list)):
                    requirements_str = "\n".join(f"{k}. {v}" if isinstance(params['requirements'], dict) else f"{i+1}. {v}" 
                                              for i, (k, v) in enumerate(params['requirements'].items() if isinstance(params['requirements'], dict) else enumerate(params['requirements'])))
                    params['requirements'] = requirements_str
                else:
                    return self.format_response(
                        "Invalid requirements format",
                        status="error",
                        error="Requirements must be a string containing a numbered list or bullet points"
                    )
            except Exception as e:
                return self.format_response(
                    "Invalid requirements format",
                    status="error",
                    error=f"Could not process requirements: {str(e)}"
                )

        try:
            script_title = params['title']
            script_reasoning = params['reasoning']
            script_requirements = params['requirements']
            
            script_description = f"""Title: {script_title}
Reasoning: {script_reasoning}
Requirements: {script_requirements}"""
            
            # Setup script path
            script_dir = os.path.join(g.PROJ_VSCODE_DIR_PATH, "python_tool")
            os.makedirs(script_dir, exist_ok=True)
            script_path = os.path.join(script_dir, script_title)
            
            # Track if we're creating a new script or modifying existing
            # is_new_script = not os.path.exists(script_path)
            
            # Create context chat for LLM interactions
            context_chat = Chat(debug_title="Python Tool")
            context_chat.add_message(Role.SYSTEM, "You are a Python programming assistant.")
            
            # Handle script implementation
            # if is_new_script:
            final_script = self.request_implementation(context_chat, script_description)
            # else:
            #     evaluation = self.evaluate_existing_script(script_path, context_chat, script_description)
                
            #     if evaluation['decision'] == 'keep':
            #         with open(script_path, 'r') as f:
            #             final_script = f.read()
            #     else:
            #         final_script = self.request_implementation(context_chat, script_description)
            
            # Write script
            with open(script_path, "w") as f:
                f.write(final_script)
            
            # Execute script
            execution_details, success = self.execute_script(script_path)
            
            if not success:
                # Request fixed implementation with error context
                fix_chat = context_chat.deep_copy()
                fix_chat.add_message(
                    Role.USER,
                    f"""Fix this Python script that failed with the following error:

ERROR OUTPUT:
{execution_details}

CURRENT SCRIPT:
```python
{final_script}
```

Requirements for the fix:
{script_requirements}

Respond with ONLY the complete fixed Python code, no explanations or markdown."""
                )
                
                fixed_script = LlmRouter.generate_completion(
                    fix_chat,
                    ["qwen2.5-coder:7b-instruct", "claude-3-5-sonnet-latest"],
                    silent_reasoning=True
                )
                
                # Clean response to ensure we only get code
                if "```python" in fixed_script:
                    fixed_script = fixed_script[fixed_script.find("```python") + 9:fixed_script.rfind("```")]
                fixed_script = fixed_script.strip()
                
                # Write fixed script
                with open(script_path, "w") as f:
                    f.write(fixed_script)
                
                # Try executing the fixed script
                execution_details, success = self.execute_script(script_path)
                
                if not success:
                    return self.format_response(
                        reasoning=f"Script execution failed even after fix attempt",
                        status="error",
                        error=execution_details,
                        script_path=script_path,
                        fixed_script=fixed_script
                    )
                
                return self.format_response(
                    reasoning="Script fixed and executed successfully",
                    status="success",
                    stdout=execution_details,
                    script_path=script_path,
                    fixed_script=fixed_script
                )
            
            return self.format_response(
                reasoning="Script executed successfully",
                status="success",
                stdout=execution_details,
                script_path=script_path
            )

        except Exception as e:
            return self.format_response(
                reasoning=f"Error in Python tool execution: {str(e)}",
                status="error",
                error=str(e)
            )