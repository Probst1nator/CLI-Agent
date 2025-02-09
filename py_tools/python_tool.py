from typing import Any, Dict, Tuple, Optional
import tempfile
import os
import sys
import subprocess
import json
from termcolor import colored

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
                    "description": "Detailed description of what the Python script should do, including input/output specifications, data handling requirements, and expected behavior"
                }
            },
            required_params=["reasoning", "title", "requirements"],
            example_usage="""
            {
                "reasoning": "Need to create a visualization of data points in 3D space",
                "tool": "python",
                "title": "plot_3d_scatter.py",
                "requirements": "Create a 3D scatter plot using matplotlib. The script should:
                1. Generate sample data points in 3D space
                2. Create an interactive 3D scatter plot
                3. Add proper axis labels and title
                4. Save the plot as 'scatter_3d.png'
                5. Display the plot in a window"
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

1. Implements a 50x50 grid using numpy arrays
2. Initializes random starting state with 30% live cells
3. Applies Conway's rules:
   - Live cell survives with 2-3 neighbors
   - Dead cell revives with exactly 3 neighbors
4. Uses pygame to display animated grid with interactive features:
   - Left click to toggle cells alive/dead
   - Space bar to pause/resume simulation
   - 'r' key to randomize grid state
   - Up/Down arrows to speed up/slow down simulation
   - 'c' key to clear the grid
5. Updates every 200ms (adjustable with arrow keys)
6. Displays current generation count and simulation speed
7. Runs until 'q' is pressed

The simulation should provide real-time feedback for all user interactions and maintain smooth performance throughout execution. The interface should be intuitive and responsive, allowing users to experiment with different patterns and observe their evolution."
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

The implementation must:
1. Use type hints for all functions and variables
2. Include comprehensive error handling
3. Include docstrings and comments
4. Be self-contained and reusable
5. Follow PEP 8 style guidelines

Respond with ONLY the Python code, no explanations or markdown."""
        )
        
        response = LlmRouter.generate_completion(
            implement_chat,
            ["claude-3-5-sonnet-latest", "gpt-4", "qwen2.5-coder:7b-instruct"],
            silent_reasoning=True
        )
        
        # Clean response to ensure we only get code
        if "```python" in response:
            response = response[response.find("```python") + 9:response.rfind("```")]
        return response.strip()

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

    async def execute(self, params: Dict[str, Any]) -> ToolResponse:
        """Execute the Python tool with comprehensive error handling and script management."""
        if not self.validate_params(params):
            return self.format_response(
                "Invalid parameters provided",
                status="error",
                error="Missing required parameters"
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
            is_new_script = not os.path.exists(script_path)
            
            # Create context chat for LLM interactions
            context_chat = Chat()
            context_chat.add_message(Role.SYSTEM, "You are a Python programming assistant.")
            
            # Handle script implementation
            if is_new_script:
                final_script = self.request_implementation(context_chat, script_description)
            else:
                evaluation = self.evaluate_existing_script(script_path, context_chat, script_description)
                
                if evaluation['decision'] == 'keep':
                    with open(script_path, 'r') as f:
                        final_script = f.read()
                else:
                    final_script = self.request_implementation(context_chat, script_description)
            
            # Write script
            with open(script_path, "w") as f:
                f.write(final_script)
            
            # Execute script
            execution_details, success = self.execute_script(script_path)
            
            if not success:
                error_analysis = self.handle_execution_error(execution_details, context_chat)
                if error_analysis['requires_rewrite']:
                    # Recursively try again with a fresh implementation
                    return await self.execute(params)
                
                return self.format_response(
                    reasoning=f"Script execution failed: {error_analysis['analysis']}",
                    status="error",
                    error=execution_details,
                    error_analysis=error_analysis,
                    script_path=script_path
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