# tools/paralleltodos.py
"""
This file implements the 'paralleltodos' tool. It allows an agent to specify
multiple file paths and a single prompt, then spawns concurrent, read-only 
sub-agents to execute that prompt on each path.
"""

import os
import re
import asyncio
from typing import Dict, Any

from core.globals import g
from core.chat import Chat, Role
from core.llm_router import LlmRouter
from shared.common_utils import extract_blocks
from termcolor import colored

class paralleltodos:
    """
    A tool to execute a single prompt across multiple directories in parallel
    using sandboxed, read-only sub-agents.
    """
    
    @staticmethod
    def get_delim() -> str:
        """Provides the delimiter for this tool."""
        return 'paralleltodos'

    @staticmethod
    def get_tool_info() -> dict:
        """Provides standardized documentation for this tool."""
        return {
            "name": "paralleltodos",
            "description": "Executes a single prompt on a list of directories in parallel. Each directory is processed by an independent, read-only agent. Useful for applying the same analysis or task across multiple components of a codebase.",
            "example": (
                "<paralleltodos>\n"
                "  <path>./src/api</path>\n"
                "  <path>./src/ui</path>\n"
                "  <prompt>Analyze the code in this directory and write a short summary of its purpose and key dependencies.</prompt>\n"
                "</paralleltodos>"
            )
        }

    @staticmethod
    def _get_readonly_tools() -> Dict[str, Any]:
        """
        Filters the globally available tools to a safe, read-only subset for sub-agents.
        """
        # This function relies on the main agent having already discovered all tools
        # and populated g.tool_classes.
        if not hasattr(g, 'tool_classes') or not g.tool_classes:
            return {}

        allowed_tool_names = [
            'readfile', 'grepsearch', 'searchweb', 'think', 
            'response', 'todos', 'viewpermissions', 'getuserfiles'
        ]
        
        readonly_tools = {
            name: cls for name, cls in g.tool_classes.items() 
            if name in allowed_tool_names
        }
        return readonly_tools

    @staticmethod
    def _generate_readonly_system_prompt(readonly_tools: Dict[str, Any]) -> str:
        """
        Constructs a system prompt specifically for the read-only sub-agents,
        listing only the tools they are permitted to use.
        """
        base_instructions = (
            "**SYSTEM: Read-Only Agent Protocol**\n\n"
            "You are a specialized, read-only AI agent. Your capabilities are strictly limited to "
            "observing and analyzing files and information. You CANNOT make any modifications.\n"
            "Your goal is to follow the user's prompt within your designated directory.\n\n"
            "### Guiding Principles (Your Core Logic)\n"
            "1.  **OBSERVE & DECOMPOSE**: Analyze the user's request and break it down into the required steps. Start with: `<think>...</think>`.\n"
            "2.  **PLAN & TRACK**: Use `<todos>...</todos>` to outline your plan and track your progress."
        )
        
        tool_docs_parts = []
        # We only document executable tools, not cognitive/structural ones like think/todos/response.
        documented_tools = {k: v for k, v in readonly_tools.items() if k not in ['think', 'todos', 'response']}
        
        for tool_name, tool_class in sorted(documented_tools.items()):
            info = tool_class.get_tool_info()
            doc_string = f"- **{tool_name}**: {info['description']}\n  - *Example*: `{info['example']}`"
            tool_docs_parts.append(doc_string)
        
        tool_docs = "3. **EXECUTE**: Use one of the following tools to perform an action. Use only ONE tool at a time.\n\n" + "\n".join(tool_docs_parts)

        final_step = "4.  **RESPOND**: Once you have gathered all necessary information, provide a final, comprehensive answer using the `<response>...</response>` tag. This tag concludes your work and terminates your session."

        return f"{base_instructions}\n{tool_docs}\n{final_step}"

    @staticmethod
    async def _run_sub_agent(path: str, prompt: str, readonly_tools: Dict[str, Any], readonly_system_prompt: str) -> str:
        """
        The core logic for a single, isolated, read-only sub-agent.
        """
        original_cwd = os.getcwd()
        try:
            # 1. Setup Environment
            if not os.path.isdir(path):
                return f"Error: Path '{path}' is not a valid directory."
            os.chdir(path)

            # 2. Initialize Chat and LLM Router
            chat = Chat(debug_title=f"Sub-Agent({os.path.basename(path)})")
            chat.set_instruction_message(readonly_system_prompt)
            chat.add_message(Role.USER, prompt)
            
            llm_router = LlmRouter()
            
            # 3. Sub-agent Execution Loop
            MAX_TURNS = 10 
            for _ in range(MAX_TURNS):
                assistant_response = await llm_router.chat_with_retry(
                    chat=chat,
                    temperature=0.0, # Use low temperature for focused, deterministic sub-agents
                    model_key=None # Use default model
                )
                
                if not assistant_response:
                    return f"Sub-agent for '{path}' failed to generate a response."

                chat.add_message(Role.ASSISTANT, assistant_response)

                # Check for a final response, which terminates the loop
                response_match = re.search(r'<response>(.*?)</response>', assistant_response, re.DOTALL)
                if response_match:
                    return response_match.group(1).strip()

                # 4. Execute Tools
                tool_was_executed = False
                for tool_name, tool_class in readonly_tools.items():
                    # extract_blocks returns the content *inside* the tags
                    blocks = extract_blocks(assistant_response, [tool_name])
                    if blocks:
                        tool_was_executed = True
                        # Run the first block found for this tool
                        block_content = blocks[0]
                        tool_result = tool_class.run(block_content)
                        
                        if asyncio.iscoroutine(tool_result):
                            tool_result = await tool_result
                        
                        if tool_result:
                            context_msg = f"<context>\n{tool_result}\n</context>"
                            chat.add_message(Role.ASSISTANT, context_msg)
                
                # If no tool was found, the agent might be stuck in a conversational loop.
                # We break to return its last message instead of looping infinitely.
                if not tool_was_executed:
                    return f"Sub-agent for '{path}' finished without a final <response> tag. Last message was: {assistant_response}"

            return f"Sub-agent for '{path}' exceeded maximum turn limit of {MAX_TURNS}."

        except Exception as e:
            # Capture any unexpected errors during sub-agent execution.
            return f"An unexpected error occurred in sub-agent for '{path}': {str(e)}"
        finally:
            # CRITICAL: Ensure we always return to the original working directory.
            os.chdir(original_cwd)

    @staticmethod
    async def run(content: str) -> str:
        """
        Parses the input, sets up, and runs the parallel sub-agents.
        """
        # 1. Parse paths and prompt from the input content
        path_matches = re.findall(r'<path>(.*?)</path>', content, re.DOTALL)
        prompt_match = re.search(r'<prompt>(.*?)</prompt>', content, re.DOTALL)

        if not path_matches or not prompt_match:
            return "Error: <paralleltodos> requires one or more <path> tags and one <prompt> tag."
            
        paths = [p.strip() for p in path_matches]
        prompt = prompt_match.group(1).strip()

        # 2. Setup the read-only environment for all sub-agents
        readonly_tools = paralleltodos._get_readonly_tools()
        if not readonly_tools:
            return "Error: Could not load read-only tools for sub-agents. Main agent may not be initialized correctly."
            
        readonly_system_prompt = paralleltodos._generate_readonly_system_prompt(readonly_tools)

        # 3. Create and run all sub-agent tasks concurrently
        tasks = [
            paralleltodos._run_sub_agent(path, prompt, readonly_tools, readonly_system_prompt)
            for path in paths
        ]
        
        print(colored(f"🤖 Spawning {len(paths)} read-only sub-agents to work in parallel...", "cyan"))
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 4. Aggregate and format the results
        output_parts = ["Parallel execution finished. Aggregating results:\n"]
        for path, result in zip(paths, results):
            output_parts.append(f"--- Results for path: {path} ---")
            if isinstance(result, Exception):
                output_parts.append(f"Error: {str(result)}")
            else:
                output_parts.append(result)
            output_parts.append("--- End of Results ---")
            
        return "\n\n".join(output_parts)