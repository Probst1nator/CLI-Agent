import os
import re
import json
import asyncio
from typing import List, Dict, Optional, Coroutine, Any
from termcolor import colored

from py_classes.cls_util_base import UtilBase
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_classes.enum_ai_strengths import AIStrengths

def _run_async_safely(coro: Coroutine) -> Any:
    """Helper function to run async code from sync context safely."""
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            try: import nest_asyncio; nest_asyncio.apply()
            except ImportError: pass
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)

class FileMind(UtilBase):
    """
    An intelligent utility that analyzes a task, discovers relevant files, and
    proposes a structured, executable plan for the agent.
    """
    MAX_DISCOVERY_ITERATIONS = 10

    @staticmethod
    def run(task_description: str) -> str:
        """
        Analyzes a task and the filesystem to create an action plan.

        Args:
            task_description: A natural language description of the coding task.

        Returns:
            str: A JSON string containing a high-level plan and a precise,
                 copy-pasteable `tool_code` block for the agent to execute next.
        """
        print(colored(f"🤖 FileMind Activated. Task: {task_description}", "magenta"))
        context_chat = Chat.load_from_json()
        full_context_prompt = f"Task: {task_description}\nHistory:\n{context_chat.get_messages_as_string(-3)}"

        try:
            # Stage 1: Discover relevant files using an internal LLM loop
            discovered_files = _run_async_safely(FileMind._discover_files(full_context_prompt))
            if not discovered_files:
                return json.dumps({"error": "No relevant files were found or selected during discovery."})

            # Stage 2: Generate a plan and a precise command for the agent
            plan = _run_async_safely(FileMind._generate_plan_and_command(full_context_prompt, discovered_files))
            
            return json.dumps({"result": plan}, indent=2)

        except Exception as e:
            import traceback
            return json.dumps({"error": f"FileMind critical error: {e}", "details": traceback.format_exc()})

    @staticmethod
    async def _discover_files(full_context: str) -> Dict[str, str]:
        """Uses an LLM to discover relevant files via `ls` and `cat` commands."""
        print(colored("  > Stage 1: Discovering relevant files...", "yellow"))
        relevant_files = {}
        instruction = f"""You are a file discovery agent. Your goal is to find all files relevant to the task.
Based on the task and file contents, issue one of three commands: `ls <path>`, `cat <path>`, or `done()`.
You have {FileMind.MAX_DISCOVERY_ITERATIONS} iterations.

TASK:
{full_context}
"""
        current_path = "."
        for i in range(FileMind.MAX_DISCOVERY_ITERATIONS):
            discovery_chat = Chat(instruction, debug_title=f"FileMind-Discover-{i+1}")
            
            # Get directory listing
            try:
                dir_contents = os.listdir(current_path)
                dir_listing = {"path": current_path, "contents": dir_contents}
            except Exception as e:
                dir_listing = {"path": current_path, "error": str(e), "contents": []}
            
            prompt = f"You have already found these relevant files: {list(relevant_files.keys()) or 'None'}.\n"
            prompt += f"Current directory '{dir_listing.get('path', current_path)}' listing:\n{json.dumps(dir_listing.get('contents', []), indent=2)}\n\nWhat is your next command?"
            discovery_chat.add_message(Role.USER, prompt)

            response = await LlmRouter.generate_completion(discovery_chat, strengths=[AIStrengths.REASONING], temperature=0.0)
            command = response.strip().replace("`", "")

            if command.startswith("ls"):
                path_to_list = command.split(" ", 1)[-1].strip() or "."
                current_path = os.path.join(current_path, path_to_list)
                print(f"  > Executing ls: {current_path}")
            elif command.startswith("cat"):
                filepath = command.split(" ", 1)[-1].strip()
                print(f"  > Executing cat: {filepath}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    relevant_files[filepath] = content
                except Exception as e:
                    print(f"    └─ Error reading {filepath}: {e}")
            elif command.startswith("done()"):
                print(colored("  > File discovery complete.", "green"))
                break
        return relevant_files

    @staticmethod
    async def _generate_plan_and_command(full_context: str, discovered_files: Dict[str, str]) -> Dict[str, Any]:
        """After discovery, ask an LLM to generate a plan and a precise tool_code block."""
        
        instruction = """You are a senior software architect. Your job is to create a plan for another AI agent.
Analyze the user's task and the provided file context.
You MUST respond with a single JSON object containing:
1.  A "plan" key with your concise, high-level strategy.
2.  A "tool_code" key with the exact, single `EditFile.run` command to execute the plan.

Example Response:
{
  "plan": "Refactor the login function in `src/api/auth.js` to use async/await and add error handling for the database call.",
  "tool_code": "EditFile.run(filepath='src/api/auth.js')"
}
"""
        planning_chat = Chat(instruction, debug_title="FileMind-PlanGenerator")
        context_block = "\n\n".join(f"--- FILE: {p} ---\n{c[:800]}...\n" for p, c in discovered_files.items())
        prompt = f"TASK:\n{full_context}\n\nDISCOVERED FILES:\n{context_block}\n\nNow, provide the JSON object containing the plan and the exact `tool_code`."
        planning_chat.add_message(Role.USER, prompt)

        response = await LlmRouter.generate_completion(planning_chat, strengths=[AIStrengths.STRONG], temperature=0.0)
        
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                plan_json = json.loads(match.group(0))
                print(colored("  > Stage 2: Plan generated successfully.", "green"))
                return plan_json
        except json.JSONDecodeError:
            return {"error": "Failed to generate a valid JSON plan."}
        return {"error": "No JSON plan was generated."}