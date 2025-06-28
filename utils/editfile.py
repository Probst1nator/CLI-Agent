import os
import re
import json
import asyncio
from typing import Optional, Coroutine, Any
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
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError: pass
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)

def _extract_block(text: str, language: str) -> Optional[str]:
    """Helper to robustly extract content from a markdown code block."""
    pattern = re.compile(rf'```{language}\n(.*?)\n```', re.DOTALL)
    match = pattern.search(text)
    if match: return match.group(1).strip()
    pattern = re.compile(r'```\n(.*?)\n```', re.DOTALL)
    match = pattern.search(text)
    if match: return match.group(1).strip()
    return None

class EditFile(UtilBase):
    """
    Intelligently edits a single file using a two-step LLM process.
    It first decides on an edit strategy, then generates and applies the content.
    """

    @staticmethod
    def run(filepath: str) -> str:
        """
        Reads a file, asks an AI for an edit strategy, gets the content, and applies it.

        Args:
            filepath (str): The path to the file to be edited.

        Returns:
            str: A JSON string with a 'result' key on success, or an 'error' key on failure.
        """
        print(colored(f"📝 EditFile activated for: {filepath}", "cyan"))
        try:
            context_chat = Chat.load_from_json()
            original_content = ""
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f: original_content = f.read()
            
            # STEP 1: Decide on the Edit Strategy (fast, reasoning model)
            instruction_decide = "You are an AI programmer. Decide the best strategy for editing a file. Respond with ONLY the strategy name: `REPLACE_WHOLE_FILE` or `REPLACE_LINE_RANGE`."
            decision_chat = context_chat.deep_copy().set_instruction_message(instruction_decide)
            prompt_decide = f"Based on our conversation, which strategy for `{filepath}`?\n--- FILE CONTENT ---\n{original_content}\n--- END ---"
            decision_chat.add_message(Role.USER, prompt_decide)
            strategy = _run_async_safely(LlmRouter.generate_completion(decision_chat, strengths=[AIStrengths.REASONING], temperature=0.0)).strip()
            print(colored(f"  ├─ AI chose strategy: {strategy}", "yellow"))

            # STEP 2: Get Content Based on Strategy (powerful code model)
            content_chat = context_chat.deep_copy()
            action_summary = ""

            if strategy == "REPLACE_WHOLE_FILE":
                instruction = "You are an expert AI programmer. Provide the full, new content for the file. You MUST respond with a single code block. Do not add any other text."
                content_chat.set_instruction_message(instruction).add_message(Role.USER, f"Provide the complete new content for `{filepath}`.")
                print(colored("  ├─ Step 2: Requesting full content from a stronger model...", "white"))
                response = _run_async_safely(LlmRouter.generate_completion(content_chat, strengths=[AIStrengths.CODE, AIStrengths.STRONG], temperature=0.0))
                new_content = _extract_block(response, "")
                if new_content is None: return json.dumps({"error": "Could not extract code block from AI response."})
                with open(filepath, 'w', encoding='utf-8') as f: f.write(new_content)
                action_summary = "Replaced entire file content."

            elif strategy == "REPLACE_LINE_RANGE":
                instruction = "You are an expert AI programmer. Provide a patch. You MUST respond with a single JSON object in a ```json block with `start_line`, `end_line`, and `new_content` keys."
                content_chat.set_instruction_message(instruction).add_message(Role.USER, f"Provide the JSON patch for `{filepath}`.\n--- ORIGINAL ---\n{original_content}\n--- END ---")
                print(colored("  ├─ Step 2: Requesting JSON patch from a stronger model...", "white"))
                response = _run_async_safely(LlmRouter.generate_completion(content_chat, strengths=[AIStrengths.CODE, AIStrengths.STRONG], temperature=0.0))
                json_str = _extract_block(response, "json")
                if json_str is None: return json.dumps({"error": "Could not extract JSON block from AI response."})
                patch_data = json.loads(json_str)
                lines = original_content.splitlines(True)
                start, end, new_snippet = patch_data['start_line'], patch_data['end_line'], patch_data['new_content']
                new_lines = lines[:start - 1] + new_snippet.splitlines(True) + lines[end:]
                with open(filepath, 'w', encoding='utf-8') as f: f.writelines(new_lines)
                action_summary = f"Patched lines {start}-{end}."
            else:
                return json.dumps({"error": f"AI provided an unknown strategy: '{strategy}'."})

            result = {"result": {"status": "Success", "path": os.path.abspath(filepath), "action": action_summary}}
            print(colored(f"  └─ Success! {action_summary}", "green"))
            return json.dumps(result, indent=2)

        except Exception as e:
            import traceback
            return json.dumps({"error": f"An unexpected error in EditFile: {e}", "details": traceback.format_exc()})