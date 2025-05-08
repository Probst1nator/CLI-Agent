from typing import Any, Optional, Union, List
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_util_base import UtilBase

class ToBool(UtilBase):
    """
    A utility for converting a raw prompt (optionally with some content) to a boolean value.
    """
    @staticmethod
    def run(prompt: str, content: Optional[str]=None) -> bool:
        """
        Send a prompt to an llm and get a boolean response.
        Use this as a intelligent heuristic for dynamic coding requirements.
        Args:
            prompt: The prompt to send to the language model
            content: Optional content (raw file content, command output, long text, etc.)
        Returns:
            A boolean value
        """
        # Implement the LLM calling logic here
        instruction = """Analyze the following question and determine if the answer should be 'yes' or 'no'.

First, write out your step-by-step reasoning process:
1. Identify the key parts of the question
2. Consider relevant context and implications
3. Evaluate different perspectives or possibilities
4. Weigh the evidence

After your reasoning, provide your final answer as ONLY 'yes' or 'no' on the last line."""
        
        if content:
            combined_prompt = f"```content\n{content}\n```"
            combined_prompt += f"Question: {prompt}\n\nPlease analyze and provide a careful reasoning before answering with yes or no."
        else:
            combined_prompt = f"Question: {prompt}\n\nPlease analyze and provide a careful reasoning before answering with yes or no."
        
        chat = Chat(instruction)
        chat.add_message(Role.USER, combined_prompt)
        
        full_response = LlmRouter.generate_completion(chat, strengths=[AIStrengths.REASONING], exclude_reasoning_tokens=False, force_local=True)
        
        # Extract just the final yes/no answer
        lines = full_response.strip().split('\n')
        final_line = lines[-1].lower()
        
        return "yes" in final_line and "no" not in final_line  # Ensure we don't get confused by "not yes" or similar