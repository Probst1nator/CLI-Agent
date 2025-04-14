from typing import Any, Optional, Union, List

from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import AIStrengths, LlmRouter
from py_classes.cls_util_base import UtilBase

class ToBool(UtilBase):
    """
    A utility for converting a raw prompt (optionally with some content) to a boolean value.
    """
    
    @staticmethod
    def run(prompt: str, content: Optional[str]=None) -> bool:
        """
        Send a prompt to an ai and get a boolean response.
        Use this as a intelligent in-code heuristic for dynamic decision making.
        
        Args:
            prompt: The prompt to send to the language model
            content: Optional content (raw file contents, long command output, etc.)
            
        Returns:
            An ai determined boolean value (True or False)
        """
        # Implement the LLM calling logic here
        # This is a placeholder implementation
        instruction = "Answer only with a yes or no"
        
        if content:
            combined_prompt = f"```content\n{content}\n```"
            combined_prompt += f"Instruction: {instruction}\n\nPrompt: {prompt}"
        else:
            combined_prompt = prompt
                    
        chat = Chat(instruction)
        chat.add_message(Role.USER, combined_prompt)
            
        response = LlmRouter.generate_completion(chat, strengths=[AIStrengths.REASONING], exclude_reasoning_tokens=True, force_local=True).strip()
        
        return "yes" in response.lower()