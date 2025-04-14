from typing import Any, Optional, Union, List

from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import AIStrengths, LlmRouter
from py_classes.cls_util_base import UtilBase

class AskLlm(UtilBase):
    """
    A utility for asking questions to a language model.
    
    This utility sends prompts to a language model and returns the response.
    """
    
    @staticmethod
    def run(prompt: str, content: Optional[str]=None, instruction: str="Answer only with a yes or no") -> str:
        """
        Send a prompt to a language model and get the response.
        Use this as a more intelligent heuristic for decision making.
        
        Args:
            prompt: The prompt to send to the language model
            content: Optional content (Like file content or other potentially large text)
            instruction: Instructions for how the model should respond
            
        Returns:
            The string response from the language model
            
        Example:
            ```python
            # Check if a code snippet contains TypeScript
            result = AskLlmUtil.run(
                prompt="Does this code contain TypeScript?", 
                content=file.read(),
                instruction="Answer only with 'yes' or 'no'"
            )
            does_contain_typescript = result == "yes"
            ```
        """
        # Implement the LLM calling logic here
        # This is a placeholder implementation
        
        if content:
            combined_prompt = f"```content\n{content}\n```"
            combined_prompt += f"Instruction: {instruction}\n\nPrompt: {prompt}"
        else:
            combined_prompt = prompt
                    
        chat = Chat(instruction)
        chat.add_message(Role.USER, combined_prompt)
            
        response = LlmRouter.generate_completion(chat, strengths=[AIStrengths.REASONING], exclude_reasoning_tokens=True, force_local=True).strip()
        
        return response