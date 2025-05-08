# from typing import Optional, Dict, Any, List
# import os

# from py_classes.cls_chat import Chat, Role
# from py_classes.cls_llm_router import LlmRouter
# from py_classes.enum_ai_strengths import AIStrengths
# from py_classes.cls_util_base import UtilBase

# class AuthorFile(UtilBase):
#     """
#     A utility for generating file content based on a prompt and additional context.
    
#     This utility uses a language model to create or modify file content based on 
#     the provided prompt, file extension, and any additional context. It's useful
#     for creating configuration files, scripts, documentation, or any other text-based
#     files with AI assistance.
#     """
    
#     @staticmethod
#     def run(
#         filepath: str,
#         prompt: str,
#         filetype: str = "py",
#         additional_context: Optional[str] = None,
#         strengths: List[AIStrengths] = [AIStrengths.CODE],
#         temperature: float = 0.1
#     ) -> str:
#         """
#         Generate file content based on a prompt and optional context.
        
#         Args:
#             filepath: The path where the generated file will be saved
#             prompt: The prompt describing what kind of file to generate
#             filetype: The file extension or type (e.g., "py", "js", "md", "json")
#             additional_context: Optional additional context to help guide the generation
#             strengths: List of AI strengths to target for this generation
#             temperature: Temperature setting for generation (lower = more deterministic)
            
#         Returns:
#             The absolute path to the created file
#         """
#         import os
        
#         if not prompt:
#             raise ValueError("Prompt cannot be empty.")
#         if not filepath:
#             raise ValueError("Output filepath cannot be empty.")
        
#         # Ensure the output directory exists
#         output_dir = os.path.dirname(filepath)
#         if output_dir:
#             os.makedirs(output_dir, exist_ok=True)
        
#         # Create the system prompt
#         system_prompt = f"""Use your deep thinking subroutine, you always start with <thinking> and </thinking> tags before responding.
# You are an expert file content generator.
# Your task is to generate content for a {filetype} file.
# You should output ONLY the content of the file, with no explanations or markdown formatting.
# The content should be ready to be written directly to the file.
# """
        
#         # Create the user prompt
#         user_prompt = f"Generate content for a {filetype} file with the following requirements:\n\n{prompt}"
        
#         # Add additional context if provided
#         if additional_context:
#             user_prompt += f"\n\nAdditional context:\n{additional_context}"
        
#         # Create the chat
#         chat = Chat(system_prompt)
#         chat.add_message(Role.USER, user_prompt)
        
#         # Generate the completion
#         content = LlmRouter.generate_completion(
#             chat=chat,
#             strengths=strengths,
#             temperature=temperature,
#             exclude_reasoning_tokens=True
#         )
        
#         # Write the content to the file
#         try:
#             with open(filepath, 'w') as file:
#                 file.write(content)
            
#             return os.path.abspath(filepath)
#         except Exception as e:
#             raise RuntimeError(f"Failed to write content to file: {e}")


# # --- Example Usage (for testing) ---
# if __name__ == "__main__":
#     # Create a temporary directory for output
#     temp_dir = "temp_generated_files"
#     if not os.path.exists(temp_dir):
#         os.makedirs(temp_dir)

#     output_path1 = os.path.join(temp_dir, "example.py")
    
#     try:
#         # Example 1: Generate a Python script
#         saved_path1 = AuthorFile.run(
#             filepath=output_path1,
#             filetype="py",
#             prompt="Create a function that calculates the Fibonacci sequence up to n terms.",
#             additional_context="Make sure to include proper type hints and documentation."
#         )
#         print(f"File saved successfully to: {saved_path1}")
#         print(f"File exists: {os.path.exists(saved_path1)}")
        
#         # Print the content of the generated file
#         with open(saved_path1, 'r') as f:
#             print("\nGenerated content:")
#             print("=" * 40)
#             print(f.read())
#             print("=" * 40)
#     except Exception as e:
#         print(f"Test failed: {e}")
#         import traceback
#         traceback.print_exc()
    
#     # Optional: Clean up generated files/folders afterwards
#     # import shutil
#     # if os.path.exists(temp_dir):
#     #     print(f"\nCleaning up temporary directory: {temp_dir}")
#     #     shutil.rmtree(temp_dir) 