import base64
import os
import asyncio
from typing import Any, Coroutine, Dict
from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_util_base import UtilBase

def _run_async_safely(coro: Coroutine) -> Any:
    """
    Helper function to run async code from sync context safely.
    Handles both cases: when called from sync context and from async context.
    """
    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
        # We're in an async context - this is the tricky case
        # Instead of using thread pools (which cause signal issues), 
        # let's try to use nest_asyncio to allow nested event loops
        try:
            import nest_asyncio
            nest_asyncio.apply(loop)
            return asyncio.run(coro)
        except ImportError:
            # nest_asyncio not available, fall back to a blocking wait approach
            # Create a task and busy-wait for it (not ideal but avoids threading issues)
            task = loop.create_task(coro)
            import time
            while not task.done():
                time.sleep(0.001)  # Small sleep to prevent busy waiting
            return task.result()
    except RuntimeError:
        # No event loop running, we can use asyncio.run() safely
        return asyncio.run(coro)


class ViewImage(UtilBase):
    """
    A utility for viewing an image and getting a description of it.
    
    This utility allows viewing an image and getting a description of it.
    """
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["describe image", "view picture", "analyze image", "what is in this image"],
            "use_cases": [
                "Describe the contents of the image 'screenshot.png'.",
                "What does the image at 'images/graph.jpg' show?",
                "Analyze the attached picture and tell me what it is."
            ],
            "arguments": {
                "file_path": "The path to the image file.",
                "prompt": "A prompt to guide the description."
            },
            "code_examples": [
                {
                    "description": "Describe an image",
                    "code": "from utils.viewimage import ViewImage\nresult = ViewImage.run(file_path='image.png')"
                },
                {
                    "description": "Describe an image with a specific prompt",
                    "code": "from utils.viewimage import ViewImage\nresult = ViewImage.run(file_path='image.png', prompt='Focus on the text in the image.')"
                }
            ]
        }


    @staticmethod
    def _run_logic(
        file_path: str,
        prompt: str = "Please describe this image in detail. Read out text and symbols if there are any."
    ) -> str:
        """
        Inspect an image, optionally using a prompt.
        
        Args:
            file_path: str, Path to the image file
            prompt: str, Prompt to guide the image description and focus attention on specific features
            
        Returns:
            str, A text description of the image
        """
        # Validate image path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Convert image file to base64
        with open(file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Create instruction for image analysis
        instruction = """You are an ai tasked with describing an image for further processing.
        Provide a clear and informative description of the image and its contents.
        Focus on relevant details including:
        - Main subjects and objects
        - Visual composition and layout
        - Text content if any
        - Context and setting
        - Any notable features or unusual elements
        
        Your description should be thorough but focused on what is asked.
        """
        
        chat = Chat(instruction, debug_title="Viewing Image")
        
        # Add user message with the custom prompt
        chat.add_message(Role.USER, prompt)
        
        # The image is passed directly to the LlmRouter now.
        try:
            response = _run_async_safely(LlmRouter.generate_completion(
                chat,
                base64_images=[base64_image],
                strengths=[AIStrengths.VISION], 
                exclude_reasoning_tokens=True,
                hidden_reason="Viewing Image"
            ))
            return response
        except Exception as e:
            return f"Error analyzing image: {str(e)}"


# Module-level run function for CLI-Agent compatibility  
def run(file_path: str, prompt: str = "Please describe this image in detail. Read out text and symbols if there are any.") -> str:
    """
    Module-level wrapper for ViewImage._run_logic() to maintain compatibility with CLI-Agent.
    
    Args:
        file_path (str): Path to the image file
        prompt (str): Prompt for image analysis
        
    Returns:
        str: Image analysis result or error message
    """
    return ViewImage._run_logic(file_path=file_path, prompt=prompt)
