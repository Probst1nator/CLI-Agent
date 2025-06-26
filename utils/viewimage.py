import base64
import os

from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_util_base import UtilBase


class ViewImage(UtilBase):
    """
    A utility for viewing an image and getting a description of it.
    
    This utility allows viewing an image and getting a description of it.
    """
    
    @staticmethod
    async def run(
        file_path: str,
        prompt: str = "This image shows a screenshot of the current screen. There should be a video player playing a video. What is the name of the video and whats the like to dislike ratio?"
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
        instruction = """You are an expert at analyzing and describing images.
        Provide a clear and informative description of the image shown.
        Focus on relevant details including:
        - Main subjects and objects
        - Visual composition and layout
        - Text content if any
        - Context and setting
        - Any notable features or unusual elements
        
        Your description should be thorough but focused on what matters.
        """
        
        chat = Chat(instruction, debug_title="Viewing Image")
        
        # Add user message with the custom prompt
        chat.add_message(Role.USER, prompt)
        
        # Add the image to the chat
        chat.add_image_message(base64_image, role=Role.USER)
        
        try:
            response = await LlmRouter.generate_completion(
                chat, 
                strengths=[AIStrengths.STRONG], 
                exclude_reasoning_tokens=True,
                hidden_reason="Viewing Image"
            )
            return response
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
