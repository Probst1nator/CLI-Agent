from typing import Optional
import base64
import os

from py_classes.cls_chat import Chat, Role
from py_classes.cls_llm_router import LlmRouter
from py_classes.enum_ai_strengths import AIStrengths
from py_classes.cls_util_base import UtilBase


class ImageToText(UtilBase):
    """
    A utility for converting an image to descriptive text.
    
    This utility allows converting an image file into a detailed text 
    description using vision-capable language models.
    """
    
    @staticmethod
    def run(
        image_path: str,
        prompt: Optional[str] = "Describe this image in detail."
    ) -> str:
        """
        Convert an image to descriptive text.
        
        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt to guide the image description and focus attention on specific features
            
        Returns:
            A text description of the image
        """
        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Convert image file to base64
        with open(image_path, "rb") as image_file:
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
        
        # Create the chat object with the instruction
        chat = Chat(instruction, debug_title="Image to Text")
        
        # Add user message with the custom prompt
        chat.add_message(Role.USER, prompt)
        
        # Generate the completion with the base64 image
        response = LlmRouter.generate_completion(
            chat,
            base64_images=[base64_image],
            strengths=[AIStrengths.VISION],
            exclude_reasoning_tokens=True,
            hidden_reason="Image to Text"
        )
        
        return response
