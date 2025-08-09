
import base64
import os
import markpickle
import asyncio
from typing import Dict, Any

try:
    from py_classes.cls_util_base import UtilBase
    from py_classes.cls_llm_router import LlmRouter
    from py_classes.enum_ai_strengths import AIStrengths
except ImportError:
    print("Warning: Could not import required classes. Using mock classes for standalone testing.")
    class UtilBase:
        pass
    class LlmRouter:
        @classmethod
        async def generate_completion(cls, *args, **kwargs):
            return "Mock response: Could not process image due to missing dependencies."
    class AIStrengths:
        VISION = "vision"

class ViewImage(UtilBase):
    """
    A utility to analyze an image using the OpenAI API and return a description.
    """
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """
        Provides standardized metadata for the tool.
        """
        return {
            "keywords": ["image", "vision", "analysis", "describe", "openai"],
            "use_cases": [
                "Describe the contents of an image file.",
                "Analyze a screenshot to understand what is visible."
            ],
            "arguments": {
                "file_path": "The path to the image file to be analyzed.",
                "prompt": "The prompt to guide the image analysis."
            }
        }

    @staticmethod
    def _run_logic(file_path: str, prompt: str = "Please describe this image in detail.") -> str:
        """
        The core implementation of the utility using LlmRouter with vision capabilities.
        """
        if not os.path.exists(file_path):
            return markpickle.dumps({"error": f"File not found at {file_path}"})

        try:
            # Read and encode the image as base64
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Use LlmRouter with vision capabilities to analyze the image
            async def analyze_image():
                return await LlmRouter.generate_completion(
                    chat=prompt,
                    strengths=[AIStrengths.VISION],
                    base64_images=[base64_image],
                    hidden_reason=""
                )
            
            # Run the async function
            analysis_result = asyncio.run(analyze_image())
            
            if not analysis_result or not analysis_result.strip():
                return markpickle.dumps({"error": "No vision-capable models available or analysis failed."})
            
            success_payload = {
                "result": {
                    "status": "Success",
                    "message": "Image analysis was successful.",
                    "data": analysis_result
                }
            }
            return markpickle.dumps(success_payload)

        except Exception as e:
            return markpickle.dumps({"error": f"An unexpected error occurred: {str(e)}"})

def run(file_path: str, prompt: str = "Please describe this image in detail.") -> str:
    """
    Public, module-level entry point for the agent.
    """
    return ViewImage._run_logic(file_path=file_path, prompt=prompt)

