import os
from typing import List, Tuple
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

class StableDiffusion:
    """
    A class to handle Stable Diffusion operations including listing downloaded models
    and generating images from text prompts.
    """

    def __init__(self, cache_dir: str = None):
        """
        Initialize the StableDiffusionHandler.

        Args:
            cache_dir (str, optional): The directory where models are cached.
                                       Defaults to ~/.cache/huggingface/diffusers.
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/diffusers")
        self.pipe = None

    def list_downloaded_models(self) -> List[str]:
        """
        List all downloaded Stable Diffusion models in the cache directory.

        Returns:
            List[str]: A list of model names found in the cache directory.
        """
        if not os.path.exists(self.cache_dir):
            print("No models found. The cache directory does not exist.")
            return []
        
        models = [d for d in os.listdir(self.cache_dir) if os.path.isdir(os.path.join(self.cache_dir, d))]
        return models

    def load_model(self, model_id: str):
        """
        Load a Stable Diffusion model.

        Args:
            model_id (str): The identifier of the model to load.
        """
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")
        print(f"Model '{model_id}' loaded successfully.")

    def generate_image(self, prompt: str, output_path: str, 
                       num_inference_steps: int = 50, 
                       guidance_scale: float = 7.5) -> Tuple[Image.Image, str]:
        """
        Generate an image from a text prompt using the loaded Stable Diffusion model.

        Args:
            prompt (str): The text prompt to generate the image from.
            output_path (str): The path where the generated image will be saved.
            num_inference_steps (int, optional): Number of denoising steps. Defaults to 50.
            guidance_scale (float, optional): Scale for classifier-free guidance. Defaults to 7.5.

        Returns:
            Tuple[Image.Image, str]: The generated image and the path where it was saved.

        Raises:
            ValueError: If no model has been loaded.
        """
        if self.pipe is None:
            raise ValueError("No model loaded. Please load a model first using load_model().")

        image = self.pipe(prompt, num_inference_steps=num_inference_steps, 
                          guidance_scale=guidance_scale).images[0]
        image.save(output_path)
        print(f"Image generated and saved as '{output_path}'")
        return image, output_path

# Example usage
if __name__ == "__main__":
    handler = StableDiffusion()
    
    # List downloaded models
    models = handler.list_downloaded_models()
    if models:
        print("Downloaded Stable Diffusion models:")
        for model in models:
            print(f"- {model}")
    else:
        print("No downloaded models found.")

    # Load a model
    handler.load_model("CompVis/stable-diffusion-v1-4")

    # Generate an image
    prompt = "A beautiful landscape with mountains and a lake"
    handler.generate_image(prompt, "generated_image.png")