import os
import torch
from PIL import Image
from typing import Optional, Any, Dict, Tuple

# --- Required Hugging Face Imports ---
try:
    from diffusers import AutoPipelineForText2Image
except ImportError:
    print("Error: Required libraries not found. Please install them:")
    print("pip install torch diffusers transformers accelerate Pillow")
    # You might want to raise the error or exit depending on your application structure
    raise

# Assume these base class imports exist from your project structure
from py_classes.cls_util_base import UtilBase
# from py_classes.cls_llm_router import LlmRouter # Not directly used here anymore
# from py_classes.enum_ai_strengths import AIStrengths # Not directly used here


class GenerateImage(UtilBase):
    """
    A utility for generating an image based on a text prompt using Hugging Face
    diffusers library (e.g., Stable Diffusion) and saving it to a file.

    This utility loads a specified diffusion model pipeline, generates an image,
    and saves the result to the specified path. It includes basic caching for
    loaded pipelines within the same process execution.
    """

    _loaded_pipelines: Dict[Tuple[str, torch.dtype, torch.device], Any] = {}

    @staticmethod
    def _get_pipeline(
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Loads or retrieves a cached diffusion pipeline."""
        cache_key = (model_id, dtype, device)
        if cache_key in GenerateImage._loaded_pipelines:
            print(f"Using cached pipeline for {model_id} on {device} with {dtype}")
            return GenerateImage._loaded_pipelines[cache_key]

        print(f"Loading pipeline {model_id} to {device} with {dtype}...")

        # Use AutoPipeline for flexibility with safety checker disabled
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,         # Disable safety checker
            feature_extractor=None,      # No need for feature extractor when safety checker is disabled
            requires_safety_checker=False # Explicitly disable safety checking
        ).to(device)

        # Optional: Enable memory-efficient optimizations if available/needed
        # pipeline.enable_xformers_memory_efficient_attention() # requires pip install xformers
        # pipeline.enable_attention_slicing() # Alternative if xformers not available

        print(f"Pipeline {model_id} loaded successfully.")
        GenerateImage._loaded_pipelines[cache_key] = pipeline
        return pipeline

    @staticmethod
    def _parse_size(size_str: str) -> Optional[Tuple[int, int]]:
        """Parses WxH string like '1024x1024' into (width, height)."""
        if not size_str or 'x' not in size_str:
            return None
        try:
            width_str, height_str = size_str.lower().split('x')
            width = int(width_str)
            height = int(height_str)
            # Basic sanity check for reasonable dimensions
            if width > 0 and height > 0 and width <= 4096 and height <= 4096:
                 # Ensure divisible by 8 for many models
                width = (width // 8) * 8
                height = (height // 8) * 8
                return width, height
            else:
                print(f"Warning: Invalid dimensions in size string '{size_str}'.")
                return None
        except ValueError:
            print(f"Warning: Could not parse size string '{size_str}'.")
            return None


    @staticmethod
    def run(
        path: str,
        prompt: str,
        negative_prompt: str = "blurry, low quality, deformed"
    ) -> str:
        """
        Generates an image based on a text prompt and saves it to the specified path.
        
        Args:
            path: The file path where the generated image should be saved
            prompt: The text description of the image to generate
            negative_prompt: Optional text describing elements to avoid in the image
            
        Returns:
            The absolute path to the saved image file
        """
        size: str = "512x512"
        model_id: str = "runwayml/stable-diffusion-v1-5"
        num_inference_steps: int = 30
        guidance_scale: float = 7.5
        seed: Optional[int] = None
        
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        if not path:
            raise ValueError("Output path cannot be empty.")

        # Ensure the output directory exists
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Determine the best device and precision to use
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        try:
            # Load or retrieve cached pipeline
            pipeline = GenerateImage._get_pipeline(model_id, device, dtype)
            
            # Parse size string to dimensions
            dimensions = GenerateImage._parse_size(size) or (512, 512)
            width, height = dimensions
            
            # Set up generator if seed provided
            generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
            
            # Generate the image
            pipeline_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "generator": generator
            }
            call_args = {k: v for k, v in pipeline_args.items() if v is not None}
            
            with torch.no_grad():
                output = pipeline(**call_args)
                image = output.images[0]
            
            # Save the image
            image.save(path)
            
            return os.path.abspath(path)
            
        except Exception as e:
            print(f"Error generating image: {e}")
            raise RuntimeError(f"Failed to generate or save image: {e}")


# --- Example Usage (for testing) ---
if __name__ == "__main__":
    # Create a temporary directory for output
    temp_dir = "temp_generated_images_diffusers"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    output_path1 = os.path.join(temp_dir, "cat_on_windowsill_sd15.png")
    output_path2 = os.path.join(temp_dir, "nested/cyberpunk_city_sdxl.jpg") # Test nested path
    output_path3 = os.path.join(temp_dir, "astronaut_horse_cpu.png") # Test CPU (will be SLOW)

    # --- Basic SD v1.5 Test (GPU recommended) ---
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        print("\n--- Test Case 1: Stable Diffusion v1.5 (GPU/MPS) ---")
        try:
            saved_path1 = GenerateImage.run(
                path=output_path1,
                prompt="A photorealistic high quality image of a fluffy ginger cat sitting on a sunny windowsill, looking curious",
                model_id="runwayml/stable-diffusion-v1-5",
                size="512x512",
                num_inference_steps=40, # Slightly more steps
                guidance_scale=8.0,
                negative_prompt="blurry, low quality, deformed, text, signature",
                seed=12345 # Add seed for reproducibility
            )
            print(f"Image saved successfully to: {saved_path1}")
            assert os.path.exists(saved_path1)
            print(f"File exists: {os.path.exists(saved_path1)}")
        except Exception as e:
            print(f"Test Case 1 FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n--- Skipping Test Case 1: No CUDA/MPS GPU detected ---")


    # --- SDXL Test (Requires more VRAM, GPU highly recommended) ---
    # Note: SDXL often benefits from a refiner model too, which adds complexity.
    # This example uses only the base model.
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        print("\n--- Test Case 2: Stable Diffusion XL Base (GPU/MPS) ---")
        try:
            saved_path2 = GenerateImage.run(
                path=output_path2,
                prompt="Cyberpunk city skyline at night, neon lights reflecting on wet streets, cinematic lighting, hyperrealistic, 8k",
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                size="1024x1024", # Default SDXL size
                num_inference_steps=35,
                guidance_scale=7.0,
                negative_prompt="cartoon, drawing, sketch, low quality, daylight",
                seed=54321
            )
            print(f"Image saved successfully to: {saved_path2}")
            assert os.path.exists(saved_path2)
            print(f"File exists: {os.path.exists(saved_path2)}")
        except Exception as e:
            print(f"Test Case 2 FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n--- Skipping Test Case 2: No CUDA/MPS GPU detected ---")

    # --- CPU Test (Force CPU - EXPECT THIS TO BE VERY SLOW) ---
    # print("\n--- Test Case 3: Stable Diffusion v1.5 (CPU - VERY SLOW) ---")
    # print("WARNING: This test forces CPU execution and will take a long time.")
    # # Temporarily override device detection for testing (not recommended for production)
    # original_cuda_check = torch.cuda.is_available
    # original_mps_check = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    # torch.cuda.is_available = lambda: False
    # if hasattr(torch.backends, "mps"):
    #      torch.backends.mps.is_available = lambda: False
    #
    # try:
    #      saved_path3 = GenerateImage.run(
    #          path=output_path3,
    #          prompt="Astronaut riding a horse on the moon, detailed illustration",
    #          model_id="runwayml/stable-diffusion-v1-5",
    #          size="512x512",
    #          num_inference_steps=15, # Keep steps low for CPU test
    #          guidance_scale=7.0,
    #          seed=9876
    #      )
    #      print(f"Image saved successfully to: {saved_path3}")
    #      assert os.path.exists(saved_path3)
    #      print(f"File exists: {os.path.exists(saved_path3)}")
    # except Exception as e:
    #      print(f"Test Case 3 FAILED: {e}")
    #      import traceback
    #      traceback.print_exc()
    # finally:
    #      # Restore original device checks
    #      torch.cuda.is_available = original_cuda_check
    #      if hasattr(torch.backends, "mps"):
    #            torch.backends.mps.is_available = lambda: original_mps_check
    # print("CPU Test finished (or skipped).")

    print("\n--- Test Case 4: Empty Prompt (expect ValueError) ---")
    try:
        GenerateImage.run(path=os.path.join(temp_dir, "error.png"), prompt="")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Test Case 4 FAILED with unexpected error: {e}")

    print("\n--- Finished testing ---")

    # Optional: Clean up generated files/folders afterwards
    # import shutil
    # if os.path.exists(temp_dir):
    #     print(f"\nCleaning up temporary directory: {temp_dir}")
    #     shutil.rmtree(temp_dir)