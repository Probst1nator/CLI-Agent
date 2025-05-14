import os
import torch
from PIL import Image
from typing import Optional, Any, Dict, Tuple, Callable
from functools import lru_cache
import importlib
import warnings

# Import base class
from py_classes.cls_util_base import UtilBase

class GenerateImage(UtilBase):
    """
    A utility for generating an image based on a text prompt using Hugging Face
    diffusers library (e.g., Stable Diffusion) and saving it to a file.
    
    This utility lazily loads the specified diffusion model pipeline only when needed,
    generates an image, and saves the result to the specified path. It includes 
    basic caching for loaded pipelines within the same process execution.
    """
    # Dictionary to store loaded pipelines
    _loaded_pipelines: Dict[Tuple[str, torch.dtype, torch.device], Any] = {}
    
    # Lazy imports
    _diffusers_imports = None
    
    @classmethod
    def _ensure_dependencies(cls):
        """Lazily import dependencies only when needed."""
        if cls._diffusers_imports is None:
            try:
                # Create a dictionary of lazy-loaded imports
                cls._diffusers_imports = {
                    'AutoPipelineForText2Image': importlib.import_module('diffusers').AutoPipelineForText2Image
                }
                return True
            except ImportError:
                error_msg = (
                    "Required libraries not found. Please install them:\n"
                    "pip install torch diffusers transformers accelerate Pillow"
                )
                warnings.warn(error_msg)
                return False
        return True

    @classmethod
    def _get_pipeline(
        cls,
        model_id: str,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Loads or retrieves a cached diffusion pipeline on demand."""
        if not cls._ensure_dependencies():
            raise ImportError("Required dependencies not available. See warnings for details.")
            
        cache_key = (model_id, dtype, device)
        
        # Return cached pipeline if available
        if cache_key in cls._loaded_pipelines:
            print(f"Using cached pipeline for {model_id} on {device} with {dtype}")
            return cls._loaded_pipelines[cache_key]
            
        # Lazily load the pipeline only when requested
        print(f"Loading pipeline {model_id} to {device} with {dtype}...")
        
        # Get the AutoPipelineForText2Image class from our lazy imports
        AutoPipelineForText2Image = cls._diffusers_imports['AutoPipelineForText2Image']
        
        # Create the pipeline
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,         # Disable safety checker
            feature_extractor=None,      # No need for feature extractor when safety checker is disabled
            requires_safety_checker=False # Explicitly disable safety checking
        ).to(device)
        
        # Optional: Enable memory-efficient optimizations if available
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except (AttributeError, ImportError):
            try:
                pipeline.enable_attention_slicing()
                print("Enabled attention slicing")
            except AttributeError:
                print("Could not enable memory optimizations")
        
        print(f"Pipeline {model_id} loaded successfully.")
        cls._loaded_pipelines[cache_key] = pipeline
        return pipeline
    
    @classmethod
    def unload_pipeline(cls, model_id: str = None, device: torch.device = None, dtype: torch.dtype = None):
        """
        Unload specified pipelines to free memory.
        If no parameters provided, unloads all pipelines.
        """
        keys_to_remove = []
        
        for key in cls._loaded_pipelines.keys():
            key_model_id, key_dtype, key_device = key
            if (model_id is None or key_model_id == model_id) and \
               (device is None or key_device == device) and \
               (dtype is None or key_dtype == dtype):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            # Get the pipeline
            pipeline = cls._loaded_pipelines[key]
            # Move to CPU first if on GPU to help with CUDA memory
            if hasattr(pipeline, 'to') and str(pipeline.device) != 'cpu':
                try:
                    pipeline.to('cpu')
                except:
                    pass  # Best effort to move to CPU
            # Remove from cache
            del cls._loaded_pipelines[key]
            
        # Force garbage collection to help free memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return len(keys_to_remove)
    
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
    
    @classmethod
    def run(
        cls,
        path: str,
        prompt: str,
        negative_prompt: str = "blurry, low quality, deformed",
        model_id: str = "runwayml/stable-diffusion-v1-5",
        size: str = "512x512",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> str:
        """
        Generates an image based on a text prompt and saves it to the specified path.
        
        Args:
            path: The file path where the generated image should be saved
            prompt: The text description of the image to generate
            negative_prompt: Optional text describing elements to avoid in the image
            model_id: The model ID to use for generation
            size: Image size in format "WxH" (e.g., "512x512")
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Optional seed for reproducibility
            
        Returns:
            The absolute path to the saved image file
        """
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
            # Load or retrieve cached pipeline (lazy loading happens here)
            pipeline = cls._get_pipeline(model_id, device, dtype)
            
            # Parse size string to dimensions
            dimensions = cls._parse_size(size) or (512, 512)
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
    
    # --- Basic SD v1.5 Test (GPU recommended) ---
    if torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("\n--- Test Case 1: Stable Diffusion v1.5 (GPU/MPS) ---")
        try:
            saved_path1 = GenerateImage.run(
                path=output_path1,
                prompt="A photorealistic high quality image of a fluffy ginger cat sitting on a sunny windowsill, looking curious",
                model_id="runwayml/stable-diffusion-v1-5",
                size="512x512",
                num_inference_steps=40,
                guidance_scale=8.0,
                negative_prompt="blurry, low quality, deformed, text, signature",
                seed=12345
            )
            print(f"Image saved successfully to: {saved_path1}")
            assert os.path.exists(saved_path1)
            print(f"File exists: {os.path.exists(saved_path1)}")
            
            # Test unloading the pipeline
            print("Testing pipeline unloading...")
            unloaded = GenerateImage.unload_pipeline()
            print(f"Unloaded {unloaded} pipeline(s)")
            
        except Exception as e:
            print(f"Test Case 1 FAILED: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n--- Skipping Test Case 1: No CUDA/MPS GPU detected ---")
        
    print("\n--- Test Case 2: Empty Prompt (expect ValueError) ---")
    try:
        GenerateImage.run(path=os.path.join(temp_dir, "error.png"), prompt="")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    except Exception as e:
        print(f"Test Case 2 FAILED with unexpected error: {e}")
        
    print("\n--- Finished testing ---")