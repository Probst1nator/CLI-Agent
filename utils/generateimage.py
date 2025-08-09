import os
import gc
import json
import sys
import torch
import warnings
from typing import Optional, Dict, Any

# Add parent directory to path to import py_classes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_classes.cls_util_base import UtilBase

try:
    import diffusers
    import transformers
    import accelerate
    from diffusers import FluxPipeline
    from diffusers.utils import logging as diffusers_logging
    diffusers_logging.set_verbosity_error()
    _DEPS_INSTALLED = True
except ImportError:
    _DEPS_INSTALLED = False
    FluxPipeline = None  # Define as None when dependencies aren't available

# Check for optional bitsandbytes (only needed for quantization)
try:
    import bitsandbytes
    _BITSANDBYTES_AVAILABLE = True
except ImportError:
    _BITSANDBYTES_AVAILABLE = False

class GenerateImage(UtilBase):
    """
    A utility for generating images using the FLUX model.
    Optimized for running on hardware with limited VRAM (e.g., 8GB).
    """
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["create image", "draw picture", "generate art", "diffusion model", "text to image", "dall-e", "midjourney"],
            "use_cases": [
                "Generate an image of a red fox in a snowy forest.",
                "Create a photorealistic picture of a futuristic city.",
                "Draw a logo for a coffee shop."
            ],
            "arguments": {
                "path": "The file path where the generated image will be saved (e.g., 'images/fox.png').",
                "prompt": "A detailed text description of the image to generate.",
                "width": "The width of the output image in pixels.",
                "height": "The height of the output image in pixels.",
                "seed": "An integer for reproducible image generation.",
                "num_inference_steps": "Number of diffusion steps. Lower is faster, higher may be better quality.",
                "enable_quantization": "Use 4-bit quantization to save VRAM (recommended for <12GB VRAM).",
                "enable_cpu_offloading": "Offload parts of the model to system RAM to save VRAM (recommended for <12GB VRAM)."
            },
            "code_examples": [
                {
                    "description": "Generate a simple image",
                    "code": "from utils.generateimage import GenerateImage\nresult = GenerateImage.run(path='output.png', prompt='A cat sitting on a table')"
                },
                {
                    "description": "Generate a more complex image with specific dimensions",
                    "code": "from utils.generateimage import GenerateImage\nresult = GenerateImage.run(path='output.png', prompt='A photorealistic image of a futuristic city at night', width=1024, height=768)"
                }
            ]
        }
    
    _pipeline: Optional['FluxPipeline'] = None  # Use string annotation
    _pipeline_options: Dict[str, Any] = {}

    @classmethod
    def _ensure_dependencies(cls) -> bool:
        """Checks if all required dependencies are installed."""
        if not _DEPS_INSTALLED:
            warnings.warn(
                "Required packages not found. Please install them to use this utility:\n"
                "pip install torch diffusers transformers accelerate"
            )
        return _DEPS_INSTALLED

    @classmethod
    def _initialize_pipeline(
        cls,
        enable_quantization: bool = False,
        enable_cpu_offloading: bool = False
    ) -> bool:
        """
        Initializes and configures the FLUX pipeline.
        This method is called automatically by `run` if the pipeline is not loaded.
        """
        if not cls._ensure_dependencies():
            return False

        options = {
            "enable_quantization": enable_quantization,
            "enable_cpu_offloading": enable_cpu_offloading
        }

        if cls._pipeline is not None and cls._pipeline_options == options:
            return True # Pipeline already loaded with the same options

        if cls._pipeline is not None:
            cls.unload() # Unload existing pipeline if options differ

        cls._pipeline_options = options
        print("Initializing FLUX pipeline...")
        print(f"Options: Quantization={'Enabled' if enable_quantization else 'Disabled'}, CPU Offloading={'Enabled' if enable_cpu_offloading else 'Disabled'}")

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == 'cpu':
                print("Warning: Running on CPU. This will be very slow.")
                if enable_quantization:
                    warnings.warn("Quantization is not supported on CPU and will be disabled.")
                    enable_quantization = False

            model_id = "black-forest-labs/FLUX.1-schnell"
            pipe_kwargs = {"torch_dtype": torch.bfloat16}

            if enable_quantization:
                if not _BITSANDBYTES_AVAILABLE:
                    warnings.warn("Quantization requested but bitsandbytes is not available. Install with: pip install bitsandbytes")
                    enable_quantization = False
                else:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    pipe_kwargs["transformer_loading_kwargs"] = {
                        "quantization_config": quantization_config
                    }

            cls._pipeline = FluxPipeline.from_pretrained(model_id, **pipe_kwargs)

            if enable_cpu_offloading:
                cls._pipeline.enable_model_cpu_offload()
                print("Enabled CPU offloading.")
            else:
                cls._pipeline.to(device)

            cls._pipeline.enable_attention_slicing()
            # Additional memory optimizations
            if hasattr(cls._pipeline, 'enable_vae_slicing'):
                cls._pipeline.enable_vae_slicing()
            if hasattr(cls._pipeline, 'enable_vae_tiling'):
                cls._pipeline.enable_vae_tiling()
            print("FLUX pipeline initialized successfully.")
            return True

        except Exception as e:
            cls._pipeline = None
            print(f"Error initializing FLUX pipeline: {e}")
            return False

    @classmethod
    def unload(cls) -> Dict[str, Any]:
        """Unloads the pipeline and clears memory."""
        if cls._pipeline is not None:
            # Check if model is on CUDA before trying to move it
            if hasattr(cls._pipeline.transformer, 'device') and 'cuda' in str(cls._pipeline.transformer.device):
                cls._pipeline.to('cpu')

            del cls._pipeline
            cls._pipeline = None
            cls._pipeline_options = {}

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("FLUX pipeline unloaded and memory cleared.")
            return {"status": "success", "message": "Pipeline unloaded."}
        
        return {"status": "success", "message": "Pipeline was not loaded."}

    @classmethod
    def get_memory_usage(cls) -> Dict[str, Any]:
        """Returns current VRAM usage if CUDA is available."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            return {
                "vram_allocated_gb": f"{allocated:.2f}",
                "vram_reserved_gb": f"{reserved:.2f}"
            }
        return {"message": "CUDA not available."}

    @staticmethod
    def _run_logic(
        path: str,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        num_inference_steps: int = 4,
        enable_quantization: bool = False,
        enable_cpu_offloading: bool = False
    ) -> str:
        """
        Generates an image with FLUX and saves it to the specified path.

        Args:
            path: The file path to save the image (e.g., 'output.png').
            prompt: The text prompt to generate the image from.
            width: The width of the image. Defaults to 1024.
            height: The height of the image. Defaults to 1024.
            seed: A seed for reproducible results.
            num_inference_steps: The number of steps for the diffusion process.
            enable_quantization: Load model in 4-bit for lower VRAM usage.
            enable_cpu_offloading: Offload parts of the model to CPU to save VRAM.

        Returns:
            A JSON string with the result or an error message.
        """
        try:
            if not GenerateImage._initialize_pipeline(enable_quantization, enable_cpu_offloading):
                 raise RuntimeError("Pipeline initialization failed.")

            if not path or not prompt:
                raise ValueError("Path and prompt cannot be empty.")

            output_dir = os.path.dirname(path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Clear cache before generation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None and torch.cuda.is_available() else None
            
            with torch.no_grad():
                image = GenerateImage._pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                ).images[0]

            image.save(path)
            abs_path = os.path.abspath(path)
            
            # Clean up after generation to save memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return json.dumps({
                "result": {
                    "image_path": abs_path,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "seed": seed,
                    "steps": num_inference_steps
                }
            })

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            return json.dumps({"error": error_message})


# Module-level run function for CLI-Agent compatibility
def run(path: str, prompt: str, width: int = 1024, height: int = 1024, seed: Optional[int] = None, num_inference_steps: int = 4, enable_quantization: bool = False, enable_cpu_offloading: bool = False) -> str:
    """
    Module-level wrapper for GenerateImage._run_logic() to maintain compatibility with CLI-Agent.
    
    Args:
        path (str): The file path to save the image
        prompt (str): The text prompt to generate the image from
        width (int): The width of the image. Defaults to 1024
        height (int): The height of the image. Defaults to 1024
        seed (Optional[int]): A seed for reproducible results
        num_inference_steps (int): The number of steps for the diffusion process
        enable_quantization (bool): Load model in 4-bit for lower VRAM usage
        enable_cpu_offloading (bool): Offload parts of the model to CPU to save VRAM
        
    Returns:
        str: JSON string with result or error
    """
    return GenerateImage._run_logic(path=path, prompt=prompt, width=width, height=height, seed=seed, num_inference_steps=num_inference_steps, enable_quantization=enable_quantization, enable_cpu_offloading=enable_cpu_offloading)

if __name__ == '__main__':
    # Example usage for testing
    if _DEPS_INSTALLED:
        # For 8GB VRAM, enable quantization and cpu offloading
        # For >12GB VRAM, you can set both to False for faster generation
        is_8gb_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) < 12 if torch.cuda.is_available() else False
        
        print(f"\n--- Generating image ({'8GB VRAM optimized' if is_8gb_vram else 'High VRAM mode'}) ---")
        result_json = GenerateImage._run_logic(
            path="flux_test_image.png",
            prompt="A majestic lion overlooking the savannah at sunset, cinematic lighting",
            width=512,  # Smaller size for testing
            height=512,
            seed=123,
            enable_quantization=is_8gb_vram and _BITSANDBYTES_AVAILABLE,
            enable_cpu_offloading=is_8gb_vram
        )
        print(f"Result: {result_json}")

        print("\n--- Checking memory usage ---")
        print(GenerateImage.get_memory_usage())

        print("\n--- Unloading model ---")
        print(GenerateImage.unload())

        print("\n--- Memory usage after unload ---")
        print(GenerateImage.get_memory_usage())
    else:
        print("Skipping example: Dependencies not installed.")
        GenerateImage._ensure_dependencies()