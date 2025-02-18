import typing
from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

def generate_white_image(
    output_path: typing.Union[str, Path],
    prompt: str = "A pure white background, minimalist, clean",
    width: int = 512,
    height: int = 512
) -> None:
    # Initialize the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    
    # Generate the image
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    
    # Save the image
    output_path = Path(output_path)
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    downloads_path = Path.home() / "Downloads" / "white_sd.png"
    generate_white_image(downloads_path)
