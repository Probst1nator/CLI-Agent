import numpy as np
import multiprocessing as mp
from PIL import Image
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
import argparse

@dataclass
class MandelbrotConfig:
    width: int = 800
    height: int = 600
    max_iter: int = 100
    x_min: float = -2.0
    x_max: float = 1.0
    y_min: float = -1.5
    y_max: float = 1.5
    cpu_cores: int = 4

def mandelbrot_point(c: complex, max_iter: int) -> int:
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def process_row(args: Tuple[int, np.ndarray, MandelbrotConfig]) -> Tuple[int, np.ndarray]:
    y_idx, x_range, config = args
    row = np.zeros(config.width, dtype=np.uint8)
    y = config.y_min + (y_idx * (config.y_max - config.y_min) / config.height)
    
    for x_idx in range(config.width):
        x = config.x_min + (x_idx * (config.x_max - config.x_min) / config.width)
        c = complex(x, y)
        row[x_idx] = mandelbrot_point(c, config.max_iter)
    return y_idx, row

def generate_mandelbrot(config: MandelbrotConfig) -> None:
    image_array = np.zeros((config.height, config.width), dtype=np.uint8)
    pool = mp.Pool(processes=config.cpu_cores)
    
    # Prepare arguments for each row
    row_args = [(y, np.arange(config.width), config) for y in range(config.height)]
    
    # Process rows with progress visualization
    for y_idx, row in pool.imap_unordered(process_row, row_args):
        image_array[y_idx] = row
        if y_idx % 10 == 0:  # Update visualization every 10 rows
            # Normalize and convert to image
            normalized = (image_array * 255 / config.max_iter).astype(np.uint8)
            img = Image.fromarray(normalized, 'L')
            img.show()
            time.sleep(0.1)  # Slow down visualization
            img.close()
    
    pool.close()
    pool.join()
    
    # Final image
    normalized = (image_array * 255 / config.max_iter).astype(np.uint8)
    final_img = Image.fromarray(normalized, 'L')
    final_img.save('mandelbrot.png')
    final_img.show()

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Mandelbrot Set with adjustable CPU cores')
    parser.add_argument('--cores', type=int, default=4, help='Number of CPU cores to use')
    parser.add_argument('--width', type=int, default=800, help='Image width')
    parser.add_argument('--height', type=int, default=600, help='Image height')
    parser.add_argument('--max-iter', type=int, default=100, help='Maximum iterations')
    
    args = parser.parse_args()
    
    config = MandelbrotConfig(
        width=args.width,
        height=args.height,
        max_iter=args.max_iter,
        cpu_cores=args.cores
    )
    
    print(f"Generating Mandelbrot set using {config.cpu_cores} CPU cores...")
    generate_mandelbrot(config)

if __name__ == "__main__":
    main()
