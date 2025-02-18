import pyglet
from typing import Tuple

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MAX_ITER = 256

# Julia set parameters
C = complex(-0.7, 0.27015)

def julia_set_point(c: complex, z0: complex, max_iter: int) -> int:
    """
    Calculate the number of iterations for a given point in the Julia set.
    
    :param c: The constant used to generate the Julia set
    :param z0: The initial complex number (point)
    :param max_iter: Maximum number of iterations
    :return: Number of iterations before divergence or max_iter if it diverges
    """
    z = z0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

def create_julia_set(width: int, height: int, x_min: float, x_max: float, y_min: float, y_max: float, c: complex, max_iter: int) -> Tuple[Tuple[int, ...], ...]:
    """
    Create a Julia set using the given parameters.
    
    :param width: Width of the image
    :param height: Height of the image
    :param x_min: Minimum real part of the complex plane
    :param x_max: Maximum real part of the complex plane
    :param y_min: Minimum imaginary part of the complex plane
    :param y_max: Maximum imaginary part of the complex plane
    :param c: The constant used to generate the Julia set
    :param max_iter: Maximum number of iterations
    :return: A tuple representing the Julia set image
    """
    pixels = [[0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            # Map pixel coordinates to complex plane
            zx = (x / width) * (x_max - x_min) + x_min
            zy = (y / height) * (y_max - y_min) + y_min
            z = complex(zx, zy)
            pixels[y][x] = julia_set_point(c, z, max_iter)
    return tuple(tuple(row) for row in pixels)

def main():
    # Create a Julia set image
    julia_image = create_julia_set(WINDOW_WIDTH, WINDOW_HEIGHT, -2.0, 1.0, -1.5, 1.5, C, MAX_ITER)
    
    # Create a Pyglet window
    window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
    
    @window.event
    def on_draw():
        window.clear()
        batch = pyglet.graphics.Batch()
        for y in range(WINDOW_HEIGHT):
            for x in range(WINDOW_WIDTH):
                color = julia_image[y][x]
                # Map iteration count to color (simple grayscale for now)
                intensity = min(color / MAX_ITER * 255, 255)
                batch.add(1, pyglet.gl.GL_POINTS, None,
                          ('v2i', (x, y)),
                          ('c3B', (intensity, intensity, intensity)))
    
    # Run the application
    pyglet.app.run()

if __name__ == "__main__":
    main()