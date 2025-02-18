import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union

def mandelbrot(
    width: int = 800,
    height: int = 600,
    max_iter: int = 100,
    x_range: Tuple[float, float] = (-2, 1),
    y_range: Tuple[float, float] = (-1.5, 1.5)
) -> np.ndarray:
    """
    Generate the Mandelbrot set visualization.
    
    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels
        max_iter (int): Maximum number of iterations
        x_range (Tuple[float, float]): Range for real component
        y_range (Tuple[float, float]): Range for imaginary component
    
    Returns:
        np.ndarray: Array representing the Mandelbrot set
    """
    x = np.linspace(x_range[0], x_range[1], width)
    y = np.linspace(y_range[0], y_range[1], height)
    c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
    z = np.zeros_like(c)
    divtime = max_iter + np.zeros(z.shape, dtype=int)

    for i in range(max_iter):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2

    return divtime

def plot_mandelbrot(
    width: int = 800,
    height: int = 600,
    cmap: str = 'hot'
) -> None:
    """
    Plot the Mandelbrot set using matplotlib.
    
    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels
        cmap (str): Colormap to use for visualization
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(
        mandelbrot(width, height),
        cmap=cmap,
        extent=[-2, 1, -1.5, 1.5]
    )
    plt.colorbar(label='Iteration count')
    plt.title('Mandelbrot Set')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.show()

if __name__ == "__main__":
    plot_mandelbrot()
