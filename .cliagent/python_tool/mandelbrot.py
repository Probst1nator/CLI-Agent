import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def create_mandelbrot_set(width: int = 800, height: int = 600, max_iter: int = 100) -> np.ndarray:
    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5
    
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
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

def plot_mandelbrot(width: int = 800, height: int = 600, max_iter: int = 100) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(create_mandelbrot_set(width, height, max_iter),
               cmap='hot',
               extent=[-2, 1, -1.5, 1.5])
    plt.colorbar(label='Iteration count')
    plt.title('Mandelbrot Set')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.show()

if __name__ == "__main__":
    plot_mandelbrot()
