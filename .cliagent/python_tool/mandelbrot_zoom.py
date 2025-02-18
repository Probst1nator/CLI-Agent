import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Optional
import numba

@numba.jit(nopython=True)
def mandelbrot_iteration(x: float, y: float, max_iter: int) -> int:
    c = complex(x, y)
    z = 0j
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

@numba.jit(nopython=True)
def generate_mandelbrot_frame(xmin: float, xmax: float, ymin: float, ymax: float, 
                            width: int, height: int, max_iter: int) -> np.ndarray:
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            result[i, j] = mandelbrot_iteration(x[j], y[i], max_iter)
    return result

class MandelbrotZoom:
    def __init__(self, width: int = 800, height: int = 600, 
                 max_iter: int = 100, frames: int = 300) -> None:
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.frames = frames
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.zoom_point = (-0.7435, 0.1314)  # Interesting zoom point
        self.zoom_factor = 0.9
        
    def calculate_bounds(self, frame: int) -> Tuple[float, float, float, float]:
        zoom = self.zoom_factor ** frame
        center_x, center_y = self.zoom_point
        width = 3.0 * zoom
        height = width * (self.height / self.width)
        
        xmin = center_x - width/2
        xmax = center_x + width/2
        ymin = center_y - height/2
        ymax = center_y + height/2
        
        return xmin, xmax, ymin, ymax
    
    def update(self, frame: int) -> None:
        xmin, xmax, ymin, ymax = self.calculate_bounds(frame)
        mandel = generate_mandelbrot_frame(xmin, xmax, ymin, ymax, 
                                         self.width, self.height, self.max_iter)
        
        if hasattr(self, 'im'):
            self.im.remove()
        
        self.im = self.ax.imshow(mandel, cmap='hot', extent=[xmin, xmax, ymin, ymax])
        self.ax.set_title(f'Mandelbrot Zoom Frame {frame}')
    
    def animate(self) -> None:
        self.ax.set_axis_off()
        anim = FuncAnimation(self.fig, self.update, frames=self.frames,
                           interval=1000/60, blit=False)  # 60 FPS
        plt.show()

if __name__ == "__main__":
    mandelbrot = MandelbrotZoom(width=800, height=600, max_iter=100, frames=300)
    mandelbrot.animate()
