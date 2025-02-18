import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Any

class MandelbrotZoomer:
    def __init__(self) -> None:
        self.width: int = 800
        self.height: int = 800
        self.max_iter: int = 100
        self.fps: int = 60
        self.zoom_speed: float = 0.95
        
        # Initial view coordinates
        self.x_min: float = -2.0
        self.x_max: float = 1.0
        self.y_min: float = -1.5
        self.y_max: float = 1.5
        
        # Target point for zooming (interesting part of the set)
        self.target_x: float = -0.7435
        self.target_y: float = 0.1314
        
        self.setup_plot()

    def mandelbrot(self, h: int, w: int, max_iter: int) -> np.ndarray:
        y, x = np.ogrid[self.y_min:self.y_max:h*1j, self.x_min:self.x_max:w*1j]
        c = x + y*1j
        z = c
        divtime = max_iter + np.zeros(z.shape, dtype=int)

        for i in range(max_iter):
            z = z**2 + c
            diverge = z*np.conj(z) > 2**2
            div_now = diverge & (divtime == max_iter)
            divtime[div_now] = i
            z[diverge] = 2

        return divtime

    def setup_plot(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        plt.tight_layout()

    def update(self, frame: int) -> Tuple[Any, ...]:
        # Calculate zoom factor
        zoom_factor: float = self.zoom_speed
        
        # Update boundaries while maintaining aspect ratio
        x_center: float = (self.x_min + self.x_max) / 2
        y_center: float = (self.y_min + self.y_max) / 2
        
        x_range: float = (self.x_max - self.x_min)
        y_range: float = (self.y_max - self.y_min)
        
        # Move center towards target
        x_center += (self.target_x - x_center) * 0.05
        y_center += (self.target_y - y_center) * 0.05
        
        # Apply zoom
        self.x_min = x_center - (x_range * zoom_factor) / 2
        self.x_max = x_center + (x_range * zoom_factor) / 2
        self.y_min = y_center - (y_range * zoom_factor) / 2
        self.y_max = y_center + (y_range * zoom_factor) / 2
        
        # Generate new frame
        mandel = self.mandelbrot(self.height, self.width, self.max_iter)
        
        # Clear previous frame and plot new one
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        img = self.ax.imshow(mandel, cmap='hot', extent=[self.x_min, self.x_max, self.y_min, self.y_max])
        return (img,)

    def animate(self) -> None:
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=None,  # Infinite frames
            interval=1000//self.fps,  # 60 FPS
            blit=True
        )
        plt.show()

if __name__ == "__main__":
    zoomer = MandelbrotZoomer()
    zoomer.animate()
