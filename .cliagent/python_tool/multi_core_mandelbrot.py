import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Optional
from numba import jit
import numpy.typing as npt

@jit(nopaque=True)
def mandelbrot_point(c: complex, max_iter: int) -> float:
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def calculate_mandelbrot(
    xmin: float, 
    xmax: float, 
    ymin: float, 
    ymax: float, 
    width: int, 
    height: int, 
    max_iter: int
) -> npt.NDArray:
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
    
    # Split the computation across CPU cores
    with mp.Pool() as pool:
        result = np.array(pool.map(
            lambda row: [mandelbrot_point(point, max_iter) for point in row],
            [c[i, :] for i in range(width)]
        ))
    
    return result

class MandelbrotZoom:
    def __init__(
        self,
        width: int = 800,
        height: int = 800,
        max_iter: int = 100,
        frames: int = 200,
        zoom_speed: float = 0.95
    ):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.frames = frames
        self.zoom_speed = zoom_speed
        
        # Initial view coordinates
        self.x_center: float = -0.7435
        self.y_center: float = 0.1314
        self.zoom_level: float = 1.0
        
        # Setup the plot
        self.fig, self.ax = plt.subplots()
        self.img = None
        
    def update(self, frame: int) -> None:
        # Calculate the new zoom level
        self.zoom_level *= self.zoom_speed
        
        # Calculate the new boundaries
        width = 3.0 * self.zoom_level
        height = 3.0 * self.zoom_level
        
        xmin = self.x_center - width/2
        xmax = self.x_center + width/2
        ymin = self.y_center - height/2
        ymax = self.y_center + height/2
        
        # Generate the new frame
        mandel = calculate_mandelbrot(
            xmin, xmax, ymin, ymax,
            self.width, self.height,
            self.max_iter
        )
        
        # Update or create the image
        if self.img is None:
            self.img = self.ax.imshow(
                mandel.T,
                cmap='hot',
                extent=[xmin, xmax, ymin, ymax]
            )
        else:
            self.img.set_array(mandel.T)
            self.img.set_extent([xmin, xmax, ymin, ymax])
        
        return [self.img]
    
    def animate(self) -> None:
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=self.frames,
            interval=1000/60,  # 60 FPS
            blit=True
        )
        
        plt.show()

if __name__ == "__main__":
    # Set up multiprocessing to use all available cores
    mp.freeze_support()
    
    # Create and run the animation
    mandelbrot = MandelbrotZoom(
        width=800,
        height=800,
        max_iter=100,
        frames=200,
        zoom_speed=0.95
    )
    mandelbrot.animate()
