import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Union

class MandelbrotZoom:
    def __init__(
        self, 
        width: int = 800, 
        height: int = 600, 
        max_iter: int = 100,
        zoom_speed: float = 0.9
    ) -> None:
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.zoom_speed = zoom_speed
        self.zoom_point = (-0.7435, 0.1314)  # Interesting point to zoom into
        self.zoom_factor = 1.0
        
    def mandelbrot(self, h: int, w: int, max_iter: int, zoom: float) -> np.ndarray:
        y, x = np.ogrid[
            1.4/zoom:-1.4/zoom:h*1j, 
            -2/zoom:0.8/zoom:w*1j
        ]
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

    def update(self, frame: int) -> Tuple[plt.Artist, ...]:
        self.zoom_factor *= self.zoom_speed
        plt.clf()
        mandel = self.mandelbrot(
            self.height, 
            self.width, 
            self.max_iter, 
            self.zoom_factor
        )
        plt.imshow(mandel, cmap='hot', extent=[-2, 0.8, -1.4, 1.4])
        plt.title(f'Zoom factor: {self.zoom_factor:.2f}x')
        return plt.gca(),

    def animate(self) -> None:
        fig = plt.figure(figsize=(10, 8))
        anim = FuncAnimation(
            fig, 
            self.update, 
            frames=200,
            interval=50, 
            blit=True
        )
        plt.show()

if __name__ == "__main__":
    mandelbrot_zoom = MandelbrotZoom()
    mandelbrot_zoom.animate()
