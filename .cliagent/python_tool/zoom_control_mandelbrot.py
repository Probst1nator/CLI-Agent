import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Any

class MandelbrotZoom:
    def __init__(self, width: int = 800, height: int = 600, max_iter: int = 100):
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.fig, self.ax = plt.subplots()
        # Starting coordinates for interesting Mandelbrot area
        self.x_center: float = -0.7435
        self.y_center: float = 0.1314
        self.zoom_factor: float = 0.8
        self.current_zoom: float = 1.0
        self.zoom_speed: float = 1.05

    def mandelbrot(self, h: int, w: int, max_iter: int, zoom: float) -> np.ndarray:
        y, x = np.ogrid[
            self.y_center-2/zoom:self.y_center+2/zoom:h*1j,
            self.x_center-2/zoom:self.x_center+2/zoom:w*1j
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

    def update(self, frame: int) -> Tuple[Any, ...]:
        self.current_zoom *= self.zoom_speed
        plt.clf()
        mandel = self.mandelbrot(self.height, self.width, self.max_iter, self.current_zoom)
        plt.imshow(mandel, cmap='hot', extent=[-2, 2, -2, 2])
        plt.title(f'Zoom level: {self.current_zoom:.2f}x')
        return plt.gca(),

    def animate(self) -> None:
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=200,
            interval=50,
            blit=True
        )
        plt.show()

if __name__ == "__main__":
    mandelbrot_zoom = MandelbrotZoom()
    mandelbrot_zoom.animate()
