import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Any
from matplotlib.colors import hsv_to_rgb

class RainbowSpiralAnimator:
    def __init__(self, num_points: int = 100, num_frames: int = 100) -> None:
        self.num_points: int = num_points
        self.num_frames: int = num_frames
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scatter: Any = None
        self.animation: Any = None

    def init_animation(self) -> Tuple[Any, ...]:
        self.scatter._offsets3d = ([], [], [])
        return (self.scatter,)

    def animate(self, frame: int) -> Tuple[Any, ...]:
        t: np.ndarray = np.linspace(0, 10*np.pi, self.num_points)
        x: np.ndarray = np.cos(t + frame/10.0)
        y: np.ndarray = np.sin(t + frame/10.0)
        z: np.ndarray = t/10.0

        # Create rainbow colors based on position
        colors: np.ndarray = np.zeros((self.num_points, 3))
        colors[:, 0] = (z / z.max()) % 1.0  # Hue based on height
        colors[:, 1] = 0.8  # Saturation
        colors[:, 2] = 0.8  # Value
        rgb_colors: np.ndarray = hsv_to_rgb(colors)

        self.scatter._offsets3d = (x, y, z)
        self.scatter.set_color(rgb_colors)
        return (self.scatter,)

    def create_animation(self) -> None:
        t: np.ndarray = np.linspace(0, 10*np.pi, self.num_points)
        x: np.ndarray = np.cos(t)
        y: np.ndarray = np.sin(t)
        z: np.ndarray = t/10.0
        
        self.scatter = self.ax.scatter(x, y, z, c='b', s=50)
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([0, 4])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Rainbow 3D Animated Spiral')

        self.animation = FuncAnimation(
            self.fig,
            self.animate,
            init_func=self.init_animation,
            frames=self.num_frames,
            interval=50,
            blit=True
        )
        plt.show()

def main() -> None:
    spiral: RainbowSpiralAnimator = RainbowSpiralAnimator()
    spiral.create_animation()

if __name__ == "__main__":
    main()
