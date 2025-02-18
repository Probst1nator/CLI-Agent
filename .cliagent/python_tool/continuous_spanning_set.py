import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy.typing as npt

@dataclass
class SpanningSet:
    vectors: npt.NDArray[np.float64]
    color: str = 'blue'
    alpha: float = 0.6

class SpanVisualizer:
    def __init__(
        self, 
        spanning_set: SpanningSet,
        n_points: int = 50,
        animation_frames: int = 100,
        interval: int = 50
    ) -> None:
        self.spanning_set = spanning_set
        self.n_points = n_points
        self.frames = animation_frames
        self.interval = interval
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def generate_surface_points(self, t: float) -> Tuple[npt.NDArray[np.float64], ...]:
        u = np.linspace(-2, 2, self.n_points)
        v = np.linspace(-2, 2, self.n_points)
        U, V = np.meshgrid(u, v)
        
        # Create a morphing surface using spanning set vectors
        scale = (np.sin(t * 2 * np.pi) + 1) / 2  # Oscillating scale factor
        X = (self.spanning_set.vectors[0][0] * U + 
             self.spanning_set.vectors[1][0] * V) * scale
        Y = (self.spanning_set.vectors[0][1] * U + 
             self.spanning_set.vectors[1][1] * V) * scale
        Z = (self.spanning_set.vectors[0][2] * U + 
             self.spanning_set.vectors[1][2] * V) * scale
        
        return X, Y, Z

    def update(self, frame: int) -> None:
        self.ax.clear()
        t = frame / self.frames
        
        # Generate and plot the surface
        X, Y, Z = self.generate_surface_points(t)
        surf = self.ax.plot_surface(
            X, Y, Z,
            color=self.spanning_set.color,
            alpha=self.spanning_set.alpha
        )
        
        # Plot the spanning vectors
        for vector in self.spanning_set.vectors:
            self.ax.quiver(
                0, 0, 0,
                vector[0], vector[1], vector[2],
                color='red',
                arrow_length_ratio=0.1
            )
        
        # Set labels and limits
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Spanning Set Visualization (t={t:.2f})')
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_zlim([-3, 3])

    def animate(self) -> None:
        anim = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=self.frames,
            interval=self.interval,
            blit=False
        )
        plt.show()

def main() -> None:
    # Define spanning set vectors
    vectors = np.array([
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 1.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    # Create spanning set and visualizer
    spanning_set = SpanningSet(vectors=vectors, color='skyblue')
    visualizer = SpanVisualizer(
        spanning_set=spanning_set,
        n_points=30,
        animation_frames=120,
        interval=50
    )
    
    # Run animation
    visualizer.animate()

if __name__ == "__main__":
    main()
