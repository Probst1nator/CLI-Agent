import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import List, Tuple, Optional, Union, Callable

class SpanningSetVisualizer:
    def __init__(
        self, 
        spanning_set: np.ndarray,
        linear_coefficients: np.ndarray,
        n_points: int = 100,
        animation_frames: int = 50,
        interval: int = 100
    ):
        self.spanning_set = spanning_set
        self.coefficients = linear_coefficients
        self.n_points = n_points
        self.frames = animation_frames
        self.interval = interval
        
        # Initialize the figure and 3D axis
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Generate parametric surface points
        self.u = np.linspace(0, 2 * np.pi, n_points)
        self.v = np.linspace(0, np.pi, n_points)
        
    def generate_linear_combination(self, t: float) -> np.ndarray:
        """Generate a point that's a linear combination of the spanning set."""
        time_varying_coeffs = self.coefficients * (1 + 0.5 * np.sin(t))
        return np.sum(self.spanning_set * time_varying_coeffs[:, np.newaxis], axis=0)
        
    def generate_surface(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a surface that depends on the linear combination."""
        center = self.generate_linear_combination(t)
        
        x = center[0] + np.outer(np.cos(self.u), np.sin(self.v))
        y = center[1] + np.outer(np.sin(self.u), np.sin(self.v))
        z = center[2] + np.outer(np.ones(np.size(self.u)), np.cos(self.v))
        
        return x, y, z
    
    def animate(self, frame: int) -> None:
        """Animation function for matplotlib."""
        self.ax.clear()
        
        # Plot spanning set vectors
        for vector in self.spanning_set:
            self.ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], 
                         color='r', alpha=0.5)
        
        # Plot generated surface
        t = 2 * np.pi * frame / self.frames
        x, y, z = self.generate_surface(t)
        self.ax.plot_surface(x, y, z, color='b', alpha=0.3)
        
        # Plot linear combination point
        point = self.generate_linear_combination(t)
        self.ax.scatter([point[0]], [point[1]], [point[2]], 
                       color='g', s=100, label='Linear Combination')
        
        # Set labels and limits
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Spanning Set and Generated Surface')
        self.ax.legend()
        
    def show(self) -> None:
        """Display the animation."""
        anim = animation.FuncAnimation(
            self.fig, self.animate, 
            frames=self.frames, 
            interval=self.interval
        )
        plt.show()

def main() -> None:
    # Define spanning set (3D vectors)
    spanning_set = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    
    # Define coefficients for linear combination
    coefficients = np.array([0.3, 0.3, 0.3, 0.1])
    
    # Create and show visualization
    visualizer = SpanningSetVisualizer(
        spanning_set=spanning_set,
        linear_coefficients=coefficients,
        animation_frames=100,
        interval=50
    )
    visualizer.show()

if __name__ == "__main__":
    main()
