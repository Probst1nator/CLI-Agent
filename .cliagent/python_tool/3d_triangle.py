import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Optional

class Triangle3D:
    def __init__(
        self,
        vertices: List[Tuple[float, float, float]] = [
            (0, 0, 0),
            (1, 0, 0),
            (0.5, np.sqrt(0.75), 0)
        ]
    ):
        self.vertices = np.array(vertices)
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_plot()
        
    def setup_plot(self) -> None:
        """Setup the initial plot parameters."""
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Rotating Triangle')
        # Set equal aspect ratio
        self.ax.set_box_aspect([1,1,1])
        
    def update(self, frame: int) -> None:
        """Update function for animation."""
        self.ax.clear()
        self.setup_plot()
        
        # Rotate vertices
        theta = np.radians(frame)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        rotated_vertices = np.dot(self.vertices, rotation_matrix)
        
        # Draw the triangle
        vertices = np.vstack((rotated_vertices, rotated_vertices[0]))  # Close the triangle
        self.ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'b-')
        
        # Fill the triangle
        self.ax.plot_trisurf(
            rotated_vertices[:, 0],
            rotated_vertices[:, 1],
            rotated_vertices[:, 2],
            color='red',
            alpha=0.6
        )
        
        # Set consistent view limits
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_zlim([-1.5, 1.5])
        
    def animate(self, frames: int = 360, interval: int = 50) -> None:
        """Create and display the animation."""
        anim = FuncAnimation(
            self.fig,
            self.update,
            frames=frames,
            interval=interval,
            blit=False
        )
        plt.show()

def main() -> None:
    """Main function to create and show the 3D triangle animation."""
    triangle = Triangle3D()
    triangle.animate()

if __name__ == "__main__":
    main()
