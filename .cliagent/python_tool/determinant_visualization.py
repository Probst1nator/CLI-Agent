import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Rectangle
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
import matplotlib.gridspec as gridspec

class DeterminantVisualizer:
    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        self.ax1 = plt.subplot(gs[0])
        self.ax2 = plt.subplot(gs[1])
        
        # Initial 2x2 matrix
        self.matrix: np.ndarray = np.array([[1, 0], [0, 1]])
        self.frames: int = 60
        self.animation: Optional[FuncAnimation] = None
        
    def draw_parallelogram(self, matrix: np.ndarray, ax: plt.Axes, title: str) -> None:
        ax.clear()
        # Create vertices of parallelogram
        vertices: List[Tuple[float, float]] = [
            (0, 0),
            (matrix[0][0], matrix[0][1]),
            (matrix[1][0], matrix[1][1]),
            (matrix[0][0] + matrix[1][0], matrix[0][1] + matrix[1][1])
        ]
        
        # Draw parallelogram
        ax.add_patch(plt.Polygon(vertices, fill=True, alpha=0.3))
        
        # Draw vectors
        ax.arrow(0, 0, matrix[0][0], matrix[0][1], head_width=0.1, color='r', length_includes_head=True)
        ax.arrow(0, 0, matrix[1][0], matrix[1][1], head_width=0.1, color='b', length_includes_head=True)
        
        # Set limits and aspect ratio
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.grid(True)
        ax.set_aspect('equal')
        ax.set_title(title)
        
        # Add determinant value
        det: float = np.linalg.det(matrix)
        ax.text(-1.8, 1.8, f'Determinant: {det:.2f}', bbox=dict(facecolor='white', alpha=0.7))

    def animate(self, frame: int) -> None:
        t: float = frame / self.frames
        
        # Original matrix
        self.draw_parallelogram(self.matrix, self.ax1, "Original Matrix")
        
        # Matrix with swapped rows (animate the swap)
        swapped_matrix: np.ndarray = np.array([
            (1-t) * self.matrix[0] + t * self.matrix[1],
            (1-t) * self.matrix[1] + t * self.matrix[0]
        ])
        self.draw_parallelogram(swapped_matrix, self.ax2, "Swapping Rows")

    def show(self) -> None:
        self.animation = FuncAnimation(
            self.fig, 
            self.animate, 
            frames=self.frames, 
            interval=50, 
            repeat=True
        )
        plt.tight_layout()
        plt.show()

# Create and show the visualization
visualizer = DeterminantVisualizer()
visualizer.show()
