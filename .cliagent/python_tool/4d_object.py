import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Tesseract:
    def __init__(self, size: float = 1.0) -> None:
        # Initialize vertices of a tesseract (4D hypercube)
        self.vertices: np.ndarray = np.array([
            [x, y, z, w] for x in [-size, size]
            for y in [-size, size]
            for z in [-size, size]
            for w in [-size, size]
        ])
        
        # Define edges connecting vertices
        self.edges: List[Tuple[int, int]] = []
        for i in range(16):
            for j in range(i + 1, 16):
                # Connect vertices that differ in only one coordinate
                if sum(abs(self.vertices[i] - self.vertices[j]) > 0.1) == 1:
                    self.edges.append((i, j))

    def project_4d_to_3d(self, w_rotation: float, distance: float = 2.0) -> np.ndarray:
        # Project 4D points to 3D using perspective projection
        points_4d = self.vertices.copy()
        
        # Rotate in 4D (in the xw plane)
        rotation_matrix = np.eye(4)
        rotation_matrix[0, 0] = np.cos(w_rotation)
        rotation_matrix[0, 3] = -np.sin(w_rotation)
        rotation_matrix[3, 0] = np.sin(w_rotation)
        rotation_matrix[3, 3] = np.cos(w_rotation)
        
        points_4d = points_4d @ rotation_matrix.T
        
        # Perform perspective projection
        w = points_4d[:, 3]
        scale_factors = 1 / (distance - w)
        points_3d = points_4d[:, :3] * scale_factors[:, np.newaxis]
        
        return points_3d

    def visualize(self, num_frames: int = 100) -> None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for frame in range(num_frames):
            ax.clear()
            
            # Set axis limits and labels
            ax.set_xlim([-2, 2])
            ax.set_ylim([-2, 2])
            ax.set_zlim([-2, 2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Project 4D points to 3D
            w_rotation = frame * 2 * np.pi / num_frames
            points_3d = self.project_4d_to_3d(w_rotation)
            
            # Draw edges
            for edge in self.edges:
                points = points_3d[list(edge)]
                ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-')
            
            plt.pause(0.05)
        
        plt.show()

if __name__ == "__main__":
    tesseract = Tesseract(size=1.0)
    tesseract.visualize(num_frames=100)
