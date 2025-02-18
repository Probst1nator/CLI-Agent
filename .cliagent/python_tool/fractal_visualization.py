import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List
import sys

class Fractal3D:
    """A class to generate and visualize 3D Mandelbulb fractals."""

    def __init__(self, size: int = 100, max_iterations: int = 20, power: float = 8.0):
        """
        Initialize the fractal generator.

        Args:
            size: Resolution of the 3D grid
            max_iterations: Maximum number of iterations for each point
            power: Power parameter for the Mandelbulb formula
        """
        self.size: int = size
        self.max_iterations: int = max_iterations
        self.power: float = power
        self.points: List[Tuple[float, float, float]] = []

    def _transform_point(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Transform a point using the Mandelbulb formula.

        Args:
            x, y, z: Coordinates of the point

        Returns:
            Transformed coordinates (x, y, z)
        """
        try:
            r: float = np.sqrt(x*x + y*y + z*z)
            theta: float = np.arctan2(np.sqrt(x*x + y*y), z)
            phi: float = np.arctan2(y, x)
            
            r_n: float = r ** self.power
            theta_n: float = theta * self.power
            phi_n: float = phi * self.power
            
            return (
                r_n * np.sin(theta_n) * np.cos(phi_n),
                r_n * np.sin(theta_n) * np.sin(phi_n),
                r_n * np.cos(theta_n)
            )
        except Exception as e:
            print(f"Error in point transformation: {e}")
            return (0.0, 0.0, 0.0)

    def generate(self) -> None:
        """Generate the fractal points."""
        try:
            self.points = []
            spacing: np.ndarray = np.linspace(-1.5, 1.5, self.size)
            
            for x in spacing:
                for y in spacing:
                    for z in spacing:
                        c_x, c_y, c_z = x, y, z
                        x1, y1, z1 = 0.0, 0.0, 0.0
                        
                        for _ in range(self.max_iterations):
                            x1, y1, z1 = self._transform_point(x1, y1, z1)
                            x1, y1, z1 = x1 + c_x, y1 + c_y, z1 + c_z
                            
                            if np.sqrt(x1*x1 + y1*y1 + z1*z1) > 2:
                                self.points.append((x, y, z))
                                break
                                
        except Exception as e:
            print(f"Error generating fractal: {e}")
            sys.exit(1)

    def visualize(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the fractal.

        Args:
            save_path: Optional path to save the visualization
        """
        try:
            if not self.points:
                raise ValueError("No points to visualize. Run generate() first.")
                
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            points = np.array(self.points)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', alpha=0.1, s=1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Mandelbulb Fractal')
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
            
        except Exception as e:
            print(f"Error visualizing fractal: {e}")
            sys.exit(1)

if __name__ == "__main__":
    try:
        fractal = Fractal3D(size=50)  # Use smaller size for faster rendering
        fractal.generate()
        fractal.visualize()
    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)