import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
from mpl_toolkits.mplot3d import Axes3D

def create_helix(
    t_range: Tuple[float, float], 
    num_points: int = 1000,
    radius: float = 1.0,
    height_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a 3D helix curve.
    
    Args:
        t_range: Range of the parameter t (start, end)
        num_points: Number of points to generate
        radius: Radius of the helix
        height_scale: Scale factor for height
        
    Returns:
        Tuple of x, y, z coordinates as numpy arrays
    """
    t = np.linspace(t_range[0], t_range[1], num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = height_scale * t
    return x, y, z

def plot_3d_curve(
    x: np.ndarray, 
    y: np.ndarray, 
    z: np.ndarray,
    title: str = "3D Helix"
) -> None:
    """
    Plots a 3D curve.
    
    Args:
        x, y, z: Coordinates of the curve
        title: Title of the plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x, y, z, 'b-', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.show()

def main() -> None:
    # Create a helix curve
    x, y, z = create_helix(
        t_range=(0, 10*np.pi),
        num_points=1000,
        radius=2.0,
        height_scale=0.5
    )
    
    # Plot the curve
    plot_3d_curve(x, y, z)

if __name__ == "__main__":
    main()
