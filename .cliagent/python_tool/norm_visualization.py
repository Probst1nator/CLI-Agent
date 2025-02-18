import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List

def calculate_norm_points(
    resolution: int = 20,
    radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate points for visualizing the norm in R³.
    
    Args:
        resolution: Number of points per dimension
        radius: Radius of the sphere
        
    Returns:
        Tuple of x, y, z coordinates and corresponding norms
    """
    x = np.linspace(-radius, radius, resolution)
    y = np.linspace(-radius, radius, resolution)
    z = np.linspace(-radius, radius, resolution)
    
    X, Y, Z = np.meshgrid(x, y, z)
    norms = np.sqrt(X**2 + Y**2 + Z**2)
    
    return X, Y, Z, norms

def plot_norm_sphere(
    resolution: int = 20,
    radius: float = 1.0,
    alpha: float = 0.1
) -> None:
    """
    Plot the norm visualization in R³.
    
    Args:
        resolution: Number of points per dimension
        radius: Radius of the sphere
        alpha: Transparency level of the plot
    """
    X, Y, Z, norms = calculate_norm_points(resolution, radius)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the norm spheres
    scatter = ax.scatter(X, Y, Z, c=norms, alpha=alpha, cmap='viridis')
    
    # Plot unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.1, color='blue')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization of Norm in R³')
    
    # Add colorbar
    plt.colorbar(scatter, label='Norm Value')
    
    plt.show()

if __name__ == "__main__":
    plot_norm_sphere(resolution=20, radius=1.0, alpha=0.1)
