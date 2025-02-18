import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional

def visualize_subspaces(
    dimensions: int = 3,
    subspaces: List[Tuple[np.ndarray, str]] = None,
    points: int = 100
) -> None:
    """
    Visualize linear subspaces in 2D or 3D space.
    
    Args:
        dimensions: Number of dimensions (2 or 3)
        subspaces: List of tuples containing basis vectors and labels
        points: Number of points to plot
    """
    fig = plt.figure(figsize=(10, 10))
    
    if dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    # Default subspaces if none provided
    if subspaces is None:
        if dimensions == 3:
            subspaces = [
                (np.array([[1, 0, 0], [0, 1, 0]]), "xy-plane"),
                (np.array([[1, 0, 0]]), "x-axis"),
                (np.array([[0, 1, 0]]), "y-axis"),
                (np.array([[0, 0, 1]]), "z-axis")
            ]
        else:
            subspaces = [
                (np.array([[1, 0]]), "x-axis"),
                (np.array([[0, 1]]), "y-axis")
            ]
    
    # Generate and plot points for each subspace
    colors = ['r', 'b', 'g', 'y', 'c', 'm']
    for idx, (basis, label) in enumerate(subspaces):
        # Generate random coefficients
        coeffs = np.random.randn(basis.shape[0], points)
        
        # Generate points in the subspace
        points_in_subspace = np.dot(basis.T, coeffs)
        
        if dimensions == 3:
            ax.scatter(
                points_in_subspace[0],
                points_in_subspace[1],
                points_in_subspace[2],
                c=colors[idx % len(colors)],
                label=label,
                alpha=0.6
            )
            ax.set_zlabel('Z')
        else:
            ax.scatter(
                points_in_subspace[0],
                points_in_subspace[1],
                c=colors[idx % len(colors)],
                label=label,
                alpha=0.6
            )
    
    # Set plot properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{dimensions}D Linear Subspaces Visualization')
    ax.legend()
    
    # Set equal aspect ratio
    if dimensions == 2:
        ax.set_aspect('equal')
    else:
        ax.set_box_aspect([1, 1, 1])
    
    # Set limits
    limit = 3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    if dimensions == 3:
        ax.set_zlim(-limit, limit)
    
    plt.grid(True)
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Visualize 3D subspaces
    visualize_subspaces(dimensions=3)
    
    # Visualize 2D subspaces
    visualize_subspaces(dimensions=2)
    
    # Custom subspace example (2D line through origin)
    custom_subspaces = [(np.array([[1, 1]]), "y=x line")]
    visualize_subspaces(dimensions=2, subspaces=custom_subspaces)
