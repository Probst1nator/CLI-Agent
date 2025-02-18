import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional

def plot_subspace_and_complement(
    basis_vectors: List[np.ndarray],
    grid_points: int = 10,
    scale: float = 2.0
) -> None:
    """
    Plot a subspace and its orthogonal complement in R³.
    
    Args:
        basis_vectors: List of numpy arrays representing the basis of the subspace
        grid_points: Number of points to use in the grid
        scale: Scale factor for the plot
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Stack basis vectors and get orthogonal complement
    V = np.stack(basis_vectors)
    V_orth = np.eye(3) - V.T @ np.linalg.pinv(V.T)
    
    # Create grid points
    x = np.linspace(-scale, scale, grid_points)
    y = np.linspace(-scale, scale, grid_points)
    X, Y = np.meshgrid(x, y)
    
    # Plot subspace (in blue)
    if len(basis_vectors) == 1:
        t = np.linspace(-scale, scale, 100)
        v = basis_vectors[0]
        ax.plot(t*v[0], t*v[1], t*v[2], 'b-', linewidth=2, label='Subspace')
    elif len(basis_vectors) == 2:
        Z = np.zeros_like(X)
        for i in range(grid_points):
            for j in range(grid_points):
                point = np.array([X[i,j], Y[i,j], 0])
                transformed = V.T @ V @ point
                Z[i,j] = transformed[2]
        ax.plot_surface(X, Y, Z, alpha=0.3, color='b', label='Subspace')
    
    # Plot orthogonal complement (in red)
    if V_orth.any():  # If orthogonal complement exists
        Z = np.zeros_like(X)
        for i in range(grid_points):
            for j in range(grid_points):
                point = np.array([X[i,j], Y[i,j], 0])
                transformed = V_orth @ point
                Z[i,j] = transformed[2]
        ax.plot_surface(X, Y, Z, alpha=0.3, color='r', label='Orthogonal Complement')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Subspace and its Orthogonal Complement')
    
    # Set axis limits
    ax.set_xlim([-scale, scale])
    ax.set_ylim([-scale, scale])
    ax.set_zlim([-scale, scale])
    
    # Add legend
    ax.legend()
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example 1: Subspace spanned by a single vector
    v1 = np.array([1, 1, 1]) / np.sqrt(3)  # normalized vector
    plot_subspace_and_complement([v1])
    
    # Example 2: Subspace spanned by two vectors
    v2 = np.array([1, 0, 0])
    v3 = np.array([0, 1, 0])
    plot_subspace_and_complement([v2, v3])
