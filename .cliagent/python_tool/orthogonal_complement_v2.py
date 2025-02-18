import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from mpl_toolkits.mplot3d import Axes3D

def visualize_orthogonal_complement(
    subspace_vectors: List[np.ndarray],
    scale: float = 2.0
) -> None:
    """
    Visualizes a subspace and its orthogonal complement in R³.
    
    Args:
        subspace_vectors: List of vectors that span the subspace
        scale: Scale factor for visualization
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate orthogonal complement
    basis = np.array(subspace_vectors)
    orthogonal = np.eye(3) - basis.T @ np.linalg.pinv(basis.T)
    orth_vectors = [v for v in orthogonal.T if not np.allclose(v, 0)]
    
    # Plot original subspace vectors
    for v in subspace_vectors:
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color='blue', alpha=0.8, label='Subspace')
    
    # Plot orthogonal complement vectors
    for v in orth_vectors:
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color='red', alpha=0.8, label='Orthogonal Complement')
    
    # Set plot limits and labels
    ax.set_xlim([-scale, scale])
    ax.set_ylim([-scale, scale])
    ax.set_zlim([-scale, scale])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Subspace and Orthogonal Complement in R³')
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define a subspace (e.g., a plane through origin)
    vectors = [np.array([1, 0, 0]), np.array([0, 1, 0])]
    visualize_orthogonal_complement(vectors)
