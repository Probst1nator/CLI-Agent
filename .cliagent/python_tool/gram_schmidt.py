import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from numpy.typing import NDArray

def gram_schmidt_visualization(vectors: List[NDArray]) -> Tuple[List[NDArray], List[Tuple]]:
    """
    Implements and visualizes the Gram-Schmidt orthonormalization process.
    
    Args:
        vectors: List of input vectors to orthonormalize
        
    Returns:
        Tuple containing orthonormalized vectors and plotting coordinates
    """
    # Initialize result vectors and plotting coordinates
    orthonormal_vectors: List[NDArray] = []
    plot_coordinates: List[Tuple] = []
    
    for i, v in enumerate(vectors):
        # Project v onto all previous orthogonal vectors
        projection = np.zeros_like(v, dtype=float)
        for u in orthonormal_vectors:
            projection += np.dot(v, u) * u
            
        # Compute the orthogonal component
        orthogonal = v - projection
        
        # Normalize the vector
        normalized = orthogonal / np.linalg.norm(orthogonal)
        orthonormal_vectors.append(normalized)
        
        # Store coordinates for plotting
        plot_coordinates.append((v[0], v[1], normalized[0], normalized[1]))

    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot original vectors
    for i, v in enumerate(vectors):
        plt.subplot(1, 2, 1)
        plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
                  color=f'C{i}', label=f'v{i+1}')
    plt.title('Original Vectors')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    # Plot orthonormalized vectors
    for i, v in enumerate(orthonormal_vectors):
        plt.subplot(1, 2, 2)
        plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
                  color=f'C{i}', label=f'u{i+1}')
    plt.title('Orthonormalized Vectors')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return orthonormal_vectors, plot_coordinates

# Example usage
if __name__ == "__main__":
    # Define two non-orthogonal vectors
    v1: NDArray = np.array([1.0, 0.0])
    v2: NDArray = np.array([1.0, 1.0])
    
    vectors: List[NDArray] = [v1, v2]
    orthonormal_vectors, coordinates = gram_schmidt_visualization(vectors)
    
    # Print the results
    print("Original vectors:")
    for i, v in enumerate(vectors):
        print(f"v{i+1}: {v}")
    
    print("\nOrthonormalized vectors:")
    for i, v in enumerate(orthonormal_vectors):
        print(f"u{i+1}: {v}")
