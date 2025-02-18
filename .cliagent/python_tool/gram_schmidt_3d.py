import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
import numpy.typing as npt

def gram_schmidt_3d(vectors: List[npt.NDArray[np.float64]]) -> List[npt.NDArray[np.float64]]:
    """
    Perform Gram-Schmidt orthonormalization in 3D space.
    
    Args:
        vectors: List of input vectors to orthonormalize
    Returns:
        List of orthonormalized vectors
    """
    orthonormal_vectors: List[npt.NDArray[np.float64]] = []
    
    for i, v in enumerate(vectors):
        # Start with the original vector
        u = v.copy()
        
        # Subtract projections of all previous orthogonal vectors
        for u_prev in orthonormal_vectors:
            projection = np.dot(v, u_prev) * u_prev
            u = u - projection
            
        # Normalize the vector
        norm = np.linalg.norm(u)
        if norm > 1e-10:  # Avoid division by zero
            u = u / norm
            orthonormal_vectors.append(u)
    
    return orthonormal_vectors

def plot_vectors_3d(original: List[npt.NDArray[np.float64]], 
                   orthonormal: List[npt.NDArray[np.float64]]) -> None:
    """
    Plot original and orthonormalized vectors in 3D space.
    
    Args:
        original: List of original vectors
        orthonormal: List of orthonormalized vectors
    """
    fig = plt.figure(figsize=(12, 6))
    
    # Plot original vectors
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Original Vectors')
    for v in original:
        ax1.quiver(0, 0, 0, v[0], v[1], v[2], color='b', alpha=0.5)
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_zlim([-1.5, 1.5])
    
    # Plot orthonormal vectors
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Orthonormal Vectors')
    for v in orthonormal:
        ax2.quiver(0, 0, 0, v[0], v[1], v[2], color='r', alpha=0.5)
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_zlim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.show()

def main() -> None:
    # Define three non-orthogonal vectors
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 1.0, 0.0])
    v3 = np.array([1.0, 1.0, 1.0])
    
    original_vectors = [v1, v2, v3]
    orthonormal_vectors = gram_schmidt_3d(original_vectors)
    
    # Print vectors
    print("Original vectors:")
    for i, v in enumerate(original_vectors, 1):
        print(f"v{i}: {v}")
    
    print("\nOrthonormal vectors:")
    for i, v in enumerate(orthonormal_vectors, 1):
        print(f"u{i}: {v}")
        
    # Verify orthogonality by computing dot products
    print("\nVerifying orthogonality (dot products should be close to 0):")
    for i in range(len(orthonormal_vectors)):
        for j in range(i + 1, len(orthonormal_vectors)):
            dot_product = np.dot(orthonormal_vectors[i], orthonormal_vectors[j])
            print(f"u{i+1} · u{j+1} = {dot_product:.10f}")
    
    # Plot the vectors
    plot_vectors_3d(original_vectors, orthonormal_vectors)

if __name__ == "__main__":
    main()
