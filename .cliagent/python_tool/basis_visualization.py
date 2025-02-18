import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
import numpy.typing as npt

def plot_basis_and_vector(basis_vectors: List[npt.NDArray[np.float64]], 
                         target_vector: npt.NDArray[np.float64],
                         coefficients: Optional[List[float]] = None,
                         title: str = "Visualization") -> None:
    """Plot basis vectors and target vector in 3D with optional linear combination"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot origin
    ax.scatter([0], [0], [0], color='black', s=100)
    
    # Plot basis vectors
    colors = ['r', 'g', 'b']
    for i, vector in enumerate(basis_vectors):
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], 
                 color=colors[i], alpha=0.8, label=f'Basis vector {i+1}')
    
    # Plot target vector
    ax.quiver(0, 0, 0, target_vector[0], target_vector[1], target_vector[2], 
             color='purple', alpha=0.8, label='Target vector')
    
    if coefficients:
        linear_comb = sum(c * v for c, v in zip(coefficients, basis_vectors))
        ax.quiver(0, 0, 0, linear_comb[0], linear_comb[1], linear_comb[2],
                 color='orange', alpha=0.5, label='Linear combination')
    
    # Setup plot properties
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]).T.flatten()
    mean = np.mean(limits)
    for dim in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
        dim(mean - 2, mean + 2)
    
    ax.legend()
    plt.show(block=False)
    plt.pause(0.1)

# Main execution
if __name__ == "__main__":
    # Standard basis vectors
    standard_basis = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), 
        np.array([0.0, 0.0, 1.0])
    ]
    
    # Target vector and its coefficients
    target = np.array([1.5, 1.0, 0.5])
    coeffs = [1.5, 1.0, 0.5]
    
    # Create visualizations
    plot_basis_and_vector(standard_basis, target, 
                         title="Standard Basis and Target Vector")
    plot_basis_and_vector(standard_basis, target, coeffs,
                         title="With Linear Combination")
    
    input("Press Enter to close plots...")
    plt.close('all')
