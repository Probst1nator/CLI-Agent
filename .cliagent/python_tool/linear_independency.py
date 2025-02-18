import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from mpl_toolkits.mplot3d import Axes3D

def generate_field_data(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n_points: int = 20
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate data for two fields."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Define two different fields
    Z1 = np.sin(X) * np.cos(Y)  # Field 1
    Z2 = X**2 - Y**2            # Field 2
    
    return X, Y, Z1, Z2

def plot_linear_independence(
    X: np.ndarray,
    Y: np.ndarray,
    Z1: np.ndarray,
    Z2: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5
) -> None:
    """Plot the two fields and their linear combination."""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot Field 1
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
    ax1.set_title('Field 1')
    
    # Plot Field 2
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8)
    ax2.set_title('Field 2')
    
    # Plot Linear Combination
    ax3 = fig.add_subplot(133, projection='3d')
    Z_combined = alpha * Z1 + beta * Z2
    surf3 = ax3.plot_surface(X, Y, Z_combined, cmap='magma', alpha=0.8)
    ax3.set_title(f'Linear Combination\n(α={alpha}, β={beta})')
    
    # Add labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

def main() -> None:
    """Main function to demonstrate linear independence."""
    # Generate field data
    X, Y, Z1, Z2 = generate_field_data(
        x_range=(-2, 2),
        y_range=(-2, 2),
        n_points=50
    )
    
    # Plot the fields and their linear combination
    plot_linear_independence(X, Y, Z1, Z2, alpha=0.7, beta=0.3)

if __name__ == "__main__":
    main()
