import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List

def plot_3d_vector(vector: List[float], origin: List[float] = [0, 0, 0]) -> None:
    """
    Plot a 3D vector starting from the origin.
    
    Args:
        vector (List[float]): The vector coordinates [x, y, z]
        origin (List[float]): The starting point of the vector [x, y, z]
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the vector
    ax.quiver(origin[0], origin[1], origin[2],
              vector[0], vector[1], vector[2],
              color='r', arrow_length_ratio=0.1)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    max_range = np.array([vector[0], vector[1], vector[2]]).max()
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Add a title
    plt.title('3D Vector Visualization')
    
    plt.show()

# Example usage
vector: List[float] = [2, 3, 4]
plot_3d_vector(vector)
