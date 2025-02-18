import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List

def generate_sphere(radius: float, center: Tuple[float, float, float], points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate points for a sphere."""
    phi = np.linspace(0, 2*np.pi, points)
    theta = np.linspace(0, np.pi, points)
    phi, theta = np.meshgrid(phi, theta)
    
    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)
    return x, y, z

def generate_circle(radius: float, center: Tuple[float, float, float], points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate points for a circle."""
    theta = np.linspace(0, 2*np.pi, points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(theta, center[2])
    return x, y, z

def main() -> None:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Head
    head_x, head_y, head_z = generate_sphere(1.0, (0, 0, 0))
    ax.plot_surface(head_x, head_y, head_z, color='yellow', alpha=0.6)
    
    # Eyes
    left_eye_x, left_eye_y, left_eye_z = generate_sphere(0.15, (-0.3, 0.3, 0.7))
    right_eye_x, right_eye_y, right_eye_z = generate_sphere(0.15, (0.3, 0.3, 0.7))
    ax.plot_surface(left_eye_x, left_eye_y, left_eye_z, color='black')
    ax.plot_surface(right_eye_x, right_eye_y, right_eye_z, color='black')
    
    # Smile
    smile_radius = 0.5
    smile_center = (0, 0, 0.3)
    smile_x, smile_y, smile_z = generate_circle(smile_radius, smile_center)
    # Only plot the bottom half of the circle for the smile
    half_length = len(smile_x) // 2
    ax.plot(smile_x[half_length:], smile_y[half_length:], smile_z[half_length:], 
            color='black', linewidth=3)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.title('3D Smiley Face')
    plt.show()

if __name__ == "__main__":
    main()
