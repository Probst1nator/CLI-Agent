import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Tuple, Union
from mpl_toolkits.mplot3d import Axes3D

def create_spanning_set_animation(
    spanning_set: np.ndarray,
    generated_sets: List[np.ndarray],
    rotation_speed: float = 2.0,
    interval: int = 50
) -> Tuple[plt.Figure, animation.FuncAnimation]:
    """
    Creates a 3D animation of spanning sets generating other sets.
    
    Args:
        spanning_set: Base vectors forming the spanning set (Nx3 array)
        generated_sets: List of sets to be generated (each is Nx3 array)
        rotation_speed: Speed of rotation in degrees per frame
        interval: Time between frames in milliseconds
    
    Returns:
        Tuple of (Figure, Animation)
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Spanning Set Animation')
    
    # Initialize the plot limits
    max_range = max(
        np.max(np.abs(spanning_set)),
        max(np.max(np.abs(set_)) for set_ in generated_sets)
    )
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # Animation update function
    def update(frame: int) -> None:
        ax.cla()  # Clear the current axes
        
        # Plot spanning set
        ax.scatter(
            spanning_set[:, 0],
            spanning_set[:, 1],
            spanning_set[:, 2],
            color='blue',
            label='Spanning Set',
            s=100
        )
        
        # Plot current generated set
        current_set = generated_sets[frame % len(generated_sets)]
        ax.scatter(
            current_set[:, 0],
            current_set[:, 1],
            current_set[:, 2],
            color='red',
            label=f'Generated Set {frame % len(generated_sets) + 1}',
            s=100
        )
        
        # Draw lines from origin to spanning set points
        for point in spanning_set:
            ax.plot(
                [0, point[0]],
                [0, point[1]],
                [0, point[2]],
                'b--',
                alpha=0.5
            )
        
        # Set labels and limits
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Spanning Set Animation')
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.legend()
        
        # Rotate view
        ax.view_init(elev=30, azim=frame * rotation_speed)

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=100,  # Number of frames
        interval=interval,
        blit=False
    )
    
    return fig, anim

def main() -> None:
    # Define spanning set (3D vectors)
    spanning_set = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    # Define sets to be generated
    generated_sets = [
        np.array([
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 1]
        ]),
        np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
        ]),
        np.array([
            [1, 1, 2],
            [2, 1, 1],
            [1, 2, 1]
        ])
    ]
    
    # Create animation
    fig, anim = create_spanning_set_animation(
        spanning_set,
        generated_sets,
        rotation_speed=2.0,
        interval=50
    )
    
    # Show the animation
    plt.show()

if __name__ == "__main__":
    main()
