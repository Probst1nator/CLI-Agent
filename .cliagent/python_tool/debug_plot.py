import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Any

class SpiralAnimation:
    def __init__(self) -> None:
        # Create figure and 3D axes
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize spiral parameters
        self.t = np.linspace(0, 10*np.pi, 1000)
        self.x = np.sin(self.t)
        self.y = np.cos(self.t)
        self.z = self.t/10
        
        # Initialize the line
        self.line, = self.ax.plot([], [], [], lw=2)
        
        # Set axis labels and limits
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 4)
        
        # Animation frame counter
        self.frame_num = 0

    def update(self, frame: int) -> Tuple[Any, ...]:
        # Update the line data
        self.frame_num = (self.frame_num + 1) % len(self.t)
        idx = self.frame_num
        
        # Get current coordinates
        x_data = self.x[:idx]
        y_data = self.y[:idx]
        z_data = self.z[:idx]
        
        # Create rainbow colors based on z-coordinate
        colors = plt.cm.rainbow(z_data/max(self.z))
        
        # Update the line with new coordinates and colors
        self.line.set_data(x_data, y_data)
        self.line.set_3d_properties(z_data)
        self.line.set_color(colors)
        
        return (self.line,)

    def animate(self) -> None:
        # Create animation
        anim = FuncAnimation(
            self.fig, 
            self.update, 
            frames=len(self.t),
            interval=20, 
            blit=True
        )
        
        # Display the animation
        plt.show()

if __name__ == "__main__":
    spiral = SpiralAnimation()
    spiral.animate()
