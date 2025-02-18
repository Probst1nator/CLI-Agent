import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ForceSystem:
    mass: float  # in kg
    spring_k: float  # spring constant in N/m
    gravity: float = 9.81  # m/s^2
    
    def calculate_forces(self, displacement: float) -> Tuple[float, float]:
        weight_force: float = self.mass * self.gravity
        spring_force: float = self.spring_k * displacement
        return weight_force, spring_force

class MeasurementDeviceVisualization:
    def __init__(self, force_system: ForceSystem) -> None:
        self.force_system = force_system
        self.fig, self.ax = plt.subplots(figsize=(8, 10))
        plt.subplots_adjust(bottom=0.25)
        
        # Initial displacement
        self.displacement: float = 0.0
        
        # Create the plot
        self.weight_line: Optional[plt.Line2D] = None
        self.spring_line: Optional[plt.Line2D] = None
        self.force_arrows: List[plt.Arrow] = []
        
        # Create slider
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.displacement_slider = Slider(
            ax=ax_slider,
            label='Displacement (m)',
            valmin=-0.5,
            valmax=0.5,
            valinit=self.displacement
        )
        
        # Connect the slider to the update function
        self.displacement_slider.on_changed(self.update)
        
        # Initial plot
        self.setup_plot()
        self.update(self.displacement)

    def setup_plot(self) -> None:
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.set_title('Force Measurement Device')

    def draw_system(self, displacement: float) -> None:
        # Clear previous arrows
        for arrow in self.force_arrows:
            arrow.remove()
        self.force_arrows.clear()
        
        if self.weight_line is not None:
            self.weight_line.remove()
        if self.spring_line is not None:
            self.spring_line.remove()

        # Draw the weight
        weight_pos: float = -displacement
        self.weight_line = plt.plot([-0.5, 0.5], [weight_pos, weight_pos], 'k-', linewidth=3)[0]
        
        # Draw the spring (simplified)
        spring_points: int = 20
        x: np.ndarray = np.linspace(-0.2, 0.2, spring_points)
        y: np.ndarray = np.linspace(0, weight_pos, spring_points)
        self.spring_line = plt.plot(x, y, 'b-')[0]
        
        # Calculate and draw forces
        weight_force, spring_force = self.force_system.calculate_forces(displacement)
        
        # Draw force arrows
        # Weight force (downward)
        weight_arrow = plt.arrow(0, weight_pos, 0, -weight_force/50,
                               head_width=0.1, head_length=0.1, fc='r', ec='r')
        # Spring force (upward)
        spring_arrow = plt.arrow(0, weight_pos, 0, spring_force/50,
                               head_width=0.1, head_length=0.1, fc='b', ec='b')
        
        self.force_arrows.extend([weight_arrow, spring_arrow])
        
        # Add force labels
        self.ax.text(0.5, weight_pos, f'Weight: {weight_force:.1f} N', horizontalalignment='left')
        self.ax.text(0.5, weight_pos + 0.2, f'Spring: {spring_force:.1f} N', horizontalalignment='left')

    def update(self, val: float) -> None:
        self.displacement = val
        self.ax.clear()
        self.setup_plot()
        self.draw_system(self.displacement)
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    # Create a force system with a 1 kg mass and spring constant of 20 N/m
    system = ForceSystem(mass=1.0, spring_k=20.0)
    
    # Create and show the visualization
    vis = MeasurementDeviceVisualization(system)
    plt.show()
