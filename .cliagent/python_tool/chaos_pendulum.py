import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class DoublePendulum:
    L1: float  # Length of first rod
    L2: float  # Length of second rod
    m1: float  # Mass of first bob
    m2: float  # Mass of second bob
    g: float = 9.81  # Gravitational acceleration
    theta1: float = math.pi/2  # Initial angle of first pendulum
    theta2: float = math.pi/2  # Initial angle of second pendulum
    omega1: float = 0.0  # Initial angular velocity of first pendulum
    omega2: float = 0.0  # Initial angular velocity of second pendulum
    
    def derivatives(self, state: List[float]) -> List[float]:
        theta1, omega1, theta2, omega2 = state
        
        # Constants for equation simplification
        c1 = (self.m1 + self.m2) * self.L1
        c2 = self.m2 * self.L2
        c3 = self.m2 * self.L1 * self.L2
        
        # Calculate derivatives
        dtheta1 = omega1
        dtheta2 = omega2
        
        # Complex equations for angular accelerations
        num1 = -self.g * (2 * self.m1 + self.m2) * np.sin(theta1)
        num2 = -self.m2 * self.g * np.sin(theta1 - 2 * theta2)
        num3 = -2 * np.sin(theta1 - theta2) * self.m2
        num4 = omega2**2 * self.L2 + omega1**2 * self.L1 * np.cos(theta1 - theta2)
        den = self.L1 * (2 * self.m1 + self.m2 - self.m2 * np.cos(2 * theta1 - 2 * theta2))
        
        domega1 = (num1 + num2 + num3 * num4) / den
        
        num1 = 2 * np.sin(theta1 - theta2)
        num2 = omega1**2 * self.L1 * (self.m1 + self.m2)
        num3 = self.g * (self.m1 + self.m2) * np.cos(theta1)
        num4 = omega2**2 * self.L2 * self.m2 * np.cos(theta1 - theta2)
        den = self.L2 * (2 * self.m1 + self.m2 - self.m2 * np.cos(2 * theta1 - 2 * theta2))
        
        domega2 = num1 * (num2 + num3 + num4) / den
        
        return [dtheta1, domega1, dtheta2, domega2]

class ChaosPendulumAnimation:
    def __init__(self, pendulum: DoublePendulum, dt: float = 0.05):
        self.pendulum = pendulum
        self.dt = dt
        self.time: float = 0
        self.state: List[float] = [
            pendulum.theta1,
            pendulum.omega1,
            pendulum.theta2,
            pendulum.omega2
        ]
        self.trail_x: List[float] = []
        self.trail_y: List[float] = []
        
        # Setup the animation
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-(pendulum.L1 + pendulum.L2) * 1.2, (pendulum.L1 + pendulum.L2) * 1.2)
        self.ax.set_ylim(-(pendulum.L1 + pendulum.L2) * 1.2, (pendulum.L1 + pendulum.L2) * 1.2)
        
        # Initialize the line objects
        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.trail, = self.ax.plot([], [], 'b-', alpha=0.3)
        
    def get_coords(self) -> Tuple[List[float], List[float]]:
        theta1, _, theta2, _ = self.state
        
        x1 = self.pendulum.L1 * np.sin(theta1)
        y1 = -self.pendulum.L1 * np.cos(theta1)
        
        x2 = x1 + self.pendulum.L2 * np.sin(theta2)
        y2 = y1 - self.pendulum.L2 * np.cos(theta2)
        
        return [0, x1, x2], [0, y1, y2]
    
    def animate(self, frame: int) -> Tuple[plt.Line2D, ...]:
        # Update state using RK4 method
        k1 = self.pendulum.derivatives(self.state)
        k2 = self.pendulum.derivatives([self.state[i] + k1[i] * self.dt/2 for i in range(4)])
        k3 = self.pendulum.derivatives([self.state[i] + k2[i] * self.dt/2 for i in range(4)])
        k4 = self.pendulum.derivatives([self.state[i] + k3[i] * self.dt for i in range(4)])
        
        for i in range(4):
            self.state[i] += (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) * self.dt/6
            
        x, y = self.get_coords()
        
        # Update trail
        self.trail_x.append(x[2])
        self.trail_y.append(y[2])
        
        # Keep only last 50 points for trail
        if len(self.trail_x) > 50:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
            
        self.line.set_data(x, y)
        self.trail.set_data(self.trail_x, self.trail_y)
        
        return self.line, self.trail

def main() -> None:
    # Create pendulum with initial conditions
    pendulum = DoublePendulum(
        L1=1.0,    # Length of first rod
        L2=1.0,    # Length of second rod
        m1=1.0,    # Mass of first bob
        m2=1.0,    # Mass of second bob
        theta1=np.pi/2,    # Initial angle of first pendulum
        theta2=np.pi/2,    # Initial angle of second pendulum
    )
    
    # Create animation
    anim = ChaosPendulumAnimation(pendulum)
    animation = FuncAnimation(
        anim.fig,
        anim.animate,
        frames=None,
        interval=50,
        blit=True
    )
    
    plt.grid(True)
    plt.title("Double Pendulum Chaos")
    plt.show()

if __name__ == "__main__":
    main()
