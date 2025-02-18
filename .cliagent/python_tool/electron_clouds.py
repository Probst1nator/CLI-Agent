import numpy as np
import plotly.graph_objects as go
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math

@dataclass
class ElectronOrbital:
    n: int  # Principal quantum number
    l: int  # Angular momentum quantum number
    m: int  # Magnetic quantum number

class QuantumVisualizer:
    def __init__(self, grid_points: int = 50, radius: float = 5.0):
        self.grid_points = grid_points
        self.radius = radius
        self.x, self.y, self.z = self._create_mesh_grid()
        
    def _create_mesh_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create a 3D mesh grid for visualization."""
        points = np.linspace(-self.radius, self.radius, self.grid_points)
        return np.meshgrid(points, points, points)
    
    def _spherical_harmonics(self, orbital: ElectronOrbital) -> np.ndarray:
        """Calculate spherical harmonics for given quantum numbers."""
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        theta = np.arccos(self.z/r)
        phi = np.arctan2(self.y, self.x)
        
        # Radial part (simplified)
        R = np.exp(-r/orbital.n) * (2*r/orbital.n)**orbital.l
        
        # Angular part (simplified for common orbitals)
        if orbital.l == 0:  # s orbital
            Y = 1/np.sqrt(4*np.pi)
        elif orbital.l == 1:  # p orbital
            if orbital.m == 0:
                Y = np.sqrt(3/(4*np.pi)) * np.cos(theta)
            elif orbital.m == 1:
                Y = -np.sqrt(3/(8*np.pi)) * np.sin(theta) * np.exp(1j*phi)
        
        return (R * Y).real
    
    def visualize_orbital(self, orbital: ElectronOrbital, iso_value: float = 0.02) -> None:
        """Create an interactive 3D visualization of the electron orbital."""
        wave_function = self._spherical_harmonics(orbital)
        probability_density = np.abs(wave_function)**2
        
        fig = go.Figure(data=go.Isosurface(
            x=self.x.flatten(),
            y=self.y.flatten(),
            z=self.z.flatten(),
            value=probability_density.flatten(),
            isomin=0,
            isomax=iso_value,
            surface_count=5,
            colorscale='Viridis',
            opacity=0.5
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)'
            ),
            title=f'Electron Orbital (n={orbital.n}, l={orbital.l}, m={orbital.m})',
            width=800,
            height=800
        )
        
        fig.show()

def main() -> None:
    # Create visualizer instance
    visualizer = QuantumVisualizer(grid_points=50, radius=5.0)
    
    # Visualize different orbitals
    # 1s orbital
    visualizer.visualize_orbital(ElectronOrbital(n=1, l=0, m=0))
    
    # 2p orbital
    visualizer.visualize_orbital(ElectronOrbital(n=2, l=1, m=0))
    
    # 2p_x orbital
    visualizer.visualize_orbital(ElectronOrbital(n=2, l=1, m=1))

if __name__ == "__main__":
    main()
