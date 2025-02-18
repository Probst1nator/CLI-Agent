import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class ElectronState:
    energy: float
    principal_quantum_number: int
    position: Tuple[float, float, float]
    probability_density: np.ndarray

class ElectronCloudSimulation:
    def __init__(self, grid_size: int = 20, initial_energy: float = -13.6):
        self.grid_size = grid_size
        self.current_state = ElectronState(
            energy=initial_energy,
            principal_quantum_number=1,
            position=(0., 0., 0.),
            probability_density=self._calculate_probability_density()
        )
        
    def _calculate_probability_density(self) -> np.ndarray:
        """Calculate the probability density for the 1s orbital."""
        x = np.linspace(-5, 5, self.grid_size)
        y = np.linspace(-5, 5, self.grid_size)
        z = np.linspace(-5, 5, self.grid_size)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Simplified 1s orbital probability density
        R = np.sqrt(X**2 + Y**2 + Z**2)
        psi = np.exp(-R)  # Simplified radial wavefunction
        return np.abs(psi)**2
    
    def observe_electron(self) -> Tuple[Tuple[float, float, float], float]:
        """Simulate an electron observation."""
        # Random position based on probability density
        pos = tuple(np.random.normal(0, 1, 3))  # type: Tuple[float, float, float]
        return pos, self.current_state.energy
    
    def apply_perturbation(self, delta_energy: float = 1.0) -> None:
        """Apply an energy perturbation to the system."""
        self.current_state.energy += delta_energy
        self.current_state.probability_density *= (1 + 0.1 * np.random.random(self.current_state.probability_density.shape))
        
    def visualize(self) -> None:
        """Visualize the electron cloud."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = y = z = np.linspace(-5, 5, self.grid_size)
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Plot probability density isosurface
        probability = self.current_state.probability_density
        ax.scatter(X, Y, Z, c=probability, alpha=0.1)
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Electron Cloud Probability Density')
        plt.show()

def main() -> None:
    simulation = ElectronCloudSimulation()
    
    while True:
        print("\nElectron Cloud Simulation")
        print("1. Observe electron")
        print("2. Apply perturbation")
        print("3. Visualize")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            position, energy = simulation.observe_electron()
            print(f"Electron observed at position: {position}")
            print(f"Current energy: {energy:.2f} eV")
        
        elif choice == '2':
            simulation.apply_perturbation()
            print("Perturbation applied to the system")
        
        elif choice == '3':
            simulation.visualize()
        
        elif choice == '4':
            print("Exiting simulation")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
