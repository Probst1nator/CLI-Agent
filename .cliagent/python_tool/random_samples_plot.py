import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple

def generate_and_plot_density(n_samples: int = 1000,
                            range_min: float = 0.0,
                            range_max: float = 1.0) -> Tuple[plt.Figure, plt.Axes]:
    # Generate random samples
    samples: NDArray[np.float64] = np.random.uniform(range_min, range_max, n_samples)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot density
    ax.hist(samples, bins=50, density=True, alpha=0.7, color='blue', label='Density')
    
    # Customize plot
    ax.set_title('Density Plot of Uniform Random Samples')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Return figure and axis for potential further customization
    return fig, ax

# Generate plot and display
fig, ax = generate_and_plot_density()
plt.show()
