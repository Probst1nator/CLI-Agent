import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def mantle_prophet(x: List[float], y: List[float]) -> Tuple[plt.Figure, Dict[str, np.ndarray]]:
    """
    Visualize the Mantle Prophet using Python with matplotlib.
    
    Args:
        x (List[float]): Time series data points (x-axis)
        y (List[float]): Observed values (y-axis)
    
    Returns:
        Tuple[plt.Figure, Dict[str, np.ndarray]]: Figure object and dictionary with computed values
    """
    # Convert inputs to numpy arrays
    x_array = np.array(x)
    y_array = np.array(y)
    
    # Calculate trend using polynomial fit
    z = np.polyfit(x_array, y_array, 2)
    trend = np.poly1d(z)
    
    # Calculate seasonality using Fourier series
    period = len(x_array)
    seasonality = np.zeros_like(y_array)
    for i in range(1, 3):
        seasonality += np.sin(2 * np.pi * i * x_array / period)
    
    # Calculate residuals
    fitted = trend(x_array) + seasonality
    residuals = y_array - fitted
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle('Mantle Prophet Decomposition', fontsize=14)
    
    # Original Data
    axes[0].plot(x_array, y_array, 'b-', label='Original')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    
    # Trend
    axes[1].plot(x_array, trend(x_array), 'r-', label='Trend')
    axes[1].set_ylabel('Trend')
    axes[1].legend()
    
    # Seasonality
    axes[2].plot(x_array, seasonality, 'g-', label='Seasonality')
    axes[2].set_ylabel('Seasonality')
    axes[2].legend()
    
    # Residuals
    axes[3].plot(x_array, residuals, 'k-', label='Residuals')
    axes[3].set_ylabel('Residuals')
    axes[3].legend()
    
    plt.tight_layout()
    
    # Return figure and computed values
    return fig, {
        'trend': trend(x_array),
        'seasonality': seasonality,
        'residuals': residuals,
        'fitted': fitted
    }

# Example usage
if __name__ == "__main__":
    # Generate sample data
    x_data = list(range(100))
    y_data = [np.sin(x/10) + x/50 + np.random.normal(0, 0.1) for x in x_data]
    
    # Create visualization
    fig, components = mantle_prophet(x_data, y_data)
    plt.show()
