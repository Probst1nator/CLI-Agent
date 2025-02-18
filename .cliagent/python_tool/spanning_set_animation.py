import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

class SpanningSetVisualizer:
    def __init__(self, 
                 basis_vectors: List[np.ndarray],
                 grid_size: int = 10,
                 animation_frames: int = 100) -> None:
        self.basis_vectors = basis_vectors
        self.grid_size = grid_size
        self.animation_frames = animation_frames
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Custom colormap for visualization
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        self.cmap = LinearSegmentedColormap.from_list('custom', colors)
        
    def setup_plot(self) -> None:
        """Setup the initial plot with grid and labels."""
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_xlim(-self.grid_size, self.grid_size)
        self.ax.set_ylim(-self.grid_size, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.set_title('Spanning Set Generation Animation', fontsize=12)
        self.ax.set_xlabel('X-axis', fontsize=10)
        self.ax.set_ylabel('Y-axis', fontsize=10)
        
    def generate_span_points(self, t: float) -> List[np.ndarray]:
        """Generate points in the span for a given time t."""
        span_points = []
        for i in np.linspace(-2, 2, 20):
            for j in np.linspace(-2, 2, 20):
                point = t * (i * self.basis_vectors[0] + j * self.basis_vectors[1])
                span_points.append(point)
        return span_points
    
    def animate(self, frame: int) -> Tuple[plt.Artist, ...]:
        """Animation function for each frame."""
        self.ax.clear()
        self.setup_plot()
        
        t = frame / self.animation_frames
        
        # Plot basis vectors
        for i, vector in enumerate(self.basis_vectors):
            self.ax.quiver(0, 0, vector[0], vector[1], 
                         angles='xy', scale_units='xy', scale=1,
                         color=self.cmap(i/len(self.basis_vectors)),
                         label=f'Basis Vector {i+1}')
        
        # Plot span points
        span_points = self.generate_span_points(t)
        x_coords = [p[0] for p in span_points]
        y_coords = [p[1] for p in span_points]
        scatter = self.ax.scatter(x_coords, y_coords, 
                                c=np.random.random(len(span_points)),
                                cmap=self.cmap, alpha=0.3, s=20)
        
        # Add legend and progress indicator
        self.ax.legend(loc='upper right')
        self.ax.text(-self.grid_size + 1, -self.grid_size + 1, 
                    f'Progress: {int(t*100)}%',
                    bbox=dict(facecolor='white', alpha=0.7))
        
        return scatter,
    
    def create_animation(self, 
                        save_path: Optional[str] = None,
                        interval: int = 50) -> FuncAnimation:
        """Create and optionally save the animation."""
        anim = FuncAnimation(self.fig, self.animate, 
                           frames=self.animation_frames,
                           interval=interval, blit=False)
        
        if save_path:
            anim.save(save_path, writer='pillow')
            
        return anim

def main() -> None:
    # Define basis vectors
    basis_vectors = [np.array([1, 1]), np.array([-1, 1])]
    
    # Create visualizer instance
    visualizer = SpanningSetVisualizer(basis_vectors)
    
    # Create animation
    animation = visualizer.create_animation(interval=50)
    
    # Show the animation
    plt.show()

if __name__ == "__main__":
    main()
