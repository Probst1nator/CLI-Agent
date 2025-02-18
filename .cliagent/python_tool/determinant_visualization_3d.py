import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, TextBox
from typing import Tuple, List, Optional
import matplotlib.animation as animation

class DeterminantVisualizer:
    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(15, 8))
        self.matrix: np.ndarray = np.array([[1, 0, 0], 
                                          [0, 1, 0], 
                                          [0, 0, 1]], dtype=float)
        self.setup_plot()
        
    def setup_plot(self) -> None:
        # 3D visualization
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_matrix = self.fig.add_subplot(122)
        
        # Add controls
        self.swap_button_ax = plt.axes([0.15, 0.05, 0.1, 0.075])
        self.swap_button = Button(self.swap_button_ax, 'Swap Rows')
        self.swap_button.on_clicked(self.swap_rows)
        
        # Text display for determinant
        self.det_text = self.fig.text(0.5, 0.95, '', ha='center')
        self.update_visualization()
        
    def calculate_determinant(self) -> float:
        return np.linalg.det(self.matrix)
    
    def update_visualization(self) -> None:
        self.ax_3d.clear()
        self.ax_matrix.clear()
        
        # Update 3D visualization
        vectors = self.matrix.T
        origin = np.zeros((3))
        
        # Plot basis vectors
        colors = ['r', 'g', 'b']
        for i, vector in enumerate(vectors):
            self.ax_3d.quiver(origin[0], origin[1], origin[2],
                            vector[0], vector[1], vector[2],
                            color=colors[i], alpha=0.8)
            
        # Plot parallelpiped
        xx, yy = np.meshgrid([0, 1], [0, 1])
        for i in range(2):
            for j in range(2):
                self.ax_3d.plot_surface(xx, yy, np.zeros((2, 2)) + i,
                                      alpha=0.2, color=colors[i])
        
        # Set 3D plot properties
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('3D Visualization of Matrix Transformation')
        
        # Update matrix visualization
        self.ax_matrix.imshow(self.matrix, cmap='coolwarm')
        for i in range(3):
            for j in range(3):
                self.ax_matrix.text(j, i, f'{self.matrix[i, j]:.2f}',
                                  ha='center', va='center')
                
        self.ax_matrix.set_title('Matrix Values')
        
        # Update determinant text
        det = self.calculate_determinant()
        self.det_text.set_text(f'Determinant: {det:.2f}')
        
        plt.draw()
        
    def swap_rows(self, event: Optional[plt.MouseEvent]) -> None:
        # Swap first two rows
        self.matrix[[0, 1]] = self.matrix[[1, 0]]
        self.update_visualization()
        
    def animate(self, frame: int) -> None:
        angle = frame * np.pi / 180
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                  [np.sin(angle), np.cos(angle), 0],
                                  [0, 0, 1]])
        self.ax_3d.view_init(elev=20, azim=angle)
        plt.draw()
        
    def run(self) -> None:
        ani = animation.FuncAnimation(self.fig, self.animate,
                                    frames=360, interval=50)
        plt.show()

if __name__ == "__main__":
    visualizer = DeterminantVisualizer()
    visualizer.run()
