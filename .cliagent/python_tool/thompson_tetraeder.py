import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from typing import List, Tuple, Optional
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import sys

class ThompsonTetraeder:
    def __init__(self) -> None:
        # Define vertices of the tetrahedron
        self.vertices: np.ndarray = np.array([
            [0, 0, 0],  # vertex 0
            [1, 0, 0],  # vertex 1
            [0.5, np.sqrt(0.75), 0],  # vertex 2
            [0.5, np.sqrt(0.75)/3, np.sqrt(2/3)]  # vertex 3
        ])
        
        # Define faces using vertex indices
        self.faces: List[List[int]] = [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ]
        
        self.current_angle: float = 0.0
        self.setup_plot()

    def setup_plot(self) -> None:
        self.fig: Figure = plt.figure(figsize=(8, 8))
        self.ax: Axes3D = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.update_plot()

    def rotate_vertices(self, angle: float) -> np.ndarray:
        rotation_matrix: np.ndarray = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        return np.dot(self.vertices, rotation_matrix)

    def update_plot(self) -> None:
        self.ax.clear()
        rotated_vertices: np.ndarray = self.rotate_vertices(self.current_angle)
        
        # Create the faces
        faces_coords: List[np.ndarray] = [[rotated_vertices[idx] for idx in face] for face in self.faces]
        
        # Plot the tetrahedron
        collection: Poly3DCollection = Poly3DCollection(faces_coords, alpha=0.5)
        collection.set_facecolor(['cyan', 'magenta', 'yellow', 'green'])
        self.ax.add_collection3d(collection)
        
        # Set plot limits and labels
        self.ax.set_xlim([-1, 2])
        self.ax.set_ylim([-1, 2])
        self.ax.set_zlim([-1, 2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        plt.draw()

    def on_click(self, event: Optional[plt.MouseEvent]) -> None:
        if event is not None and event.button == 1:  # Left click
            self.current_angle += np.pi/12
        elif event is not None and event.button == 3:  # Right click
            self.current_angle -= np.pi/12
        self.update_plot()

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Interactive Thompson Tetraeder")
        self.setGeometry(100, 100, 800, 800)
        
        # Create main widget and layout
        main_widget: QWidget = QWidget()
        self.setCentralWidget(main_widget)
        layout: QVBoxLayout = QVBoxLayout(main_widget)
        
        # Create tetraeder and add to layout
        self.tetraeder: ThompsonTetraeder = ThompsonTetraeder()
        layout.addWidget(FigureCanvasQTAgg(self.tetraeder.fig))

def main() -> None:
    app: QApplication = QApplication(sys.argv)
    window: MainWindow = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
