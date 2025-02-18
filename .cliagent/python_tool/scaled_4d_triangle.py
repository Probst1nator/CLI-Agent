import numpy as np
import pygame
from typing import List, Tuple, Optional
from math import cos, sin

class FourDTriangle:
    def __init__(self, size: int = 800) -> None:
        pygame.init()
        self.size: int = size
        self.screen = pygame.display.set_mode((size, size))
        self.clock = pygame.time.Clock()
        self.scale: float = 200.0
        self.angles: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6 rotation angles for 4D
        
        # Define 4D triangle vertices
        self.vertices: List[np.ndarray] = [
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0])
        ]

    def rotate_4d(self, points: List[np.ndarray], angles: List[float]) -> List[np.ndarray]:
        rotated: List[np.ndarray] = []
        for point in points:
            # Apply multiple 4D rotations
            p = point.copy()
            # XY rotation
            p = np.array([
                cos(angles[0])*p[0] - sin(angles[0])*p[1],
                sin(angles[0])*p[0] + cos(angles[0])*p[1],
                p[2],
                p[3]
            ])
            # XZ rotation
            p = np.array([
                cos(angles[1])*p[0] - sin(angles[1])*p[2],
                p[1],
                sin(angles[1])*p[0] + cos(angles[1])*p[2],
                p[3]
            ])
            rotated.append(p)
        return rotated

    def project_3d(self, points: List[np.ndarray], w_plane: float = 2.0) -> List[np.ndarray]:
        projected_3d: List[np.ndarray] = []
        for point in points:
            # Perspective projection from 4D to 3D
            factor: float = 1.0 / (w_plane - point[3])
            p3d = np.array([
                point[0] * factor,
                point[1] * factor,
                point[2] * factor
            ])
            projected_3d.append(p3d)
        return projected_3d

    def project_2d(self, points: List[np.ndarray], z_plane: float = 2.0) -> List[Tuple[float, float]]:
        projected_2d: List[Tuple[float, float]] = []
        for point in points:
            # Perspective projection from 3D to 2D
            factor: float = 1.0 / (z_plane - point[2])
            x: float = point[0] * factor * self.scale + self.size/2
            y: float = point[1] * factor * self.scale + self.size/2
            projected_2d.append((x, y))
        return projected_2d

    def draw(self, points_2d: List[Tuple[float, float]]) -> None:
        self.screen.fill((0, 0, 0))
        # Draw edges
        for i in range(len(points_2d)):
            for j in range(i + 1, len(points_2d)):
                pygame.draw.line(self.screen, (255, 255, 255), 
                               points_2d[i], points_2d[j], 2)
        # Draw vertices
        for point in points_2d:
            pygame.draw.circle(self.screen, (255, 0, 0), 
                             (int(point[0]), int(point[1])), 5)

    def run(self) -> None:
        running: bool = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    if event.buttons[0]:  # Left mouse button
                        self.angles[0] += event.rel[0] * 0.01
                        self.angles[1] += event.rel[1] * 0.01
                elif event.type == pygame.MOUSEWHEEL:
                    self.scale += event.y * 10

            # Apply transformations
            rotated_points = self.rotate_4d(self.vertices, self.angles)
            projected_3d = self.project_3d(rotated_points)
            projected_2d = self.project_2d(projected_3d)

            self.draw(projected_2d)
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    triangle = FourDTriangle(size=800)
    triangle.run()
