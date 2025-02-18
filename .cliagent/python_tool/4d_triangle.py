import numpy as np
from typing import List, Tuple, Optional
import pygame
from pygame.locals import *
from math import cos, sin
import sys

class FourDTriangle:
    def __init__(self, scale: float = 100):
        # Define 4D triangle vertices (using a simplex in 4D)
        self.vertices: List[np.ndarray] = [
            np.array([0., 0., 0., 0.]),
            np.array([1., 0., 0., 0.]),
            np.array([0.5, 0.866, 0., 0.]),
            np.array([0.5, 0.289, 0.816, 0.])
        ]
        self.scale: float = scale
        self.angles: List[float] = [0., 0., 0., 0., 0., 0.]  # 6 rotation angles for 4D

    def rotate_4d(self, vertices: List[np.ndarray], angle_xy: float, angle_xz: float, 
                  angle_xw: float, angle_yz: float, angle_yw: float, angle_zw: float) -> List[np.ndarray]:
        # Create rotation matrices for each plane
        rotations = []
        # XY rotation
        rot_xy = np.array([
            [cos(angle_xy), -sin(angle_xy), 0, 0],
            [sin(angle_xy), cos(angle_xy), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        rotations.append(rot_xy)

        # Add other rotation matrices (XZ, XW, YZ, YW, ZW)
        # ... (similar pattern for other rotation matrices)

        # Apply all rotations
        result = []
        for vertex in vertices:
            rotated = vertex.copy()
            for rotation in rotations:
                rotated = np.dot(rotation, rotated)
            result.append(rotated)
        return result

    def project_3d(self, point: np.ndarray, w_distance: float = 2.) -> np.ndarray:
        # Project from 4D to 3D using perspective projection
        w = 1 / (w_distance - point[3])
        return np.array([point[0] * w, point[1] * w, point[2] * w])

    def project_2d(self, point: np.ndarray, distance: float = 5.) -> Tuple[float, float]:
        # Project from 3D to 2D
        z = 1 / (distance - point[2])
        return (point[0] * z * self.scale, point[1] * z * self.scale)

def main() -> None:
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("4D Triangle Visualization")
    clock = pygame.time.Clock()

    triangle = FourDTriangle()
    rotation_speed: float = 0.02

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # Update rotation angles
        triangle.angles = [(angle + rotation_speed) % (2 * np.pi) 
                         for angle in triangle.angles]

        # Clear screen
        screen.fill((0, 0, 0))

        # Rotate and project vertices
        rotated = triangle.rotate_4d(
            triangle.vertices, *triangle.angles
        )
        
        projected_3d = [triangle.project_3d(v) for v in rotated]
        projected_2d = [triangle.project_2d(v) for v in projected_3d]

        # Draw the triangle
        for i in range(len(projected_2d)):
            for j in range(i + 1, len(projected_2d)):
                start_pos = (int(projected_2d[i][0] + width//2),
                           int(projected_2d[i][1] + height//2))
                end_pos = (int(projected_2d[j][0] + width//2),
                          int(projected_2d[j][1] + height//2))
                pygame.draw.line(screen, (255, 255, 255), start_pos, end_pos)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
