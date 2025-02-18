import pygame
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class Vector2D:
    x: float
    y: float

    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)

class VectorSpace:
    def __init__(self, width: int = 800, height: int = 600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Interactive Vector Space")
        self.clock = pygame.time.Clock()
        self.origin = Vector2D(width // 2, height // 2)
        self.scale = 40  # pixels per unit
        self.vectors: List[Vector2D] = []
        self.selected_vector: Optional[int] = None
        self.dragging = False

    def screen_to_vector(self, pos: Tuple[int, int]) -> Vector2D:
        return Vector2D((pos[0] - self.origin.x) / self.scale,
                       (self.origin.y - pos[1]) / self.scale)

    def vector_to_screen(self, vector: Vector2D) -> Tuple[int, int]:
        return (int(self.origin.x + vector.x * self.scale),
                int(self.origin.y - vector.y * self.scale))

    def draw_grid(self) -> None:
        # Draw grid lines
        for x in range(0, self.width, self.scale):
            pygame.draw.line(self.screen, (200, 200, 200),
                           (x, 0), (x, self.height))
        for y in range(0, self.height, self.scale):
            pygame.draw.line(self.screen, (200, 200, 200),
                           (0, y), (self.width, y))

        # Draw axes
        pygame.draw.line(self.screen, (0, 0, 0),
                        (0, self.origin.y), (self.width, self.origin.y), 2)
        pygame.draw.line(self.screen, (0, 0, 0),
                        (self.origin.x, 0), (self.origin.x, self.height), 2)

    def draw_vector(self, vector: Vector2D, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
        end_pos = self.vector_to_screen(vector)
        origin_screen = (int(self.origin.x), int(self.origin.y))
        pygame.draw.line(self.screen, color, origin_screen, end_pos, 2)
        # Draw arrow head
        angle = np.arctan2(origin_screen[1] - end_pos[1],
                          end_pos[0] - origin_screen[0])
        arrow_size = 10
        arrow_angle = np.pi / 6
        pygame.draw.line(self.screen, color, end_pos,
                        (end_pos[0] - arrow_size * np.cos(angle - arrow_angle),
                         end_pos[1] + arrow_size * np.sin(angle - arrow_angle)), 2)
        pygame.draw.line(self.screen, color, end_pos,
                        (end_pos[0] - arrow_size * np.cos(angle + arrow_angle),
                         end_pos[1] + arrow_size * np.sin(angle + arrow_angle)), 2)

    def run(self) -> None:
        running = True
        while running:
            self.screen.fill((255, 255, 255))
            self.draw_grid()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mouse_pos = pygame.mouse.get_pos()
                        vector = self.screen_to_vector(mouse_pos)
                        self.vectors.append(vector)
                    elif event.button == 3:  # Right click
                        self.vectors = []

            # Draw all vectors
            for vector in self.vectors:
                self.draw_vector(vector)

            # Draw vector addition if there are at least 2 vectors
            if len(self.vectors) >= 2:
                result = self.vectors[0]
                for v in self.vectors[1:]:
                    result = result + v
                self.draw_vector(result, (0, 255, 0))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

if __name__ == "__main__":
    vector_space = VectorSpace()
    vector_space.run()
