import pygame
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto

class Direction(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()

@dataclass
class GameConfig:
    width: int = 800
    height: int = 600
    grid_size: int = 20
    fps: int = 15
    background_color: Tuple[int, int, int] = (0, 0, 0)
    snake_color: Tuple[int, int, int] = (0, 255, 0)
    food_color: Tuple[int, int, int] = (255, 0, 0)

class SnakeGame:
    """A classic Snake game implementation using Pygame."""

    def __init__(self, config: GameConfig = GameConfig()):
        """Initialize the game with given configuration."""
        self.config = config
        pygame.init()
        self.screen = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self) -> None:
        """Reset the game state to initial values."""
        self.snake: List[Tuple[int, int]] = [(self.config.width // 2, self.config.height // 2)]
        self.direction = Direction.RIGHT
        self.food = self._generate_food()
        self.score = 0
        self.game_over = False

    def _generate_food(self) -> Tuple[int, int]:
        """Generate new food position."""
        while True:
            x = random.randrange(0, self.config.width, self.config.grid_size)
            y = random.randrange(0, self.config.height, self.config.grid_size)
            if (x, y) not in self.snake:
                return (x, y)

    def handle_input(self) -> bool:
        """Handle user input. Returns False if game should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN
                elif event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_r and self.game_over:
                    self.reset_game()
        return True

    def update(self) -> None:
        """Update game state."""
        if self.game_over:
            return

        head = list(self.snake[0])
        if self.direction == Direction.UP:
            head[1] -= self.config.grid_size
        elif self.direction == Direction.DOWN:
            head[1] += self.config.grid_size
        elif self.direction == Direction.LEFT:
            head[0] -= self.config.grid_size
        elif self.direction == Direction.RIGHT:
            head[0] += self.config.grid_size

        # Check collision with walls
        if (head[0] < 0 or head[0] >= self.config.width or
            head[1] < 0 or head[1] >= self.config.height):
            self.game_over = True
            return

        # Check collision with self
        if tuple(head) in self.snake[1:]:
            self.game_over = True
            return

        self.snake.insert(0, tuple(head))

        # Check if food is eaten
        if tuple(head) == self.food:
            self.score += 1
            self.food = self._generate_food()
        else:
            self.snake.pop()

    def draw(self) -> None:
        """Draw the current game state."""
        self.screen.fill(self.config.background_color)

        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.screen, self.config.snake_color,
                           (*segment, self.config.grid_size, self.config.grid_size))

        # Draw food
        pygame.draw.rect(self.screen, self.config.food_color,
                        (*self.food, self.config.grid_size, self.config.grid_size))

        if self.game_over:
            font = pygame.font.Font(None, 50)
            text = font.render(f"Game Over! Score: {self.score} (Press R to restart)", True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.config.width/2, self.config.height/2))
            self.screen.blit(text, text_rect)

        pygame.display.flip()

    def run(self) -> None:
        """Main game loop."""
        try:
            running = True
            while running:
                running = self.handle_input()
                self.update()
                self.draw()
                self.clock.tick(self.config.fps)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            pygame.quit()

if __name__ == "__main__":
    game = SnakeGame()
    game.run()