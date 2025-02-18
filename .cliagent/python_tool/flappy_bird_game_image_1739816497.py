import pygame
import sys
from typing import Tuple, List

def initialize_pygame() -> Tuple[pygame.Surface, pygame.time.Clock]:
    """
    Initialize the Pygame window and clock.

    Returns:
        A tuple containing the Pygame window and clock.
    """
    try:
        pygame.init()
        window: pygame.Surface = pygame.display.set_mode((800, 600))
        clock: pygame.time.Clock = pygame.time.Clock()
        return window, clock
    except pygame.error as e:
        print(f"Failed to initialize Pygame: {e}")
        sys.exit(1)

def load_images() -> List[pygame.Surface]:
    """
    Load the images from CleanPNG.

    Returns:
        A list of Pygame surfaces representing the images.
    """
    try:
        # Load the images from CleanPNG
        background_image: pygame.Surface = pygame.image.load("background.png")
        player_image: pygame.Surface = pygame.image.load("player.png")
        enemy_image: pygame.Surface = pygame.image.load("enemy.png")
        return [background_image, player_image, enemy_image]
    except pygame.error as e:
        print(f"Failed to load images: {e}")
        sys.exit(1)

def draw_images(window: pygame.Surface, images: List[pygame.Surface]) -> None:
    """
    Draw the images on the Pygame window.

    Args:
        window: The Pygame window.
        images: A list of Pygame surfaces representing the images.
    """
    try:
        # Draw the background image
        window.blit(images[0], (0, 0))  # (0, 0) is the top-left corner of the window
        # Draw the player image
        window.blit(images[1], (100, 100))  # (100, 100) is the position of the player
        # Draw the enemy image
        window.blit(images[2], (300, 300))  # (300, 300) is the position of the enemy
    except pygame.error as e:
        print(f"Failed to draw images: {e}")
        sys.exit(1)

def handle_events() -> None:
    """
    Handle Pygame events.
    """
    try:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Quit the game when the window is closed
                pygame.quit()
                sys.exit(0)
    except pygame.error as e:
        print(f"Failed to handle events: {e}")
        sys.exit(1)

def main() -> None:
    """
    The main function.
    """
    window, clock = initialize_pygame()
    images = load_images()
    while True:
        handle_events()
        draw_images(window, images)
        # Update the window
        pygame.display.update()
        # Cap the frame rate to 60 FPS
        clock.tick(60)

if __name__ == "__main__":
    main()