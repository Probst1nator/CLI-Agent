import pygame
import sys
from typing import Tuple

def initialize_graphics() -> Tuple[pygame.Surface, pygame.display.Info]:
    """
    Initialize the graphics for the game.

    Returns:
        A tuple containing the game window surface and the display info.
    """
    # Initialize the pygame modules
    pygame.init()

    # Set up some constants
    WINDOW_WIDTH: int = 800
    WINDOW_HEIGHT: int = 600

    # Create the game window
    window: pygame.Surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    # Get the display info
    display_info: pygame.display.Info = pygame.display.Info()

    return window, display_info

def handle_events() -> None:
    """
    Handle events for the game.

    This function will handle the QUIT event to prevent the game from crashing.
    """
    for event in pygame.event.get():
        # Check if the user wants to quit
        if event.type == pygame.QUIT:
            # Quit the game
            pygame.quit()
            sys.exit()

def update_game_state(window: pygame.Surface) -> None:
    """
    Update the game state.

    Args:
        window: The game window surface.
    """
    # Fill the window with a black color
    window.fill((0, 0, 0))

    # Update the display
    pygame.display.update()

def main() -> None:
    """
    The main function of the game.

    This function will initialize the graphics, handle events, and update the game state.
    """
    try:
        # Initialize the graphics
        window, display_info = initialize_graphics()

        # Game loop
        while True:
            # Handle events
            handle_events()

            # Update the game state
            update_game_state(window)

            # Limit the frame rate to 60 FPS
            pygame.time.Clock().tick(60)

    except pygame.error as e:
        # Handle pygame errors
        print(f"Pygame error: {e}")

    except Exception as e:
        # Handle any other errors
        print(f"Error: {e}")

if __name__ == "__main__":
    main()