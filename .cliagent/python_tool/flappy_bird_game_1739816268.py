import pygame
import sys
from typing import Tuple

def initialize_pygame() -> Tuple[pygame.Surface, pygame.time.Clock]:
    """
    Initialize Pygame and create a game window.

    Returns:
        A tuple containing the game window surface and the game clock.
    """
    try:
        # Initialize Pygame
        pygame.init()
        
        # Set the dimensions of the game window
        window_dimensions = (800, 600)
        
        # Create the game window surface
        window_surface = pygame.display.set_mode(window_dimensions)
        
        # Set the title of the game window
        pygame.display.set_caption("Simple Game")
        
        # Create the game clock
        game_clock = pygame.time.Clock()
        
        return window_surface, game_clock
    
    except pygame.error as e:
        print(f"Failed to initialize Pygame: {e}")
        sys.exit(1)

def handle_user_input() -> None:
    """
    Handle user input.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Quit the game if the user closes the window
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN:
            # Handle keyboard input
            if event.key == pygame.K_ESCAPE:
                # Quit the game if the user presses the escape key
                pygame.quit()
                sys.exit(0)

def update_game_state(window_surface: pygame.Surface, game_clock: pygame.time.Clock) -> None:
    """
    Update the game state.

    Args:
        window_surface: The game window surface.
        game_clock: The game clock.
    """
    try:
        # Fill the window surface with a black background
        window_surface.fill((0, 0, 0))
        
        # Draw a white rectangle in the center of the window
        rectangle_color = (255, 255, 255)
        rectangle_dimensions = (50, 50)
        rectangle_x = (window_surface.get_width() - rectangle_dimensions[0]) // 2
        rectangle_y = (window_surface.get_height() - rectangle_dimensions[1]) // 2
        pygame.draw.rect(window_surface, rectangle_color, (rectangle_x, rectangle_y, rectangle_dimensions[0], rectangle_dimensions[1]))
        
        # Update the display
        pygame.display.update()
        
        # Cap the frame rate to 60 FPS
        game_clock.tick(60)
    
    except pygame.error as e:
        print(f"Failed to update the game state: {e}")

def check_collision() -> bool:
    """
    Check for collision.

    Returns:
        True if a collision occurs, False otherwise.
    """
    try:
        # Simulate a collision by checking if the rectangle is at the center of the window
        # In a real game, you would replace this with actual collision detection logic
        rectangle_x = (pygame.display.get_surface().get_width() - 50) // 2
        rectangle_y = (pygame.display.get_surface().get_height() - 50) // 2
        if rectangle_x == 375 and rectangle_y == 275:
            return True
        else:
            return False
    
    except pygame.error as e:
        print(f"Failed to check for collision: {e}")
        return False

def update_score(score: int) -> int:
    """
    Update the score.

    Args:
        score: The current score.

    Returns:
        The updated score.
    """
    try:
        # Simulate scoring by incrementing the score by 1
        # In a real game, you would replace this with actual scoring logic
        return score + 1
    
    except Exception as e:
        print(f"Failed to update the score: {e}")
        return score

def main() -> None:
    """
    The main function.
    """
    window_surface, game_clock = initialize_pygame()
    score = 0
    
    while True:
        handle_user_input()
        
        # Check for collision and update the score if a collision occurs
        if check_collision():
            score = update_score(score)
            print(f"Collision detected! Score: {score}")
        
        update_game_state(window_surface, game_clock)

if __name__ == "__main__":
    main()