import sys
from typing import Tuple, Optional
import pygame

# Color constants
YELLOW: Tuple[int, int, int] = (255, 255, 0)  # Pikachu's body color
BLACK: Tuple[int, int, int] = (0, 0, 0)       # Eyes color
RED: Tuple[int, int, int] = (255, 0, 0)       # Cheeks color
WHITE: Tuple[int, int, int] = (255, 255, 255)  # Background color

class PikachuDrawer:
    """Class to handle drawing Pikachu using Pygame."""
    
    def __init__(self, width: int = 800, height: int = 600) -> None:
        """
        Initialize the Pikachu drawer.

        Args:
            width: Window width in pixels
            height: Window height in pixels
        """
        self.width = width
        self.height = height
        self.screen: Optional[pygame.Surface] = None

    def initialize_pygame(self) -> None:
        """Initialize Pygame and create the window."""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pikachu Drawing")
        except pygame.error as e:
            print(f"Failed to initialize Pygame: {e}")
            sys.exit(1)

    def draw_pikachu(self) -> None:
        """Draw Pikachu's components on the screen."""
        if not self.screen:
            return

        try:
            # Fill background
            self.screen.fill(WHITE)

            # Draw body (large circle)
            center = (self.width // 2, self.height // 2)
            pygame.draw.circle(self.screen, YELLOW, center, 100)

            # Draw eyes (two small black circles)
            left_eye = (center[0] - 30, center[1] - 20)
            right_eye = (center[0] + 30, center[1] - 20)
            pygame.draw.circle(self.screen, BLACK, left_eye, 10)
            pygame.draw.circle(self.screen, BLACK, right_eye, 10)

            # Draw cheeks (red circles)
            left_cheek = (center[0] - 50, center[1] + 20)
            right_cheek = (center[0] + 50, center[1] + 20)
            pygame.draw.circle(self.screen, RED, left_cheek, 15)
            pygame.draw.circle(self.screen, RED, right_cheek, 15)

            # Update display
            pygame.display.flip()

        except pygame.error as e:
            print(f"Error drawing Pikachu: {e}")
            self.cleanup()
            sys.exit(1)

    def run(self) -> None:
        """Main run loop for the Pikachu drawer."""
        self.initialize_pygame()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            self.draw_pikachu()
            pygame.time.delay(100)  # Add small delay to reduce CPU usage

        self.cleanup()

    def cleanup(self) -> None:
        """Clean up Pygame resources."""
        pygame.quit()

def main() -> None:
    """Main function to create and run the Pikachu drawer."""
    drawer = PikachuDrawer()
    drawer.run()

if __name__ == "__main__":
    main()