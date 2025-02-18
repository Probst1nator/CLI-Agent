import pygame
import sys
import random

# Initialize Pygame
pygame.init()

class Game:
    """
    A simple game implementation using Pygame.

    Attributes:
        screen_width (int): The width of the game screen.
        screen_height (int): The height of the game screen.
        player_size (int): The size of the player object.
        player_pos (list): The position of the player object.
        enemy_size (int): The size of the enemy object.
        enemy_pos (list): The position of the enemy object.
        score (int): The current score.
    """

    def __init__(self) -> None:
        """
        Initializes the game with default values.
        """
        self.screen_width: int = 800
        self.screen_height: int = 600
        self.player_size: int = 50
        self.player_pos: list = [self.screen_width / 2, self.screen_height / 2]
        self.enemy_size: int = 50
        self.enemy_pos: list = [random.randint(0, self.screen_width - self.enemy_size), random.randint(0, self.screen_height - self.enemy_size)]
        self.score: int = 0
        self.screen: pygame.display = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock: pygame.time.Clock = pygame.time.Clock()

    def handle_input(self) -> None:
        """
        Handles user input to move the player object.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Get a list of all keys currently being pressed down
        keys = pygame.key.get_pressed()

        # Move the player object based on the keys being pressed
        if keys[pygame.K_UP]:
            # Move up
            self.player_pos[1] -= 5
        if keys[pygame.K_DOWN]:
            # Move down
            self.player_pos[1] += 5
        if keys[pygame.K_LEFT]:
            # Move left
            self.player_pos[0] -= 5
        if keys[pygame.K_RIGHT]:
            # Move right
            self.player_pos[0] += 5

        # Ensure the player object doesn't move off the screen
        if self.player_pos[0] < 0:
            self.player_pos[0] = 0
        elif self.player_pos[0] > self.screen_width - self.player_size:
            self.player_pos[0] = self.screen_width - self.player_size
        if self.player_pos[1] < 0:
            self.player_pos[1] = 0
        elif self.player_pos[1] > self.screen_height - self.player_size:
            self.player_pos[1] = self.screen_height - self.player_size

    def update(self) -> None:
        """
        Updates the game state, including enemy movement and collision detection.
        """
        # Move the enemy object
        self.enemy_pos[0] += 2
        if self.enemy_pos[0] > self.screen_width:
            self.enemy_pos[0] = -self.enemy_size
            self.enemy_pos[1] = random.randint(0, self.screen_height - self.enemy_size)

        # Check for collision between the player and enemy objects
        if (self.player_pos[0] < self.enemy_pos[0] + self.enemy_size and
            self.player_pos[0] + self.player_size > self.enemy_pos[0] and
            self.player_pos[1] < self.enemy_pos[1] + self.enemy_size and
            self.player_pos[1] + self.player_size > self.enemy_pos[1]):
            # Collision detected, reset the enemy object and increment the score
            self.enemy_pos[0] = -self.enemy_size
            self.enemy_pos[1] = random.randint(0, self.screen_height - self.enemy_size)
            self.score += 1

    def render(self) -> None:
        """
        Renders the game state to the screen.
        """
        # Fill the screen with a black background
        self.screen.fill((0, 0, 0))

        # Draw the player object
        pygame.draw.rect(self.screen, (255, 255, 255), (self.player_pos[0], self.player_pos[1], self.player_size, self.player_size))

        # Draw the enemy object
        pygame.draw.rect(self.screen, (255, 0, 0), (self.enemy_pos[0], self.enemy_pos[1], self.enemy_size, self.enemy_size))

        # Draw the score
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        # Update the display
        pygame.display.flip()

def main() -> None:
    """
    The main entry point of the game.
    """
    try:
        game: Game = Game()
        while True:
            game.handle_input()
            game.update()
            game.render()
            game.clock.tick(60)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()