import pygame
import sys
import random

def initialize_pygame() -> None:
    """
    Initialize the Pygame library.
    """
    pygame.init()

def create_game_window(width: int, height: int) -> pygame.display:
    """
    Create a game window with the specified width and height.

    Args:
        width (int): The width of the game window.
        height (int): The height of the game window.

    Returns:
        pygame.display: The game window.
    """
    return pygame.display.set_mode((width, height))

class Bird:
    """
    A class representing a bird in the game.
    """

    def __init__(self, x: int, y: int) -> None:
        """
        Initialize a bird object.

        Args:
            x (int): The initial x-coordinate of the bird.
            y (int): The initial y-coordinate of the bird.
        """
        self.x = x
        self.y = y
        self.width = 50
        self.height = 50
        self.velocity = 0

    def draw(self, window: pygame.display) -> None:
        """
        Draw the bird on the game window.

        Args:
            window (pygame.display): The game window.
        """
        pygame.draw.rect(window, (255, 0, 0), (self.x, self.y, self.width, self.height))

    def update(self) -> None:
        """
        Update the bird's position based on its velocity.
        """
        self.velocity += 0.5  # gravity
        self.y += self.velocity

    def jump(self) -> None:
        """
        Make the bird jump.
        """
        self.velocity = -10

class Pipe:
    """
    A class representing a pipe in the game.
    """

    def __init__(self, x: int, y: int) -> None:
        """
        Initialize a pipe object.

        Args:
            x (int): The initial x-coordinate of the pipe.
            y (int): The initial y-coordinate of the pipe.
        """
        self.x = x
        self.y = y
        self.width = 80
        self.height = 400
        self.gap = 150

    def draw(self, window: pygame.display) -> None:
        """
        Draw the pipe on the game window.

        Args:
            window (pygame.display): The game window.
        """
        pygame.draw.rect(window, (0, 255, 0), (self.x, 0, self.width, self.y))
        pygame.draw.rect(window, (0, 255, 0), (self.x, self.y + self.gap, self.width, 600 - (self.y + self.gap)))

def check_collision(bird: Bird, pipe: Pipe) -> bool:
    """
    Check if the bird collides with the pipe.

    Args:
        bird (Bird): The bird object.
        pipe (Pipe): The pipe object.

    Returns:
        bool: True if the bird collides with the pipe, False otherwise.
    """
    # check if the bird's x-coordinate is within the pipe's x-coordinate range
    if bird.x + bird.width > pipe.x and bird.x < pipe.x + pipe.width:
        # check if the bird's y-coordinate is within the pipe's y-coordinate range
        if bird.y < pipe.y or bird.y + bird.height > pipe.y + pipe.gap:
            return True
    return False

def main() -> None:
    """
    The main function of the game.
    """
    try:
        # initialize Pygame
        initialize_pygame()

        # create a game window
        window = create_game_window(800, 600)
        pygame.display.set_caption("Flappy Bird")

        # create a clock object
        clock = pygame.time.Clock()

        # create a bird object
        bird = Bird(100, 100)

        # create a pipe object
        pipe = Pipe(600, random.randint(100, 400))

        # initialize score
        score = 0

        # game loop
        while True:
            # event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        bird.jump()

            # update game state
            bird.update()
            pipe.x -= 5  # move the pipe to the left

            # check for collision
            if check_collision(bird, pipe):
                print("Game Over!")
                print(f"Final Score: {score}")
                pygame.quit()
                sys.exit()

            # check if the pipe has moved off the screen
            if pipe.x < -pipe.width:
                pipe.x = 800  # reset the pipe's x-coordinate
                pipe.y = random.randint(100, 400)  # reset the pipe's y-coordinate
                score += 1  # increment the score

            # draw everything
            window.fill((135, 206, 235))  # fill the window with a light blue color
            bird.draw(window)
            pipe.draw(window)
            font = pygame.font.Font(None, 72)
            text = font.render(str(score), True, (0, 0, 0))
            window.blit(text, (400, 20))

            # update the display
            pygame.display.update()

            # cap the frame rate
            clock.tick(60)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()