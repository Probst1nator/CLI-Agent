import pygame
import random
import math
from typing import List, Tuple

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)
SHOOT_DELAY = 150  # Milliseconds between shots

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Asteroids")

class Bullet:
    def __init__(self, x: float, y: float, angle: float):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 7.0
        self.radius = 2

    def update(self) -> None:
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        if self.x < 0:
            self.x = WIDTH
        elif self.x > WIDTH:
            self.x = 0
        if self.y < 0:
            self.y = HEIGHT
        elif self.y > HEIGHT:
            self.y = 0

    def draw(self) -> None:
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius)

    def collides_with(self, asteroid: 'Asteroid') -> bool:
        distance = math.sqrt((self.x - asteroid.x)**2 + (self.y - asteroid.y)**2)
        return distance < asteroid.radius + self.radius

class Ship:
    def __init__(self):
        self.x: float = WIDTH // 2
        self.y: float = HEIGHT // 2
        self.angle: float = 0
        self.speed: float = 0
        self.bullets: List[Bullet] = []
        self.last_shot: int = 0

    def shoot(self) -> None:
        current_time = pygame.time.get_ticks()
        if current_time - self.last_shot >= SHOOT_DELAY:
            self.bullets.append(Bullet(self.x, self.y, self.angle))
            self.last_shot = current_time

    def draw(self) -> None:
        points: List[Tuple[float, float]] = []
        for i in range(3):
            angle = self.angle + i * math.pi * 2 / 3
            points.append((self.x + math.cos(angle) * 20, self.y + math.sin(angle) * 20))
        pygame.draw.polygon(screen, WHITE, points)
        
        for bullet in self.bullets:
            bullet.draw()

    def update(self) -> None:
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        if self.x < 0:
            self.x = WIDTH
        elif self.x > WIDTH:
            self.x = 0
        if self.y < 0:
            self.y = HEIGHT
        elif self.y > HEIGHT:
            self.y = 0

        for bullet in self.bullets:
            bullet.update()

class Asteroid:
    def __init__(self):
        self.x: float = random.randint(0, WIDTH)
        self.y: float = random.randint(0, HEIGHT)
        self.speed_x: float = random.uniform(-2, 2)
        self.speed_y: float = random.uniform(-2, 2)
        self.radius: int = random.randint(10, 30)

    def draw(self) -> None:
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius)

    def update(self) -> None:
        self.x += self.speed_x
        self.y += self.speed_y

        if self.x < 0:
            self.x = WIDTH
        elif self.x > WIDTH:
            self.x = 0
        if self.y < 0:
            self.y = HEIGHT
        elif self.y > HEIGHT:
            self.y = 0

def main() -> None:
    ship = Ship()
    asteroids: List[Asteroid] = [Asteroid() for _ in range(10)]
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    ship.speed = 5
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    ship.speed = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            ship.angle -= 0.1
        if keys[pygame.K_RIGHT]:
            ship.angle += 0.1
        if keys[pygame.K_SPACE]:
            ship.shoot()

        # Update game state
        ship.update()
        
        # Check collisions
        for bullet in ship.bullets[:]:
            for asteroid in asteroids[:]:
                if bullet.collides_with(asteroid):
                    if bullet in ship.bullets:  # Check if bullet still exists
                        ship.bullets.remove(bullet)
                    if asteroid in asteroids:  # Check if asteroid still exists
                        asteroids.remove(asteroid)
                        asteroids.append(Asteroid())
                    break

        # Draw everything
        screen.fill((0, 0, 0))
        ship.draw()
        for asteroid in asteroids:
            asteroid.update()
            asteroid.draw()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
