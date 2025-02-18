import pygame
import math
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pygame import Vector2

@dataclass
class GameObject:
    position: Vector2
    velocity: Vector2
    rotation: float
    size: float
    active: bool

class AsteroidsGame:
    def __init__(self, width: int = 800, height: int = 600) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Player properties
        self.player = GameObject(
            position=Vector2(width/2, height/2),
            velocity=Vector2(0, 0),
            rotation=0,
            size=20,
            active=True
        )
        
        # Game objects
        self.bullets: List[GameObject] = []
        self.asteroids: List[GameObject] = []
        self.score: int = 0
        
        # Spawn initial asteroids
        for _ in range(4):
            self.spawn_asteroid()

    def spawn_asteroid(self) -> None:
        side = random.randint(0, 3)
        if side == 0:  # Top
            pos = Vector2(random.randint(0, self.width), 0)
        elif side == 1:  # Right
            pos = Vector2(self.width, random.randint(0, self.height))
        elif side == 2:  # Bottom
            pos = Vector2(random.randint(0, self.width), self.height)
        else:  # Left
            pos = Vector2(0, random.randint(0, self.height))
            
        self.asteroids.append(GameObject(
            position=pos,
            velocity=Vector2(random.uniform(-2, 2), random.uniform(-2, 2)),
            rotation=random.uniform(0, 360),
            size=random.uniform(20, 50),
            active=True
        ))

    def handle_input(self) -> None:
        keys = pygame.key.get_pressed()
        
        # Rotation
        if keys[pygame.K_LEFT]:
            self.player.rotation -= 5
        if keys[pygame.K_RIGHT]:
            self.player.rotation += 5
            
        # Thrust
        if keys[pygame.K_UP]:
            thrust = Vector2(0, -0.2).rotate(-self.player.rotation)
            self.player.velocity += thrust
            
        # Shooting
        if keys[pygame.K_SPACE]:
            if len(self.bullets) < 5:  # Limit bullets
                direction = Vector2(0, -10).rotate(-self.player.rotation)
                self.bullets.append(GameObject(
                    position=self.player.position.copy(),
                    velocity=direction,
                    rotation=0,
                    size=2,
                    active=True
                ))

    def update(self) -> None:
        # Update player
        self.player.position += self.player.velocity
        self.player.velocity *= 0.99  # Friction
        
        # Wrap around screen
        self.player.position.x %= self.width
        self.player.position.y %= self.height
        
        # Update bullets
        for bullet in self.bullets[:]:
            bullet.position += bullet.velocity
            if not (0 <= bullet.position.x <= self.width and 0 <= bullet.position.y <= self.height):
                self.bullets.remove(bullet)
                
        # Update asteroids
        for asteroid in self.asteroids[:]:
            asteroid.position += asteroid.velocity
            asteroid.position.x %= self.width
            asteroid.position.y %= self.height
            
            # Check collision with bullets
            for bullet in self.bullets[:]:
                if (asteroid.position - bullet.position).length() < asteroid.size:
                    if asteroid.size > 20:
                        # Split asteroid
                        for _ in range(2):
                            self.asteroids.append(GameObject(
                                position=asteroid.position.copy(),
                                velocity=Vector2(random.uniform(-2, 2), random.uniform(-2, 2)),
                                rotation=random.uniform(0, 360),
                                size=asteroid.size / 2,
                                active=True
                            ))
                    self.score += 100
                    self.asteroids.remove(asteroid)
                    self.bullets.remove(bullet)
                    break
            
            # Check collision with player
            if (asteroid.position - self.player.position).length() < asteroid.size + self.player.size:
                self.running = False

    def draw(self) -> None:
        self.screen.fill((0, 0, 0))
        
        # Draw player
        points = [
            Vector2(0, -20).rotate(-self.player.rotation),
            Vector2(15, 20).rotate(-self.player.rotation),
            Vector2(-15, 20).rotate(-self.player.rotation)
        ]
        points = [(p + self.player.position) for p in points]
        pygame.draw.polygon(self.screen, (255, 255, 255), points, 1)
        
        # Draw bullets
        for bullet in self.bullets:
            pygame.draw.circle(self.screen, (255, 255, 255), bullet.position, bullet.size)
        
        # Draw asteroids
        for asteroid in self.asteroids:
            pygame.draw.circle(self.screen, (255, 255, 255), asteroid.position, asteroid.size, 1)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    game = AsteroidsGame()
    game.run()
