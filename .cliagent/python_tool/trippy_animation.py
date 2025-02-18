import pygame
import pyaudio
import numpy as np
import colorsys
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass
import random

@dataclass
class ParticleSystem:
    particles: List[Tuple[float, float, float, float]]  # x, y, speed, angle
    max_particles: int = 100
    
    def update(self, volume: float) -> None:
        for i, (x, y, speed, angle) in enumerate(self.particles):
            new_x = x + math.cos(angle) * speed * volume
            new_y = y + math.sin(angle) * speed * volume
            new_speed = speed * 0.99
            new_angle = angle + 0.1
            
            if 0 <= new_x <= WIDTH and 0 <= new_y <= HEIGHT:
                self.particles[i] = (new_x, new_y, new_speed, new_angle)
            else:
                self.particles[i] = self.create_particle()
    
    def create_particle(self) -> Tuple[float, float, float, float]:
        return (
            random.randint(0, WIDTH),
            random.randint(0, HEIGHT),
            random.uniform(1, 5),
            random.uniform(0, 2 * math.pi)
        )

class AudioVisualizer:
    def __init__(self, width: int = 1200, height: int = 800) -> None:
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Trippy Audio Visualizer")
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024
        )
        
        # Visual elements
        self.time: float = 0
        self.color_offset: float = 0
        self.particle_system = ParticleSystem([
            ParticleSystem.create_particle(None) for _ in range(100)
        ])
    
    def get_audio_data(self) -> float:
        try:
            data = np.frombuffer(self.stream.read(1024, exception_on_overflow=False), dtype=np.float32)
            return np.abs(data).mean() * 20
        except Exception:
            return 0.0
    
    def create_gradient(self, volume: float) -> pygame.Surface:
        surface = pygame.Surface((self.width, self.height))
        for y in range(self.height):
            hue = (y / self.height + self.time / 10.0) % 1.0
            rgb = [int(255 * x) for x in colorsys.hsv_to_rgb(hue, 0.8, 0.8)]
            pygame.draw.line(surface, rgb, (0, y), (self.width, y))
        return surface
    
    def draw_frame(self, volume: float) -> None:
        # Create trippy background
        gradient = self.create_gradient(volume)
        self.screen.blit(gradient, (0, 0))
        
        # Update and draw particles
        self.particle_system.update(volume)
        for x, y, _, _ in self.particle_system.particles:
            size = int(5 + volume * 20)
            hue = (self.time / 10.0 + x / self.width) % 1.0
            color = [int(255 * x) for x in colorsys.hsv_to_rgb(hue, 0.9, 0.9)]
            pygame.draw.circle(self.screen, color, (int(x), int(y)), size)
        
        # Draw central pattern
        center_x, center_y = self.width // 2, self.height // 2
        num_points = 12
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi + self.time
            radius = 100 + volume * 200
            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            size = 20 + volume * 30
            hue = (self.time / 5.0 + i / num_points) % 1.0
            color = [int(255 * x) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
            pygame.draw.circle(self.screen, color, (int(x), int(y)), int(size))
            
            # Draw connecting lines
            next_i = (i + 1) % num_points
            next_angle = (next_i / num_points) * 2 * math.pi + self.time
            next_x = center_x + math.cos(next_angle) * radius
            next_y = center_y + math.sin(next_angle) * radius
            pygame.draw.line(self.screen, color, (x, y), (next_x, next_y), 3)
    
    def run(self) -> None:
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            volume = self.get_audio_data()
            self.draw_frame(volume)
            pygame.display.flip()
            
            self.time += 0.05
            clock.tick(60)
        
        self.cleanup()
    
    def cleanup(self) -> None:
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        pygame.quit()

if __name__ == "__main__":
    WIDTH, HEIGHT = 1200, 800
    visualizer = AudioVisualizer(WIDTH, HEIGHT)
    visualizer.run()
