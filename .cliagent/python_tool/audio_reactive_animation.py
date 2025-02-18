import pygame
import numpy as np
from typing import Tuple, List, Optional
import pyaudio
import struct
from colorsys import hsv_to_rgb
import math

class AudioVisualizer:
    def __init__(self, width: int = 800, height: int = 600):
        # Initialize Pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Audio Reactive Visualizer")
        
        # Audio setup
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.p = pyaudio.PyAudio()
        
        # Try to get the default input device
        try:
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
        except Exception as e:
            print(f"Could not initialize audio: {e}")
            self.stream = None

        # Visualization parameters
        self.points: List[Tuple[float, float]] = []
        self.time: float = 0
        self.running: bool = True
        self.clock = pygame.time.Clock()

    def get_audio_data(self) -> float:
        if not self.stream:
            return 0.0
        
        try:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            data_int = struct.unpack(str(self.CHUNK) + 'h', data)
            amplitude = np.mean(np.abs(data_int)) / 32768.0  # Normalize
            return min(amplitude * 5, 1.0)  # Scale and cap the amplitude
        except Exception:
            return 0.0

    def create_spiral_points(self, amplitude: float) -> List[Tuple[float, float]]:
        points = []
        num_points = 150
        for i in range(num_points):
            angle = i * 0.1 + self.time
            radius = (i * 2 + amplitude * 100) * (1 + amplitude)
            x = self.width/2 + math.cos(angle) * radius
            y = self.height/2 + math.sin(angle) * radius
            points.append((x, y))
        return points

    def run(self) -> None:
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            # Get audio data
            amplitude = self.get_audio_data()

            # Clear screen with fade effect
            self.screen.fill((0, 0, 0, 10))

            # Create and draw spiral
            self.points = self.create_spiral_points(amplitude)
            
            # Draw points with color based on amplitude
            for i in range(len(self.points)-1):
                start_pos = self.points[i]
                end_pos = self.points[i+1]
                
                # Create color based on position and amplitude
                hue = (i / len(self.points) + self.time * 0.1) % 1.0
                rgb = hsv_to_rgb(hue, 1.0, amplitude + 0.5)
                color = tuple(int(x * 255) for x in rgb)
                
                # Draw line with varying thickness
                thickness = int(amplitude * 5) + 1
                pygame.draw.line(self.screen, color, start_pos, end_pos, thickness)

            pygame.display.flip()
            self.time += 0.05
            self.clock.tick(60)

    def cleanup(self) -> None:
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        pygame.quit()

if __name__ == "__main__":
    try:
        visualizer = AudioVisualizer()
        visualizer.run()
    finally:
        visualizer.cleanup()
