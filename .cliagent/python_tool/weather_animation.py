import pygame
import sys
import requests
import json
from typing import Tuple, Dict, Optional
from datetime import datetime
import os
from moviepy.editor import ImageSequenceClip
import tempfile

class WeatherAnimation:
    def __init__(self, width: int = 800, height: int = 600):
        """Initialize the weather animation system."""
        pygame.init()
        self.width: int = width
        self.height: int = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Bavarian Weather Animation")
        self.clock = pygame.time.Clock()
        self.frames: list = []
        self.weather_data: Optional[Dict] = None
        self.temp_dir = tempfile.mkdtemp()

    def get_weather_data(self) -> None:
        """Fetch weather data for Bavaria from OpenWeatherMap API."""
        try:
            API_KEY = "YOUR_API_KEY_HERE"  # Replace with actual API key
            url = f"http://api.openweathermap.org/data/2.5/weather?q=Munich,DE&appid={API_KEY}"
            response = requests.get(url)
            response.raise_for_status()
            self.weather_data = response.json()
        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            sys.exit(1)

    def draw_sun(self, position: Tuple[int, int]) -> None:
        """Draw sun animation frame."""
        pygame.draw.circle(self.screen, (255, 255, 0), position, 40)
        rays = [(60, 0), (42, 42), (0, 60), (-42, 42),
                (-60, 0), (-42, -42), (0, -60), (42, -42)]
        for ray in rays:
            end_pos = (position[0] + ray[0], position[1] + ray[1])
            pygame.draw.line(self.screen, (255, 255, 0),
                           position, end_pos, 4)

    def draw_cloud(self, position: Tuple[int, int]) -> None:
        """Draw cloud animation frame."""
        cloud_color = (200, 200, 200)
        pygame.draw.circle(self.screen, cloud_color,
                         (position[0], position[1]), 30)
        pygame.draw.circle(self.screen, cloud_color,
                         (position[0] + 20, position[1] - 10), 30)
        pygame.draw.circle(self.screen, cloud_color,
                         (position[0] + 40, position[1]), 30)

    def draw_rain(self, position: Tuple[int, int]) -> None:
        """Draw rain animation frame."""
        rain_color = (0, 0, 255)
        for i in range(0, 60, 20):
            pygame.draw.line(self.screen, rain_color,
                           (position[0] + i, position[1]),
                           (position[0] + i - 10, position[1] + 20), 3)

    def create_frame(self, frame_number: int) -> str:
        """Create and save a single animation frame."""
        self.screen.fill((135, 206, 235))  # Sky blue background
        
        # Animate based on weather condition
        weather_condition = self.weather_data.get('weather', [{}])[0].get('main', 'Clear')
        
        base_position = (self.width // 2, self.height // 2)
        offset = int(10 * pygame.math.sin(frame_number / 5))
        
        if weather_condition == 'Clear':
            self.draw_sun((base_position[0], base_position[1] + offset))
        elif weather_condition == 'Clouds':
            self.draw_cloud((base_position[0] + offset, base_position[1]))
        elif weather_condition in ['Rain', 'Drizzle']:
            self.draw_cloud((base_position[0] + offset, base_position[1]))
            self.draw_rain((base_position[0], base_position[1] + 40))

        frame_path = os.path.join(self.temp_dir, f"frame_{frame_number}.png")
        pygame.image.save(self.screen, frame_path)
        return frame_path

    def create_animation(self) -> None:
        """Create and save the weather animation."""
        try:
            frame_paths = []
            for i in range(60):  # Create 60 frames
                frame_path = self.create_frame(i)
                frame_paths.append(frame_path)
                self.clock.tick(30)

            clip = ImageSequenceClip(frame_paths, fps=30)
            clip.write_videofile("bavarian_weather.mp4")

        except Exception as e:
            print(f"Error creating animation: {e}")
            sys.exit(1)
        finally:
            # Cleanup temporary files
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)

    def run(self) -> None:
        """Main animation loop."""
        try:
            self.get_weather_data()
            self.create_animation()
            
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                self.screen.fill((135, 206, 235))
                frame_number = pygame.time.get_ticks() // 33 % 60
                self.create_frame(frame_number)
                pygame.display.flip()
                self.clock.tick(30)

        except Exception as e:
            print(f"Runtime error: {e}")
        finally:
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    animation = WeatherAnimation()
    animation.run()