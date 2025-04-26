import pygame
import random
import os
import time
import math # Need math for calculating ray positions and vectors

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_width = 800
screen_height = 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Frog Splatter Simulation")

# Colors
black = (0, 0, 0)
gray = (100, 100, 100)
red = (255, 0, 0)
white = (255, 255, 255) # For the fading light

# Road properties
road_height = 50
road_y = screen_height - road_height

# Car properties
car_speed = 5
# Define new desired sizes for the images
new_car_width = 150
new_car_height = 75
new_frog_width = 50
new_frog_height = 50

# Image paths (using the CORRECT paths in the home directory and .jpg/.png extension)
car_image_path = os.path.expanduser("~/green_bmw.jpg")
frog_image_path = os.path.expanduser("~/blue_frog.jpg")
angel_image_path = os.path.expanduser("~/angel.png") # Assuming the angel is a PNG


print(f"Attempting to load car image from: {car_image_path}")
print(f"Attempting to load frog image from: {frog_image_path}")
print(f"Attempting to load angel image from: {angel_image_path}")


# Load images first to get original dimensions
try:
    car_image_orig = pygame.image.load(car_image_path).convert_alpha()
    frog_image_orig = pygame.image.load(frog_image_path).convert_alpha()
    angel_image_orig = pygame.image.load(angel_image_path).convert_alpha()

except pygame.error as e:
    print(f"Error loading images: {e}")
    print(f"Make sure image files exist and are valid: '{car_image_path}', '{frog_image_path}', '{angel_image_path}'")
    pygame.quit()
    exit() # Exit if images can't be loaded

# Scale images
car_image = pygame.transform.scale(car_image_orig, (new_car_width, new_car_height))
frog_image = pygame.transform.scale(frog_image_orig, (new_frog_width, new_frog_height))

# Scale angel proportionally based on the desired width (same as car width)
angel_original_aspect = angel_image_orig.get_width() / angel_image_orig.get_height()
scaled_angel_width = new_car_width # Start with car width
scaled_angel_height = int(scaled_angel_width / angel_original_aspect)
# If the user wants it slimmer, we could reduce scaled_angel_width here, e.g., scaled_angel_width = int(new_car_width * 0.8)
# Let's stick to car width for now, as "slimmer" might mean less tall if the original is wide.
# If the original is tall, scaling to car width will make it shorter, which might be "slimmer" visually.
# Let's scale to car width and see how it looks.
angel_image = pygame.transform.scale(angel_image_orig, (scaled_angel_width, scaled_angel_height))
print(f"Scaled angel image to: {scaled_angel_width}x{scaled_angel_height}")


# Define a consistent off-screen top-left position for the angel's descent start and return target
ANGEL_OFFSCREEN_TOPLEFT = (-scaled_angel_width, -scaled_angel_height) # Start completely off-screen top-left


# Car, Frog, and Angel setup function (initial setup)
def initial_setup():
    car_rect = car_image.get_rect()
    car_rect.bottom = road_y
    car_rect.left = -car_rect.width # Start off-screen left

    frog_rect = frog_image.get_rect()
    frog_rect.bottom = road_y
    frog_rect.centerx = screen_width // 2 # Center the frog horizontally

    # Angel initial position (set to the consistent start position)
    angel_rect = angel_image.get_rect()
    angel_rect.topleft = ANGEL_OFFSCREEN_TOPLEFT


    frog_alive = True
    splatter_particles = []
    last_frog_pos = frog_rect.center # Store initial position

    # Angel state and movement variables
    angel_state = 'idle' # 'idle', 'flying_down', 'flying_up'
    angel_target_pos = last_frog_pos # Angel flies to where the frog died
    angel_return_speed = 8 # Speed for flying back up (pixels per frame)
    angel_descent_duration = 90 # frames for the angel's descent (controls speed) - ADJUSTED DURATION
    angel_descent_timer = 0 # Timer for the descent flight


    return car_rect, frog_rect, frog_alive, splatter_particles, last_frog_pos, angel_rect, angel_state, angel_target_pos, angel_return_speed, angel_descent_duration, angel_descent_timer


# Splatter properties
splatter_count = 50 # Number of splatter particles
splatter_speed_range = (1, 5) # Min/Max speed for particles
splatter_lifetime = 30 # How many frames particles last

# State machine
STATE_PLAYING = 'playing'
STATE_ANGEL_DESCENT = 'angel_descent' # New state for angel flying down
STATE_RESPAWNING = 'respawning' # Combined state for fading light and frog

current_state = STATE_PLAYING

# Respawning timing and effects
fading_duration = 90 # frames for the fading effect (1.5 seconds at 60 fps)
fading_timer = 0
last_frog_pos = (screen_width // 2, road_y - new_frog_height // 2) # Default position

# Variables for fading alpha (need to be accessible in drawing)
light_alpha = 0
frog_alpha = 0

# Sun effect properties
sun_radius = 40 # Radius of the central sun circle
ray_length = 60 # Length of the sun rays
ray_width = 10 # Width of the sun rays
num_rays = 12 # Number of rays

# Game loop
running = True
car_rect, frog_rect, frog_alive, splatter_particles, last_frog_pos, angel_rect, angel_state, angel_target_pos, angel_return_speed, angel_descent_duration, angel_descent_timer = initial_setup()
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- State Logic ---
    if current_state == STATE_PLAYING:
        # Move the car
        car_rect.x += car_speed

        # Move angel up if flying up
        if angel_state == 'flying_up':
            # Calculate direction vector from current pos to top-left off-screen
            direction = pygame.math.Vector2(ANGEL_OFFSCREEN_TOPLEFT) - pygame.math.Vector2(angel_rect.topleft)
            # Normalize the vector and multiply by speed
            if direction.length() > 0: # Avoid division by zero
                 direction = direction.normalize() * angel_return_speed
                 angel_rect.x += direction.x
                 angel_rect.y += direction.y

            # Check if angel is off-screen top-left
            if angel_rect.right < 0 and angel_rect.bottom < 0:
                angel_state = 'idle'
                # Reset angel position to start point when idle
                angel_rect.topleft = ANGEL_OFFSCREEN_TOPLEFT
                print("Angel returned to heaven and reset.")


        # Check for collision if frog is alive
        if frog_alive and car_rect.colliderect(frog_rect):
            print("Collision detected!")
            frog_alive = False
            last_frog_pos = frog_rect.center # Store position for respawn effect
            # Create splatter particles
            splatter_origin = frog_rect.center
            for _ in range(splatter_count):
                angle = random.uniform(0, 360)
                speed = random.uniform(*splatter_speed_range)
                # Calculate velocity components
                vel_x = speed * pygame.math.Vector2(1, 0).rotate(angle).x
                vel_y = speed * pygame.math.Vector2(1, 0).rotate(angle).y
                splatter_particles.append({
                    'pos': list(splatter_origin),
                    'vel': [vel_x, vel_y],
                    'lifetime': splatter_lifetime
                })

        # Check if car is off-screen and frog is dead to start angel descent
        if car_rect.left > screen_width and not frog_alive:
            print("Car off-screen and frog dead, starting angel descent...")
            current_state = STATE_ANGEL_DESCENT # Transition to the new state
            angel_state = 'flying_down' # Set angel state
            angel_descent_timer = angel_descent_duration # Start the descent timer
            angel_target_pos = last_frog_pos # Angel flies to where the frog died
            # Car stays off-screen, don't reset its position yet
            # Fading effect and frog respawn don't start yet


    elif current_state == STATE_ANGEL_DESCENT:
        angel_descent_timer -= 1

        # --- Update Angel Position during flying_down ---
        if angel_descent_timer >= 0:
            # Calculate interpolation factor (0 at start, 1 at end of descent)
            t = 1 - (angel_descent_timer / angel_descent_duration)
            t = max(0, min(1, t)) # Clamp t between 0 and 1

            # Linear interpolation between start and target position
            # Angel's center moves from ANGEL_OFFSCREEN_TOPLEFT to angel_target_pos
            angel_rect.center = (int(ANGEL_OFFSCREEN_TOPLEFT[0] + (angel_target_pos[0] - ANGEL_OFFSCREEN_TOPLEFT[0]) * t),
                                 int(ANGEL_OFFSCREEN_TOPLEFT[1] + (angel_target_pos[1] - ANGEL_OFFSCREEN_TOPLEFT[1]) * t))

        if angel_descent_timer <= 0:
            print("Angel descent ended, starting respawning effect...")
            # Angel has reached the target position, transition to respawning
            current_state = STATE_RESPAWNING
            fading_timer = fading_duration # Start the fading timer for light/frog
            # Angel stays at the target position during respawning


    elif current_state == STATE_RESPAWNING:
        fading_timer -= 1

        # Calculate alpha based on timer progress *every frame in this state*
        # Light alpha goes from 255 to 0
        light_alpha = int((fading_timer / fading_duration) * 255)
        light_alpha = max(0, light_alpha) # Ensure alpha doesn't go below 0

        # Frog alpha goes from 0 to 255
        frog_alpha = int((1 - (fading_timer / fading_duration)) * 255)
        frog_alpha = min(255, frog_alpha) # Ensure alpha doesn't go above 255

        # Angel stays at the target position during this state
        angel_rect.center = angel_target_pos


        if fading_timer <= 0:
            print("Respawning effect ended, resetting car and returning to playing state.")
            # Respawn frog (it's already drawn with full alpha on the last frame of respawning)
            frog_alive = True # Set alive for the next playing state
            # Reset car position
            car_rect.left = -car_rect.width
            # Clear old splatter particles for the new round
            splatter_particles = []
            current_state = STATE_PLAYING # Return to playing state

            # --- Trigger Angel Flight Up ---
            angel_state = 'flying_up'
            # Angel is already at the target position (last_frog_pos)


    # --- Update Splatter Particles (always update regardless of state) ---
    for particle in splatter_particles[:]: # Iterate over a copy to allow removal
        particle['pos'][0] += particle['vel'][0]
        particle['pos'][1] += particle['vel'][1]
        particle['lifetime'] -= 1
        if particle['lifetime'] <= 0:
            splatter_particles.remove(particle)

    # --- Drawing ---
    screen.fill(black) # Sky
    pygame.draw.rect(screen, gray, (0, road_y, screen_width, road_height)) # Road

    # Draw frog
    if current_state == STATE_PLAYING and frog_alive:
        screen.blit(frog_image, frog_rect)
    elif current_state == STATE_RESPAWNING:
        # Draw the fading-in frog at the respawn position
        # Create a temporary surface for the frog with alpha *here* in the drawing loop
        temp_frog_surface = frog_image.copy()
        temp_frog_surface.set_alpha(frog_alpha)
        # Need to calculate the rect for the temporary surface at the correct position
        temp_frog_rect = temp_frog_surface.get_rect(center=last_frog_pos)
        screen.blit(temp_frog_surface, temp_frog_rect)


    # Draw fading light effect if in respawning state and timer > 0
    if current_state == STATE_RESPAWNING and fading_timer > 0:
        # Create a temporary surface for the sun effect
        # Make it large enough to contain the central circle and rays
        sun_surface_size = (sun_radius + ray_length) * 2
        sun_surface = pygame.Surface((sun_surface_size, sun_surface_size), pygame.SRCALPHA)
        sun_center = (sun_surface_size // 2, sun_surface_size // 2)

        # Draw central circle
        pygame.draw.circle(sun_surface, white, sun_center, sun_radius)

        # Draw rays
        for i in range(num_rays):
            angle = (360 / num_rays) * i
            # Calculate start and end points for the ray
            start_pos = pygame.math.Vector2(sun_radius, 0).rotate(angle) + sun_center
            end_pos = pygame.math.Vector2(sun_radius + ray_length, 0).rotate(angle) + sun_center

            # Draw lines for rays
            pygame.draw.line(sun_surface, white, start_pos, end_pos, ray_width)


        # Apply alpha to the sun surface
        sun_surface.set_alpha(light_alpha)

        # Blit the sun surface centered at the last frog position
        sun_rect = sun_surface.get_rect(center=last_frog_pos)
        screen.blit(sun_surface, sun_rect)

    # Draw the angel if not idle
    if angel_state != 'idle':
        screen.blit(angel_image, angel_rect)


    # Draw car (always draw, its position is handled by state)
    screen.blit(car_image, car_rect)

    # Draw splatter particles (always draw)
    for particle in splatter_particles:
        pygame.draw.circle(screen, red, (int(particle['pos'][0]), int(particle['pos'][1])), 2) # Draw small red circles

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60) # 60 frames per second

pygame.quit()
# Note: SystemExit is expected here as Pygame is shutting down.
# exit() # No need for explicit exit() after pygame.quit() in a script
