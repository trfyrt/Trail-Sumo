import cv2
import cv2.aruco as aruco
import pygame
import sys
import numpy as np
from random import choice, randint, uniform
from particles import Particle, ExplodingParticle, FloatingParticle
from constants import SCREEN_WIDTH, SCREEN_HEIGHT
from sort import Sort

# --- ArUco Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# --- Pygame Setup ---
pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("ArUco Particle Trail with SORT Tracking")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0) # For tracked bounding box
PARTICLE_COLORS = [(255, 200, 0), (255, 100, 0), (255, 255, 0)] # Yellow, Orange, Gold

# Particle Group
all_particles = pygame.sprite.Group()

# --- SORT Tracker Initialization ---
mot_tracker = Sort() # Initialize the SORT tracker

# --- Particle Emission Function ---
def emit_particles(pos, num_particles=3, particle_type='exploding'):
    """
    Emits particles from a given position in the Pygame window.
    """
    for _ in range(num_particles):
        color = choice(PARTICLE_COLORS)
        # Random direction vector
        angle = uniform(0, 2 * np.pi)
        direction = pygame.math.Vector2(np.cos(angle), np.sin(angle))
        speed = randint(50, 150) # Pixels per second

        if particle_type == 'exploding':
            ExplodingParticle(all_particles, pos, color, direction, speed)
        elif particle_type == 'floating':
            FloatingParticle(all_particles, pos, color, direction, speed)
        else: # Default to generic particle
            Particle(all_particles, pos, color, direction, speed)

# --- Main Loop ---
running = True
clock = pygame.time.Clock() # For delta time

while running:
    # Calculate delta time for consistent movement across different frame rates
    dt = clock.tick(60) / 1000.0 # Delta time in seconds

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- OpenCV (ArUco Detection) ---
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Prepare detections for SORT tracker
    detections = []
    if ids is not None:
        for i, corner in enumerate(corners):
            c = corner[0] # Get the 4 corners of the marker
            # Calculate bounding box from corners: [x_min, y_min, width, height]
            x_min = np.min(c[:, 0])
            y_min = np.min(c[:, 1])
            x_max = np.max(c[:, 0])
            y_max = np.max(c[:, 1])
            width = x_max - x_min
            height = y_max - y_min
            detections.append([x_min, y_min, width, height])

    detections = np.array(detections) if detections else np.empty((0, 4))

    # Update the SORT tracker with current detections
    # tracked_objects will be [[x,y,w,h,id],[x,y,w,h,id],...]
    tracked_objects = mot_tracker.update(detections)

    # --- Drawing on OpenCV Frame and Emitting Particles ---
    for obj in tracked_objects:
        x, y, w, h, obj_id = obj.astype(int) # Get x, y, width, height, and ID

        # Calculate center of the tracked bounding box
        center_x_cv = x + w // 2
        center_y_cv = y + h // 2

        # Draw bounding box and ID on the OpenCV frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        cv2.putText(frame, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
        cv2.circle(frame, (center_x_cv, center_y_cv), 5, RED, -1) # Draw center on OpenCV feed

        # Map OpenCV coordinates to Pygame coordinates for particle emission
        frame_height, frame_width, _ = frame.shape
        pygame_x = int(center_x_cv * (SCREEN_WIDTH / frame_width))
        pygame_y = int(center_y_cv * (SCREEN_HEIGHT / frame_height))

        # Emit particles from the tracked marker's current location
        emit_particles((pygame_x, pygame_y), num_particles=3, particle_type='exploding') # Adjust num_particles for density

    cv2.imshow('ArUco Detection (Webcam) with SORT Tracking', frame)

    # --- Pygame Drawing ---
    screen.fill(BLACK) # Clear screen

    # Update and draw particles
    all_particles.update(dt)
    all_particles.draw(screen)

    pygame.display.flip() # Update Pygame display

    # --- Exit Condition ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()