import pygame
import cv2
import numpy as np
from particles import Particle, ExplodingParticle, FloatingParticle
from constants import SCREEN_WIDTH, SCREEN_HEIGHT
from sort import Sort
from random import choice, randint, uniform # Import for particle generation

# --- Pygame Initialization ---
pygame.init()
display_surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Robot Tracking with ArUco and Particle Trails")
clock = pygame.time.Clock()

particle_group = pygame.sprite.Group()

# --- OpenCV and ArUco Initialization ---
# Choose the appropriate dictionary for your ArUco markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Initialize camera (0 for default webcam, change if you have multiple cameras or IP camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    pygame.quit()
    exit()

# Set camera resolution (optional, might affect performance)
# Match this to your SCREEN_WIDTH/HEIGHT for consistent scaling
cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# --- SORT Initialization ---
mot_tracker = Sort(max_age=5, min_hits=3) # Adjust max_age and min_hits as needed

# --- Particle Spawning Functions (Adapted for ArUco) ---
def spawn_particles_at_location(x, y, n: int, particle_type="normal"):
    for _ in range(n):
        pos = (x, y)
        if particle_type == "normal":
            color = choice(("red", "green", "blue"))
            direction = pygame.math.Vector2(uniform(-1, 1), uniform(-1, 1))
            direction = direction.normalize()
            speed = randint(50, 400)
            Particle(particle_group, pos, color, direction, speed)
        elif particle_type == "exploding":
            color = choice(("red", "yellow", "orange"))
            direction = pygame.math.Vector2(uniform(-0.2, 0.2), uniform(-1, 0))
            direction = direction.normalize()
            speed = randint(50, 400)
            ExplodingParticle(particle_group, pos, color, direction, speed)
        elif particle_type == "floating":
            color = "white"
            direction = pygame.math.Vector2(0, -1)
            speed = randint(50, 100)
            FloatingParticle(particle_group, pos, color, direction, speed)


# --- Main Loop ---
running = True
while running:
    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Capture Frame from Camera ---
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    # Flip frame horizontally (optional, depending on camera orientation and setup)
    # If your camera is directly above and the robot moves intuitively on screen, you might not need this.
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale for ArUco detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(gray)

    detections = []
    if ids is not None:
        for i, corner in enumerate(corners):
            # Calculate bounding box [x, y, w, h] from corner points
            x_min = int(min(c[0] for c in corner[0]))
            y_min = int(min(c[1] for c in corner[0]))
            x_max = int(max(c[0] for c in corner[0]))
            y_max = int(max(c[1] for c in corner[0]))
            width = x_max - x_min
            height = y_max - y_min
            detections.append([x_min, y_min, width, height])

            # Optional: Draw bounding box and ID on the OpenCV frame for debugging in a separate window
            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # cv2.putText(frame, f"ID: {ids[i][0]}", (x_min, y_min - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update SORT tracker with current detections
    # detections need to be a numpy array
    if len(detections) > 0:
        tracked_objects = mot_tracker.update(np.array(detections))
    else:
        tracked_objects = mot_tracker.update(np.empty((0, 4))) # No detections

    # --- Process Tracked Objects and Spawn Particles ---
    for obj in tracked_objects:
        x, y, w, h, obj_id = obj.astype(int)
        center_x = x + w // 2
        center_y = y + h // 2

        # Spawn particles at the center of the tracked ArUco marker
        # We want a trail, so spawn a small number of particles continuously
        spawn_particles_at_location(center_x, center_y, 3, particle_type="normal")
        # You can add other particle types if desired:
        # spawn_particles_at_location(center_x, center_y, 1, particle_type="floating")


    # --- Pygame Rendering ---
    # Clear screen with black background
    display_surface.fill("black")

    # Draw particles
    particle_group.draw(display_surface)

    # --- Update Particles and Display ---
    dt = clock.tick() / 1000.0 # Time since last frame in seconds
    particle_group.update(dt)

    pygame.display.update()

    # Optional: Display the raw camera feed in a separate OpenCV window for debugging
    # cv2.imshow("Camera Feed (for debugging)", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit this debug window
    #     running = False

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows() # Close any OpenCV windows
pygame.quit()