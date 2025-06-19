import cv2
import cv2.aruco as aruco
import pygame
import sys
import numpy as np
from random import choice, randint, uniform
from particles import Particle, ExplodingParticle, FloatingParticle
from constants import SCREEN_WIDTH, SCREEN_HEIGHT
from sort import Sort

# --- Fungsi Mapping Koordinat dari Kamera ke Pygame ---
def map_cv_to_pygame(x, y, frame_shape, screen_size):
    """
    Mengubah koordinat dari OpenCV (kamera) ke koordinat Pygame.
    """
    frame_height, frame_width = frame_shape
    screen_width, screen_height = screen_size

    mapped_x = int(x * (screen_width / frame_width))
    mapped_y = int(y * (screen_height / frame_height))
    return mapped_x, mapped_y

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

# Warna
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)  # Untuk kotak pelacakan
CYAN = (0, 255, 255)  # Untuk titik corner
PARTICLE_COLORS = [(255, 200, 0), (255, 100, 0), (255, 255, 0)]  # Kuning, oranye, emas

# Kelompok partikel
all_particles = pygame.sprite.Group()

# --- SORT Tracker Initialization ---
mot_tracker = Sort()

# --- Particle Emission Function ---
def emit_particles(pos, num_particles=3, particle_type='exploding'):
    """
    Emit partikel dari posisi tertentu di layar Pygame.
    """
    for _ in range(num_particles):
        color = choice(PARTICLE_COLORS)
        angle = uniform(0, 2 * np.pi)
        direction = pygame.math.Vector2(np.cos(angle), np.sin(angle))
        speed = randint(50, 150)
        if particle_type == 'exploding':
            ExplodingParticle(all_particles, pos, color, direction, speed)
        elif particle_type == 'floating':
            FloatingParticle(all_particles, pos, color, direction, speed)
        else:
            Particle(all_particles, pos, color, direction, speed)

# --- Main Loop ---
running = True
clock = pygame.time.Clock()

while running:
    dt = clock.tick(60) / 1000.0  # Delta time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- OpenCV Capture & ArUco Detection ---
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    detections = []
    if ids is not None:
        for i, corner in enumerate(corners):
            c = corner[0]
            x_min = np.min(c[:, 0])
            y_min = np.min(c[:, 1])
            x_max = np.max(c[:, 0])
            y_max = np.max(c[:, 1])
            width = x_max - x_min
            height = y_max - y_min
            detections.append([x_min, y_min, width, height])

    detections = np.array(detections) if detections else np.empty((0, 4))

    tracked_objects = mot_tracker.update(detections)

    # --- Gambar bounding box dan partikel ---
    for obj in tracked_objects:
        x, y, w, h, obj_id = obj.astype(int)
        center_x_cv = x + w // 2
        center_y_cv = y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        cv2.putText(frame, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
        cv2.circle(frame, (center_x_cv, center_y_cv), 5, RED, -1)

        # Mapping ke Pygame
        pygame_x, pygame_y = map_cv_to_pygame(center_x_cv, center_y_cv, frame.shape[:2], (SCREEN_WIDTH, SCREEN_HEIGHT))
        emit_particles((pygame_x, pygame_y), num_particles=3, particle_type='exploding')

    # --- Gambar Sudut ArUco ke Layar Pygame ---
    if ids is not None:
        for i, corner in enumerate(corners):
            c = corner[0]
            for j in range(4):  # Keempat sudut
                cx, cy = c[j]
                px, py = map_cv_to_pygame(cx, cy, frame.shape[:2], (SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.draw.circle(screen, CYAN, (px, py), 5)

    # --- Tampilkan OpenCV Frame ---
    cv2.imshow('ArUco Detection (Webcam) with SORT Tracking', frame)

    # --- Update Pygame ---
    screen.fill(BLACK)
    all_particles.update(dt)
    all_particles.draw(screen)
    pygame.display.flip()

    # --- Keluar saat tekan 'q' ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()