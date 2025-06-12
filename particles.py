import pygame
from random import choice, randint, uniform
from constants import SCREEN_WIDTH, SCREEN_HEIGHT

class Particle(pygame.sprite.Sprite):
    def __init__(self, group, pos, color, direction, speed):
        super().__init__(group)
        self.image = pygame.Surface((5, 5))
        self.image.fill(color)
        self.rect = self.image.get_rect(center=pos)

        self.pos = pygame.math.Vector2(pos)
        self.direction = direction
        self.speed = speed
        self.lifetime = randint(500, 1500) # milliseconds
        self.creation_time = pygame.time.get_ticks()

    def update(self, dt):
        self.pos += self.direction * self.speed * dt
        self.rect.center = round(self.pos.x), round(self.pos.y)

        if pygame.time.get_ticks() - self.creation_time > self.lifetime:
            self.kill()

class ExplodingParticle(Particle):
    def __init__(self, group, pos, color, direction, speed):
        super().__init__(group, pos, color, direction, speed)
        self.image = pygame.Surface((7, 7))
        self.image.fill(color)

    def update(self, dt):
        super().update(dt)
        self.speed *= 0.95 # Slow down over time

class FloatingParticle(Particle):
    def __init__(self, group, pos, color, direction, speed):
        super().__init__(group, pos, color, direction, speed)
        self.image = pygame.Surface((3, 3))
        self.image.fill(color)
        self.gravity = 50 # Simulate a slight upward float then fall

    def update(self, dt):
        self.direction.y += self.gravity * dt # Apply gravity
        super().update(dt)
        self.speed *= 0.98 # Slow down