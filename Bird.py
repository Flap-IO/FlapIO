"""Bird: gravity + flap physics, collision rectangle and drawing."""
import pygame
from settings import (GRAVITY, FLAP_STRENGTH, BIRD_SIZE, BIRD_START_X,
                      SCREEN_HEIGHT, YELLOW)


class Bird:
    def __init__(self, x=BIRD_START_X, y=SCREEN_HEIGHT // 2 - BIRD_SIZE // 2):
        self.x = x
        self.y = float(y)
        self.velocity = 0.0  # current vertical speed (negative = going up)

        # rectangle used for drawing and collision detection
        self.rect = pygame.Rect(int(self.x), int(self.y), BIRD_SIZE, BIRD_SIZE)

    def flap(self):
        """Called when the player (or the AI) flaps."""
        self.velocity = FLAP_STRENGTH

    def update(self):
        """Gravity constantly pulls the bird down."""
        self.velocity += GRAVITY
        self.y += self.velocity
        self.rect.y = round(self.y)  # keep the rectangle in sync with y

    def hit_bounds(self):
        """True if the bird touched the ceiling or the floor."""
        return self.y < 0 or self.y + BIRD_SIZE > SCREEN_HEIGHT

    def draw(self, screen):
        pygame.draw.rect(screen, YELLOW, self.rect)
        eye = (self.rect.right - 8, self.rect.top + 8)   # small eye + beak
        pygame.draw.circle(screen, (0, 0, 0), eye, 3)
        pygame.draw.rect(screen, (255, 120, 0),
                         (self.rect.right, self.rect.top + 12, 6, 6))
