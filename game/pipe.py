"""Pipe: a top + bottom obstacle with a gap in between."""
import random
import pygame
from settings import SCREEN_HEIGHT, PIPE_WIDTH, PIPE_GAP, PIPE_SPEED, GREEN


class Pipe:
    def __init__(self, x, rng=random):
        self.x = x  # horizontal position of the pipe

        # random height for the gap's TOP edge
        self.gap_y = rng.randint(100, SCREEN_HEIGHT - 100 - PIPE_GAP)

        self.top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.gap_y)
        self.bottom_rect = pygame.Rect(
            self.x, self.gap_y + PIPE_GAP,
            PIPE_WIDTH, SCREEN_HEIGHT - (self.gap_y + PIPE_GAP))

        self.passed = False  # becomes True once the bird gets past (scoring)

    def update(self):
        self.x -= PIPE_SPEED
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.top_rect)
        pygame.draw.rect(screen, GREEN, self.bottom_rect)

    def off_screen(self):
        return self.x + PIPE_WIDTH < 0

    def collide(self, bird_rect):
        return self.top_rect.colliderect(bird_rect) or self.bottom_rect.colliderect(bird_rect)
