"""FlappyBirdGame: Bird + Pipes + scoring + game over.

Headless by default (no window) so AI training is fast.
The window is only created the first time render() is called.
"""
import random
import pygame
from settings import (SCREEN_WIDTH, SCREEN_HEIGHT, FPS, SPAWN_INTERVAL,
                      PIPE_WIDTH, SKY_BLUE, WHITE)
from game.bird import Bird
from game.pipe import Pipe


class FlappyBirdGame:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.screen = None
        self.clock = None
        self.font = None
        self.reset()

    # ------------------------------------------------------------ game logic
    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.bird = Bird()
        self.pipes = [Pipe(SCREEN_WIDTH, self.rng)]  # first pipe starts off-screen right
        self.score = 0
        self.game_over = False
        self.frame_count = 0

    def update(self):
        """Advance the game by exactly one frame."""
        if self.game_over:
            return
        self.frame_count += 1
        self.bird.update()

        if self.frame_count % SPAWN_INTERVAL == 0:
            self.pipes.append(Pipe(SCREEN_WIDTH, self.rng))

        for pipe in self.pipes:
            pipe.update()
            if not pipe.passed and pipe.x + PIPE_WIDTH < self.bird.x:
                pipe.passed = True
                self.score += 1
            if pipe.collide(self.bird.rect):
                self.game_over = True

        self.pipes = [p for p in self.pipes if not p.off_screen()]

        if self.bird.hit_bounds():
            self.game_over = True

    # ------------------------------------------------------------- rendering
    def _init_display(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("FlapIO")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 48)

    def render(self, fps=FPS):
        if self.screen is None:
            self._init_display()
        pygame.event.pump()  # keeps the window responsive

        self.screen.fill(SKY_BLUE)
        for pipe in self.pipes:
            pipe.draw(self.screen)
        self.bird.draw(self.screen)

        text = self.font.render(str(self.score), True, WHITE)
        self.screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, 20))

        if self.game_over:
            msg = self.font.render("Game Over", True, WHITE)
            self.screen.blit(msg, (SCREEN_WIDTH // 2 - msg.get_width() // 2,
                                   SCREEN_HEIGHT // 2 - 30))
            small = pygame.font.SysFont(None, 28).render(
                "SPACE = restart", True, WHITE)
            self.screen.blit(small, (SCREEN_WIDTH // 2 - small.get_width() // 2,
                                     SCREEN_HEIGHT // 2 + 10))

        pygame.display.flip()
        if fps:
            self.clock.tick(fps)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
