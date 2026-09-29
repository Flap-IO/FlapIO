import pygame
import random

# ---------------- SETTINGS ----------------
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
PIPE_WIDTH = 70
PIPE_GAP = 150          # space between top and bottom pipe
PIPE_SPEED = 3          # how fast pipes move left

class Pipe:
    def __init__(self, x):
        self.x = x  # horizontal position of the pipe

        # pick a random height for the gap's top edge
        self.gap_y = random.randint(100, SCREEN_HEIGHT - 100 - PIPE_GAP)

        # top pipe: from top of screen (0) down to gap_y
        self.top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.gap_y)

        # bottom pipe: starts after the gap, goes to bottom of screen
        self.bottom_rect = pygame.Rect(
            self.x,
            self.gap_y + PIPE_GAP,
            PIPE_WIDTH,
            SCREEN_HEIGHT - (self.gap_y + PIPE_GAP)
        )

        self.passed = False  # used later for scoring

    def update(self):
        # move both pipes left every frame
        self.x -= PIPE_SPEED
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self, screen):
        # draw both pipes as green rectangles
        pygame.draw.rect(screen, (0, 200, 0), self.top_rect)
        pygame.draw.rect(screen, (0, 200, 0), self.bottom_rect)

    def off_screen(self):
        # true once the pipe has moved fully off the left edge
        return self.x + PIPE_WIDTH < 0

    def collide(self, bird_rect):
        # returns True if the bird's rectangle touches either pipe
        return self.top_rect.colliderect(bird_rect) or self.bottom_rect.colliderect(bird_rect)


# ---------------- MAIN LOOP EXAMPLE ----------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

pipes = [Pipe(SCREEN_WIDTH)]   # start with one pipe off-screen to the right
SPAWN_INTERVAL = 90            # frames between new pipes
frame_count = 0

running = True
while running:
    clock.tick(60)  # 60 frames per second
    frame_count += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # spawn a new pipe every SPAWN_INTERVAL frames
    if frame_count % SPAWN_INTERVAL == 0:
        pipes.append(Pipe(SCREEN_WIDTH))

    # update pipes and remove ones that left the screen
    for pipe in pipes:
        pipe.update()
    pipes = [p for p in pipes if not p.off_screen()]

    # draw everything
    screen.fill((135, 206, 235))  # sky blue background
    for pipe in pipes:
        pipe.draw(screen)
    pygame.display.flip()

pygame.quit()