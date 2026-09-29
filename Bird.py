import pygame
from settings import GRAVITY, FLAP_STRENGTH, BIRD_SIZE, YELLOW

class Bird:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.velocity = 0 # current vertical speed

# rectangle used for drawing and collision detection
        self.rect = pygame.Rect(self.x,self.y,BIRD_SIZE,BIRD_SIZE)
        def flap(self):                            # called when the player space/clicks
            self.velocity = FLAP_STRENGTH                    # sets velocity to a negative number so the bird moves up

        def update(self):                   # gravity constantly pulls the bird down
            self.velocity += GRAVITY
            self.y += self.velocity

            self.react.y = self.y                           # keeps the rectangle in sync with the birds position