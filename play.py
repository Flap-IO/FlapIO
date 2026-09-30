"""Play FlapIO yourself.  SPACE / click = flap, R or SPACE after crash = restart, ESC = quit.

Run:  python play.py
"""
import pygame
from game.flappy_game import FlappyBirdGame


def main():
    game = FlappyBirdGame()
    running = True
    while running:
        flap = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r or (event.key == pygame.K_SPACE and game.game_over):
                    game.reset()
                elif event.key == pygame.K_SPACE:
                    flap = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game.game_over:
                    game.reset()
                else:
                    flap = True

        if flap:
            game.bird.flap()
        game.update()
        game.render()
    game.close()


if __name__ == "__main__":
    pygame.init()
    main()
