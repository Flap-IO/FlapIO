"""FlappyBirdEnv: wraps the teammates' game so PPO can train on it.

Put this file at:  env/flappy_env.py

It only uses things from the UML diagram:
    game.bird.y, game.bird.velocity, game.bird.flap()
    game.pipes  (each pipe has .x and .gap_y)
    game.score, game.game_over
    game.reset(), game.update(), game.render()
Gymnasium rules:
    reset() -> (obs, info)
    step(action) -> (obs, reward, terminated, truncated, info)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game.flappy_game import FlappyBirdGame

# ---- numbers copied from the game (CHANGE these if your game uses others) ----
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
PIPE_WIDTH = 70
PIPE_GAP = 150        # gap_y is assumed to be the TOP edge of the gap
BIRD_X = 80           # bird's fixed horizontal position
MAX_SPEED = 10        # used only to scale velocity into about -1..1

# ---- reward numbers (easy to tune) ----
REWARD_SURVIVE = 0.1  # every frame the bird stays alive
REWARD_PIPE = 5.0     # passing a pipe
REWARD_CRASH = -10.0  # crashing
MAX_STEPS = 5000      # end very long games so training doesn't get stuck


class FlappyBirdEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.game = FlappyBirdGame()
        self.steps = 0
        self.last_score = 0

        # Actions: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)

        # Observation: 4 numbers, each scaled to between -1 and 1
        # [bird height, bird speed, distance to next pipe, height of gap centre]
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.steps = 0
        self.last_score = self.game.score
        return self._get_obs(), {}

    def step(self, action):
        if action == 1:
            self.game.bird.flap()      # the AI chose to flap
        self.game.update()             # advance the game by one frame
        self.steps += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = bool(self.game.game_over)   # bird crashed
        truncated = self.steps >= MAX_STEPS      # time limit
        info = {"score": self.game.score}

        if self.render_mode == "human":
            self.game.render()
        return obs, reward, terminated, truncated, info

    def _next_pipe(self):
        """First pipe that is still in front of (or on top of) the bird."""
        for pipe in self.game.pipes:
            if pipe.x + PIPE_WIDTH > BIRD_X:
                return pipe
        return self.game.pipes[-1]

    def _get_obs(self):
        """What the AI 'sees' this frame."""
        bird = self.game.bird
        pipe = self._next_pipe()
        gap_center = pipe.gap_y + PIPE_GAP / 2

        obs = np.array([
            bird.y / SCREEN_HEIGHT * 2 - 1,        # bird height (-1 top .. 1 bottom)
            bird.velocity / MAX_SPEED,             # bird speed (negative = going up)
            (pipe.x - BIRD_X) / SCREEN_WIDTH,      # distance to next pipe
            gap_center / SCREEN_HEIGHT * 2 - 1,    # height of the gap centre
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def _compute_reward(self):
        """Small reward for living, big reward for a pipe, penalty for crashing."""
        if self.game.game_over:
            return REWARD_CRASH
        reward = REWARD_SURVIVE
        if self.game.score > self.last_score:    # score went up = passed a pipe
            reward += REWARD_PIPE
        self.last_score = self.game.score
        return reward

    def close(self):
        if hasattr(self.game, "close"):
            self.game.close()
