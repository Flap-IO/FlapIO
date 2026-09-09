# FlapIO

![FlapIO banner](assets/banner.jpeg)

> **Status: in development.** This README describes the project as planned — code and results below will be filled in as each part is built.

## About

FlapIO will be a reinforcement learning agent that learns to play Flappy Bird from scratch. We're building a custom Flappy Bird clone in Pygame, wrapping it as a Gymnasium environment, and training a PPO agent (via Stable-Baselines3) against it — no hardcoded rules, no human demonstrations, just reward-driven trial and error. The goal is for the agent to progress from flapping randomly to consistently clearing pipes, purely through training.

## How it will work

1. **The game** (`FlappyBirdGame`) — a Pygame implementation with a `Bird` (gravity + flap) and a stream of `Pipe` obstacles. Will support both a rendered mode (for humans/demos) and a headless mode (fast, no display, for training).
2. **The environment** (`FlappyBirdEnv`) — a `gymnasium.Env` subclass wrapping the game, exposing:
   - **Observation space**: bird's vertical position, bird's velocity, horizontal distance to the next pipe, vertical position of the pipe gap
   - **Action space**: `Discrete(2)` — flap or do nothing
   - **Reward**: small positive reward per frame survived, larger reward for passing a pipe, penalty on collision
3. **The agent** — PPO (Stable-Baselines3), trained against the environment for several hundred thousand timesteps. PPO clips how much the policy can change per update, which keeps training stable even while the reward function is still being tuned.
4. **Evaluation** — checkpoints saved at multiple training stages (e.g. 10k, 100k, 500k steps) to record and compare the agent's progression, not just its final performance.

## Project structure

```
flapio/
├── game/
│   ├── bird.py          # Bird physics
│   ├── pipe.py          # Pipe spawning and movement
│   └── flappy_game.py   # Core game loop, collision, scoring, rendering
├── env/
│   └── flappy_env.py    # Gymnasium wrapper around the game
├── train.py             # Training script (PPO via Stable-Baselines3)
├── evaluate.py          # Runs a trained model with rendering, records demo
├── assets/
│   └── banner.svg
├── logs/                # TensorBoard training logs
├── models/              # Saved model checkpoints
└── README.md
```

## Tech stack

| Component | Library |
|---|---|
| Game engine | Pygame |
| RL environment interface | Gymnasium |
| RL algorithm | PPO (Stable-Baselines3) |
| Training monitoring | TensorBoard |

## Installation

```bash
git clone https://github.com/<your-username>/flapio.git
cd flapio
pip install -r requirements.txt
```

`requirements.txt`:
```
pygame
gymnasium
stable-baselines3[extra]
numpy
tensorboard
```

## Usage (once built)

**Play the game yourself:**
```bash
python game/flappy_game.py
```

**Train the agent:**
```bash
python train.py
```

**Watch training progress live:**
```bash
tensorboard --logdir ./logs
```

**Evaluate a trained model:**
```bash
python evaluate.py --model models/ppo_flapio_final.zip
```

## Results

*Not yet available — will be added once training is complete: final average score, reward curve, checkpoint comparison, training time, hyperparameters used.*

## Team

Built as a 4-person Advanced Programming Lab project:

| Role | Responsibility |
|---|---|
| Game Developer | `FlappyBirdGame`, `Bird`, `Pipe` — core Pygame implementation |
| RL Environment Engineer | `FlappyBirdEnv` — Gymnasium wrapper, observation/action space, reward shaping |
| ML/Training Engineer | PPO training loop, hyperparameter tuning, checkpointing |
| Evaluation & Documentation | Demo recording, reward-curve analysis, report, presentation |

## Future improvements

- Compare PPO against DQN or A2C on the same environment
- Experiment with pixel-based (CNN) observations instead of hand-crafted state features
- Add difficulty scaling (variable pipe gap/speed) to test generalization

## License

MIT
