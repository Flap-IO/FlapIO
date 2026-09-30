# FlapIO 🐦

[![Status](https://img.shields.io/badge/status-in%20development-yellow)](#roadmap)
[![License: Apache--2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

![FlapIO banner](Banner.jpeg)

> An AI project that learns to play Flappy Bird on a custom Pygame clone. The main agent is a **PPO reinforcement-learning agent** (Stable-Baselines3) that learns from scratch through reward-driven trial and error. A **supervised-learning (behaviour-cloning) agent** is included as a baseline to compare against.

---

## Table of contents

- [About](#about)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [The PPO agent](#the-ppo-agent)
- [The supervised-learning agent](#the-supervised-learning-agent)
- [Results](#results)
- [Team](#team)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## About

FlapIO is a 3-person Advanced Programming Lab project. The goal is to watch an agent go from doing nothing useful to consistently clearing pipes, and to compare two very different ways of getting there:

1. **Reinforcement learning (PPO)** learns from the reward signal only, with zero hand-coded flying logic.
2. **Supervised learning (behaviour cloning)** copies a rule-based teacher from recorded examples.

The project has four moving parts, built by three team members (see [Team](#team)):

1. A **game** the agent can play against
2. An **environment** that exposes that game through the standard Gymnasium interface
3. An **RL agent** (PPO) that learns a policy against that environment
4. A **supervised agent** trained on teacher demonstrations (baseline)

## How it works

1. **The game** (`FlappyBirdGame`) is a Pygame implementation with a `Bird` (gravity + flap) and a stream of `Pipe` obstacles. It runs **headless** (no window) for fast training and opens a window only when `render()` is called.
2. **The environment** (`FlappyBirdEnv`) is a `gymnasium.Env` wrapper around the game:
   - **Observation space**: 4 values scaled to `[-1, 1]`: bird height, bird velocity, distance to next pipe, height of the gap centre
   - **Action space**: `Discrete(2)`: do nothing or flap
   - **Reward**: `+0.1` per frame survived, `+5` for passing a pipe, `-10` on crash
3. **The PPO agent** is trained against the environment for several hundred thousand timesteps. PPO clips how much the policy can change per update, which keeps training stable.
4. **The supervised agent** learns `observation -> teacher action` with a small neural network, refined with DAgger rounds.
5. **Evaluation**: checkpoints saved at multiple training stages (10k, 100k, 500k steps) to compare progression.

## Project structure

```
FlapIO/
├── game/
│   ├── __init__.py
│   ├── bird.py              Bird physics (gravity, flap, collision rect)
│   ├── pipe.py              Pipe spawning, movement, collision
│   └── flappy_game.py       Core loop: scoring, game over, headless mode, rendering
├── env/
│   ├── __init__.py
│   └── flappy_env.py        Gymnasium wrapper around the game
├── agent/                   PPO agent (config, policy, agent, dummy env)
├── settings.py              All game constants in one place
├── play.py                  Play the game yourself
├── supervised_learning.py   Behaviour-cloning agent (collect / train / evaluate / play)
├── train.py                 PPO training script
├── predict.py               obs -> action from a saved PPO checkpoint
├── evaluate.py              🚧 render + record a trained agent
├── requirements.txt
├── models/                  generated (git-ignored)
├── data/                    generated (git-ignored)
├── logs/                    generated (git-ignored, TensorBoard)
└── README.md
```

## Tech stack

| Component                | Library                 |
| ------------------------ | ----------------------- |
| Game engine              | Pygame                  |
| RL environment interface | Gymnasium               |
| RL algorithm             | PPO (Stable-Baselines3) |
| Neural nets (RL)         | PyTorch                 |
| Supervised baseline      | scikit-learn            |
| Training monitoring      | TensorBoard             |

## Installation

```bash
git clone https://github.com/Flap-IO/FlapIO.git
cd FlapIO
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Always run commands from the repository root, since the imports (`game.`, `env.`, `settings`) depend on it.

## Usage

**Play the game yourself** (SPACE / click = flap, R = restart, ESC = quit):

```bash
python play.py
```

**Train the PPO agent:**

```bash
python train.py --timesteps 1000000
python train.py --dummy-env --timesteps 20000   # quick smoke test
```

**Watch training progress:**

```bash
tensorboard --logdir ./logs
```

**Supervised-learning agent:**

```bash
python supervised_learning.py train --samples 60000 --dagger 3
python supervised_learning.py evaluate --episodes 10
python supervised_learning.py play              # watch the AI in a window
```

**Get an action from a trained PPO model:**

```bash
python predict.py --model models/ppo_flapio_final.zip
```

## The PPO agent

The `agent/` package, `train.py` and `predict.py` are self-contained.

| File | Role |
|---|---|
| `agent/config.py` | `PPOConfig`: learning rate, `n_steps`, `batch_size`, `gamma`, `gae_lambda`, `clip_range`, network sizes, checkpoint/eval frequency. Change hyperparameters here only. |
| `agent/policy.py` | `FlapIOFeaturesExtractor`: small MLP that embeds the 4-value state before SB3's actor/critic heads. |
| `agent/agent.py` | `FlapIOAgent` wrapping SB3's `PPO`: `predict(obs)`, `train(total_timesteps)`, `save(path)`, `load(path)`. |
| `agent/dummy_env.py` | Mock env with the same observation/action space, for testing without the game. |
| `train.py` | Vectorized env, PPO training, checkpoints, periodic evaluation, final save. |
| `predict.py` | Loads a checkpoint and turns one observation into one action. |

## The supervised-learning agent

`supervised_learning.py` is a **behaviour-cloning** baseline:

1. A rule-based **teacher** plays the game. Each frame we record `(observation, teacher action)`. A little random noise (3%) is injected so the data also contains recovery situations.
2. An **MLP classifier** (scikit-learn, two hidden layers of 32) learns to predict the teacher's action. Flap frames are rare, so they are oversampled.
3. **DAgger rounds**: the trained model plays, the teacher relabels the states the model visited, and the model is retrained on the combined data. This fixes drift into states the teacher never showed.

It exposes the same style of API as the PPO agent: `predict(obs)`, `save()`, `load()`.

## Results

<!-- TODO: fill in once PPO training against the real env is done -->
*Not yet available.* This section will report: final average score, the reward curve, a checkpoint-by-checkpoint comparison (10k / 100k / 500k steps), PPO vs supervised baseline, total training time, and the hyperparameters used.

## Team

Built as a 3-person Advanced Programming Lab project:

| Role                             | Responsibility                                                                                          | Status |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- | ------ |
| Game Developer                   | `FlappyBirdGame`, `Bird`, `Pipe`: core Pygame implementation                                            | ✅ done |
| RL Environment & ML/Training Engineer | `FlappyBirdEnv` (Gymnasium wrapper, observation/action space, reward shaping) and the PPO agent (training loop, policy network, hyperparameter tuning, checkpointing) | ✅ done |
| Evaluation & Documentation       | Demo recording, reward-curve analysis, report, presentation                                             | 🚧 pending |

## Roadmap

- [x] `Bird` class (gravity, flap impulse, collision rect)
- [x] `FlappyBirdGame`: `Bird` + `Pipe` in one loop, headless mode, scoring
- [x] `FlappyBirdEnv`: Gymnasium wrapper, reward shaping
- [x] Supervised-learning (behaviour cloning) baseline
- [ ] Swap `agent/dummy_env.py` for the real env in `train.py`
- [ ] Full PPO training run + checkpoint comparison
- [ ] `evaluate.py`: render + record a trained agent playing
- [ ] Compare PPO vs supervised baseline (and optionally DQN / A2C)
- [ ] Experiment with pixel-based (CNN) observations
- [ ] Add difficulty scaling (variable pipe gap/speed)

## Contributing

This is a lab project with a fixed 3-person team; not currently accepting outside PRs. Feel free to open an issue if you spot a bug.

## License

Apache-2.0. See [LICENSE](LICENSE).
