import argparse
import os
import random

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

from env.flappy_env import FlappyBirdEnv
from settings import BIRD_SIZE, PIPE_GAP

MODEL_PATH = "models/supervised_flapio.joblib"
DATA_PATH = "data/expert_data.npz"


# --------------------------------------------------------------- the teacher
def expert_action(env, margin=15):

    game = env.game
    pipe = env._next_pipe()
    gap_center = pipe.gap_y + PIPE_GAP / 2
    bird_center = game.bird.y + BIRD_SIZE / 2
    return 1 if (bird_center > gap_center + margin and game.bird.velocity >= 0) else 0


# ------------------------------------------------------------ the AI (model)
class SupervisedAgent:

    def __init__(self, model=None):
        self.model = model

    def predict(self, obs):
        return int(self.model.predict(np.asarray(obs, dtype=np.float32).reshape(1, -1))[0])

    def save(self, path=MODEL_PATH):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path=MODEL_PATH):
        return cls(joblib.load(path))


# ------------------------------------------------------------ data collection
def collect(env, n_samples, policy=None, noise=0.03, max_steps=2000):
    
    X, y, scores = [], [], []
    while len(X) < n_samples:
        obs, _ = env.reset()
        for _ in range(max_steps):
            label = expert_action(env)
            X.append(obs)
            y.append(label)
            if policy is not None:
                action = policy.predict(obs)
            elif random.random() < noise:
                action = random.randint(0, 1)
            else:
                action = label
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated or len(X) >= n_samples:
                break
        scores.append(info["score"])
    return np.array(X, np.float32), np.array(y, np.int64), scores


# ------------------------------------------------------------------ training
def fit(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    flap_idx = np.where(y_tr == 1)[0]
    extra = np.random.choice(flap_idx, size=max(0, (y_tr == 0).sum() - len(flap_idx)))
    X_bal = np.concatenate([X_tr, X_tr[extra]])
    y_bal = np.concatenate([y_tr, y_tr[extra]])

    model = MLPClassifier(hidden_layer_sizes=(32, 32), activation="relu",
                          max_iter=300, early_stopping=True, random_state=0)
    model.fit(X_bal, y_bal)
    pred = model.predict(X_te)
    print(f"   validation accuracy: {accuracy_score(y_te, pred):.3f} | "
          f"flap recall: {recall_score(y_te, pred):.3f}")
    return model


def evaluate(agent, episodes=10, max_steps=3000, render=False):
    env = FlappyBirdEnv(render_mode="human" if render else None)
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        for _ in range(max_steps):
            obs, _, terminated, truncated, info = env.step(agent.predict(obs))
            if terminated or truncated:
                break
        scores.append(info["score"])
        print(f"   episode {ep + 1}: score {info['score']}")
    env.close()
    print(f"average score: {np.mean(scores):.1f} | best: {max(scores)}")
    return scores


def train(samples, dagger_rounds, noise):
    env = FlappyBirdEnv()
    print(f"[1/3] Collecting {samples} teacher samples...")
    X, y, scores = collect(env, samples, noise=noise)
    print(f"      teacher scores (with noise): mean {np.mean(scores):.1f}, flap share {y.mean():.2%}")
    os.makedirs("data", exist_ok=True)
    np.savez(DATA_PATH, X=X, y=y)

    print("[2/3] Training the network...")
    agent = SupervisedAgent(fit(X, y))

    for r in range(dagger_rounds):
        print(f"[DAgger {r + 1}/{dagger_rounds}] model plays, teacher relabels...")
        Xn, yn, sc = collect(env, samples // 2, policy=agent)
        print(f"      model scores this round: mean {np.mean(sc):.1f}")
        X, y = np.concatenate([X, Xn]), np.concatenate([y, yn])
        agent = SupervisedAgent(fit(X, y))

    print("[3/3] Saving model ->", MODEL_PATH)
    agent.save()
    return agent


# ---------------------------------------------------------------------- CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FlapIO supervised-learning agent")
    p.add_argument("command", choices=["train", "evaluate", "play"])
    p.add_argument("--samples", type=int, default=60000)
    p.add_argument("--dagger", type=int, default=3, help="number of DAgger rounds")
    p.add_argument("--noise", type=float, default=0.03, help="random-action rate while collecting")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--model", default=MODEL_PATH)
    args = p.parse_args()

    if args.command == "train":
        train(args.samples, args.dagger, args.noise)
    elif args.command == "evaluate":
        evaluate(SupervisedAgent.load(args.model), args.episodes)
    else:
        evaluate(SupervisedAgent.load(args.model), episodes=5, max_steps=100000, render=True)
