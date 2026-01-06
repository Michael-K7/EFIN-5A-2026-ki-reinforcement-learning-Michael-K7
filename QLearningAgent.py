# QLearningAgent.py
import random
import pickle
import numpy as np
from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        alpha=0.20,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.10,
        epsilon_decay=0.9997
    ):
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.n_actions = 2  # 0=nichts, 1=springen
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        q = self.q_table[state]
        max_q = float(np.max(q))
        best = [i for i, v in enumerate(q) if float(v) == max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state, done):
        q_sa = float(self.q_table[state][action])
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.q_table[next_state]))
        self.q_table[state][action] = q_sa + self.alpha * (target - q_sa)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="trained_q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, path="trained_q_table.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32), data)
