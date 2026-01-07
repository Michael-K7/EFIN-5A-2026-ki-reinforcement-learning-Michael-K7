# QLearningAgent.py
import random
import pickle


class QLearningAgent:
    """
    Vereinfachter Q-Learning Agent (leicht zu erklären):

    Q(s,a) = "wie gut ist Aktion a im Zustand s?"
    Update (nach einem Schritt):
        Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))

    - alpha: Lernrate (wie stark neue Erfahrung zählt)
    - gamma: Discount (wie wichtig Zukunft ist)
    - epsilon: Exploration (wie oft zufällig probiert wird)
    """

    def __init__(self, alpha=0.20, gamma=0.99, epsilon=1.0, epsilon_min=0.10, epsilon_decay=0.9997):
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.n_actions = 2  # 0 = nichts, 1 = springen

        # q_table: dict mit Key = state, Value = Liste mit Q-Werten pro Aktion
        # Beispiel: q_table[state] = [Q(state,0), Q(state,1)]
        self.q_table = {}

    def _ensure_state_exists(self, state):
        """Falls ein Zustand noch nie gesehen wurde: Q-Werte mit 0 initialisieren."""
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(self.n_actions)]

    def choose_action(self, state):
        """
        Epsilon-greedy:
        - Mit Wahrscheinlichkeit epsilon: zufällige Aktion (Exploration)
        - Sonst: beste bekannte Aktion (Exploitation)
        """
        self._ensure_state_exists(state)

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        q_values = self.q_table[state]
        best_q = max(q_values)

        # Bei Gleichstand zufällig unter den besten wählen
        best_actions = [a for a, q in enumerate(q_values) if q == best_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning Update:
        target = reward + gamma * max_a' Q(next_state, a')   (wenn nicht done)
        Q(state, action) <- Q + alpha * (target - Q)
        """
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * max(self.q_table[next_state])

        # Temporal Difference (TD) Learning
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Epsilon langsam reduzieren, aber nicht unter epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="trained_q_table.pkl"):
        """Q-Tabelle speichern (damit play() sie später laden kann)."""
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path="trained_q_table.pkl"):
        """Q-Tabelle laden."""
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)

#Zustand s = was der Agent “sieht” (bei dir: Distanz-Bin, Höhe, etc.)
# Aktion a = springen oder nichts
# Reward = gut wenn Hindernis geschafft, schlecht bei Crash
# Q(s,a) = “wie gut ist diese Aktion in diesem Zustand”
# Update-Formel: aktuelles Q wird Richtung “reward + Zukunft” verschoben (mit alpha und gamma)
# epsilon-greedy = manchmal zufällig probieren, sonst bestes nehmen