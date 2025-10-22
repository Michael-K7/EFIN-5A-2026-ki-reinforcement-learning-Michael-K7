from Game import Game
import pickle
from QLearningAgent import QLearningAgent
from collections import defaultdict
import numpy as np

# neues Spiel ohne KI
boringGame = Game()
boringGame.play_game()


# neues Spiel mit KI
coolGame = Game()
agent = QLearningAgent(coolGame)
# die gelernten Parameter (Q-Table) laden und dem agenten übergeben
with open("trained_q_table.pkl", "rb") as f:
    q_table_data = pickle.load(f)
    # Sicherstellen, dass q_table nach dem Laden vom Typ defaultdict ist
    agent.q_table = defaultdict(lambda: np.zeros(9), q_table_data)
coolGame.set_agent(agent)
coolGame.play_game()