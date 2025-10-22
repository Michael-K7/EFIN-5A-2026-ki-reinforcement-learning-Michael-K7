import random
import copy
import numpy as np
from collections import defaultdict
from Game import Game

class QLearningAgent:
    def __init__(self, game, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.game = game
        self.q_table = defaultdict(lambda: np.zeros(9))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state(self, board):
        # Übersetzt das Board in ein Tupel von Strings für das Hashing im Q-Table
        return tuple(tuple(row) for row in board)

    def choose_action(self):
        state = self.get_state(self.game.board)
        if random.uniform(0, 1) < self.epsilon:
            # Explore: Wähle eine zufällige gültige Aktion
            return random.choice(self.get_valid_actions(self.game.board))
        else:
            # Exploit: die beste bekannte Aktion wählen
            q_values = self.q_table[state]
            valid_actions = self.get_valid_actions(self.game.board)
            max_q_value = max([q_values[a] for a in valid_actions])
            best_actions = [a for a in valid_actions if q_values[a] == max_q_value]
            return random.choice(best_actions)

    def update_q_table(self, board, action, reward, next_board):
        state = self.get_state(board)
        next_state = self.get_state(next_board)
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])

    def get_valid_actions(self, board):
        actions = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    actions.append(i * 3 + j)
        return actions

    def reset(self):
        self.q_table = defaultdict(lambda: np.zeros(9))


    def train_agent(self, episodes=100):
        bot_wins = []
        opponent_wins = []
        draws = []
        
        bot_won = 0
        opponent_won = 0
        draw = 0

        for _ in range(episodes):
            self.game = Game()
            done = False

            while not done:
                # Zug des Gegners
                opponent_action = random.choice(self.get_valid_actions(self.game.board))
                self.game.board[opponent_action // 3][opponent_action % 3] = "X"
                if self.game.check_winner("X"):
                    self.update_q_table(board, action, -1, next_board)
                    done = True
                    opponent_won += 1
                elif self.game.is_board_full():
                    self.update_q_table(board, action, 0, next_board)
                    done = True
                    draw += 1
                else:
                    # Bot's Zug
                    action = self.choose_action()
                    row, col = divmod(action, 3)
                    if self.game.is_valid_move(row, col):
                        board = copy.deepcopy(self.game.board)
                        self.game.board[row][col] = "O"
                        next_board = self.game.board
                        if self.game.check_winner("O"):
                            self.update_q_table(board, action, 10, next_board)
                            done = True
                            bot_won += 1
                        elif self.game.is_board_full():
                            self.update_q_table(board, action, 0, next_board)
                            draw += 1
                            done = True
            
            # Sammeln der Statistik für diese Episode
            bot_wins.append(bot_won)
            opponent_wins.append(opponent_won)
            draws.append(draw)

        return bot_wins, opponent_wins, draws