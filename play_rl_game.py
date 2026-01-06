# play_rl_game.py
import os
from Game import Game
from QLearningAgent import QLearningAgent

QTABLE_PATH = "trained_q_table.pkl"

# Schnell trainieren:
RENDER_TRAINING = False
TRAIN_EPISODES = 12000

AUTOSAVE_EVERY = 500
PRINT_EVERY = 100


def train(episodes=TRAIN_EPISODES, render_training=RENDER_TRAINING):
    game = Game()
    agent = QLearningAgent()

    # Fenster kann offen bleiben, aber ohne render ist es turbo
    game.render_enabled = render_training

    last100_cleared = []
    last100_reward = []

    for ep in range(1, episodes + 1):
        # Curriculum: erste 2000 Episoden "easy" (fixe Hindernisse)
        game.training_easy = (ep <= 2000)

        state = game.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = game.step(action)

            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if render_training:
                game.render(extra_text=f"TRAIN ep={ep}/{episodes} eps={agent.epsilon:.3f} easy={game.training_easy}")
                game.clock.tick(120)
            else:
                # 0 = kein FPS Limit -> maximal schnell
                game.clock.tick(0)

        # Stats
        last100_cleared.append(info.get("cleared", 0))
        last100_reward.append(total_reward)
        if len(last100_cleared) > 100:
            last100_cleared.pop(0)
            last100_reward.pop(0)

        agent.decay_epsilon()

        if ep % PRINT_EVERY == 0:
            avg_c = sum(last100_cleared) / len(last100_cleared)
            avg_r = sum(last100_reward) / len(last100_reward)
            print(f"Episode {ep}/{episodes} | avg_cleared(last100)={avg_c:.2f} | avg_reward(last100)={avg_r:.2f} | eps={agent.epsilon:.3f} | easy={game.training_easy}")

        if ep % AUTOSAVE_EVERY == 0:
            agent.save(QTABLE_PATH)
            print("Zwischenspeicher gespeichert!")

    agent.save(QTABLE_PATH)
    print(f"\n✅ Training fertig. Gespeichert als: {QTABLE_PATH}")
    print("Q-table size:", len(agent.q_table), "\n")


def play():
    if not os.path.exists(QTABLE_PATH):
        print("⚠️ Keine trained_q_table.pkl gefunden. Trainiere zuerst (t).")
        return

    game = Game()
    agent = QLearningAgent(epsilon=0.0)  # nur exploit
    agent.load(QTABLE_PATH)

    print("Loaded Q-table size:", len(agent.q_table))

    game.render_enabled = True
    game.training_easy = False

    while True:
        state = game.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            state, reward, done, info = game.step(action)
            game.render(extra_text="PLAY (trained)")
            game.clock.tick(60)


if __name__ == "__main__":
    mode = input("Modus wählen: (t)rain oder (p)lay ? ").strip().lower()
    if mode.startswith("t"):
        train()
    else:
        play()
