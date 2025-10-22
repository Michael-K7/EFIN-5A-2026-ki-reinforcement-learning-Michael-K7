import random

class Game:
    def __init__(self):
        self.agent = None
        self.board = [[" " for _ in range(3)] for _ in range(3)]

    def is_valid_move(self, row, col):
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == " "

    def check_winner(self, player):
        for i in range(3):
            if all([self.board[i][j] == player for j in range(3)]) or all([self.board[j][i] == player for j in range(3)]):
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == player or self.board[0][2] == self.board[1][1] == self.board[2][0] == player:
            return True
        return False

    def is_board_full(self):
        return all(self.board[i][j] != " " for i in range(3) for j in range(3))

    def print_board(self):
        print("  0   1   2")
        for i, row in enumerate(self.board):
            print(i, " | ".join(row))
            if i < 2:
                print("  ---+---+---")

    def computer_move(self):
        if self.agent:
            action = self.agent.choose_action()
            row, col = divmod(action, 3)
            self.board[row][col] = "O"         
        else:
            while True:
                row, col = random.randint(0, 2), random.randint(0, 2)
                if self.is_valid_move(row, col):
                    self.board[row][col] = "O"
                    break

    def set_agent(self, agent):
        self.agent = agent

    def play_game(self):
        self.board = [[" " for _ in range(3)] for _ in range(3)]
        print("Spiel beginnt")
        self.print_board()

        while True:
            try:
                move = input("Dein Zug (gib Zeile und Spalte als [Zeile, Spalte] an): ")
                row, col = eval(move)
                if not self.is_valid_move(row, col):
                    print("Ungültiger Zug, versuch es nochmal.")
                    continue
                self.board[row][col] = "X"
            except (ValueError, SyntaxError):
                print("Ungültiges Format. Verwende [Zeile, Spalte].")
                continue

            if self.check_winner("X"):
                print("Herzlichen Glückwunsch! Du hast gewonnen!")
                break
            if all(self.board[i][j] != " " for i in range(3) for j in range(3)):
                print("Es ist ein Unentschieden!")
                break

            print("Computerzug:")
            self.computer_move()

            self.print_board()

            if self.check_winner("O"):
                print("Der Computer hat gewonnen. Versuch es nochmal!")
                break

            if all(self.board[i][j] != " " for i in range(3) for j in range(3)):
                print("Es ist ein Unentschieden!")
                break
