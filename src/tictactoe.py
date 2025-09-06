import random
import time
from copy import deepcopy
from typing import Final

import numpy as np
import torch
from qiskit_algorithms.utils import algorithm_globals

SEED: Final[int] = 42


def set_seed(seed: int = 42):
    """
    Setting seed
    """
    algorithm_globals.random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# If you do not want to set seed, you can comment out the below set_seed() function.
# set_seed(SEED)


class TicTacToe:
    """
    reset_board: init board
    get_state: get the current board
    display_board: display board
    is_on_board: confirm to be on board
    is_valid_action: confirm to be valid action
    place: place O or X
    get_possible_actions: get the possible action
    _is_win: judge winning
    _is_draw: judge draw
    gameover: finish game
    checkwinner: judge who wins
    """

    def __init__(self, board_size=3):
        self.board_size = board_size
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.reset_board()

    def reset_board(self) -> None:
        self.board = torch.zeros(
            [self.board_size, self.board_size],
            dtype=torch.int32,
            device=self.device,
        )
        self.board_history = []

    def get_state(self, player: int) -> torch.tensor:
        return deepcopy(self.board) * player

    def display_board(self, sleep_secs=0.8) -> None:
        print("  0 1 2")
        for i, j in enumerate(self.board):
            print(
                " ".join(
                    [str(i)]
                    + ["O" if x == 1 else "X" if x == -1 else " " for x in j]
                )
            )
        print()
        time.sleep(sleep_secs)

    def is_on_board(self, row: int, col: int) -> bool:
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def is_valid_action(self, action: list, player: int) -> bool:
        if action is None:
            return True

        row, col = action
        if self.is_on_board(row, col) and self.board[row, col] == 0:
            return True
        return False

    def place(self, action: list, player: int) -> bool:
        if action is None:
            return True

        if not self.is_valid_action(action, player):
            print("Invalid action")
            return False

        row, col = action
        self.board[row, col] = player
        self.board_history.append(action)
        return True

    def get_possible_actions(self) -> list:
        available_actions = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == 0:
                    available_actions.append((row, col))
        return available_actions

    def _is_win(self) -> int:
        num_r = torch.sum(self.board, axis=0)
        num_c = torch.sum(self.board, axis=1)
        for i in range(3):
            if num_r[i] == 3 or num_c[i] == 3:
                return 1
            elif num_r[i] == -3 or num_c[i] == -3:
                return -1
        num_rd = self.board[0, 2] + self.board[1, 1] + self.board[2, 0]
        num_ld = self.board[0, 0] + self.board[1, 1] + self.board[2, 2]
        if num_rd == 3 or num_ld == 3:
            return 1
        elif num_rd == -3 or num_ld == -3:
            return -1
        return 0

    def _is_draw(self, available_actions: list) -> bool:
        if len(available_actions) == 0:
            return True
        else:
            return False

    def gameover(self) -> bool:
        if self._is_draw(self.get_possible_actions()) or self._is_win() != 0:
            return True
        else:
            return False

    def checkwinner(self) -> int:
        if self._is_draw(self.get_possible_actions()):
            return 0
        else:
            return self._is_win()

    def get_board_history(self) -> list:
        return self.board_history


if __name__ == "__main__":
    tictactoe = TicTacToe()
    tictactoe.display_board()
    tictactoe.place([0, 0], 1)
    tictactoe.display_board()
    tictactoe.place([1, 1], -1)
    tictactoe.display_board()
    tictactoe.place([0, 1], 1)
    tictactoe.display_board()
    tictactoe.place([2, 1], -1)
    tictactoe.display_board()
    tictactoe.place([0, 2], 1)
    tictactoe.display_board()
    print(tictactoe.checkwinner())
