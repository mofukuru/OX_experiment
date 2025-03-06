#%% 必要パッケージのインポート
import time
import random
import torch
import numpy as np
from copy import deepcopy
from qiskit_algorithms.utils import algorithm_globals

#%% seed値の固定
# seed = 42
# algorithm_globals.random_seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

#%% OXゲーム
class TicTacToe:
    """
    reset_board: 盤面の初期化
    get_state: 現在の盤面の取得
    display_board: 盤面の表示
    is_on_board: 盤面上にあるかの判定
    is_valid_action: 有効な手なのかの判断
    place: 駒の配置
    get_possible_actions: とりえる行動の取得
    _is_win: 勝ち判定
    _is_draw: 引き分け判定
    gameover: ゲームの終了
    checkwinner: 勝者の判定
    """
    def __init__(self, board_size=3):
        self.board_size = board_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.reset_board()

    def reset_board(self) -> torch.tensor:
        self.board = torch.zeros([self.board_size,self.board_size],dtype=torch.int32,device=self.device)

    def get_state(self, player: int) -> torch.tensor:
        return deepcopy(self.board)* player

    def display_board(self, sleep_secs=0.8) -> None:
        print("  0 1 2")
        for i, j in enumerate(self.board):
            print(" ".join([str(i)] + ["O" if x == 1 else "X" if x == -1 else " " for x in j]))
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
        num_rd = self.board[0,2]+self.board[1,1]+self.board[2,0]
        num_ld = self.board[0,0]+self.board[1,1]+self.board[2,2]
        if num_rd == 3 or num_ld == 3:
            return 1
        elif num_rd == -3 or num_ld == -3:
            return -1
        return 0

    def _is_draw(self, available_actions: list) -> bool:
        if (len(available_actions) == 0):
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

#%% その他
'''
参考文献
* qiskit machine learning [https://qiskit-community.github.io/qiskit-machine-learning/]
* 自己対戦で強化学習する三目並べ AI をPyTorchで実装 [https://qiita.com/ydclab_P002/items/95a5bdcf1e0d17cf9e1e]
* Othello実装~機械学習基礎~ 授業資料
'''

"""
仕様書的な何か
OXゲームのAIを作成
古典, ハイブリッド, 量子NNの三種
discount, epsilonなどハイパーパラメータの調整
訓練誤差, 勝率, 盤面の表示
I'm really sorry for yammy spaggeti!
"""