import torch
from torch import nn, optim
import random
import numpy as np

from src_no_seed.network_for_real_device import CQCNN_sampler_HE

#%% seed値の固定
# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

#%% エージェントの実装
class Agent:
    """
    train: stopをFalseにすることによって訓練をさせる
    eval: stopをTrueにすることによって訓練を止める
    check_state: 現在の盤面の取得
    check_actions: とりえる行動の取得
    """
    def __init__(self, player: int):
        self.player = player
        self.stop = False

    def train(self) -> bool:
        self.stop = False

    def eval(self) -> bool:
        self.stop = True

    def check_state(self, tictactoc: classmethod) -> torch.tensor:
        return tictactoc.get_state(self.player)

    def check_actions(self, tictactoc: classmethod) -> list:
        return tictactoc.get_possible_actions()

#%% 古典量子ハイブリッドNNを使ったエージェント
class CQCAgent(Agent):
    def __init__(self, player=1, board_size=3, network=None, num_of_qubit=18, featuremap_reps=1, ansatz_reps=1):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = 0.1
        self.board_size = board_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if network == "CQCCNN_Z":
            NN = CQCCNN_Z(board_size=self.board_size, n_qubits=num_of_qubit)
        elif network == "CQCCNN_ZZ":
            NN = CQCCNN_ZZ(board_size=self.board_size, n_qubits=num_of_qubit)
        elif network == "CQCCNN_T":
            NN = CQCCNN_T(board_size=self.board_size, n_qubits=num_of_qubit)
        elif network == "CQCCNN_H":
            NN = CQCCNN_H(board_size=self.board_size, n_qubits=num_of_qubit)
        elif network == "CQCNN_sampler_ZR":
            NN = CQCNN_sampler_ZR(board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_ZZR":
            NN = CQCNN_sampler_ZZR(board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_TR":
            NN = CQCNN_sampler_TR(board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_HR":
            NN = CQCNN_sampler_HR(board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_ZE":
            NN = CQCNN_sampler_ZE(board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_ZZE":
            NN = CQCNN_sampler_ZZE(board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_TE":
            NN = CQCNN_sampler_TE(board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_HE":
            NN = CQCNN_sampler_HE(board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_estimator_ZR":
            NN = CQCNN_estimator_ZR(board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_estimator_ZZR":
            NN = CQCNN_estimator_ZZR(board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_estimator_TR":
            NN = CQCNN_estimator_TR(board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_estimator_HR":
            NN = CQCNN_estimator_HR(board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_estimator_ZE":
            NN = CQCNN_estimator_ZE(board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_estimator_ZZE":
            NN = CQCNN_estimator_ZZE(board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_estimator_TE":
            NN = CQCNN_estimator_TE(board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_estimator_HE":
            NN = CQCNN_estimator_HE(board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        else:
            print("no networks")
            exit(1)

        self.HNN = NN
        self.HNN.to(self.device)
        self.optimizer = optim.Adam(self.HNN.parameters())

    def action(self, tictactoc: classmethod) -> int:
        board = self.check_state(tictactoc)
        possible_actions = self.check_actions(tictactoc)
        if len(possible_actions) == 0:
            return None
        # epsilon-greedy法
        if not self.stop and random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            best_action, _ = self._get_the_best(board, possible_actions)
            return best_action

    def train(self):
        super().train()
        self.HNN.train()

    def eval(self):
        super().eval()
        self.HNN.eval()

    def get_qvalues(self, state):
        state = state.to(torch.float32).to(self.device)
        state = state.view(-1)
        qvalues = self.HNN(state).view(self.board_size, self.board_size)
        return qvalues

    def _get_the_best(self, board, possible_moves):
        qvalues = self.get_qvalues(board)
        best_move = None
        best_q_value = -float('inf')
        for mv_x, mv_y in possible_moves:
            q_value = qvalues[mv_x, mv_y]
            if q_value > best_q_value:
                best_q_value = q_value
                best_move = (mv_x, mv_y)
        return best_move, best_q_value

    def update(self, tictactoc, state, action, reward):
        if self.stop:
            return None

        with torch.no_grad():
            board = self.check_state(tictactoc)
            moves = self.check_actions(tictactoc)
            _, best_value = self._get_the_best(board, moves)
            next_max = max(0, best_value)
            target_q = torch.tensor(reward + self.discount * next_max)
            target_q = target_q.to(torch.float32).to(self.device)

        x, y = action
        old_qvalue = self.get_qvalues(state)[x, y]
        loss = nn.functional.huber_loss(old_qvalue, target_q)
        self.loss_v = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self.loss_v

#%% 人間
class Human(Agent):
    """
    action: 入力から行動へ
    """

    def __init__(self, player=1):
        super().__init__(player)
        self.stop = True

    def action(self, tictactoc: classmethod) -> None:
        moves = self.check_actions(tictactoc)

        while True:
            try:
                row, col = map(int, input("please input: {row col} ").split())
                if (row, col) in moves:
                    return row, col
                else:
                    print("Invalid input!")
            except ValueError:
                print("Invalid input!!")

#%% ランダムポリシー
class RandomPolicy(Agent):
    """
    action: とりえる行動からランダムに取得
    """

    def __init__(self, player=1):
        super().__init__(player)
        self.stop = True

    def action(self, tictactoc: classmethod) -> int:
        moves = self.check_actions(tictactoc)

        row, col = random.choice(moves)
        return row, col

