import torch
from torch import nn, optim
import random
import numpy as np

from src.network_re import (
    CCNN,
    CCNN2,
    CNN_instead_sampler,
    CNN_instead_estimator,
    CQCCNN_Z,
    CQCCNN_ZZ,
    CQCCNN_T,
    CQCCNN_H,
    CQCNN_sampler_ZR,
    CQCNN_sampler_ZZR,
    CQCNN_sampler_TR,
    CQCNN_sampler_HR,
    CQCNN_sampler_ZE,
    CQCNN_sampler_ZZE,
    CQCNN_sampler_TE,
    CQCNN_sampler_HE,
    CQCNN_estimator_ZR,
    CQCNN_estimator_ZZR,
    CQCNN_estimator_TR,
    CQCNN_estimator_HR,
    CQCNN_estimator_ZE,
    CQCNN_estimator_ZZE,
    CQCNN_estimator_TE,
    CQCNN_estimator_HE,
    QCNN_Z,
    QCNN_ZZ,
    QCNN_T,
    QCNN_H,
    # CQCCNN_Z_network,
    # CQCCNN_ZZ_network,
    # CQCCNN_T_network,
    # CQCCNN_H_network,
    CQCNN_sampler_ZR_network,
    CQCNN_sampler_ZZR_network,
    CQCNN_sampler_TR_network,
    CQCNN_sampler_HR_network,
    CQCNN_sampler_ZE_network,
    CQCNN_sampler_ZZE_network,
    CQCNN_sampler_TE_network,
    CQCNN_sampler_HE_network,
    # CQCNN_estimator_ZR_network,
    # CQCNN_estimator_ZZR_network,
    # CQCNN_estimator_TR_network,
    # CQCNN_estimator_HR_network,
    # CQCNN_estimator_ZE_network,
    # CQCNN_estimator_ZZE_network,
    # CQCNN_estimator_TE_network,
    # CQCNN_estimator_HE_network,
    # QCNN_Z_network,
    # QCNN_ZZ_network,
    # QCNN_T_network,
    # QCNN_H_network,
    # QCNN_Z_different_state,
    # QCNN_ZZ_different_state,
    # QCNN_T_different_state,
    # QCNN_H_different_state,
    # QCNN_Z_network_different_state,
    # QCNN_ZZ_network_different_state,
    # QCNN_T_network_different_state,
    # QCNN_H_network_different_state,
)

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

#%% 古典NNを使ったエージェント
class CNNAgent(Agent):
    def __init__(self, player=1, board_size=3, network=None, n_qubits=1):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = 0.1
        self.lr = 0.01
        self.board_size = board_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if network == "CCNN":
            NN = CCNN()
            target_NN = CCNN()
        if network == "CCNN2":
            NN = CCNN2()
            target_NN = CCNN2()
        if network == "CNN_instead_sampler":
            NN = CNN_instead_sampler(n_qubits=n_qubits)
            target_NN = CNN_instead_sampler(n_qubits=n_qubits)
            for param in NN.linear3.parameters():
               param.requires_grad = False
        if network == "CNN_instead_estimator":
            NN = CNN_instead_estimator(n_qubits=n_qubits)
            target_NN = CNN_instead_estimator(n_qubits=n_qubits)
            for param in NN.linear3.parameters():
               param.requires_grad = False
        self.NN = NN
        self.target_NN = target_NN
        self.target_NN.load_state_dict(self.NN.state_dict())
        self.target_NN.eval()
        self.NN.to(self.device)
        self.optimizer = optim.Adam(self.NN.parameters())

    def action(self, tictactoc: classmethod) -> int:
        board = self.check_state(tictactoc)
        possible_actions = self.check_actions(tictactoc)
        if len(possible_actions) == 0:
            return None
        if not self.stop and random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            best_action, _ = self._get_the_best(board, possible_actions)
            return best_action

    def train(self):
        super().train()
        self.NN.train()

    def eval(self):
        super().eval()
        self.NN.eval()

    def get_qvalues(self, state, target=False):
        state = state.to(torch.float32).to(self.device)
        state = state.view(-1)
        qvalues = 0
        if target:
            qvalues = self.target_NN(state).view(self.board_size, self.board_size)
        else:
            qvalues = self.NN(state).view(self.board_size, self.board_size)
        return qvalues

    def _get_the_best(self, board, possible_moves, target=False):
        qvalues = self.get_qvalues(board, target)
        best_move = None
        best_q_value = -float('inf')
        for mv_x, mv_y in possible_moves:
            q_value = qvalues[mv_x, mv_y]
            if q_value > best_q_value:
                best_q_value = q_value
                best_move = (mv_x, mv_y)
        return best_move, best_q_value

    def update(self, tictactoc, state, action, reward, num_of_game):
        if self.stop:
            return None

        with torch.no_grad():
            board = self.check_state(tictactoc)
            moves = self.check_actions(tictactoc)
            _, best_value = self._get_the_best(board, moves, target=True)
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
        if num_of_game % 10 == 0:
            self.target_NN.load_state_dict(self.NN.state_dict())
        return self.loss_v

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

#%% QCNNを使ったエージェント
class QCNNAgent(Agent):
    def __init__(self, player=1, board_size=3, network=None):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = 0.1
        self.board_size = board_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if network == "QCNN_Z":
            NN = QCNN_Z(board_size=self.board_size)
        elif network == "QCNN_ZZ":
            NN = QCNN_ZZ(board_size=self.board_size)
        elif network == "QCNN_T":
            NN = QCNN_T(board_size=self.board_size)
        elif network == "QCNN_H":
            NN = QCNN_H(board_size=self.board_size)
        else:
            print("no networks")
            exit(1)

        self.QCNN = NN
        self.QCNN.to(self.device)
        self.optimizer = optim.Adam(self.QCNN.parameters())

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
        self.QCNN.train()

    def eval(self):
        super().eval()
        self.QCNN.eval()

    def get_qvalues(self, state):
        state = state.to(torch.float32).to(self.device)
        state = state.view(-1)
        qvalues = self.QCNN(state).view(self.board_size, self.board_size)
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

#%%
class CQCAgent_network(Agent):
    def __init__(self, type = 1, player=1, board_size=3, network=None, num_of_qubit=18, featuremap_reps=1, ansatz_reps=1, noised=True):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = 0.1
        self.board_size = board_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.noised = noised
        self.n_qubits = num_of_qubit
        self.type = type

        # if network == "CQCCNN_Z_network":
        #     NN = CQCCNN_Z_network(type=type, board_size=self.board_size, n_qubits=num_of_qubit, noised=noised)
        # elif network == "CQCCNN_ZZ_network":
        #     NN = CQCCNN_ZZ_network(type=type, board_size=self.board_size, n_qubits=num_of_qubit, noised=noised)
        # elif network == "CQCCNN_T_network":
        #     NN = CQCCNN_T_network(type=type, board_size=self.board_size, n_qubits=num_of_qubit, noised=noised)
        # elif network == "CQCCNN_H_network":
        #     NN = CQCCNN_H_network(type=type, board_size=self.board_size, n_qubits=num_of_qubit, noised=noised)
        if network == "CQCNN_sampler_ZR_network":
            NN = CQCNN_sampler_ZR_network(type=type, board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_ZZR_network":
            NN = CQCNN_sampler_ZZR_network(type=type, board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_TR_network":
            NN = CQCNN_sampler_TR_network(type=type, board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_HR_network":
            NN = CQCNN_sampler_HR_network(type=type, board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_ZE_network":
            NN = CQCNN_sampler_ZE_network(type=type, board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_ZZE_network":
            NN = CQCNN_sampler_ZZE_network(type=type, board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_TE_network":
            NN = CQCNN_sampler_TE_network(type=type, board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        elif network == "CQCNN_sampler_HE_network":
            NN = CQCNN_sampler_HE_network(type=type, board_size=self.board_size, num_qubits=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps)
        # elif network == "CQCNN_estimator_ZR_network":
        #     NN = CQCNN_estimator_ZR_network(type=type, board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps, noised=noised)
        # elif network == "CQCNN_estimator_ZZR_network":
        #     NN = CQCNN_estimator_ZZR_network(type=type, board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps, noised=noised)
        # elif network == "CQCNN_estimator_TR_network":
        #     NN = CQCNN_estimator_TR_network(type=type, board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps, noised=noised)
        # elif network == "CQCNN_estimator_HR_network":
        #     NN = CQCNN_estimator_HR_network(type=type, board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps, noised=noised)
        # elif network == "CQCNN_estimator_ZE_network":
        #     NN = CQCNN_estimator_ZE_network(type=type, board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps, noised=noised)
        # elif network == "CQCNN_estimator_ZZE_network":
        #     NN = CQCNN_estimator_ZZE_network(type=type, board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps, noised=noised)
        # elif network == "CQCNN_estimator_TE_network":
        #     NN = CQCNN_estimator_TE_network(type=type, board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps, noised=noised)
        # elif network == "CQCNN_estimator_HE_network":
        #     NN = CQCNN_estimator_HE_network(type=type, board_size=self.board_size, num_of_qubit=num_of_qubit, featuremap_reps=featuremap_reps, ansatz_reps=ansatz_reps, noised=noised)
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
        
        with torch.no_grad():
            alpha = 0.1
            iter = 0
            if self.type == 1:
                iter = self.n_qubits*2*2
            elif self.type == 2:
                iter = self.n_qubits*2
# TODO change the following adding noise code.
            for i in range(iter):
                if self.noised and random.random() < alpha:
                    self.optimizer.param_groups[0]["params"][2][i] = torch.normal(mean=2*np.pi, std=1, size=(1,1))
                else:
                    self.optimizer.param_groups[0]["params"][2][i] = torch.Tensor([2*np.pi])

        return self.loss_v

#%%
class QCNNAgent_network(Agent):
    def __init__(self, type=1, player=1, board_size=3, network=None, noised=True):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = 0.1
        self.board_size = board_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if network == "QCNN_Z_network":
            NN = QCNN_Z_network(type=type, board_size=self.board_size, noised=noised)
        elif network == "QCNN_ZZ_network":
            NN = QCNN_ZZ_network(type=type, board_size=self.board_size, noised=noised)
        elif network == "QCNN_T_network":
            NN = QCNN_T_network(type=type, board_size=self.board_size, noised=noised)
        elif network == "QCNN_H_network":
            NN = QCNN_H_network(type=type, board_size=self.board_size, noised=noised)
        else:
            print("no networks")
            exit(1)

        self.QCNN = NN
        self.QCNN.to(self.device)
        self.optimizer = optim.Adam(self.QCNN.parameters())

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
        self.QCNN.train()

    def eval(self):
        super().eval()
        self.QCNN.eval()

    def get_qvalues(self, state):
        state = state.to(torch.float32).to(self.device)
        state = state.view(-1)
        qvalues = self.QCNN(state).view(self.board_size, self.board_size)
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

#%% QCNNを使ったエージェント stateを2つ指定できるようになる
class QCNNAgent_different_state(Agent):
    def __init__(self, type=1, player=1, board_size=3, network=None, state_weight=1, state_weight2=1):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = 0.1
        self.board_size = board_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if network == "QCNN_Z":
            NN = QCNN_Z_different_state(board_size=self.board_size, state_weight=state_weight, state_weight2=state_weight2)
        elif network == "QCNN_ZZ":
            NN = QCNN_ZZ_different_state(board_size=self.board_size, state_weight=state_weight, state_weight2=state_weight2)
        elif network == "QCNN_T":
            NN = QCNN_T_different_state(board_size=self.board_size, state_weight=state_weight, state_weight2=state_weight2)
        elif network == "QCNN_H":
            NN = QCNN_H_different_state(board_size=self.board_size, state_weight=state_weight, state_weight2=state_weight2)
        elif network == "QCNN_Z_network":
            NN = QCNN_Z_network_different_state(type=type, board_size=self.board_size, state_weight=state_weight, state_weight2=state_weight2)
        elif network == "QCNN_ZZ_network":
            NN = QCNN_ZZ_network_different_state(type=type, board_size=self.board_size, state_weight=state_weight, state_weight2=state_weight2)
        elif network == "QCNN_T_network":
            NN = QCNN_T_network_different_state(type=type, board_size=self.board_size, state_weight=state_weight, state_weight2=state_weight2)
        elif network == "QCNN_H_network":
            NN = QCNN_H_network_different_state(type=type, board_size=self.board_size, state_weight=state_weight, state_weight2=state_weight2)
        else:
            print("no networks")
            exit(1)

        self.QCNN = NN
        self.QCNN.to(self.device)
        self.optimizer = optim.Adam(self.QCNN.parameters())

        self.old_board = None
        self.old_board2 = None

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
        self.QCNN.train()

    def eval(self):
        super().eval()
        self.QCNN.eval()

    def get_qvalues(self, state, state2):
        state = state.to(torch.float32).to(self.device)
        state2 = state2.to(torch.float32).to(self.device)
        state = state.view(-1)
        state2 = state2.view(-1)
        qvalues = self.QCNN(state, state2).view(self.board_size, self.board_size)
        return qvalues

    def _get_the_best(self, board, possible_moves):
        if self.old_board is None:
            self.old_board = board
        if self.old_board2 is None:
            self.old_board2 = board
        qvalues = self.get_qvalues(board, self.old_board)
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
        old_qvalue = self.get_qvalues(state, self.old_board2)[x, y]
        self.old_board2 = -self.old_board
        self.old_board = -board
        loss = nn.functional.huber_loss(old_qvalue, target_q)
        self.loss_v = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self.loss_v