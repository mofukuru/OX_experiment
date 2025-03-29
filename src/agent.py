import torch
from typing import Final
from torch import nn, optim
import random
import numpy as np

from src.network import (
    CCNN,
    CCNN2,
    CNN_instead_sampler,
    CNN_instead_estimator,
    CNN_QCNN_CNN,
    CNN_QNN_CNN_sampler,
    CNN_QNN_CNN_estimator,
    QCNN,
    QNN,
    CNN_QCNN_CNN_for_network,
    CNN_QNN_CNN_sampler_for_network,
    CNN_QNN_CNN_estimator_for_network
)

SEED: Final[int] = 42

def set_seed(seed: int=42):
    """
    Setting seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# If you do not want to set seed, you can comment out the below set_seed() function.
set_seed(SEED)

class Agent:
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

class Human(Agent):
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

class RandomPolicy(Agent):
    def __init__(self, player=1):
        super().__init__(player)
        self.stop = True

    def action(self, tictactoc: classmethod) -> int:
        moves = self.check_actions(tictactoc)

        row, col = random.choice(moves)
        return row, col

class CNNAgent(Agent):
    def __init__(self, player=1, board_size=3, network=None, n_qubits=7):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = 0.1
        self.lr = 0.01
        self.alpha = 0.01
        self.board_size = board_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if network == "CCNN":
            NN = CCNN()
        if network == "CCNN2":
            NN = CCNN2()
        if network == "CNN_instead_sampler":
            NN = CNN_instead_sampler(n_qubits=n_qubits)
        if network == "CNN_instead_estimator":
            NN = CNN_instead_estimator(n_qubits=n_qubits)
        self.NN = NN
        self.NN.to(self.device)
        self.optimizer = optim.Adam(self.NN.parameters())

        if network == "CNN_instead_sampler" or "CNN_instead_estimator":
            self.optimizer.param_groups[0]["params"][2].requires_grad=False

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

    def get_qvalues(self, state):
        state = state.to(torch.float32).to(self.device)
        state = state.view(-1)
        qvalues = self.NN(state).view(self.board_size, self.board_size)
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

        l2 = torch.tensor(0., requires_grad=True)
        for w in self.NN.parameters():
            l2 = l2 + torch.norm(w)**2
        loss = loss + self.alpha*l2

        self.loss_v = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("STATE_DICT")
        # print(self.optimizer.state_dict())
        # print("PARAMS")
        # print(self.optimizer.param_groups[0]["params"])
        # print("PARAMS 2")
        return self.loss_v

class CQCAgent(Agent):
    def __init__(
            self,
            embedding_type: str=None,
            ansatz_type: str=None,
            nn_network: int=None,
            player=1,
            board_size=3,
            n_qubits=10,
            feature_map_reps=1,
            ansatz_reps=1
        ):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = 0.1
        self.board_size = board_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if nn_network == 1:
            NN = CNN_QCNN_CNN(
                embedding_type=embedding_type,
                board_size=board_size,
                n_qubits=n_qubits
            )
        elif nn_network == 2:
            NN = CNN_QNN_CNN_sampler(
                embedding_type=embedding_type,
                ansatz_type=ansatz_type,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps
            )
        elif nn_network == 3:
            NN = CNN_QNN_CNN_estimator(
                embedding_type=embedding_type,
                ansatz_type=ansatz_type,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps
            )

        self.HNN = NN
        self.HNN.to(self.device)
        self.optimizer = optim.Adam(self.HNN.parameters())

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

class QAgent(Agent):
    def __init__(
            self,
            embedding_type: str=None,
            ansatz_type: str=None,
            nn_network: int=None,
            n_qubits=10,
            param_weight=1.0,
            state_weight=1.0,
            feature_map_reps=1,
            ansatz_reps=1,
            player=1,
            board_size=3,
        ):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = 0.1
        self.board_size = board_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if nn_network == 1:
            NN = QCNN(
                embedding_type=embedding_type,
                param_weight=param_weight,
                board_size=board_size,
                state_weight=state_weight
            )
        elif nn_network == 2:
            NN = QNN(
                embedding_type=embedding_type,
                ansatz_type=ansatz_type,
                param_weight=param_weight,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps
            )

        self.QNN = NN
        self.QNN.to(self.device)
        self.optimizer = optim.Adam(self.QNN.parameters())

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
        self.QNN.train()

    def eval(self):
        super().eval()
        self.QNN.eval()

    def get_qvalues(self, state):
        state = state.to(torch.float32).to(self.device)
        state = state.view(-1)
        qvalues = self.QNN(state).view(self.board_size, self.board_size)
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

class CQCAgent_network(Agent):
    def __init__(
            self,
            embedding_type: str=None,
            ansatz_type: str=None,
            nn_network: int=None,
            network_model = 1,
            player=1,
            board_size=3,
            n_qubits=18,
            feature_map_reps=1,
            ansatz_reps=1,
            noised=True,
            distance=15.0
        ):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = 0.1
        self.board_size = board_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.noised = noised
        self.n_qubits = n_qubits
        self.network_model = network_model
        self.d = distance

        if nn_network == 1:
            NN = CNN_QCNN_CNN_for_network(
                embedding_type=embedding_type,
                network_model=network_model,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps
            )
        elif nn_network == 2:
            NN = CNN_QNN_CNN_sampler_for_network(
                embedding_type=embedding_type,
                ansatz_type=ansatz_type,
                network_model=network_model,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps
            )
        elif nn_network == 3:
            NN = CNN_QNN_CNN_estimator_for_network(
                embedding_type=embedding_type,
                ansatz_type=ansatz_type,
                network_model=network_model,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps,
            )

        self.HNN = NN
        self.HNN.to(self.device)
        self.optimizer = optim.Adam(self.HNN.parameters())

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
            iter = 0
            alpha = 0.2
            noise_strength = lambda d: 1 / (10.0**(-alpha/10.0*d)) -1
            noise = noise_strength(self.d)
            if self.noise_model == 1:
                iter = self.n_qubits*2*2
            elif self.noise_model == 2:
                iter = self.n_qubits*2
            for i in range(iter):
                if self.noised:
                    self.optimizer.param_groups[0]["params"][2][i] = torch.normal(mean=self.optimizer.param_groups[0]["params"][2][i].item(), std=noise, size=(1,1))

        return self.loss_v

class QCNNAgent_different_state(Agent):
    pass
