import math
import random
from typing import Final

import numpy as np
import torch
from torch import nn, optim

from src.components import ReplayBuffer, Transition
from src.network import (
    CCNN,
    CCNN2,
    CNN_QCNN_CNN,
    QCNN,
    QNN,
    CNN_instead_estimator,
    CNN_instead_sampler,
    CNN_QCNN_CNN_for_network,
    CNN_QNN_CNN_estimator,
    CNN_QNN_CNN_estimator_for_network,
    CNN_QNN_CNN_sampler,
    CNN_QNN_CNN_sampler_for_network,
)

SEED: Final[int] = 42


def set_seed(seed: int = 42):
    """
    Setting seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# If you do not want to set seed, you can comment out the below set_seed() function.
# set_seed(SEED)


class Agent:
    def __init__(self, player: int):
        self.player = player
        self.stop = False

    def train(self) -> bool:
        self.stop = False

    def eval(self) -> bool:
        self.stop = True

    def check_state(self, tictactoe: classmethod) -> torch.tensor:
        return tictactoe.get_state(self.player)

    def check_actions(self, tictactoe: classmethod) -> list:
        return tictactoe.get_possible_actions()


class Human(Agent):
    def __init__(self, player=1):
        super().__init__(player)
        self.stop = True

    def action(self, tictactoe: classmethod) -> None:
        moves = self.check_actions(tictactoe)

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

    def action(self, tictactoe: classmethod) -> int:
        moves = self.check_actions(tictactoe)
        return random.choice(moves)


class CNNAgent(Agent):
    def __init__(
        self,
        player=1,
        board_size=3,
        network=None,
        n_qubits=7,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1000,
    ):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = 0.01
        self.alpha = 0.01
        self.board_size = board_size
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

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
            self.optimizer.param_groups[0]["params"][2].requires_grad = False

    def action(self, tictactoe: classmethod) -> int:
        board = self.check_state(tictactoe)
        possible_actions = self.check_actions(tictactoe)
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
        best_q_value = -float("inf")
        for mv_x, mv_y in possible_moves:
            q_value = qvalues[mv_x, mv_y]
            if q_value > best_q_value:
                best_q_value = q_value
                best_move = (mv_x, mv_y)
        return best_move, best_q_value

    def update(self, tictactoe, state, action, reward):
        if self.stop:
            return None

        with torch.no_grad():
            board = self.check_state(tictactoe)
            moves = self.check_actions(tictactoe)
            _, best_value = self._get_the_best(board, moves)
            next_max = max(0, best_value)
            target_q = torch.tensor(reward + self.discount * next_max)
            target_q = target_q.to(torch.float32).to(self.device)

        x, y = action
        old_qvalue = self.get_qvalues(state)[x, y]
        loss = nn.functional.huber_loss(old_qvalue, target_q)

        l2 = torch.tensor(0.0, requires_grad=True)
        for w in self.NN.parameters():
            l2 = l2 + torch.norm(w) ** 2
        loss = loss + self.alpha * l2

        self.loss_v = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self.loss_v

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * episode / self.epsilon_decay)


class CQCAgent(Agent):
    def __init__(
        self,
        embedding_type: str = None,
        ansatz_type: str = None,
        nn_network: int = None,
        player=1,
        board_size=3,
        n_qubits=10,
        feature_map_reps=1,
        ansatz_reps=1,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1000,
    ):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.board_size = board_size
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if nn_network == 1:
            NN = CNN_QCNN_CNN(
                embedding_type=embedding_type,
                board_size=board_size,
                n_qubits=n_qubits,
            )
        elif nn_network == 2:
            NN = CNN_QNN_CNN_sampler(
                embedding_type=embedding_type,
                ansatz_type=ansatz_type,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps,
            )
        elif nn_network == 3:
            NN = CNN_QNN_CNN_estimator(
                embedding_type=embedding_type,
                ansatz_type=ansatz_type,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps,
            )

        self.HNN = NN
        self.HNN.to(self.device)
        self.optimizer = optim.Adam(self.HNN.parameters())

    def action(self, tictactoe: classmethod) -> int:
        board = self.check_state(tictactoe)
        possible_actions = self.check_actions(tictactoe)
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
        best_q_value = -float("inf")
        for mv_x, mv_y in possible_moves:
            q_value = qvalues[mv_x, mv_y]
            if q_value > best_q_value:
                best_q_value = q_value
                best_move = (mv_x, mv_y)
        return best_move, best_q_value

    def update(self, tictactoe, state, action, reward):
        if self.stop:
            return None

        with torch.no_grad():
            board = self.check_state(tictactoe)
            moves = self.check_actions(tictactoe)
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

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * episode / self.epsilon_decay)


class QAgent(Agent):
    def __init__(
        self,
        embedding_type: str = None,
        ansatz_type: str = None,
        nn_network: int = None,
        n_qubits=10,
        state_weight=1.0,
        feature_map_reps=1,
        ansatz_reps=1,
        player=1,
        board_size=3,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1000,
    ):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.board_size = board_size
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if nn_network == 1:
            NN = QCNN(
                embedding_type=embedding_type,
                board_size=board_size,
                state_weight=state_weight,
            )
        elif nn_network == 2:
            NN = QNN(
                embedding_type=embedding_type,
                ansatz_type=ansatz_type,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps,
            )

        self.QNN = NN
        self.QNN.to(self.device)
        self.optimizer = optim.Adam(self.QNN.parameters())

    def action(self, tictactoe: classmethod) -> int:
        board = self.check_state(tictactoe)
        possible_actions = self.check_actions(tictactoe)
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
        best_q_value = -float("inf")
        for mv_x, mv_y in possible_moves:
            q_value = qvalues[mv_x, mv_y]
            if q_value > best_q_value:
                best_q_value = q_value
                best_move = (mv_x, mv_y)
        return best_move, best_q_value

    def update(self, tictactoe, state, action, reward):
        if self.stop:
            return None

        with torch.no_grad():
            board = self.check_state(tictactoe)
            moves = self.check_actions(tictactoe)
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

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * episode / self.epsilon_decay)


class CNNAgent_ER(Agent):
    def __init__(
        self,
        player=1,
        board_size=3,
        network=None,
        n_qubits=7,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1000,
    ):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lr = 0.01
        self.alpha = 0.01
        self.board_size = board_size
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

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
            self.optimizer.param_groups[0]["params"][2].requires_grad = False

    def action(self, tictactoe: classmethod) -> int:
        board = self.check_state(tictactoe)
        possible_actions = self.check_actions(tictactoe)
        if not possible_actions:
            return None
        if not self.stop and random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            with torch.no_grad():
                best_action, _ = self._get_the_best(board, possible_actions)
                return best_action

    def memorize(self, state, action, next_state, reward, done):
        """リプレイバッファに遷移を保存する"""
        action_index = action[0] * self.board_size + action[1]
        action_tensor = torch.tensor(
            [action_index], device=self.device, dtype=torch.long
        )
        reward_tensor = torch.tensor(
            [reward], device=self.device, dtype=torch.float32
        )
        done_tensor = torch.tensor(
            [done], device=self.device, dtype=torch.bool
        )

        self.memory.push(
            state.unsqueeze(0),
            action_tensor,
            next_state.unsqueeze(0) if next_state is not None else None,
            reward_tensor,
            done_tensor,
        )

    def learn(self):
        """経験再生を使って学習する"""
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        if len(action_batch.shape) == 1:
            action_batch = action_batch.unsqueeze(1)

        q_values = self.NN(state_batch.float()).view(self.batch_size, -1)
        state_action_values = q_values.gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states.size(0) > 0:
            next_q_values = self.NN(non_final_next_states.float()).view(
                non_final_next_states.size(0), -1
            )
            next_state_values[non_final_mask] = next_q_values.max(1)[
                0
            ].detach()

        expected_state_action_values = (
            next_state_values * self.discount
        ) + reward_batch.squeeze()

        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _get_the_best(self, board, possible_moves):
        with torch.no_grad():
            qvalues = self.NN(board.unsqueeze(0).float()).view(-1)
            best_q_value = -float("inf")
            best_move = None
            for move in possible_moves:
                move_index = move[0] * self.board_size + move[1]
                q_value = qvalues[move_index]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_move = move
            return best_move, best_q_value

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * episode / self.epsilon_decay)


class CQCAgent_ER(Agent):
    def __init__(
        self,
        embedding_type: str = None,
        ansatz_type: str = None,
        nn_network: int = None,
        player=1,
        board_size=3,
        n_qubits=10,
        feature_map_reps=1,
        ansatz_reps=1,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1000,
        capacity=10000,
        batch_size=128,
    ):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.board_size = board_size
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.memory = ReplayBuffer(capacity)
        self.batch_size = batch_size

        if nn_network == 1:
            NN = CNN_QCNN_CNN(
                embedding_type=embedding_type,
                board_size=board_size,
                n_qubits=n_qubits,
            )
        elif nn_network == 2:
            NN = CNN_QNN_CNN_sampler(
                embedding_type=embedding_type,
                ansatz_type=ansatz_type,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps,
            )
        elif nn_network == 3:
            NN = CNN_QNN_CNN_estimator(
                embedding_type=embedding_type,
                ansatz_type=ansatz_type,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps,
            )

        self.HNN = NN
        self.HNN.to(self.device)
        self.optimizer = optim.Adam(self.HNN.parameters())

    def action(self, tictactoe: classmethod) -> int:
        board = self.check_state(tictactoe)
        possible_actions = self.check_actions(tictactoe)
        if not possible_actions:
            return None
        if not self.stop and random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            with torch.no_grad():
                best_action, _ = self._get_the_best(board, possible_actions)
                return best_action

    def memorize(self, state, action, next_state, reward, done):
        action_index = action[0] * self.board_size + action[1]
        action_tensor = torch.tensor(
            [action_index], device=self.device, dtype=torch.long
        )
        reward_tensor = torch.tensor(
            [reward], device=self.device, dtype=torch.float32
        )
        done_tensor = torch.tensor(
            [done], device=self.device, dtype=torch.bool
        )
        self.memory.push(
            state.unsqueeze(0),
            action_tensor,
            next_state.unsqueeze(0) if next_state is not None else None,
            reward_tensor,
            done_tensor,
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        if len(action_batch.shape) == 1:
            action_batch = action_batch.unsqueeze(1)

        q_values = self.HNN(state_batch.float()).view(self.batch_size, -1)
        state_action_values = q_values.gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states.size(0) > 0:
            next_q_values = self.HNN(non_final_next_states.float()).view(
                non_final_next_states.size(0), -1
            )
            next_state_values[non_final_mask] = next_q_values.max(1)[
                0
            ].detach()

        expected_state_action_values = (
            next_state_values * self.discount
        ) + reward_batch.squeeze()

        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _get_the_best(self, board, possible_moves):
        with torch.no_grad():
            qvalues = self.HNN(board.unsqueeze(0).float()).view(-1)
            best_q_value = -float("inf")
            best_move = None
            for move in possible_moves:
                move_index = move[0] * self.board_size + move[1]
                q_value = qvalues[move_index]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_move = move
            return best_move, best_q_value

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * episode / self.epsilon_decay)


class QAgent_ER(Agent):
    def __init__(
        self,
        embedding_type: str = None,
        ansatz_type: str = None,
        nn_network: int = None,
        n_qubits=10,
        state_weight=1.0,
        feature_map_reps=1,
        ansatz_reps=1,
        player=1,
        board_size=3,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1000,
        capacity=10000,
        batch_size=128,
    ):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.board_size = board_size
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.memory = ReplayBuffer(capacity)
        self.batch_size = batch_size

        if nn_network == 1:
            NN = QCNN(
                embedding_type=embedding_type,
                board_size=board_size,
                state_weight=state_weight,
            )
        elif nn_network == 2:
            NN = QNN(
                embedding_type=embedding_type,
                ansatz_type=ansatz_type,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps,
            )

        self.QNN = NN
        self.QNN.to(self.device)

        self.optimizer = optim.Adam(self.QNN.parameters())

    def action(self, tictactoe: classmethod) -> int:
        board = self.check_state(tictactoe)
        possible_actions = self.check_actions(tictactoe)
        if not possible_actions:
            return None
        if not self.stop and random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            with torch.no_grad():
                best_action, _ = self._get_the_best(board, possible_actions)
                return best_action

    def memorize(self, state, action, next_state, reward, done):
        action_index = action[0] * self.board_size + action[1]
        action_tensor = torch.tensor(
            [action_index], device=self.device, dtype=torch.long
        )
        reward_tensor = torch.tensor(
            [reward], device=self.device, dtype=torch.float32
        )
        done_tensor = torch.tensor(
            [done], device=self.device, dtype=torch.bool
        )
        self.memory.push(
            state.unsqueeze(0),
            action_tensor,
            next_state.unsqueeze(0) if next_state is not None else None,
            reward_tensor,
            done_tensor,
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        if len(action_batch.shape) == 1:
            action_batch = action_batch.unsqueeze(1)

        q_values = self.QNN(state_batch.float()).view(self.batch_size, -1)
        state_action_values = q_values.gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states.size(0) > 0:
            next_q_values = self.QNN(non_final_next_states.float()).view(
                non_final_next_states.size(0), -1
            )
            next_state_values[non_final_mask] = next_q_values.max(1)[
                0
            ].detach()

        expected_state_action_values = (
            next_state_values * self.discount
        ) + reward_batch.squeeze()

        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _get_the_best(self, board, possible_moves):
        with torch.no_grad():
            qvalues = self.QNN(board.unsqueeze(0).float()).view(-1)
            best_q_value = -float("inf")
            best_move = None
            for move in possible_moves:
                move_index = move[0] * self.board_size + move[1]
                q_value = qvalues[move_index]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_move = move
            return best_move, best_q_value

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * episode / self.epsilon_decay)


class CQCAgent_network(Agent):
    def __init__(
        self,
        embedding_type: str = None,
        ansatz_type: str = None,
        nn_network: int = None,
        network_model=1,
        player=1,
        board_size=3,
        n_qubits=18,
        feature_map_reps=1,
        ansatz_reps=1,
        noised=True,
        distance=15.0,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1000,
    ):
        super().__init__(player)
        self.discount = 0.9
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.board_size = board_size
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
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
                feature_map_reps=feature_map_reps,
            )
        elif nn_network == 2:
            NN = CNN_QNN_CNN_sampler_for_network(
                embedding_type=embedding_type,
                ansatz_type=ansatz_type,
                network_model=network_model,
                board_size=board_size,
                n_qubits=n_qubits,
                feature_map_reps=feature_map_reps,
                ansatz_reps=ansatz_reps,
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

    def action(self, tictactoe: classmethod) -> int:
        board = self.check_state(tictactoe)
        possible_actions = self.check_actions(tictactoe)
        if not possible_actions:
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
        best_q_value = -float("inf")
        for mv_x, mv_y in possible_moves:
            q_value = qvalues[mv_x, mv_y]
            if q_value > best_q_value:
                best_q_value = q_value
                best_move = (mv_x, mv_y)
        return best_move, best_q_value

    def update(self, tictactoe, state, action, reward):
        if self.stop:
            return None

        with torch.no_grad():
            board = self.check_state(tictactoe)
            moves = self.check_actions(tictactoe)
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
            noise_strength = lambda d: 1 / (10.0 ** (-alpha / 10.0 * d)) - 1
            noise = noise_strength(self.d)
            if self.network_model == 1:
                iter = self.n_qubits * 2 * 2
            elif self.network_model == 2:
                iter = self.n_qubits * 2
            for i in range(iter):
                if self.noised:
                    self.optimizer.param_groups[0]["params"][2][i] = (
                        torch.normal(
                            mean=self.optimizer.param_groups[0]["params"][2][
                                i
                            ].item(),
                            std=noise,
                            size=(1, 1),
                        )
                    )

        return self.loss_v

    def update_epsilon(self, episode):
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * math.exp(-1.0 * episode / self.epsilon_decay)
