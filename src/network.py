import random
from typing import Final

import numpy as np
import torch
from qiskit.primitives import BackendEstimator, BackendSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import Estimator, Sampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import (
    SPSAEstimatorGradient,
    SPSASamplerGradient,
)
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from torch import nn
from torch.nn import functional as F

import src.set_backend as set_backend
from src.components import QCNNComponent, QIComponent, QNNComponent

REAL_DEVICE: Final[bool] = set_backend.REAL_DEVICE
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


class CCNN(nn.Module):
    """
    Making Classical Convolutional Neural Network(CCNN).
    This NN is indicated in the paper as ""Stronger""
    """

    def __init__(self):
        super().__init__()
        self.board_size = 3
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, bias=False)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 128, bias=True)
        self.linear2 = nn.Linear(128, 9, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.reshape(state, (1, 3, 3))
        x = self.conv1(x)
        x = torch.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.tanh(self.linear2(x))
        return x


class CCNN2(nn.Module):
    """
    Making Classical Convolutional Neural Network(CCNN2)
    This NN is indicated in the paper as ""Weaker""
    """

    def __init__(self):
        super().__init__()
        self.board_size = 3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, bias=False)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16, 9, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.reshape(state, (1, 3, 3))
        x = self.conv1(x)
        x = torch.flatten(x)
        x = self.tanh(self.linear(x))
        return x


class CNN_instead_sampler(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.board_size = 3
        self.linear = nn.Linear(9, n_qubits, bias=True)
        self.linear2 = nn.Linear(2**n_qubits, 9, bias=True)
        self.linear3 = nn.Linear(n_qubits, 2**n_qubits, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.linear(state)
        x = self.linear3(x)
        x = self.tanh(self.linear2(x))
        return x


class CNN_instead_estimator(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.board_size = 3
        self.linear = nn.Linear(9, n_qubits, bias=True)
        self.linear2 = nn.Linear(n_qubits, 9, bias=True)
        self.linear3 = nn.Linear(n_qubits, n_qubits, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.linear(state)
        x = self.linear3(x)
        x = self.tanh(self.linear2(x))
        return x


class CNN_QCNN_CNN(nn.Module):
    def __init__(self, embedding_type: str, board_size=3, n_qubits=10):
        super().__init__()
        self.board_size = board_size

        self.backend_num_qubits = set_backend.backend.num_qubits
        self.n_qubits = n_qubits

        self.linear1 = nn.Linear(self.board_size**2, self.n_qubits, bias=True)
        self.linear2 = nn.Linear(
            self.n_qubits // 2, self.board_size**2, bias=True
        )

        self.circuit, self.feature_map_parameters, self.ansatz_parameters = (
            QCNNComponent(
                self.n_qubits,
                ["conv", "pool"],
                [i for i in range(0, self.n_qubits, 2)],
                [i for i in range(1, self.n_qubits, 2)],
            ).make_qcnn_circuit(embedding_type=embedding_type, reps=1)
        )

        self.observable = []
        for i in range(self.n_qubits - 1, -1, -1):
            s_op = ""
            for j in range(self.n_qubits):
                if i % 2 == 0:
                    if j == i:
                        s_op += "Z"
                    else:
                        s_op += "I"
            if i % 2 == 0:
                if REAL_DEVICE:
                    self.observable.append(
                        SparsePauliOp.from_list(
                            [
                                (
                                    "I"
                                    * (self.backend_num_qubits - self.n_qubits)
                                    + s_op,
                                    1,
                                )
                            ]
                        )
                    )
                else:
                    self.observable.append(
                        SparsePauliOp.from_list([(s_op, 1)])
                    )

        self.pm = generate_preset_pass_manager(
            backend=set_backend.backend, optimization_level=1
        )
        self.isa_circuit = self.pm.run(self.circuit)
        self.estimator = (
            Estimator(backend=set_backend.backend, options={"shots": 1024})
            if REAL_DEVICE
            else BackendEstimator(
                backend=set_backend.backend,
                # options={"shots": 1024, "seed_simulator": SEED},
                options={"shots": 1024},
            )
        )
        self.qnn = EstimatorQNN(
            estimator=self.estimator,
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=self.feature_map_parameters,
            weight_params=self.ansatz_parameters,
            gradient=SPSAEstimatorGradient(self.estimator),
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1, TorchConnector(self.qnn), self.linear2, nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x


class CNN_QNN_CNN_sampler(nn.Module):
    def __init__(
        self,
        embedding_type: str,
        ansatz_type: str,
        board_size=3,
        n_qubits=7,
        feature_map_reps=1,
        ansatz_reps=1,
    ):
        super().__init__()
        self.board_size = board_size

        self.n_qubits = n_qubits
        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(
            self.n_qubits
        ).make_circuit(
            embedding_type=embedding_type,
            ansatz_type=ansatz_type,
            feature_map_reps=feature_map_reps,
            ansatz_reps=ansatz_reps,
        )

        self.pm = generate_preset_pass_manager(
            backend=set_backend.backend, optimization_level=1
        )
        self.isa_circuit = self.pm.run(self.circuit)
        self.sampler = (
            Sampler(backend=set_backend.backend, options={"shots": 1024})
            if REAL_DEVICE
            else BackendSampler(
                backend=set_backend.backend,
                # options={"shots": 1024, "seed_simulator": SEED},
                options={"shots": 1024},
            )
        )
        self.qnn = SamplerQNN(
            sampler=self.sampler,
            circuit=self.isa_circuit,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            gradient=SPSASamplerGradient(self.sampler),
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            nn.Linear(9, self.n_qubits, bias=True),
            TorchConnector(self.qnn),
            nn.Linear(2**self.n_qubits, 9, bias=True),
            nn.Tanh(),
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x


class CNN_QNN_CNN_estimator(nn.Module):
    def __init__(
        self,
        embedding_type: str,
        ansatz_type: str,
        board_size=3,
        n_qubits=10,
        feature_map_reps=1,
        ansatz_reps=1,
    ):
        super().__init__()
        self.board_size = board_size
        self.n_qubits = n_qubits
        self.linear1 = nn.Linear(9, self.n_qubits, bias=True)
        self.linear2 = nn.Linear(self.n_qubits, 9, bias=True)

        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(
            self.n_qubits
        ).make_circuit(
            embedding_type=embedding_type,
            ansatz_type=ansatz_type,
            feature_map_reps=feature_map_reps,
            ansatz_reps=ansatz_reps,
        )

        self.observable = []
        for i in range(self.n_qubits - 1, -1, -1):
            s_op = ""
            for j in range(self.n_qubits):
                if j == i:
                    s_op += "Z"
                else:
                    s_op += "I"
            if REAL_DEVICE:
                self.observable.append(
                    SparsePauliOp.from_list(
                        [
                            (
                                "I" * (self.backend_num_qubits - self.n_qubits)
                                + s_op
                            ),
                            1,
                        ]
                    )
                )
            else:
                self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))

        self.pm = generate_preset_pass_manager(
            backend=set_backend.backend, optimization_level=1
        )
        self.isa_circuit = self.pm.run(self.circuit)
        self.estimator = (
            Estimator(backend=set_backend.backend, options={"shots": 1024})
            if REAL_DEVICE
            else BackendEstimator(
                backend=set_backend.backend,
                # options={"shots": 1024, "seed_simulator": SEED},
                options={"shots": 1024},
            )
        )
        self.qnn = EstimatorQNN(
            estimator=self.estimator,
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            gradient=SPSAEstimatorGradient(self.estimator),
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1, TorchConnector(self.qnn), self.linear2, nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x


class QCNN(nn.Module):
    def __init__(
        self,
        embedding_type: str,
        board_size=3,
        state_weight=1,
    ):
        super().__init__()
        self.board_size = board_size
        self.state_weight = state_weight
        self.n_qubits = 18

        self.backend_num_qubits = set_backend.backend.num_qubits

        self.circuit, self.feature_map_parameters, self.ansatz_parameters = (
            QCNNComponent(
                18,
                ["conv", "pool"],
                [0, 2, 4, 6, 8, 10, 12, 14, 16],
                [1, 3, 5, 7, 9, 11, 13, 15, 17],
            ).make_qcnn_circuit(embedding_type=embedding_type, reps=1)
        )

        self.observable = []
        for i in range(self.n_qubits - 1, -1, -1):
            s_op = ""
            for j in range(self.n_qubits):
                if i % 2 == 0:
                    if j == i:
                        s_op += "Z"
                    else:
                        s_op += "I"
            if i % 2 == 0:
                if REAL_DEVICE:
                    self.observable.append(
                        SparsePauliOp.from_list(
                            [
                                (
                                    "I"
                                    * (self.backend_num_qubits - self.n_qubits)
                                    + s_op,
                                    1,
                                )
                            ]
                        )
                    )
                else:
                    self.observable.append(
                        SparsePauliOp.from_list([(s_op, 1)])
                    )

        self.pm = generate_preset_pass_manager(
            backend=set_backend.backend, optimization_level=1
        )
        self.isa_circuit = self.pm.run(self.circuit)
        self.estimator = (
            Estimator(backend=set_backend.backend, options={"shots": 1024})
            if REAL_DEVICE
            else BackendEstimator(
                backend=set_backend.backend,
                # options={"shots": 1024, "seed_simulator": SEED},
                options={"shots": 1024},
            )
        )
        self.qnn = EstimatorQNN(
            estimator=self.estimator,
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=self.feature_map_parameters,
            weight_params=self.ansatz_parameters,
            gradient=SPSAEstimatorGradient(self.estimator),
            input_gradients=True,
        )
        self.qcnn = nn.Sequential(TorchConnector(self.qnn), nn.Tanh())

    def forward(self, state):
        # state_weight is not used
        # state = state * self.state_weight
        x = torch.flatten(state)
        x = x.repeat(1, 2)
        x = self.qcnn(x)
        return x


class QNN(nn.Module):
    def __init__(
        self,
        embedding_type: str,
        ansatz_type: str,
        board_size=3,
        n_qubits=9,
        feature_map_reps=1,
        ansatz_reps=1,
    ):
        super().__init__()
        self.board_size = board_size
        self.backend_num_qubits = set_backend.backend.num_qubits
        (
            self.circuit,
            self.feature_map_parameters,
            self.ansatz_parameters,
        ) = QNNComponent(n_qubits).make_circuit(
            embedding_type=embedding_type,
            ansatz_type=ansatz_type,
            feature_map_reps=feature_map_reps,
            ansatz_reps=ansatz_reps,
        )

        self.observable = []
        for i in range(n_qubits - 1, -1, -1):
            s_op = ""
            for j in range(n_qubits):
                if j == i:
                    s_op += "Z"
                else:
                    s_op += "I"
            if REAL_DEVICE:
                self.observable.append(
                    SparsePauliOp.from_list(
                        [
                            (
                                "I" * (self.backend_num_qubits - n_qubits)
                                + s_op
                            ),
                            1,
                        ]
                    )
                )
            else:
                self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))

        self.pm = generate_preset_pass_manager(
            backend=set_backend.backend, optimization_level=1
        )
        self.isa_circuit = self.pm.run(self.circuit)
        self.estimator = (
            Estimator(backend=set_backend.backend, options={"shots": 1024})
            if REAL_DEVICE
            else BackendEstimator(
                backend=set_backend.backend,
                # options={"shots": 1024, "seed_simulator": SEED},
                options={"shots": 1024},
            )
        )
        self.qnn = EstimatorQNN(
            estimator=self.estimator,
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=self.feature_map_parameters,
            weight_params=self.ansatz_parameters,
            gradient=SPSAEstimatorGradient(self.estimator),
            input_gradients=True,
        )

        self.Qnn = nn.Sequential(TorchConnector(self.qnn), nn.Tanh())

    def forward(self, state):
        x = torch.flatten(state)
        x = self.Qnn(x)
        return x


class CNN_QCNN_CNN_for_network(nn.Module):
    def __init__(
        self,
        embedding_type: str,
        network_model: int,
        board_size=3,
        n_qubits=7,
        feature_map_reps=1,
    ):
        super().__init__()
        self.board_size = board_size
        self.backend = set_backend.backend
        self.n_qubits = n_qubits

        self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(
            self.n_qubits
        ).make_qcnn_circuit(
            embedding_type=embedding_type,
            network_model=network_model,
            feature_map_reps=feature_map_reps,
        )
        self.observable = []
        for i in range(self.n_qubits - 1, -1, -1):
            s_op = ""
            for j in range(self.n_qubits):
                if i % 2 == 0:
                    if j == i:
                        s_op += "Z"
                    else:
                        s_op += "I"
            if i % 2 == 0:
                if REAL_DEVICE:
                    self.observable.append(
                        SparsePauliOp.from_list(
                            [
                                (
                                    "I"
                                    * (self.backend_num_qubits - self.n_qubits)
                                    + s_op,
                                    1,
                                )
                            ]
                        )
                    )
                else:
                    self.observable.append(
                        SparsePauliOp.from_list([(s_op, 1)])
                    )

        self.pm = generate_preset_pass_manager(
            backend=set_backend.backend, optimization_level=1
        )
        self.isa_circuit = self.pm.run(self.circuit)

        self.qnn = EstimatorQNN(
            estimator=(
                Estimator(backend=set_backend.backend, options={"shots": 1000})
                if REAL_DEVICE
                else BackendEstimator(
                    backend=set_backend.backend,
                    options={"shots": 1000, "seed_simulator": SEED},
                )
            ),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=self.feature_map_parameters,
            weight_params=self.ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1, TorchConnector(self.qnn), self.linear2, nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x


class CNN_QNN_CNN_sampler_for_network(nn.Module):
    def __init__(
        self,
        embedding_type: str,
        ansatz_type: str,
        network_model: int,
        board_size=3,
        n_qubits=7,
        feature_map_reps=1,
        ansatz_reps=1,
    ):
        super().__init__()
        self.board_size = board_size
        self.backend = set_backend.backend
        self.n_qubits = n_qubits

        self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(
            self.n_qubits
        ).make_qnn_circuit(
            embedding_type=embedding_type,
            ansatz_type=ansatz_type,
            network_model=network_model,
            feature_map_reps=feature_map_reps,
            ansatz_reps=ansatz_reps,
        )
        self.pm = generate_preset_pass_manager(
            backend=self.backend, optimization_level=1
        )
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=(
                Sampler(backend=self.backend, options={"shots": 1000})
                if REAL_DEVICE
                else BackendSampler(
                    backend=self.backend,
                    options={"shots": 1000, "seed_simulator": SEED},
                )
            ),
            circuit=self.isa_circuit,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            nn.Linear(9, self.num_qubits, bias=True),
            TorchConnector(self.qnn),
            nn.Linear(2**self.num_qubits, 9, bias=True),
            nn.Tanh(),
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x


class CNN_QNN_CNN_estimator_for_network(nn.Module):
    def __init__(
        self,
        embedding_type: str,
        ansatz_type: str,
        network_model: int,
        board_size=3,
        n_qubits=10,
        feature_map_reps=1,
        ansatz_reps=1,
    ):
        super().__init__()
        self.board_size = board_size
        self.n_qubits = n_qubits
        self.linear1 = nn.Linear(9, self.n_qubits, bias=True)
        self.linear2 = nn.Linear(self.n_qubits, 9, bias=True)

        self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(
            self.n_qubits
        ).make_qnn_circuit(
            embedding_type=embedding_type,
            ansatz_type=ansatz_type,
            network_model=network_model,
            feature_map_reps=feature_map_reps,
            ansatz_reps=ansatz_reps,
        )

        self.observable = []
        for i in range(self.n_qubits - 1, -1, -1):
            s_op = ""
            for j in range(self.n_qubits):
                if j == i:
                    s_op += "Z"
                else:
                    s_op += "I"
            if REAL_DEVICE:
                self.observable.append(
                    SparsePauliOp.from_list(
                        [
                            (
                                "I" * (self.backend_num_qubits - self.n_qubits)
                                + s_op
                            ),
                            1,
                        ]
                    )
                )
            else:
                self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))

        self.pm = generate_preset_pass_manager(
            backend=set_backend.backend, optimization_level=1
        )
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = EstimatorQNN(
            estimator=(
                Estimator(backend=set_backend.backend, options={"shots": 1000})
                if REAL_DEVICE
                else BackendEstimator(
                    backend=set_backend.backend,
                    options={"shots": 1000, "seed_simulator": SEED},
                )
            ),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1, TorchConnector(self.qnn), self.linear2, nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x
