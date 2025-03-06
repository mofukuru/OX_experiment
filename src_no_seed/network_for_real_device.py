import random
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.primitives import BackendEstimator
from qiskit.primitives import BackendSampler

from qiskit_ibm_runtime import Estimator
from qiskit_ibm_runtime import Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from src_no_seed.components import QCNNComponent, QNNComponent, QIComponent
import src_no_seed.set_backend as set_backend

REAL_DEVICE = set_backend.REAL_DEVICE

#%% 古典量子ハイブリッドNN SamplerQNN
class CQCNN_sampler_HE(nn.Module):
    def __init__(self, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size

        self.num_qubits = num_qubits
        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_qubits).HE(featuremap_reps, ansatz_reps)

        x = lambda x: x
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        # self.isa_circuit = transpile(self.circuit, backend=set_backend.backend, initial_layout=range(self.num_qubits), optimization_level=1)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=set_backend.backend) if REAL_DEVICE else BackendSampler(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            interpret=x,
            output_shape=2**self.num_qubits,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            nn.Linear(9, self.num_qubits, bias=True),
            TorchConnector(self.qnn),
            nn.Linear(2**self.num_qubits, 9, bias=True),
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x