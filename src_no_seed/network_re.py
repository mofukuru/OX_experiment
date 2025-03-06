import random
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

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
## deprecated
# NETWORK = set_backend.backend_noise
#%% seed値の固定
# seed = 42
# algorithm_globals.random_seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

#%% 古典NN
class CCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.board_size = 3
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, bias=False)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 128, bias=True)
        self.linear2 = nn.Linear(128, 9, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.reshape(state, (1,3,3))
        x = self.conv1(x)
        x = torch.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.tanh(self.linear2(x))
        return x

#%% 古典NN2
class CCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.board_size = 3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, bias=False)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16, 9, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.reshape(state, (1,3,3))
        x = self.conv1(x)
        x = torch.flatten(x)
        x = self.tanh(self.linear(x))
        return x

#%%   
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

#%%
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

#%% 古典量子ハイブリッドQCNNを用いる ZFeatureMap
class CQCCNN_Z(nn.Module):
    def __init__(self, board_size=3, n_qubits = 10):
        super().__init__()
        self.board_size = board_size

        self.backend_num_qubits = set_backend.backend.num_qubits
        self.n_qubits = n_qubits

        self.linear1 = nn.Linear(9, self.n_qubits, bias=True)
        self.linear2 = nn.Linear(self.n_qubits//2, 9, bias=True)

        self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(self.n_qubits, ["conv", "pool"], [i for i in range(0, self.n_qubits, 2)], [i for i in range(1, self.n_qubits, 2)]).create_qcnn("ZFeatureMap", reps=1)

        self.observable = []
        for i in range(self.n_qubits-1, -1, -1):
            s_op = ""
            for j in range(self.n_qubits):
                if i%2 == 0:
                    if j == i:
                        s_op += "Z"
                    else:
                        s_op += "I"
            if i%2 == 0:
                self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))

        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)

        self.qnn = EstimatorQNN(
            estimator= Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=self.feature_map_parameters,
            weight_params=self.ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1,
            TorchConnector(self.qnn),
            self.linear2,
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x

#%% 古典量子ハイブリッドQCNNを用いる ZZFeatureMap
class CQCCNN_ZZ(nn.Module):
    def __init__(self, board_size=3, n_qubits = 10):
        super().__init__()
        self.board_size = board_size

        self.backend_num_qubits = set_backend.backend.num_qubits
        self.n_qubits = n_qubits

        self.linear1 = nn.Linear(9, self.n_qubits, bias=True)
        self.linear2 = nn.Linear(self.n_qubits//2, 9, bias=True)

        self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(self.n_qubits, ["conv", "pool"], [i for i in range(0, self.n_qubits, 2)], [i for i in range(1, self.n_qubits, 2)]).create_qcnn("ZZFeatureMap", reps=1)

        self.observable = []
        for i in range(self.n_qubits-1, -1, -1):
            s_op = ""
            for j in range(self.n_qubits):
                if i%2 == 0:
                    if j == i:
                        s_op += "Z"
                    else:
                        s_op += "I"
            if i%2 == 0:
                self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))

        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)

        self.qnn = EstimatorQNN(
            estimator= Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=self.feature_map_parameters,
            weight_params=self.ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1,
            TorchConnector(self.qnn),
            self.linear2,
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x

#%% 古典量子ハイブリッドQCNNを用いる TPE
class CQCCNN_T(nn.Module):
    def __init__(self, board_size=3, n_qubits = 10):
        super().__init__()
        self.board_size = board_size

        self.backend_num_qubits = set_backend.backend.num_qubits
        self.n_qubits = n_qubits

        self.linear1 = nn.Linear(9, self.n_qubits, bias=True)
        self.linear2 = nn.Linear(self.n_qubits//2, 9, bias=True)

        self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(self.n_qubits, ["conv", "pool"], [i for i in range(0, self.n_qubits, 2)], [i for i in range(1, self.n_qubits, 2)]).create_qcnn("TPE", reps=1)

        self.observable = []
        for i in range(self.n_qubits-1, -1, -1):
            s_op = ""
            for j in range(self.n_qubits):
                if i%2 == 0:
                    if j == i:
                        s_op += "Z"
                    else:
                        s_op += "I"
            if i%2 == 0:
                self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))

        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)

        self.qnn = EstimatorQNN(
            estimator= Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=self.feature_map_parameters,
            weight_params=self.ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1,
            TorchConnector(self.qnn),
            self.linear2,
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x

#%% 古典量子ハイブリッドQCNNを用いる HEE
class CQCCNN_H(nn.Module):
    def __init__(self, board_size=3, n_qubits = 10):
        super().__init__()
        self.board_size = board_size

        self.backend_num_qubits = set_backend.backend.num_qubits
        self.n_qubits = n_qubits

        self.linear1 = nn.Linear(9, self.n_qubits, bias=True)
        self.linear2 = nn.Linear(self.n_qubits//2, 9, bias=True)

        self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(self.n_qubits, ["conv", "pool"], [i for i in range(0, self.n_qubits, 2)], [i for i in range(1, self.n_qubits, 2)]).create_qcnn("HEE", reps=1)

        self.observable = []
        for i in range(self.n_qubits-1, -1, -1):
            s_op = ""
            for j in range(self.n_qubits):
                if i%2 == 0:
                    if j == i:
                        s_op += "Z"
                    else:
                        s_op += "I"
            if i%2 == 0:
                self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))

        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)

        self.qnn = EstimatorQNN(
            estimator= Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=self.feature_map_parameters,
            weight_params=self.ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1,
            TorchConnector(self.qnn),
            self.linear2,
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x

#%% 古典量子ハイブリッドNN SamplerQNN
class CQCNN_sampler_ZR(nn.Module):
    def __init__(self, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size

        self.num_qubits = num_qubits
        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_qubits).ZR(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%% 古典量子ハイブリッドNN SamplerQNN
class CQCNN_sampler_ZZR(nn.Module):
    def __init__(self, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size

        self.num_qubits = num_qubits
        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_qubits).ZZR(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%% 古典量子ハイブリッドNN SamplerQNN
class CQCNN_sampler_TR(nn.Module):
    def __init__(self, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size

        self.num_qubits = num_qubits
        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_qubits).TR(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%% 古典量子ハイブリッドNN SamplerQNN
class CQCNN_sampler_HR(nn.Module):
    def __init__(self, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size

        self.num_qubits = num_qubits
        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_qubits).HR(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%% 古典量子ハイブリッドNN SamplerQNN
class CQCNN_sampler_ZE(nn.Module):
    def __init__(self, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size

        self.num_qubits = num_qubits
        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_qubits).ZE(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%% 古典量子ハイブリッドNN SamplerQNN
class CQCNN_sampler_ZZE(nn.Module):
    def __init__(self, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size

        self.num_qubits = num_qubits
        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_qubits).ZZE(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

    #%% 古典量子ハイブリッドNN SamplerQNN
class CQCNN_sampler_TE(nn.Module):
    def __init__(self, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size

        self.num_qubits = num_qubits
        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_qubits).TE(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%% 古典量子ハイブリッドNN SamplerQNN
class CQCNN_sampler_HE(nn.Module):
    def __init__(self, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size

        self.num_qubits = num_qubits
        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_qubits).HE(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=set_backend.backend) if REAL_DEVICE else BackendSampler(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%% 古典量子ハイブリッドNN estimator
class CQCNN_estimator_ZR(nn.Module):
    def __init__(self, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        self.num_of_qubit = num_of_qubit
        self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
        self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_of_qubit).ZR(featuremap_reps, ansatz_reps)

        self.observable = []
        for i in range(self.num_of_qubit-1, -1, -1):
            s_op = ""
            for j in range(self.num_of_qubit):
                if j == i:
                    s_op += "Z"
                else:
                    s_op += "I"
            self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = EstimatorQNN(
            estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1,
            TorchConnector(self.qnn),
            self.linear2,
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x

#%% 古典量子ハイブリッドNN estimator
class CQCNN_estimator_ZZR(nn.Module):
    def __init__(self, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        self.num_of_qubit = num_of_qubit
        self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
        self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_of_qubit).ZZR(featuremap_reps, ansatz_reps)

        self.observable = []
        for i in range(self.num_of_qubit-1, -1, -1):
            s_op = ""
            for j in range(self.num_of_qubit):
                if j == i:
                    s_op += "Z"
                else:
                    s_op += "I"
            self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = EstimatorQNN(
            estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1,
            TorchConnector(self.qnn),
            self.linear2,
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x

#%% 古典量子ハイブリッドNN estimator
class CQCNN_estimator_TR(nn.Module):
    def __init__(self, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        self.num_of_qubit = num_of_qubit
        self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
        self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_of_qubit).TR(featuremap_reps, ansatz_reps)

        self.observable = []
        for i in range(self.num_of_qubit-1, -1, -1):
            s_op = ""
            for j in range(self.num_of_qubit):
                if j == i:
                    s_op += "Z"
                else:
                    s_op += "I"
            self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = EstimatorQNN(
            estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1,
            TorchConnector(self.qnn),
            self.linear2,
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x

#%% 古典量子ハイブリッドNN estimator
class CQCNN_estimator_HR(nn.Module):
    def __init__(self, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        self.num_of_qubit = num_of_qubit
        self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
        self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_of_qubit).HR(featuremap_reps, ansatz_reps)

        self.observable = []
        for i in range(self.num_of_qubit-1, -1, -1):
            s_op = ""
            for j in range(self.num_of_qubit):
                if j == i:
                    s_op += "Z"
                else:
                    s_op += "I"
            self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = EstimatorQNN(
            estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1,
            TorchConnector(self.qnn),
            self.linear2,
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x

#%% 古典量子ハイブリッドNN estimator
class CQCNN_estimator_ZE(nn.Module):
    def __init__(self, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        self.num_of_qubit = num_of_qubit
        self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
        self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_of_qubit).ZE(featuremap_reps, ansatz_reps)

        self.observable = []
        for i in range(self.num_of_qubit-1, -1, -1):
            s_op = ""
            for j in range(self.num_of_qubit):
                if j == i:
                    s_op += "Z"
                else:
                    s_op += "I"
            self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = EstimatorQNN(
            estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1,
            TorchConnector(self.qnn),
            self.linear2,
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x

#%% 古典量子ハイブリッドNN estimator
class CQCNN_estimator_ZZE(nn.Module):
    def __init__(self, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        self.num_of_qubit = num_of_qubit
        self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
        self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_of_qubit).ZZE(featuremap_reps, ansatz_reps)

        self.observable = []
        for i in range(self.num_of_qubit-1, -1, -1):
            s_op = ""
            for j in range(self.num_of_qubit):
                if j == i:
                    s_op += "Z"
                else:
                    s_op += "I"
            self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = EstimatorQNN(
            estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1,
            TorchConnector(self.qnn),
            self.linear2,
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x

#%% 古典量子ハイブリッドNN estimator
class CQCNN_estimator_TE(nn.Module):
    def __init__(self, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        self.num_of_qubit = num_of_qubit
        self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
        self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_of_qubit).TE(featuremap_reps, ansatz_reps)

        self.observable = []
        for i in range(self.num_of_qubit-1, -1, -1):
            s_op = ""
            for j in range(self.num_of_qubit):
                if j == i:
                    s_op += "Z"
                else:
                    s_op += "I"
            self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = EstimatorQNN(
            estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1,
            TorchConnector(self.qnn),
            self.linear2,
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x

#%% 古典量子ハイブリッドNN estimator
class CQCNN_estimator_HE(nn.Module):
    def __init__(self, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        self.num_of_qubit = num_of_qubit
        self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
        self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

        self.circuit, feature_map_parameters, ansatz_parameters = QNNComponent(self.num_of_qubit).HE(featuremap_reps, ansatz_reps)

        self.observable = []
        for i in range(self.num_of_qubit-1, -1, -1):
            s_op = ""
            for j in range(self.num_of_qubit):
                if j == i:
                    s_op += "Z"
                else:
                    s_op += "I"
            self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = EstimatorQNN(
            estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=feature_map_parameters,
            weight_params=ansatz_parameters,
            input_gradients=True,
        )
        self.cqnn = nn.Sequential(
            self.linear1,
            TorchConnector(self.qnn),
            self.linear2,
            nn.Tanh()
        )

    def forward(self, state):
        x = self.cqnn(state)
        return x

#%% QCNN
class QCNN_Z(nn.Module):
    def __init__(self, board_size=3, state_weight=1):
        super().__init__()
        self.board_size=board_size
        self.state_weight=state_weight

        self.backend_num_qubits = set_backend.backend.num_qubits

        self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn("ZFeatureMap", reps=1)
        if REAL_DEVICE:
            self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
            self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
            self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
            self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
            self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
            self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
            self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
            self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
            self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
        else:
            self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
            self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
            self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
            self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
            self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
            self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
            self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
            self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
            self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
        self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        # EstimatorQNNはNG
        self.qnn = EstimatorQNN(
            # estimator=Estimator(backend, options={"shots": 1000}),
            estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={"shots":1000, "seed_simulator": seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=self.feature_map_parameters,
            weight_params=self.ansatz_parameters,
            input_gradients=True,
        )
        # [goal]input_gradients=False and no using TorchConnector
        self.qcnn = nn.Sequential(
            TorchConnector(self.qnn),
            nn.Tanh()
        )

    def forward(self, state):
        state = state*self.state_weight
        '''このflattenは要らない。。'''
        x = torch.flatten(state)
        x = x.repeat(1, 2)
        x = self.qcnn(x)
        return x

class QCNN_ZZ(nn.Module):
    def __init__(self, board_size=3, state_weight=1):
        super().__init__()
        self.board_size=board_size
        self.state_weight=state_weight

        self.backend_num_qubits = set_backend.backend.num_qubits

        self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn("ZZFeatureMap", reps=1)
        if REAL_DEVICE:
            self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
            self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
            self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
            self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
            self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
            self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
            self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
            self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
            self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
        else:
            self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
            self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
            self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
            self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
            self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
            self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
            self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
            self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
            self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
        self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        # EstimatorQNNはNG
        self.qnn = EstimatorQNN(
            # estimator=Estimator(backend, options={"shots": 1000}),
            estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={"shots":1000, "seed_simulator": seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=self.feature_map_parameters,
            weight_params=self.ansatz_parameters,
            input_gradients=True,
        )
        # [goal]input_gradients=False and no using TorchConnector
        self.qcnn = nn.Sequential(
            TorchConnector(self.qnn),
            nn.Tanh()
        )

    def forward(self, state):
        state = state*self.state_weight
        x = torch.flatten(state)
        x = x.repeat(1, 2)
        x = self.qcnn(x)
        return x

class QCNN_T(nn.Module):
    def __init__(self, board_size=3, state_weight=1):
        super().__init__()
        self.board_size=board_size
        self.state_weight=state_weight

        self.backend_num_qubits = set_backend.backend.num_qubits

        self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn("TPE", reps=1)
        if REAL_DEVICE:
            self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
            self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
            self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
            self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
            self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
            self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
            self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
            self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
            self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
        else:
            self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
            self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
            self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
            self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
            self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
            self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
            self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
            self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
            self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
        self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        # EstimatorQNNはNG
        self.qnn = EstimatorQNN(
            # estimator=Estimator(backend, options={"shots": 1000}),
            estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={"shots":1000, "seed_simulator": seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=self.feature_map_parameters,
            weight_params=self.ansatz_parameters,
            input_gradients=True,
        )
        # [goal]input_gradients=False and no using TorchConnector
        self.qcnn = nn.Sequential(
            TorchConnector(self.qnn),
            nn.Tanh()
        )

    def forward(self, state):
        state = state*self.state_weight
        x = torch.flatten(state)
        x = x.repeat(1, 2)
        x = self.qcnn(x)
        return x

class QCNN_H(nn.Module):
    def __init__(self, board_size=3, state_weight=1):
        super().__init__()
        self.board_size=board_size
        self.state_weight=state_weight

        self.backend_num_qubits = set_backend.backend.num_qubits

        self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn("HEE", reps=1)
        if REAL_DEVICE:
            self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
            self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
            self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
            self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
            self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
            self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
            self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
            self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
            self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
        else:
            self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
            self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
            self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
            self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
            self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
            self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
            self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
            self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
            self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
        self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
        self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        # EstimatorQNNはNG
        self.qnn = EstimatorQNN(
            # estimator=Estimator(backend, options={"shots": 1000}),
            estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={"shots":1000, "seed_simulator": seed}),
            circuit=self.isa_circuit,
            observables=self.observable,
            input_params=self.feature_map_parameters,
            weight_params=self.ansatz_parameters,
            input_gradients=True,
        )
        # [goal]input_gradients=False and no using TorchConnector
        self.qcnn = nn.Sequential(
            TorchConnector(self.qnn),
            nn.Tanh()
        )

    def forward(self, state):
        state = state*self.state_weight
        x = torch.flatten(state)
        x = x.repeat(1, 2)
        x = self.qcnn(x)
        return x

"""--------------------------------------------------------------------------------------------------------------------------"""
#%% networkとの接続した場合を検証する typeという引数が追加されている
# class CQCCNN_Z_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, n_qubits = 10, noised=True):
#         super().__init__()
#         self.board_size = board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.backend_num_qubits = self.backend.num_qubits
#         self.n_qubits = n_qubits

#         self.linear1 = nn.Linear(9, self.n_qubits, bias=True)
#         self.linear2 = nn.Linear(self.n_qubits//2, 9, bias=True)

#         if type == 1:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(self.n_qubits, ["conv", "pool"], [i for i in range(0, self.n_qubits, 2)], [i for i in range(1, self.n_qubits, 2)]).create_qcnn_network1("ZFeatureMap", reps=1)
#         elif type == 2:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(self.n_qubits, ["conv", "pool"], [i for i in range(0, self.n_qubits, 2)], [i for i in range(1, self.n_qubits, 2)]).create_qcnn_network2("ZFeatureMap", reps=1)

#         self.observable = []
#         for i in range(self.n_qubits-1, -1, -1):
#             s_op = ""
#             for j in range(self.n_qubits):
#                 if i%2 == 0:
#                     if j == i:
#                         s_op += "Z"
#                     else:
#                         s_op += "I"
#             if i%2 == 0:
#                 self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))

#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)

#         self.qnn = EstimatorQNN(
#             estimator= Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         self.cqnn = nn.Sequential(
#             self.linear1,
#             TorchConnector(self.qnn),
#             self.linear2,
#             nn.Tanh()
#         )

#     def forward(self, state):
#         x = self.cqnn(state)
#         return x

# #%%
# class CQCCNN_ZZ_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, n_qubits = 10, noised=True):
#         super().__init__()
#         self.board_size = board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.backend_num_qubits = self.backend.num_qubits
#         self.n_qubits = n_qubits

#         self.linear1 = nn.Linear(9, self.n_qubits, bias=True)
#         self.linear2 = nn.Linear(self.n_qubits//2, 9, bias=True)

#         if type == 1:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(self.n_qubits, ["conv", "pool"], [i for i in range(0, self.n_qubits, 2)], [i for i in range(1, self.n_qubits, 2)]).create_qcnn_network1("ZZFeatureMap", reps=1)
#         elif type == 2:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(self.n_qubits, ["conv", "pool"], [i for i in range(0, self.n_qubits, 2)], [i for i in range(1, self.n_qubits, 2)]).create_qcnn_network2("ZZFeatureMap", reps=1)

#         self.observable = []
#         for i in range(self.n_qubits-1, -1, -1):
#             s_op = ""
#             for j in range(self.n_qubits):
#                 if i%2 == 0:
#                     if j == i:
#                         s_op += "Z"
#                     else:
#                         s_op += "I"
#             if i%2 == 0:
#                 self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))

#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)

#         self.qnn = EstimatorQNN(
#             estimator= Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         self.cqnn = nn.Sequential(
#             self.linear1,
#             TorchConnector(self.qnn),
#             self.linear2,
#             nn.Tanh()
#         )

#     def forward(self, state):
#         x = self.cqnn(state)
#         return x

# #%%
# class CQCCNN_T_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, n_qubits = 10, noised=True):
#         super().__init__()
#         self.board_size = board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.backend_num_qubits = self.backend.num_qubits
#         self.n_qubits = n_qubits

#         self.linear1 = nn.Linear(9, self.n_qubits, bias=True)
#         self.linear2 = nn.Linear(self.n_qubits//2, 9, bias=True)

#         if type == 1:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(self.n_qubits, ["conv", "pool"], [i for i in range(0, self.n_qubits, 2)], [i for i in range(1, self.n_qubits, 2)]).create_qcnn_network1("TPE", reps=1)
#         elif type == 2:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(self.n_qubits, ["conv", "pool"], [i for i in range(0, self.n_qubits, 2)], [i for i in range(1, self.n_qubits, 2)]).create_qcnn_network2("TPE", reps=1)

#         self.observable = []
#         for i in range(self.n_qubits-1, -1, -1):
#             s_op = ""
#             for j in range(self.n_qubits):
#                 if i%2 == 0:
#                     if j == i:
#                         s_op += "Z"
#                     else:
#                         s_op += "I"
#             if i%2 == 0:
#                 self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))

#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)

#         self.qnn = EstimatorQNN(
#             estimator= Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         self.cqnn = nn.Sequential(
#             self.linear1,
#             TorchConnector(self.qnn),
#             self.linear2,
#             nn.Tanh()
#         )

#     def forward(self, state):
#         x = self.cqnn(state)
#         return x

# #%%
# class CQCCNN_H_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, n_qubits = 10, noised=True):
#         super().__init__()
#         self.board_size = board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.backend_num_qubits = self.backend.num_qubits
#         self.n_qubits = n_qubits

#         self.linear1 = nn.Linear(9, self.n_qubits, bias=True)
#         self.linear2 = nn.Linear(self.n_qubits//2, 9, bias=True)

#         if type == 1:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(self.n_qubits, ["conv", "pool"], [i for i in range(0, self.n_qubits, 2)], [i for i in range(1, self.n_qubits, 2)]).create_qcnn_network1("HEE", reps=1)
#         elif type == 2:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(self.n_qubits, ["conv", "pool"], [i for i in range(0, self.n_qubits, 2)], [i for i in range(1, self.n_qubits, 2)]).create_qcnn_network2("HEE", reps=1)

#         self.observable = []
#         for i in range(self.n_qubits-1, -1, -1):
#             s_op = ""
#             for j in range(self.n_qubits):
#                 if i%2 == 0:
#                     if j == i:
#                         s_op += "Z"
#                     else:
#                         s_op += "I"
#             if i%2 == 0:
#                 self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))

#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)

#         self.qnn = EstimatorQNN(
#             estimator= Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         self.cqnn = nn.Sequential(
#             self.linear1,
#             TorchConnector(self.qnn),
#             self.linear2,
#             nn.Tanh()
#         )

#     def forward(self, state):
#         x = self.cqnn(state)
#         return x

#%%
class CQCNN_sampler_ZR_network(nn.Module):
    def __init__(self, type = 1, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        
        self.backend = set_backend.backend

        self.num_qubits = num_qubits

        if type == 1:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).ZR_network1(featuremap_reps, ansatz_reps)
        elif type == 2:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).ZR_network2(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%%
class CQCNN_sampler_ZZR_network(nn.Module):
    def __init__(self, type = 1, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        
        self.backend = set_backend.backend

        self.num_qubits = num_qubits

        if type == 1:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).ZZR_network1(featuremap_reps, ansatz_reps)
        elif type == 2:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).ZZR_network2(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%%
class CQCNN_sampler_TR_network(nn.Module):
    def __init__(self, type = 1, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        
        self.backend = set_backend.backend

        self.num_qubits = num_qubits

        if type == 1:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).TR_network1(featuremap_reps, ansatz_reps)
        elif type == 2:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).TR_network2(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%%
class CQCNN_sampler_HR_network(nn.Module):
    def __init__(self, type = 1, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        
        self.backend = set_backend.backend

        self.num_qubits = num_qubits

        if type == 1:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).HR_network1(featuremap_reps, ansatz_reps)
        elif type == 2:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).HR_network2(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%%
class CQCNN_sampler_ZE_network(nn.Module):
    def __init__(self, type = 1, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        
        self.backend = set_backend.backend

        self.num_qubits = num_qubits

        if type == 1:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).ZE_network1(featuremap_reps, ansatz_reps)
        elif type == 2:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).ZE_network2(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%%
class CQCNN_sampler_ZZE_network(nn.Module):
    def __init__(self, type = 1, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        
        self.backend = set_backend.backend

        self.num_qubits = num_qubits

        if type == 1:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).ZZE_network1(featuremap_reps, ansatz_reps)
        elif type == 2:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).ZZE_network2(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%%
class CQCNN_sampler_TE_network(nn.Module):
    def __init__(self, type = 1, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        
        self.backend = set_backend.backend

        self.num_qubits = num_qubits

        if type == 1:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).TE_network1(featuremap_reps, ansatz_reps)
        elif type == 2:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).TE_network2(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%%
class CQCNN_sampler_HE_network(nn.Module):
    def __init__(self, type = 1, board_size=3, num_qubits=7, featuremap_reps=1, ansatz_reps=1):
        super().__init__()
        self.board_size = board_size
        
        self.backend = set_backend.backend

        self.num_qubits = num_qubits
        if type == 1:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).HE_network1(featuremap_reps, ansatz_reps)
        elif type == 2:
            self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_qubits).HE_network2(featuremap_reps, ansatz_reps)

        self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        self.isa_circuit = self.pm.run(self.circuit)
        self.qnn = SamplerQNN(
            sampler=Sampler(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendSampler(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
            circuit=self.isa_circuit,
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

#%%
# class CQCNN_estimator_ZR_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1, noised=True):
#         super().__init__()
#         self.board_size = board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.num_of_qubit = num_of_qubit
#         self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
#         self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

#         if type == 1:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).ZR_network1(featuremap_reps, ansatz_reps)
#         elif type == 2:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).ZR_network2(featuremap_reps, ansatz_reps)

#         self.observable = []
#         for i in range(self.num_of_qubit-1, -1, -1):
#             s_op = ""
#             for j in range(self.num_of_qubit):
#                 if j == i:
#                     s_op += "Z"
#                 else:
#                     s_op += "I"
#             self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         self.qnn = EstimatorQNN(
#             estimator=Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=feature_map_parameters,
#             weight_params=ansatz_parameters,
#             input_gradients=True,
#         )
#         self.cqnn = nn.Sequential(
#             self.linear1,
#             TorchConnector(self.qnn),
#             self.linear2,
#             nn.Tanh()
#         )

#     def forward(self, state):
#         x = self.cqnn(state)
#         return x

# #%%
# class CQCNN_estimator_ZZR_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1, noised=True):
#         super().__init__()
#         self.board_size = board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.num_of_qubit = num_of_qubit
#         self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
#         self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

#         if type == 1:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).ZZR_network1(featuremap_reps, ansatz_reps)
#         elif type == 2:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).ZZR_network2(featuremap_reps, ansatz_reps)

#         self.observable = []
#         for i in range(self.num_of_qubit-1, -1, -1):
#             s_op = ""
#             for j in range(self.num_of_qubit):
#                 if j == i:
#                     s_op += "Z"
#                 else:
#                     s_op += "I"
#             self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         self.qnn = EstimatorQNN(
#             estimator=Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=feature_map_parameters,
#             weight_params=ansatz_parameters,
#             input_gradients=True,
#         )
#         self.cqnn = nn.Sequential(
#             self.linear1,
#             TorchConnector(self.qnn),
#             self.linear2,
#             nn.Tanh()
#         )

#     def forward(self, state):
#         x = self.cqnn(state)
#         return x

# #%%
# class CQCNN_estimator_TR_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1, noised=True):
#         super().__init__()
#         self.board_size = board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.num_of_qubit = num_of_qubit
#         self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
#         self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

#         if type == 1:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).TR_network1(featuremap_reps, ansatz_reps)
#         elif type == 2:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).TR_network2(featuremap_reps, ansatz_reps)

#         self.observable = []
#         for i in range(self.num_of_qubit-1, -1, -1):
#             s_op = ""
#             for j in range(self.num_of_qubit):
#                 if j == i:
#                     s_op += "Z"
#                 else:
#                     s_op += "I"
#             self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         self.qnn = EstimatorQNN(
#             estimator=Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=feature_map_parameters,
#             weight_params=ansatz_parameters,
#             input_gradients=True,
#         )
#         self.cqnn = nn.Sequential(
#             self.linear1,
#             TorchConnector(self.qnn),
#             self.linear2,
#             nn.Tanh()
#         )

#     def forward(self, state):
#         x = self.cqnn(state)
#         return x

# #%%
# class CQCNN_estimator_HR_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1, noised=True):
#         super().__init__()
#         self.board_size = board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.num_of_qubit = num_of_qubit
#         self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
#         self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

#         if type == 1:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).HR_network1(featuremap_reps, ansatz_reps)
#         elif type == 2:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).HR_network2(featuremap_reps, ansatz_reps)

#         self.observable = []
#         for i in range(self.num_of_qubit-1, -1, -1):
#             s_op = ""
#             for j in range(self.num_of_qubit):
#                 if j == i:
#                     s_op += "Z"
#                 else:
#                     s_op += "I"
#             self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         self.qnn = EstimatorQNN(
#             estimator=Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=feature_map_parameters,
#             weight_params=ansatz_parameters,
#             input_gradients=True,
#         )
#         self.cqnn = nn.Sequential(
#             self.linear1,
#             TorchConnector(self.qnn),
#             self.linear2,
#             nn.Tanh()
#         )

#     def forward(self, state):
#         x = self.cqnn(state)
#         return x

# #%%
# class CQCNN_estimator_ZE_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1, noised=True):
#         super().__init__()
#         self.board_size = board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.num_of_qubit = num_of_qubit
#         self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
#         self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

#         if type == 1:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).ZE_network1(featuremap_reps, ansatz_reps)
#         elif type == 2:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).ZE_network2(featuremap_reps, ansatz_reps)

#         self.observable = []
#         for i in range(self.num_of_qubit-1, -1, -1):
#             s_op = ""
#             for j in range(self.num_of_qubit):
#                 if j == i:
#                     s_op += "Z"
#                 else:
#                     s_op += "I"
#             self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         self.qnn = EstimatorQNN(
#             estimator=Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=feature_map_parameters,
#             weight_params=ansatz_parameters,
#             input_gradients=True,
#         )
#         self.cqnn = nn.Sequential(
#             self.linear1,
#             TorchConnector(self.qnn),
#             self.linear2,
#             nn.Tanh()
#         )

#     def forward(self, state):
#         x = self.cqnn(state)
#         return x

# #%%
# class CQCNN_estimator_ZZE_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1, noised=True):
#         super().__init__()
#         self.board_size = board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.num_of_qubit = num_of_qubit
#         self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
#         self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

#         if type == 1:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).ZZE_network1(featuremap_reps, ansatz_reps)
#         elif type == 2:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).ZZE_network2(featuremap_reps, ansatz_reps)

#         self.observable = []
#         for i in range(self.num_of_qubit-1, -1, -1):
#             s_op = ""
#             for j in range(self.num_of_qubit):
#                 if j == i:
#                     s_op += "Z"
#                 else:
#                     s_op += "I"
#             self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         self.qnn = EstimatorQNN(
#             estimator=Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=feature_map_parameters,
#             weight_params=ansatz_parameters,
#             input_gradients=True,
#         )
#         self.cqnn = nn.Sequential(
#             self.linear1,
#             TorchConnector(self.qnn),
#             self.linear2,
#             nn.Tanh()
#         )

#     def forward(self, state):
#         x = self.cqnn(state)
#         return x

# #%%
# class CQCNN_estimator_TE_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1, noised=True):
#         super().__init__()
#         self.board_size = board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.num_of_qubit = num_of_qubit
#         self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
#         self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

#         if type == 1:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).TE_network1(featuremap_reps, ansatz_reps)
#         elif type == 2:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).TE_network2(featuremap_reps, ansatz_reps)

#         self.observable = []
#         for i in range(self.num_of_qubit-1, -1, -1):
#             s_op = ""
#             for j in range(self.num_of_qubit):
#                 if j == i:
#                     s_op += "Z"
#                 else:
#                     s_op += "I"
#             self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         self.qnn = EstimatorQNN(
#             estimator=Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=feature_map_parameters,
#             weight_params=ansatz_parameters,
#             input_gradients=True,
#         )
#         self.cqnn = nn.Sequential(
#             self.linear1,
#             TorchConnector(self.qnn),
#             self.linear2,
#             nn.Tanh()
#         )

#     def forward(self, state):
#         x = self.cqnn(state)
#         return x

# #%%
# class CQCNN_estimator_HE_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, num_of_qubit=10, featuremap_reps=1, ansatz_reps=1, noised=True):
#         super().__init__()
#         self.board_size = board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.num_of_qubit = num_of_qubit
#         self.linear1 = nn.Linear(9, self.num_of_qubit, bias=True)
#         self.linear2 = nn.Linear(self.num_of_qubit, 9, bias=True)

#         if type == 1:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).HE_network1(featuremap_reps, ansatz_reps)
#         elif type == 2:
#             self.circuit, feature_map_parameters, ansatz_parameters = QIComponent(self.num_of_qubit).HE_network2(featuremap_reps, ansatz_reps)

#         self.observable = []
#         for i in range(self.num_of_qubit-1, -1, -1):
#             s_op = ""
#             for j in range(self.num_of_qubit):
#                 if j == i:
#                     s_op += "Z"
#                 else:
#                     s_op += "I"
#             self.observable.append(SparsePauliOp.from_list([(s_op, 1)]))
#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         self.qnn = EstimatorQNN(
#             estimator=Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={'shots':1000, 'seed_simulator':seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=feature_map_parameters,
#             weight_params=ansatz_parameters,
#             input_gradients=True,
#         )
#         self.cqnn = nn.Sequential(
#             self.linear1,
#             TorchConnector(self.qnn),
#             self.linear2,
#             nn.Tanh()
#         )

#     def forward(self, state):
#         x = self.cqnn(state)
#         return x

# #%%
# class QCNN_Z_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, state_weight=1, noised=True):
#         super().__init__()
#         self.board_size=board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.state_weight=state_weight

#         self.backend_num_qubits = set_backend.backend_noise.num_qubits

#         if type == 1:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network1("ZFeatureMap", reps=1)
#         elif type == 2:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network2("ZFeatureMap", reps=1)

#         if REAL_DEVICE:
#             self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
#         else:
#             self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
#         self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         # EstimatorQNNはNG
#         self.qnn = EstimatorQNN(
#             estimator=Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={"shots":1000, "seed_simulator": seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         # [goal]input_gradients=False and no using TorchConnector
#         self.qcnn = nn.Sequential(
#             TorchConnector(self.qnn),
#             nn.Tanh()
#         )

#     def forward(self, state):
#         state = state*self.state_weight
#         x = torch.flatten(state)
#         x = x.repeat(1, 2)
#         x = self.qcnn(x)
#         return x

# #%%
# class QCNN_ZZ_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, state_weight=1, noised=True):
#         super().__init__()
#         self.board_size=board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.state_weight=state_weight

#         self.backend_num_qubits = set_backend.backend_noise.num_qubits

#         if type == 1:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network1("ZZFeatureMap", reps=1)
#         elif type == 2:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network2("ZZFeatureMap", reps=1)
#         if REAL_DEVICE:
#             self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
#         else:
#             self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
#         self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         # EstimatorQNNはNG
#         self.qnn = EstimatorQNN(
#             estimator=Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={"shots":1000, "seed_simulator": seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         # [goal]input_gradients=False and no using TorchConnector
#         self.qcnn = nn.Sequential(
#             TorchConnector(self.qnn),
#             nn.Tanh()
#         )

#     def forward(self, state):
#         state = state*self.state_weight
#         x = torch.flatten(state)
#         x = x.repeat(1, 2)
#         x = self.qcnn(x)
#         return x

# #%%
# class QCNN_T_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, state_weight=1, noised=True):
#         super().__init__()
#         self.board_size=board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.state_weight=state_weight

#         self.backend_num_qubits = set_backend.backend_noise.num_qubits

#         if type == 1:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network1("TPE", reps=1)
#         elif type == 2:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network2("TPE", reps=1)
#         if REAL_DEVICE:
#             self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
#         else:
#             self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
#         self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         # EstimatorQNNはNG
#         self.qnn = EstimatorQNN(
#             estimator=Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={"shots":1000, "seed_simulator": seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         # [goal]input_gradients=False and no using TorchConnector
#         self.qcnn = nn.Sequential(
#             TorchConnector(self.qnn),
#             nn.Tanh()
#         )

#     def forward(self, state):
#         state = state*self.state_weight
#         x = torch.flatten(state)
#         x = x.repeat(1, 2)
#         x = self.qcnn(x)
#         return x

# #%%
# class QCNN_H_network(nn.Module):
#     def __init__(self, type = 1, board_size=3, state_weight=1, noised=True):
#         super().__init__()
#         self.board_size=board_size
        
#         if noised:
#             self.backend = NETWORK
#         else:
#             self.backend = set_backend.backend

#         self.state_weight=state_weight

#         self.backend_num_qubits = set_backend.backend_noise.num_qubits

#         if type == 1:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network1("HEE", reps=1)
#         elif type == 2:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network2("HEE", reps=1)
#         if REAL_DEVICE:
#             self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
#         else:
#             self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
#         self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
#         self.pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         # EstimatorQNNはNG
#         self.qnn = EstimatorQNN(
#             estimator=Estimator(backend=self.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=self.backend, options={"shots":1000, "seed_simulator": seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         # [goal]input_gradients=False and no using TorchConnector
#         self.qcnn = nn.Sequential(
#             TorchConnector(self.qnn),
#             nn.Tanh()
#         )

#     def forward(self, state):
#         state = state*self.state_weight
#         x = torch.flatten(state)
#         x = x.repeat(1, 2)
#         x = self.qcnn(x)
#         return x

# '''---------------------stateをいくつか指定できるNNを作る----------------------------------'''

# class QCNN_Z_different_state(nn.Module):
#     def __init__(self, board_size=3, state_weight=1, state_weight2=1):
#         super().__init__()
#         self.board_size=board_size
#         self.state_weight=state_weight
#         self.state_weight2=state_weight2

#         self.backend_num_qubits = set_backend.backend.num_qubits

#         self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn("ZFeatureMap", reps=1)
#         if REAL_DEVICE:
#             self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
#         else:
#             self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
#         self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
#         self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         # EstimatorQNNはNG
#         self.qnn = EstimatorQNN(
#             # estimator=Estimator(backend, options={"shots": 1000}),
#             estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={"shots":1000, "seed_simulator": seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         # [goal]input_gradients=False and no using TorchConnector
#         self.qcnn = nn.Sequential(
#             TorchConnector(self.qnn),
#             nn.Tanh()
#         )

#     def forward(self, state, state2):
#         state = state*self.state_weight
#         state2 = state2*self.state_weight2
#         state = torch.flatten(state)
#         state2 = torch.flatten(state2)
#         '''盤面状態の入れ方とかもっと工夫できそうな気がするのですよ'''
#         x = torch.cat([state, state2], dim=0)
#         x = self.qcnn(x)
#         return x

# class QCNN_ZZ_different_state(nn.Module):
#     def __init__(self, board_size=3, state_weight=1, state_weight2=1):
#         super().__init__()
#         self.board_size=board_size
#         self.state_weight=state_weight
#         self.state_weight2=state_weight2

#         self.backend_num_qubits = set_backend.backend.num_qubits

#         self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn("ZZFeatureMap", reps=1)
#         if REAL_DEVICE:
#             self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
#         else:
#             self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
#         self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
#         self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         # EstimatorQNNはNG
#         self.qnn = EstimatorQNN(
#             # estimator=Estimator(backend, options={"shots": 1000}),
#             estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={"shots":1000, "seed_simulator": seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         # [goal]input_gradients=False and no using TorchConnector
#         self.qcnn = nn.Sequential(
#             TorchConnector(self.qnn),
#             nn.Tanh()
#         )

#     def forward(self, state, state2):
#         state = state*self.state_weight
#         state2 = state2*self.state_weight2
#         state = torch.flatten(state)
#         state2 = torch.flatten(state2)
#         x = torch.cat([state, state2], dim=0)
#         x = self.qcnn(x)
#         return x

# class QCNN_T_different_state(nn.Module):
#     def __init__(self, board_size=3, state_weight=1, state_weight2=1):
#         super().__init__()
#         self.board_size=board_size
#         self.state_weight=state_weight
#         self.state_weight2=state_weight2

#         self.backend_num_qubits = set_backend.backend.num_qubits

#         self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn("TPE", reps=1)
#         if REAL_DEVICE:
#             self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
#         else:
#             self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
#         self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
#         self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         # EstimatorQNNはNG
#         self.qnn = EstimatorQNN(
#             # estimator=Estimator(backend, options={"shots": 1000}),
#             estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={"shots":1000, "seed_simulator": seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         # [goal]input_gradients=False and no using TorchConnector
#         self.qcnn = nn.Sequential(
#             TorchConnector(self.qnn),
#             nn.Tanh()
#         )

#     def forward(self, state, state2):
#         state = state*self.state_weight
#         state2 = state*self.state_weight2
#         state = torch.flatten(state)
#         state2 = torch.flatten(state2)
#         x = torch.cat([state, state2], dim=0)
#         x = self.qcnn(x)
#         return x

# class QCNN_H_different_state(nn.Module):
#     def __init__(self, board_size=3, state_weight=1, state_weight2=1):
#         super().__init__()
#         self.board_size=board_size
#         self.state_weight=state_weight
#         self.state_weight2=state_weight2

#         self.backend_num_qubits = set_backend.backend.num_qubits

#         self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn("HEE", reps=1)
#         if REAL_DEVICE:
#             self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
#         else:
#             self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
#         self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
#         self.pm = generate_preset_pass_manager(backend=set_backend.backend, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         # EstimatorQNNはNG
#         self.qnn = EstimatorQNN(
#             # estimator=Estimator(backend, options={"shots": 1000}),
#             estimator=Estimator(backend=set_backend.backend, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend, options={"shots":1000, "seed_simulator": seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         # [goal]input_gradients=False and no using TorchConnector
#         self.qcnn = nn.Sequential(
#             TorchConnector(self.qnn),
#             nn.Tanh()
#         )

#     def forward(self, state, state2):
#         state = state*self.state_weight
#         state2 = state*self.state_weight2
#         state = torch.flatten(state)
#         state2 = torch.flatten(state2)
#         x = torch.cat([state, state2], dim=0)
#         x = self.qcnn(x)
#         return x

# #%%
# class QCNN_Z_network_different_state(nn.Module):
#     def __init__(self, type = 1, board_size=3, state_weight=1, state_weight2=1):
#         super().__init__()
#         self.board_size=board_size
#         self.state_weight=state_weight
#         self.state_weight2=state_weight2

#         self.backend_num_qubits = set_backend.backend_noise.num_qubits

#         if type == 1:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network1("ZFeatureMap", reps=1)
#         elif type == 2:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network2("ZFeatureMap", reps=1)

#         if REAL_DEVICE:
#             self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
#         else:
#             self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
#         self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
#         self.pm = generate_preset_pass_manager(backend=set_backend.backend_noise, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         # EstimatorQNNはNG
#         self.qnn = EstimatorQNN(
#             # estimator=Estimator(backend, options={"shots": 1000}),
#             estimator=Estimator(backend=set_backend.backend_noise, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend_noise, options={"shots":1000, "seed_simulator": seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         # [goal]input_gradients=False and no using TorchConnector
#         self.qcnn = nn.Sequential(
#             TorchConnector(self.qnn),
#             nn.Tanh()
#         )

#     def forward(self, state, state2):
#         state = state*self.state_weight
#         state2 = state2*self.state_weight2
#         state = torch.flatten(state)
#         state2 = torch.flatten(state2)
#         x = torch.cat([state, state2], dim=0)
#         x = self.qcnn(x)
#         return x

# #%%
# class QCNN_ZZ_network_different_state(nn.Module):
#     def __init__(self, type = 1, board_size=3, state_weight=1, state_weight2=1):
#         super().__init__()
#         self.board_size=board_size
#         self.state_weight=state_weight
#         self.state_weight2=state_weight2

#         self.backend_num_qubits = set_backend.backend_noise.num_qubits

#         if type == 1:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network1("ZZFeatureMap", reps=1)
#         elif type == 2:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network2("ZZFeatureMap", reps=1)
#         if REAL_DEVICE:
#             self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
#         else:
#             self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
#         self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
#         self.pm = generate_preset_pass_manager(backend=set_backend.backend_noise, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         # EstimatorQNNはNG
#         self.qnn = EstimatorQNN(
#             # estimator=Estimator(backend, options={"shots": 1000}),
#             estimator=Estimator(backend=set_backend.backend_noise, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend_noise, options={"shots":1000, "seed_simulator": seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         # [goal]input_gradients=False and no using TorchConnector
#         self.qcnn = nn.Sequential(
#             TorchConnector(self.qnn),
#             nn.Tanh()
#         )

#     def forward(self, state, state2):
#         state = state*self.state_weight
#         state2 = state2*self.state_weight2
#         state = torch.flatten(state)
#         state2 = torch.flatten(state2)
#         x = torch.cat([state, state2], dim=0)
#         x = self.qcnn(x)
#         return x

# #%%
# class QCNN_T_network_different_state(nn.Module):
#     def __init__(self, type = 1, board_size=3, state_weight=1, state_weight2=1):
#         super().__init__()
#         self.board_size=board_size
#         self.state_weight=state_weight
#         self.state_weight2=state_weight2

#         self.backend_num_qubits = set_backend.backend_noise.num_qubits

#         if type == 1:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network1("TPE", reps=1)
#         elif type == 2:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network2("TPE", reps=1)
#         if REAL_DEVICE:
#             self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
#         else:
#             self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
#         self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
#         self.pm = generate_preset_pass_manager(backend=set_backend.backend_noise, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         # EstimatorQNNはNG
#         self.qnn = EstimatorQNN(
#             # estimator=Estimator(backend, options={"shots": 1000}),
#             estimator=Estimator(backend=set_backend.backend_noise, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend_noise, options={"shots":1000, "seed_simulator": seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         # [goal]input_gradients=False and no using TorchConnector
#         self.qcnn = nn.Sequential(
#             TorchConnector(self.qnn),
#             nn.Tanh()
#         )

#     def forward(self, state, state2):
#         state = state*self.state_weight
#         state2 = state2*self.state_weight2
#         state = torch.flatten(state)
#         state2 = torch.flatten(state2)
#         x = torch.cat([state, state2], dim=0)
#         x = self.qcnn(x)
#         return x

# #%%
# class QCNN_H_network_different_state(nn.Module):
#     def __init__(self, type = 1, board_size=3, state_weight=1, state_weight2=1):
#         super().__init__()
#         self.board_size=board_size
#         self.state_weight=state_weight
#         self.state_weight2=state_weight2

#         self.backend_num_qubits = set_backend.backend_noise.num_qubits

#         if type == 1:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network1("HEE", reps=1)
#         elif type == 2:
#             self.circuit, self.feature_map_parameters, self.ansatz_parameters = QCNNComponent(18, ["conv", "pool"], [0, 2, 4, 6, 8, 10, 12, 14, 16], [1, 3, 5, 7, 9, 11, 13, 15, 17]).create_qcnn_network2("HEE", reps=1)
#         if REAL_DEVICE:
#             self.op1 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("I"*(self.backend_num_qubits-18) + "IIIIIIIIIIIIIIIIZI", 1)])
#         else:
#             self.op1 = SparsePauliOp.from_list([("ZIIIIIIIIIIIIIIIII", 1)])
#             self.op2 = SparsePauliOp.from_list([("IIZIIIIIIIIIIIIIII", 1)])
#             self.op3 = SparsePauliOp.from_list([("IIIIZIIIIIIIIIIIII", 1)])
#             self.op4 = SparsePauliOp.from_list([("IIIIIIZIIIIIIIIIII", 1)])
#             self.op5 = SparsePauliOp.from_list([("IIIIIIIIZIIIIIIIII", 1)])
#             self.op6 = SparsePauliOp.from_list([("IIIIIIIIIIZIIIIIII", 1)])
#             self.op7 = SparsePauliOp.from_list([("IIIIIIIIIIIIZIIIII", 1)])
#             self.op8 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIZIII", 1)])
#             self.op9 = SparsePauliOp.from_list([("IIIIIIIIIIIIIIIIZI", 1)])
#         self.observable = [self.op9, self.op8, self.op7, self.op6, self.op5, self.op4, self.op3, self.op2, self.op1]
#         self.pm = generate_preset_pass_manager(backend=set_backend.backend_noise, optimization_level=1)
#         self.isa_circuit = self.pm.run(self.circuit)
#         # EstimatorQNNはNG
#         self.qnn = EstimatorQNN(
#             # estimator=Estimator(backend, options={"shots": 1000}),
#             estimator=Estimator(backend=set_backend.backend_noise, options={'shots':1000}) if REAL_DEVICE else BackendEstimator(backend=set_backend.backend_noise, options={"shots":1000, "seed_simulator": seed}),
#             circuit=self.isa_circuit,
#             observables=self.observable,
#             input_params=self.feature_map_parameters,
#             weight_params=self.ansatz_parameters,
#             input_gradients=True,
#         )
#         # [goal]input_gradients=False and no using TorchConnector
#         self.qcnn = nn.Sequential(
#             TorchConnector(self.qnn),
#             nn.Tanh()
#         )

#     def forward(self, state, state2):
#         state = state*self.state_weight
#         state2 = state2*self.state_weight2
#         state = torch.flatten(state)
#         state2 = torch.flatten(state2)
#         x = torch.cat([state, state2], dim=0)
#         x = self.qcnn(x)
#         return x