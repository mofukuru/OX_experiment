import collections
import random

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import (
    EfficientSU2,
    PauliFeatureMap,
    RealAmplitudes,
    ZFeatureMap,
    ZZFeatureMap,
)


class QCNNComponent:
    """
    Making QCNN circuit.

    Attributes:
        n_qubits: int
            num of qubits
        param_prefix: list
            prefix of the parameter that displays in the circuit
        sources: list
            convolution from the source
        sinks: list
            convolution to the sinks
    """

    def __init__(
        self, n_qubits: int, param_prefix: list, sources: list, sinks: list
    ):
        self.n_qubits = n_qubits
        self.param_prefix = param_prefix
        self.sources = sources
        self.sinks = sinks

    def conv_circuit(self, params):
        """
        Making convolutional circuit.

        Parameters:
            params
                variables that determines rotation degrees

        Returns:
            target:
                convolutional circuit
        """
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    def conv_layer(self):
        """
        Making convolutional layer according to the conv_circuit function above.

        Returns:
            qc:
                convolutional circuits
        """
        qc = QuantumCircuit(self.n_qubits, name=self.param_prefix[0])
        qubits = list(range(self.n_qubits))
        param_index = 0
        params = ParameterVector(
            self.param_prefix[0], length=self.n_qubits * 3
        )
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(
                self.conv_circuit(params[param_index : (param_index + 3)]),
                [q1, q2],
            )
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(
                self.conv_circuit(params[param_index : (param_index + 3)]),
                [q1, q2],
            )
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(self.n_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def pool_circuit(self, params):
        """
        Making pool circuit.

        Parameters:
            params:
                variables that determines rotation degrees

        Returns:
            target:
                pooling circuit
        """
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def pool_layer(self):
        """
        Making pooling layer according to the pool_circuit function above.

        Returns:
            qc:
                pooling circuits
        """
        n_qubits = len(self.sources) + len(self.sinks)
        qc = QuantumCircuit(n_qubits, name=self.param_prefix[1])
        param_index = 0
        params = ParameterVector(
            self.param_prefix[1], length=n_qubits // 2 * 3
        )
        for source, sink in zip(self.sources, self.sinks):
            qc = qc.compose(
                self.pool_circuit(params[param_index : (param_index + 3)]),
                [source, sink],
            )
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(n_qubits)
        qc.append(qc_inst, range(n_qubits))
        return qc

    def make_qcnn_circuit(self, embedding_type: str, reps: int = 1):
        """
        Assembling circuits made in conv_layer and pool_layer function.

        Parameters:
            embedding_type: str
                determine embedding circuit
            reps: int
                num of repeats

        Returns:
            circuit:
                QCNN circuit
            feature_map.parameters:
                parameters of circuit in feature_map
            ansatz.parameters:
                parameters of circuit in ansatz
        """
        data_map = lambda x: x[0] * np.pi / 2
        if embedding_type == "ZFeatureMap":
            feature_map = ZFeatureMap(
                self.n_qubits, reps=reps, data_map_func=data_map
            )
        elif embedding_type == "ZZFeatureMap":
            feature_map = ZZFeatureMap(
                self.n_qubits, reps=reps, data_map_func=data_map
            )
        elif embedding_type == "TPE":
            feature_map = QNNComponent(self.n_qubits).TPE(reps)
        elif embedding_type == "HEE":
            feature_map = QNNComponent(self.n_qubits).HEE(reps)
        else:
            print("Unknown Embedding Type.")
            exit(1)

        ansatz = QuantumCircuit(self.n_qubits, name="Ansatz")
        ansatz.compose(
            self.conv_layer(), list(range(self.n_qubits)), inplace=True
        )
        ansatz.compose(
            self.pool_layer(), list(range(self.n_qubits)), inplace=True
        )
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(feature_map, range(self.n_qubits), inplace=True)
        circuit.compose(ansatz, range(self.n_qubits), inplace=True)
        return circuit, feature_map.parameters, ansatz.parameters


class QNNComponent:
    """
    Making QNN circuit.

    Attributes:
        n_qubits: int
            num of qubits
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    def TPE(self, reps: int = 1) -> QuantumCircuit:
        """
        Making Embedding.

        Parameters:
            reps:
                num of repeats

        Returns:
            qc:
                circuit making TPE
        """
        qc = QuantumCircuit(self.n_qubits)
        theta = ParameterVector("theta", self.n_qubits)

        for _ in range(reps):
            for qubit_index in range(self.n_qubits):
                qc.rx(theta[qubit_index] * np.pi, qubit_index)

        qc.assign_parameters(theta, inplace=True)

        return qc

    def HEE(self, reps: int = 1) -> QuantumCircuit:
        """
        Making Embedding.

        Parameters:
            reps:
                num of repeats

        Returns:
            qc:
                circuit making HEE
        """
        qc = QuantumCircuit(self.n_qubits)
        theta = ParameterVector("theta", self.n_qubits)

        for _ in range(reps):
            for qubit_index in range(self.n_qubits):
                qc.rx(theta[qubit_index] * np.pi, qubit_index)
            for qubit_index in range(self.n_qubits - 1):
                qc.cx(qubit_index, qubit_index + 1)

        qc.assign_parameters(theta, inplace=True)

        return qc

    """
    Making FeatureMap+Ansatz Circuit
    Embeddings:
        ZFeatureMap: Z
        ZZFeatureMap: ZZ
        TPE: T
        HEE: H
    Ansatz:
        RealAmplitudes: R
        EfficientSU2: E
    """

    def make_circuit(
        self,
        embedding_type: str,
        ansatz_type: str,
        feature_map_reps: int = 1,
        ansatz_reps: int = 1,
    ):
        """
        Making FeatureMap+Ansatz Circuit

        Parameters:
            embedding_type: str
                ["ZFeatureMap", "ZZFeatureMap", "TPE", "HEE"]
            ansatz_type: str
                ["RealAmplitudes", "EfficientSU2"]
            feature_map_reps: int
                times to repeat feature_map
            ansatz_reps: int
                times to repeat ansatz
        """
        circuit = QuantumCircuit(self.n_qubits)

        data_map = lambda x: x[0] * np.pi / 2
        if embedding_type == "ZFeatureMap":
            feature_map = ZFeatureMap(
                self.n_qubits, reps=feature_map_reps, data_map_func=data_map
            )
        elif embedding_type == "ZZFeatureMap":
            feature_map = ZZFeatureMap(
                self.n_qubits, reps=feature_map_reps, data_map_func=data_map
            )
        elif embedding_type == "TPE":
            feature_map = self.TPE(reps=feature_map_reps)
        elif embedding_type == "HEE":
            feature_map = self.HEE(reps=feature_map_reps)
        else:
            print("Unknown embegging type.")
            exit(1)

        if ansatz_type == "RealAmplitudes":
            ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        elif ansatz_type == "EfficientSU2":
            ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        else:
            print("Unknown ansatz type.")
            exit(1)

        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)

        return circuit, feature_map.parameters, ansatz.parameters


class QIComponent(QCNNComponent, QNNComponent):
    """
    Making QNN circuit for toymodel quantum internet simulation

    Attributes:
        n_qubits: int
            num of qubits
    """

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def internet_route(self, param_name="theta1"):
        """
        Making noisy quantum channel by using rx and rz gate.
        Parameters:
            param_name:
                prefix of this parameters using in quantum circuit

        Returns:
            qc:
                circuit making noisy quantum channel
        """
        theta = ParameterVector(param_name, self.n_qubits * 2)
        qc = QuantumCircuit(self.n_qubits)
        for qubit_index in range(self.n_qubits):
            qc.rx(theta[qubit_index * 2], qubit_index)
            qc.rz(theta[qubit_index * 2 + 1], qubit_index)

        qc.assign_parameters(theta, inplace=True)

        return qc

    def make_qcnn_circuit(
        self,
        embedding_type: str,
        network_model: int,
        feature_map_reps: int = 1,
    ):
        """
        Making FeatureMap+QCNN Circuit for noisy quantum channel

        Parameters:
            embedding_type: str
                ["ZFeatureMap", "ZZFeatureMap", "TPE", "HEE"]
            network_model: int
                [1, 2]
            feature_map_reps: int
                times to repeat feature_map
        """
        data_map = lambda x: x[0] * np.pi / 2
        if embedding_type == "ZFeatureMap":
            feature_map = ZFeatureMap(
                self.n_qubits, reps=feature_map_reps, data_map_func=data_map
            )
        elif embedding_type == "ZZFeatureMap":
            feature_map = ZZFeatureMap(
                self.n_qubits, reps=feature_map_reps, data_map_func=data_map
            )
        elif embedding_type == "TPE":
            feature_map = QNNComponent(self.n_qubits).TPE(feature_map_reps)
        elif embedding_type == "HEE":
            feature_map = QNNComponent(self.n_qubits).HEE(feature_map_reps)
        else:
            print("Unknown Embedding Type.")
            exit(1)

        circuit = QuantumCircuit(self.n_qubits)
        ansatz = QuantumCircuit(self.n_qubits, name="Ansatz")
        ansatz.compose(
            self.conv_layer(), list(range(self.n_qubits)), inplace=True
        )
        ansatz.compose(
            self.pool_layer(), list(range(self.n_qubits)), inplace=True
        )
        i_route = QIComponent(self.n_qubits).internet_route()

        if network_model == 1:
            i_route2 = self.internet_route(param_name="theta2")

            circuit.compose(feature_map, inplace=True)
            circuit.barrier()
            circuit.compose(i_route, inplace=True)
            circuit.barrier()
            circuit.compose(ansatz, inplace=True)
            circuit.barrier()
            circuit.compose(i_route2, inplace=True)

            ansatz_params = (
                list(i_route.parameters)
                + list(i_route2.parameters)
                + list(ansatz.parameters)
            )
        elif network_model == 2:
            circuit.compose(feature_map, inplace=True)
            circuit.barrier()
            circuit.compose(i_route, inplace=True)
            circuit.barrier()
            circuit.compose(ansatz, inplace=True)

            ansatz_params = list(i_route.parameters) + list(ansatz.parameters)
        else:
            print("There is no such network model.")
            exit(1)

        return circuit, feature_map.parameters, ansatz_params

    def make_qnn_circuit(
        self,
        embedding_type: str,
        ansatz_type: str,
        network_model: int,
        feature_map_reps: int = 1,
        ansatz_reps: int = 1,
    ):
        """
        Making FeatureMap+Ansatz Circuit for noisy quantum channel

        Parameters:
            embedding_type: str
                ["ZFeatureMap", "ZZFeatureMap", "TPE", "HEE"]
            ansatz_type: str
                ["RealAmplitudes", "EfficientSU2"]
            network_model: int
                [1, 2]
            feature_map_reps: int
                times to repeat feature_map
            ansatz_reps: int
                times to repeat ansatz
        """
        circuit = QuantumCircuit(self.n_qubits)

        data_map = lambda x: x[0] * np.pi / 2

        if embedding_type == "ZFeatureMap":
            feature_map = ZFeatureMap(
                self.n_qubits, reps=feature_map_reps, data_map_func=data_map
            )
        elif embedding_type == "ZZFeatureMap":
            feature_map = ZZFeatureMap(
                self.n_qubits, reps=feature_map_reps, data_map_func=data_map
            )
        elif embedding_type == "TPE":
            feature_map = self.TPE(reps=feature_map_reps)
        elif embedding_type == "HEE":
            feature_map = self.HEE(reps=feature_map_reps)
        else:
            print("Unknown embegging type.")
            exit(1)

        if ansatz_type == "RealAmplitudes":
            ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        elif ansatz_type == "EfficientSU2":
            ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        else:
            print("Unknown ansatz type.")
            exit(1)

        i_route = self.internet_route()

        if network_model == 1:
            i_route2 = self.internet_route(param_name="theta2")

            circuit.compose(feature_map, inplace=True)
            circuit.barrier()
            circuit.compose(i_route, inplace=True)
            circuit.barrier()
            circuit.compose(ansatz, inplace=True)
            circuit.barrier()
            circuit.compose(i_route2, inplace=True)

            ansatz_params = (
                list(i_route.parameters)
                + list(i_route2.parameters)
                + list(ansatz.parameters)
            )
        elif network_model == 2:
            circuit.compose(feature_map, inplace=True)
            circuit.barrier()
            circuit.compose(i_route, inplace=True)
            circuit.barrier()
            circuit.compose(ansatz, inplace=True)

            ansatz_params = list(i_route.parameters) + list(ansatz.parameters)
        else:
            print("There is no such network model.")
            exit(1)

        return circuit, feature_map.parameters, ansatz_params


Transition = collections.namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayBuffer:
    """DQNの経験再生に使われるリプレイバッファ"""

    def __init__(self, capacity):
        self.memory = collections.deque([], maxlen=capacity)

    def push(self, *args):
        """遷移を保存する"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """バッチサイズ分の遷移をランダムにサンプリングする"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
