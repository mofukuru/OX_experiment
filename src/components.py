import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, ZFeatureMap, ZZFeatureMap, PauliFeatureMap

#%% QCNNを構成するためのパーツ作成
class QCNNComponent:
    def __init__(self, n_qubits, param_prefix: list, sources: list, sinks: list):
        self.n_qubits = n_qubits
        self.param_prefix = param_prefix
        self.sources = sources
        self.sinks = sinks

    def conv_circuit(self, params):
        target = QuantumCircuit(2)
        target.rz(-np.pi/2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi/2, 0)
        return target

    def conv_layer(self):
        qc = QuantumCircuit(self.n_qubits, name=self.param_prefix[0])
        qubits = list(range(self.n_qubits))
        param_index = 0
        params = ParameterVector(self.param_prefix[0], length=self.n_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(self.conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(self.conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(self.n_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def pool_circuit(self,params):
        target = QuantumCircuit(2)
        target.rz(-np.pi/2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def pool_layer(self):
        n_qubits = len(self.sources) + len(self.sinks)
        qc = QuantumCircuit(n_qubits, name=self.param_prefix[1])
        param_index = 0
        params = ParameterVector(self.param_prefix[1], length=n_qubits // 2 * 3)
        for source, sink in zip(self.sources, self.sinks):
            qc = qc.compose(self.pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(n_qubits)
        qc.append(qc_inst, range(n_qubits))
        return qc

    def create_qcnn(self, embedding_type: str, reps: int=1):
        if embedding_type == "ZFeatureMap":
            feature_map = ZFeatureMap(self.n_qubits, reps=reps)
        elif embedding_type == "ZZFeatureMap":
            feature_map = ZZFeatureMap(self.n_qubits, reps=reps)
        elif embedding_type == "TPE":
            feature_map = QNNComponent(self.n_qubits).TPE(reps)
        elif embedding_type == "HEE":
            feature_map = QNNComponent(self.n_qubits).HEE(reps)
        else:
            print("Unknown instruction")
            exit(1)

        ansatz = QuantumCircuit(self.n_qubits, name="Ansatz")
        ansatz.compose(self.conv_layer(), list(range(self.n_qubits)), inplace=True)
        ansatz.compose(self.pool_layer(), list(range(self.n_qubits)), inplace=True)
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(feature_map, range(self.n_qubits), inplace=True)
        circuit.compose(ansatz, range(self.n_qubits), inplace=True)
        return circuit, feature_map.parameters, ansatz.parameters
    
    def create_qcnn_network1(self, embedding_type: str, reps: int=1):
        if embedding_type == "ZFeatureMap":
            feature_map = ZFeatureMap(self.n_qubits, reps=reps)
        elif embedding_type == "ZZFeatureMap":
            feature_map = ZZFeatureMap(self.n_qubits, reps=reps)
        elif embedding_type == "TPE":
            feature_map = QNNComponent(self.n_qubits).TPE(reps)
        elif embedding_type == "HEE":
            feature_map = QNNComponent(self.n_qubits).HEE(reps)
        else:
            print("Unknown instruction")
            exit(1)

        ansatz = QuantumCircuit(self.n_qubits, name="Ansatz")
        ansatz.compose(self.conv_layer(), list(range(self.n_qubits)), inplace=True)
        ansatz.compose(self.pool_layer(), list(range(self.n_qubits)), inplace=True)
        i_route = QIComponent(self.n_qubits).internet_route()
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(feature_map, range(self.n_qubits), inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, range(self.n_qubits), inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        return circuit, feature_map.parameters, ansatz.parameters
    
    def create_qcnn_network2(self, embedding_type: str, reps: int=1):
        if embedding_type == "ZFeatureMap":
            feature_map = ZFeatureMap(self.n_qubits, reps=reps)
        elif embedding_type == "ZZFeatureMap":
            feature_map = ZZFeatureMap(self.n_qubits, reps=reps)
        elif embedding_type == "TPE":
            feature_map = QNNComponent(self.n_qubits).TPE(reps)
        elif embedding_type == "HEE":
            feature_map = QNNComponent(self.n_qubits).HEE(reps)
        else:
            print("Unknown instruction")
            exit(1)

        ansatz = QuantumCircuit(self.n_qubits, name="Ansatz")
        ansatz.compose(self.conv_layer(), list(range(self.n_qubits)), inplace=True)
        ansatz.compose(self.pool_layer(), list(range(self.n_qubits)), inplace=True)
        i_route = QIComponent(self.n_qubits).internet_route()
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(feature_map, range(self.n_qubits), inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, range(self.n_qubits), inplace=True)
        return circuit, feature_map.parameters, ansatz.parameters

class QNNComponent:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        
    ### Embedding

    def TPE(self, reps: int=1) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        theta = ParameterVector("theta", self.n_qubits)

        for _ in range(reps):
            for qubit_index in range(self.n_qubits):
                qc.rx(theta[qubit_index], qubit_index)

        qc.assign_parameters(theta, inplace=True)

        return qc

    def HEE(self, reps: int=1) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        theta = ParameterVector("theta", self.n_qubits)

        for _ in range(reps):
            for qubit_index in range(self.n_qubits):
                qc.rx(theta[qubit_index], qubit_index)
            for qubit_index in range(self.n_qubits-1):
                qc.cx(qubit_index, qubit_index+1)

        qc.assign_parameters(theta, inplace=True)

        return qc
    
    """make QuantumCircuit
    Embeddings:
        ZFeatureMap: Z,
        ZZFeatureMap: ZZ,
        TPE: T,
        HEE, H,
    Ansatz:
        RealAmplitudes: R,
        EfficientSU2: E,"""
        
    def ZR(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = ZFeatureMap(self.n_qubits, reps=feature_map_reps)
        ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        
        return circuit, feature_map.parameters, ansatz.parameters
    
    def ZZR(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = ZZFeatureMap(self.n_qubits, reps=feature_map_reps)
        ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        
        return circuit, feature_map.parameters, ansatz.parameters
    
    def TR(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = self.TPE(reps=feature_map_reps)
        ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        
        return circuit, feature_map.parameters, ansatz.parameters
    
    def HR(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = self.HEE(reps=feature_map_reps)
        ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        
        return circuit, feature_map.parameters, ansatz.parameters
    
    def ZE(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = ZFeatureMap(self.n_qubits, reps=feature_map_reps)
        ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        
        return circuit, feature_map.parameters, ansatz.parameters
    
    def ZZE(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = ZZFeatureMap(self.n_qubits, reps=feature_map_reps)
        ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        
        return circuit, feature_map.parameters, ansatz.parameters
    
    def TE(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = self.TPE(reps=feature_map_reps)
        ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        
        return circuit, feature_map.parameters, ansatz.parameters
    
    def HE(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = self.HEE(reps=feature_map_reps)
        ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        
        return circuit, feature_map.parameters, ansatz.parameters
    
class QIComponent:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    ### network
    def internet_route(self, param_name="theta1"):
        theta = ParameterVector(param_name, self.n_qubits*2)
        qc = QuantumCircuit(self.n_qubits)
        for qubit_index in range(self.n_qubits):
            qc.rx(theta[qubit_index*2], qubit_index)
            qc.rz(theta[qubit_index*2+1], qubit_index)
            
        qc.assign_parameters(theta, inplace=True)
            
        return qc
        
    ### Embedding

    def TPE(self, reps: int=1) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        theta = ParameterVector("theta", self.n_qubits)

        for _ in range(reps):
            for qubit_index in range(self.n_qubits):
                qc.rx(theta[qubit_index], qubit_index)

        qc.assign_parameters(theta, inplace=True)

        return qc

    def HEE(self, reps: int=1) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        theta = ParameterVector("theta", self.n_qubits)

        for _ in range(reps):
            for qubit_index in range(self.n_qubits):
                qc.rx(theta[qubit_index], qubit_index)
            for qubit_index in range(self.n_qubits-1):
                qc.cx(qubit_index, qubit_index+1)

        qc.assign_parameters(theta, inplace=True)

        return qc
    
    """make QuantumCircuit
    Embeddings:
        ZFeatureMap: Z,
        ZZFeatureMap: ZZ,
        TPE: T,
        HEE, H,
    Ansatz:
        RealAmplitudes: R,
        EfficientSU2: E,
    Network:
        1: carry qubit and measurement
        2: carry only"""
        
    def ZR_network1(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = ZFeatureMap(self.n_qubits, reps=feature_map_reps)
        ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        i_route2 = self.internet_route(param_name="theta2")
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)
        circuit.barrier()
        circuit.compose(i_route2, inplace=True)
        
        ansatz_params = list(i_route.parameters)+list(i_route2.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def ZZR_network1(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = ZZFeatureMap(self.n_qubits, reps=feature_map_reps)
        ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        i_route2 = self.internet_route(param_name="theta2")
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)
        circuit.barrier()
        circuit.compose(i_route2, inplace=True)

        ansatz_params = list(i_route.parameters)+list(i_route2.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def TR_network1(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = self.TPE(reps=feature_map_reps)
        ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        i_route2 = self.internet_route(param_name="theta2")
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)
        circuit.barrier()
        circuit.compose(i_route2, inplace=True)

        ansatz_params = list(i_route.parameters)+list(i_route2.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def HR_network1(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = self.HEE(reps=feature_map_reps)
        ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        i_route2 = self.internet_route(param_name="theta2")
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)
        circuit.barrier()
        circuit.compose(i_route2, inplace=True)
        
        ansatz_params = list(i_route.parameters)+list(i_route2.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def ZE_network1(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = ZFeatureMap(self.n_qubits, reps=feature_map_reps)
        ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        i_route2 = self.internet_route(param_name="theta2")
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)
        circuit.barrier()
        circuit.compose(i_route2, inplace=True)

        ansatz_params = list(i_route.parameters)+list(i_route2.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def ZZE_network1(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = ZZFeatureMap(self.n_qubits, reps=feature_map_reps)
        ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        i_route2 = self.internet_route(param_name="theta2")
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)
        circuit.barrier()
        circuit.compose(i_route2, inplace=True)

        ansatz_params = list(i_route.parameters)+list(i_route2.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def TE_network1(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = self.TPE(reps=feature_map_reps)
        ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        i_route2 = self.internet_route(param_name="theta2")
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)
        circuit.barrier()
        circuit.compose(i_route2, inplace=True)

        ansatz_params = list(i_route.parameters)+list(i_route2.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def HE_network1(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = self.HEE(reps=feature_map_reps)
        ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        i_route2 = self.internet_route(param_name="theta2")
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)
        circuit.barrier()
        circuit.compose(i_route2, inplace=True)

        ansatz_params = list(i_route.parameters)+list(i_route2.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    """-----------------------------------------------------------------------------------------------"""
    
    def ZR_network2(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = ZFeatureMap(self.n_qubits, reps=feature_map_reps)
        ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)

        ansatz_params = list(i_route.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def ZZR_network2(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = ZZFeatureMap(self.n_qubits, reps=feature_map_reps)
        ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)

        ansatz_params = list(i_route.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def TR_network2(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = self.TPE(reps=feature_map_reps)
        ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)

        ansatz_params = list(i_route.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def HR_network2(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = self.HEE(reps=feature_map_reps)
        ansatz = RealAmplitudes(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)

        ansatz_params = list(i_route.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def ZE_network2(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = ZFeatureMap(self.n_qubits, reps=feature_map_reps)
        ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)

        ansatz_params = list(i_route.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def ZZE_network2(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = ZZFeatureMap(self.n_qubits, reps=feature_map_reps)
        ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)

        ansatz_params = list(i_route.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def TE_network2(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = self.TPE(reps=feature_map_reps)
        ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)

        ansatz_params = list(i_route.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    
    def HE_network2(self, feature_map_reps: int=1, ansatz_reps: int=1):
        circuit = QuantumCircuit(self.n_qubits)
        
        feature_map = self.HEE(reps=feature_map_reps)
        ansatz = EfficientSU2(self.n_qubits, reps=ansatz_reps)
        i_route = self.internet_route()
        
        circuit.compose(feature_map, inplace=True)
        circuit.barrier()
        circuit.compose(i_route, inplace=True)
        circuit.barrier()
        circuit.compose(ansatz, inplace=True)

        ansatz_params = list(i_route.parameters)+list(ansatz.parameters)
        
        return circuit, feature_map.parameters, ansatz_params
    