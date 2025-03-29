import os

def train_cnn(network: str, n_qubits: int = 10):
    import torch
    from src.agent import CNNAgent
    from src.env import Environment

    agent = CNNAgent(network=network, n_qubits=n_qubits)
    print(agent.NN)

    for iter in range(25):
        agent.train()
        environment = Environment(agent, agent)
        environment.train(10)

        model_dir = "./models/train/"
        model_path = f"./models/train/train_cnn_agent_times_{iter*10+10}_n_qubits_{n_qubits}.pth"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(agent.NN.state_dict(), model_path)

def train_qcnn(
        embedding_type: str,
        n_qubits: int=10,
        param_weight: float=1.0,
        state_weight: float=1.0
    ):
    import torch
    from src.agent import QAgent
    from src.env import Environment

    nn_network = 1

    agent = QAgent(
        embedding_type=embedding_type,
        nn_network=nn_network,
        n_qubits=n_qubits,
        param_weight=param_weight,
        state_weight=state_weight,
    )
    print(agent.QNN)

    for iter in range(25):
        agent.train()
        environment = Environment(agent, agent)
        environment.train(10)

        model_dir = "./models/train/"
        model_path = f"./models/train/train_qcnn_agent_times_{iter*10+10}_n_qubits_{n_qubits}.pth"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(agent.QNN.state_dict(), model_path)

def train_qnn(
        embedding_type: str,
        ansatz_type: str,
        n_qubits: int=10,
        param_weight: float=1.0,
        state_weight: float=1.0,
        feature_map_reps: int=1,
        ansatz_reps: int=1
    ):
    import torch
    from src.agent import QAgent
    from src.env import Environment

    nn_network = 2

    agent = QAgent(
        embedding_type=embedding_type,
        ansatz_type=ansatz_type,
        nn_network=nn_network,
        n_qubits=n_qubits,
        param_weight=param_weight,
        state_weight=state_weight,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps
    )
    print(agent.QNN)

    for iter in range(25):
        agent.train()
        environment = Environment(agent, agent)
        environment.train(10)

        model_dir = "./models/train/"
        model_path = f"./models/train/train_qnn_agent_times_{iter*10+10}_n_qubits_{n_qubits}.pth"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(agent.QNN.state_dict(), model_path)

def train_cqcnn_for_network(
        noised_quantum_channel: bool=True,
        distance: float=15.0,
        embedding_type: str=None,
        ansatz_type: str=None,
        n_qubits: int=10,
        nn_network: int=None,
        network_model: int=1,
        feature_map_reps: int=1,
        ansatz_reps: int=1
    ):
    """
    nn_network:
        1: CNN_QCNN_CNN
        2: CNN_QNN_CNN_sampler
        3: CNN_QNN_CNN_estimator
    """
    import torch
    from src.agent import CQCAgent_network
    from src.env import Environment

    agent = CQCAgent_network(
        embedding_type=embedding_type,
        ansatz_type=ansatz_type,
        nn_network=nn_network,
        network_model=network_model,
        n_qubits=n_qubits,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        noised=noised_quantum_channel,
        distance=distance
    )
    print(agent.HNN)

    for iter in range(25):
        agent.train()
        environment = Environment(agent, agent)
        environment.train(10)

        model_dir = "./models/train"
        model_path = os.path.join(
            "./models/train/",
            f"train_cqcnn_agent_for_network_times_{iter*10+10}_n_qubits_{n_qubits}",
            f"_network_noise_is_{noised_quantum_channel}_distance_is_{distance}.pth",
        )
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(agent.HNN.state_dict(), model_path)
