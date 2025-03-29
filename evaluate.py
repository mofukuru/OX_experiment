import os

def evaluate_cnn(network: str, n_qubits: int=10):
    import torch
    from src.agent import RandomPolicy, CNNAgent
    from src.env import Env

    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        model_path = f"./models/train/train_cnn_agent_times_{iter*10+10}_n_qubits_{n_qubits}.pth"
        model_weights = torch.load(model_path)
        agent = CNNAgent(network=network, n_qubits=n_qubits)
        agent.NN.load_state_dict(model_weights)

        agent.eval()

        environment = Env(agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
        result_dir = "./result/"
        result_path = result_dir + f"train_cnn_agent_times_{iter*10+10}_n_qubits_{n_qubits}.json"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    with open(result_path, "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")

def evaluate_qcnn(
        embedding_type: str,
        n_qubits: int=10,
        param_weight: float=1.0,
        state_weight: float=1.0
    ):
    import torch
    from src.agent import RandomPolicy, QAgent
    from src.env import Env

    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        model_path = f"./models/train/train_qcnn_agent_times_{iter*10+10}_n_qubits_{n_qubits}.pth"
        model_weights = torch.load(model_path)
        nn_network = 1
        agent = QAgent(
            embedding_type=embedding_type,
            nn_network=nn_network,
            n_qubits=n_qubits,
            param_weight=param_weight,
            state_weight=state_weight,
        )
        agent.QNN.load_state_dict(model_weights)

        agent.eval()

        environment = Env(agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
        result_dir = "./result/"
        result_path = result_dir + f"train_qcnn_agent_times_{iter*10+10}_n_qubits_{n_qubits}.json"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    with open(result_path, "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")

def evaluate_qnn(
        embedding_type: str,
        ansatz_type: str,
        n_qubits: int=10,
        param_weight: float=1.0,
        state_weight: float=1.0,
        feature_map_reps: int=1,
        ansatz_reps: int=1
    ):
    import torch
    from src.agent import RandomPolicy, QAgent
    from src.env import Env

    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        model_path = f"./models/train/train_qnn_agent_times_{iter*10+10}_n_qubits_{n_qubits}.pth"
        model_weights = torch.load(model_path)
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
        agent.QNN.load_state_dict(model_weights)

        agent.eval()

        environment = Env(agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
        result_dir = "./result/"
        result_path = result_dir + f"train_qnn_agent_times_{iter*10+10}_n_qubits_{n_qubits}.json"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    with open(result_path, "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")

def evaluate_cqcnn_for_network(
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
    import torch
    from src.agent import RandomPolicy, CQCAgent_network
    from src.env_for_network import Env

    rate_per_NN = []
    randagent = RandomPolicy()

    for iter in range(24, 25):
        elorate = 1500
        model_path = os.path.join(
            "./models/train/",
            f"train_cqcnn_agent_for_network_times_{iter*10+10}_n_qubits_{n_qubits}",
            f"_network_noise_is_{noised_quantum_channel}_distance_is_{distance}.pth",
        )
        model_weights = torch.load(model_path)
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
        agent.HNN.load_state_dict(model_weights)

        agent.eval()

        environment = Env(agent, randagent, elorate=elorate)
        rate = environment.train(10000)
        elorate = environment.elorate
        rate_per_NN.append(rate)
        result_dir = "./result/"
        result_path = os.path.join(
            "./result/",
            f"train_cqcnn_agent_for_network_times_{iter*10+10}_n_qubits_{n_qubits}",
            f"_network_noise_is_{noised_quantum_channel}_distance_is_{distance}.json",
        )
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    with open(result_path, "a") as f:
        print(rate_per_NN, file=f)
        f.write("\n")
