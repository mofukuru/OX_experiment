import json
import os
import time

import torch

from src.agent import CNNAgent, CQCAgent, CQCAgent_network, QAgent
from src.env import Environment


def run_training_session(
    agent,
    total_episodes,
    model_name,
    network_name=None,
    nn_network=None,
    n_qubits=8,
    embedding_type=None,
    ansatz_type=None,
    feature_map_reps=1,
    ansatz_reps=1,
    epsilon_threshold=0.1,  # 新しいパラメータ: εの閾値
    early_stopping=False,  # 新しいパラメータ: 早期終了の有効化
):
    """
    汎用的な学習セッションを実行する関数。

    Args:
        agent: 使用するエージェントのインスタンス。
        total_episodes: 合計学習エピソード数。
        model_name: 保存するモデルとログのベース名。
        network_name: ネットワーク名（CNNAgentのみ）。
        epsilon_threshold: 早期終了のための閾値（デフォルト: 0.1）。
        early_stopping: 閾値に基づく早期終了を有効にするかどうか（デフォルト: False）。
    """
    start_time = time.time()
    logs = {
        "episodes": [],
        "rewards_p1": [],
        "rewards_p2": [],
        "loss_p1": [],
        "loss_p2": [],
        "epsilon": [],
        "early_stopped": False,  # 早期終了したかどうかのフラグ
        "final_epsilon": None,  # 最終的なεの値
    }

    # モデルとログの保存先ディレクトリを作成
    model_dir = "./models/train/"
    log_dir = "./logs/train/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}.pth")
    log_path = os.path.join(log_dir, f"{model_name}.json")

    agent.train()
    environment = Environment(agent, agent)

    for episode in range(total_episodes):
        # εの値を更新
        if hasattr(agent, "update_epsilon"):
            agent.update_epsilon(episode)

            # 早期終了条件のチェック
            if (
                early_stopping
                and hasattr(agent, "epsilon")
                and agent.epsilon <= epsilon_threshold
            ):
                print(
                    f"\nEarly stopping at episode {episode + 1}: epsilon = {agent.epsilon:.4f} <= {epsilon_threshold}"
                )
                logs["early_stopped"] = True
                logs["final_epsilon"] = agent.epsilon
                break

        # 1エピソード学習を実行
        loss_p1, loss_p2, reward_p1, reward_p2 = environment.train(1)

        # ログを記録
        logs["episodes"].append(episode)
        logs["rewards_p1"].extend(reward_p1)
        logs["rewards_p2"].extend(reward_p2)
        logs["loss_p1"].extend(loss_p1)
        logs["loss_p2"].extend(loss_p2)
        if hasattr(agent, "epsilon"):
            logs["epsilon"].append(agent.epsilon)

        if (episode + 1) % 100 == 0:
            epsilon_str = (
                f"Epsilon: {agent.epsilon:.4f}"
                if hasattr(agent, "epsilon")
                else ""
            )
            print(f"Episode {episode + 1}/{total_episodes} | {epsilon_str}")

    # 早期終了しなかった場合の最終εを記録
    if not logs["early_stopped"] and hasattr(agent, "epsilon"):
        logs["final_epsilon"] = agent.epsilon

    # 学習時間の計算
    training_duration = time.time() - start_time
    print(f"Training finished in {training_duration:.2f} seconds.")

    # 実際に学習したエピソード数を表示
    actual_episodes = len(logs["episodes"])
    print(f"Total episodes trained: {actual_episodes}")

    # モデルの保存
    model_to_save = (
        getattr(agent, "NN", None)
        or getattr(agent, "QNN", None)
        or getattr(agent, "HNN", None)
    )
    if model_to_save:
        torch.save(model_to_save.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # ログの保存
    agent_config = {
        k: v for k, v in agent.__dict__.items() if not k.startswith("_")
    }
    agent_config["agent_class"] = agent.__class__.__name__
    agent_config["n_qubits"] = n_qubits
    agent_config["network_name"] = network_name
    agent_config["nn_network"] = nn_network
    agent_config["embedding_type"] = embedding_type
    agent_config["ansatz_type"] = ansatz_type
    agent_config["feature_map_reps"] = feature_map_reps
    agent_config["ansatz_reps"] = ansatz_reps
    agent_config["epsilon_threshold"] = epsilon_threshold
    agent_config["early_stopping"] = early_stopping

    for key, value in agent_config.items():
        if isinstance(
            value, (torch.device, torch.nn.Module, torch.optim.Optimizer)
        ):
            agent_config[key] = str(value)

    results = {
        "training_duration": training_duration,
        "actual_episodes": actual_episodes,
        "early_stopped": logs["early_stopped"],
        "final_epsilon": logs["final_epsilon"],
        "agent_config": agent_config,
        "logs": logs,
    }

    with open(log_path, "w") as f:
        json.dump(results, f, indent=4, default=lambda o: "<not serializable>")
    print(f"Logs saved to {log_path}")


def train_cnn(
    network: str,
    n_qubits: int = 10,
    total_episodes: int = 250,
    epsilon_threshold: float = 0.1,
    early_stopping: bool = False,
):
    agent = CNNAgent(network=network, n_qubits=n_qubits)
    model_name = f"train_cnn_agent_{network}_episodes_{total_episodes}_n_qubits_{n_qubits}"
    if early_stopping:
        model_name += f"_earlystop_eps{epsilon_threshold}"
    run_training_session(
        agent,
        total_episodes,
        model_name,
        epsilon_threshold=epsilon_threshold,
        early_stopping=early_stopping,
        network_name=network,
        nn_network=None,
        n_qubits=n_qubits,
    )


def train_qcnn(
    embedding_type: str,
    n_qubits: int = 10,
    state_weight: float = 1.0,
    total_episodes: int = 250,
    epsilon_threshold: float = 0.1,
    early_stopping: bool = False,
):
    agent = QAgent(
        embedding_type=embedding_type,
        nn_network=1,
        n_qubits=n_qubits,
        state_weight=state_weight,
    )
    model_name = f"train_qcnn_agent_episodes_{total_episodes}_n_qubits_{n_qubits}_embedding_{embedding_type}"
    if early_stopping:
        model_name += f"_earlystop_eps{epsilon_threshold}"
    run_training_session(
        agent,
        total_episodes,
        model_name,
        epsilon_threshold=epsilon_threshold,
        early_stopping=early_stopping,
        embedding_type=embedding_type,
        network_name=None,
        nn_network=1,
        n_qubits=n_qubits,
    )


def train_qnn(
    embedding_type: str,
    ansatz_type: str,
    n_qubits: int = 10,
    state_weight: float = 1.0,
    feature_map_reps: int = 1,
    ansatz_reps: int = 1,
    total_episodes: int = 250,
    epsilon_threshold: float = 0.1,
    early_stopping: bool = False,
):
    agent = QAgent(
        embedding_type=embedding_type,
        ansatz_type=ansatz_type,
        nn_network=2,
        n_qubits=n_qubits,
        state_weight=state_weight,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
    )
    model_name = f"train_qnn_agent_episodes_{total_episodes}_n_qubits_{n_qubits}_embedding_{embedding_type}_ansatz_{ansatz_type}"
    if early_stopping:
        model_name += f"_earlystop_eps{epsilon_threshold}"
    run_training_session(
        agent,
        total_episodes,
        model_name,
        epsilon_threshold=epsilon_threshold,
        early_stopping=early_stopping,
        embedding_type=embedding_type,
        ansatz_type=ansatz_type,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        network_name=None,
        nn_network=2,
        n_qubits=n_qubits,
    )


def train_cqc(
    embedding_type: str,
    ansatz_type: str,
    n_qubits: int = 10,
    nn_network: int = None,
    feature_map_reps: int = 1,
    ansatz_reps: int = 1,
    total_episodes: int = 250,
    epsilon_threshold: float = 0.1,
    early_stopping: bool = False,
):
    agent = CQCAgent(
        embedding_type=embedding_type,
        ansatz_type=ansatz_type,
        nn_network=nn_network,
        n_qubits=n_qubits,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
    )
    model_name = (
        f"train_cqc_agent_episodes_{total_episodes}"
        + f"_n_qubits_{n_qubits}_embedding_{embedding_type}"
        + f"_ansatz_{ansatz_type}_nn_network_{nn_network}"
    )
    if early_stopping:
        model_name += f"_earlystop_eps{epsilon_threshold}"
    run_training_session(
        agent,
        total_episodes,
        model_name,
        epsilon_threshold=epsilon_threshold,
        early_stopping=early_stopping,
        embedding_type=embedding_type,
        ansatz_type=ansatz_type,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        network_name=None,
        nn_network=nn_network,
        n_qubits=n_qubits,
    )


def train_cqcnn_for_network(
    noised_quantum_channel: bool = True,
    distance: float = 15.0,
    embedding_type: str = None,
    ansatz_type: str = None,
    n_qubits: int = 10,
    nn_network: int = None,
    network_model: int = 1,
    feature_map_reps: int = 1,
    ansatz_reps: int = 1,
    total_episodes: int = 250,
    epsilon_threshold: float = 0.1,
    early_stopping: bool = False,
):
    agent = CQCAgent_network(
        embedding_type=embedding_type,
        ansatz_type=ansatz_type,
        nn_network=nn_network,
        network_model=network_model,
        n_qubits=n_qubits,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        noised=noised_quantum_channel,
        distance=distance,
    )
    model_name = f"train_cqcnn_network_episodes_{total_episodes}_n_qubits_{n_qubits}_noise_{noised_quantum_channel}_dist_{distance}_embedding_{embedding_type}_ansatz_{ansatz_type}_networkmodel_{network_model}"
    if early_stopping:
        model_name += f"_earlystop_eps{epsilon_threshold}"
    run_training_session(
        agent,
        total_episodes,
        model_name,
        epsilon_threshold=epsilon_threshold,
        early_stopping=early_stopping,
        embedding_type=embedding_type,
        ansatz_type=ansatz_type,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        network_name=None,
        nn_network=nn_network,
        n_qubits=n_qubits,
    )


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "cnn": train_cnn,
            "qcnn": train_qcnn,
            "qnn": train_qnn,
            "cqc": train_cqc,
            "cqc_network": train_cqcnn_for_network,
        }
    )
