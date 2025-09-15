import json
import os
import time

import fire
import torch

from src.agent import (
    CNNAgent,
    CQCAgent,
    CQCAgent_network,
    QAgent,
    RandomPolicy,
)
from src.env import Environment
from src.tictactoe import TicTacToe


def get_agent_from_config(model_path):
    """evaluate.pyから流用した、モデルをロードする関数"""
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    log_path = os.path.join("./logs/train/", f"{base_name}.json")

    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Log file not found for model: {model_path}. Expected at: {log_path}"
        )

    with open(log_path, "r") as f:
        config = json.load(f)["agent_config"]

    agent_class_name = config.get("agent_class")
    if not agent_class_name:
        raise ValueError(f"'agent_class' not found in log file: {log_path}")

    # agent_class_nameに基づいてエージェントを動的に選択・初期化
    if agent_class_name == "CNNAgent":
        agent = CNNAgent(
            network=config.get("network_name"), n_qubits=config.get("n_qubits")
        )
    elif agent_class_name == "QAgent":
        agent = QAgent(
            embedding_type=config.get("embedding_type"),
            ansatz_type=config.get("ansatz_type"),
            nn_network=config.get("nn_network"),
            n_qubits=config.get("n_qubits"),
            state_weight=config.get("state_weight"),
            feature_map_reps=config.get("feature_map_reps"),
            ansatz_reps=config.get("ansatz_reps"),
        )
    elif agent_class_name == "CQCAgent":
        agent = CQCAgent(
            embedding_type=config.get("embedding_type"),
            ansatz_type=config.get("ansatz_type"),
            nn_network=config.get("nn_network"),
            n_qubits=config.get("n_qubits"),
            feature_map_reps=config.get("feature_map_reps"),
            ansatz_reps=config.get("ansatz_reps"),
        )
    elif agent_class_name == "CQCAgent_network":
        agent = CQCAgent_network(
            embedding_type=config.get("embedding_type"),
            ansatz_type=config.get("ansatz_type"),
            nn_network=config.get("nn_network"),
            network_model=config.get("network_model"),
            n_qubits=config.get("n_qubits"),
            feature_map_reps=config.get("feature_map_reps"),
            ansatz_reps=config.get("ansatz_reps"),
            noised=config.get("noised"),
            distance=config.get("distance"),
        )
    else:
        raise ValueError(
            f"Unknown agent class '{agent_class_name}' in log file: {log_path}"
        )

    model_to_load = (
        getattr(agent, "NN", None)
        or getattr(agent, "QNN", None)
        or getattr(agent, "HNN", None)
    )
    if model_to_load and os.path.exists(model_path):
        model_to_load.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu")),
        )
    else:
        print(
            f"Warning: Model file not found at {model_path}. Agent is not loaded."
        )

    return agent


def play_and_replay(model_path1, model_path2):
    """
    2つのモデルで1回対戦し、その棋譜を再生する。
    """
    # エージェントをロード
    agent1 = get_agent_from_config(model_path1)
    agent1.eval()
    agent2 = get_agent_from_config(model_path2)
    agent2.eval()

    for i in range(2):
        # 環境を初期化してゲームを実行
        print("--- Starting Game ---")
        if i == 1:
            env = Environment(agent2, agent1)
        else:
            env = Environment(agent1, agent2)
        winner, history = env.play()

        # 結果の表示
        print("--- Game Finished ---")
        if winner == 1:
            print(f"Winner: Player 1 ({os.path.basename(model_path1)})")
        elif winner == -1:
            print(f"Winner: Player 2 ({os.path.basename(model_path2)})")
        else:
            print("Draw")
        print("\n--- Replaying Game ---")

        # 棋譜の再生
        replay_board = TicTacToe()
        replay_board.display_board(sleep_secs=0)
        player_turn = 1
        for move in history:
            replay_board.place(move, player_turn)
            replay_board.display_board()
            player_turn *= -1

        print("--- Replay Finished ---")


if __name__ == "__main__":
    fire.Fire(play_and_replay)
