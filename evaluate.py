import json
import os

import torch
import tqdm

from src.agent import (
    CNNAgent,
    CQCAgent,
    CQCAgent_network,
    QAgent,
    RandomPolicy,
)
from src.env import Environment


def get_agent_from_config(model_path):
    """
    モデルパスから設定を読み込み、エージェントを初期化する。
    """
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

    # state_dictをロード
    model_to_load = (
        getattr(agent, "NN", None)
        or getattr(agent, "QNN", None)
        or getattr(agent, "HNN", None)
    )
    if model_to_load and os.path.exists(model_path):
        model_to_load.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
    else:
        print(
            f"Warning: Model file not found at {model_path}. Agent is not loaded."
        )

    return agent


def run_evaluation(model_path1: str, model_path2: str, num_games: int = 100):
    """
    2つのモデルを対戦させ、結果を評価・保存する。

    Args:
        model_path1: プレイヤー1のモデルへのパス。
        model_path2: プレイヤー2のモデルへのパス ("random"も可)。
        num_games: 対戦するゲーム数。
    """
    # エージェントの準備
    agent1 = get_agent_from_config(model_path1)
    agent1.eval()
    print("Player 1 Agent loaded from", model_path1)

    if model_path2.lower() == "random":
        agent2 = RandomPolicy()
        print("Player 2 is a RandomPolicy agent.")
    else:
        agent2 = get_agent_from_config(model_path2)
        agent2.eval()
        print("Player 2 Agent loaded from", model_path2)

    # 結果記録用
    results = {
        "model1": model_path1,
        "model2": model_path2,
        "num_games": num_games,
        "summary": {"wins_p1": 0, "wins_p2": 0, "draws": 0},
        "game_logs": [],
    }

    for i in tqdm.tqdm(
        range(num_games),
        desc=f"Evaluating {os.path.basename(model_path1)} vs {os.path.basename(model_path2)}",
    ):
        # ゲームごとに先手・後手を入れ替える
        if i % 2 == 0:
            env = Environment(agent1, agent2)
            winner, moves = env.play()
        else:
            # agent2が先手になる
            env = Environment(agent2, agent1)
            original_winner, moves = env.play()
            # 結果を元のプレイヤー観点に戻す (e.g. agent2が勝った場合winner=1だが、記録上は-1)
            winner = original_winner * -1

        game_log = {
            "game_id": i,
            "starting_player": 1 if i % 2 == 0 else 2,
            "winner": winner,
            "move_count": len(moves),
        }
        results["game_logs"].append(game_log)

        if winner == 1:
            results["summary"]["wins_p1"] += 1
        elif winner == -1:
            results["summary"]["wins_p2"] += 1
        else:
            results["summary"]["draws"] += 1

    # レーティング（勝率）の計算
    results["summary"]["win_rate_p1"] = (
        results["summary"]["wins_p1"] / num_games
    )
    results["summary"]["win_rate_p2"] = (
        results["summary"]["wins_p2"] / num_games
    )
    results["summary"]["draw_rate"] = results["summary"]["draws"] / num_games

    print("\nEvaluation Summary:")
    print(
        f"Player 1 Wins: {results['summary']['wins_p1']} ({results['summary']['win_rate_p1']:.2%})"
    )
    print(
        f"Player 2 Wins: {results['summary']['wins_p2']} ({results['summary']['win_rate_p2']:.2%})"
    )
    print(
        f"Draws: {results['summary']['draws']} ({results['summary']['draw_rate']:.2%})"
    )

    # 結果をJSONに保存
    log_dir = "./logs/eval/"
    os.makedirs(log_dir, exist_ok=True)
    p1_name = os.path.splitext(os.path.basename(model_path1))[0]
    p2_name = (
        os.path.splitext(os.path.basename(model_path2))[0]
        if model_path2.lower() != "random"
        else "random"
    )
    log_path = os.path.join(log_dir, f"eval_{p1_name}_vs_{p2_name}.json")

    with open(log_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nEvaluation results saved to {log_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(run_evaluation)
