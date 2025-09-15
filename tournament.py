import itertools
import json
import os

import fire
import numpy as np
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

# Eloレーティング計算の定数
K_FACTOR = 32
INITIAL_RATING = 1500


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
            torch.load(model_path, map_location=torch.device("cpu"))
        )
    else:
        print(
            f"Warning: Model file not found at {model_path}. Agent is not loaded."
        )

    return agent


def calculate_expected_score(rating_a, rating_b):
    """Eloレーティングにおける期待勝率を計算する"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def play_match(model_path1, model_path2, num_games):
    """2つのモデル間で指定されたゲーム数のマッチを行う"""
    agent1 = get_agent_from_config(model_path1)
    agent1.eval()
    agent2 = get_agent_from_config(model_path2)
    agent2.eval()

    wins_p1 = 0
    draws = 0

    for i in range(num_games):
        if i % 2 == 0:
            env = Environment(agent1, agent2)
            winner, _ = env.play()
        else:
            env = Environment(agent2, agent1)
            original_winner, _ = env.play()
            winner = original_winner * -1

        if winner == 1:
            wins_p1 += 1
        elif winner == 0:
            draws += 1

    return wins_p1, draws


def run_tournament(model_dir="./models/train/", num_games_per_match=100):
    """総当たり戦トーナメントを実行する"""
    print("Starting tournament...")
    model_paths = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".pth")
    ]

    if len(model_paths) < 2:
        print(
            "Not enough models found to run a tournament (requires at least 2)."
        )
        return

    print(f"Found {len(model_paths)} models.")

    # レーティングを初期化
    ratings = {os.path.basename(path): INITIAL_RATING for path in model_paths}
    match_results = []

    # 総当たり戦の組み合わせを作成
    model_pairs = list(itertools.combinations(model_paths, 2))

    for model1_path, model2_path in tqdm.tqdm(
        model_pairs, desc="Tournament Progress"
    ):
        model1_name = os.path.basename(model1_path)
        model2_name = os.path.basename(model2_path)

        # マッチを実行
        wins_m1, draws = play_match(
            model1_path, model2_path, num_games_per_match
        )
        wins_m2 = num_games_per_match - wins_m1 - draws

        # スコアを計算 (win=1, draw=0.5, loss=0)
        score_m1 = wins_m1 + 0.5 * draws
        score_m2 = wins_m2 + 0.5 * draws

        # 期待勝率を計算
        expected_m1 = (
            calculate_expected_score(
                ratings[model1_name], ratings[model2_name]
            )
            * num_games_per_match
        )
        expected_m2 = (
            calculate_expected_score(
                ratings[model2_name], ratings[model1_name]
            )
            * num_games_per_match
        )

        # レーティングを更新
        new_rating_m1 = (
            ratings[model1_name]
            + K_FACTOR * (score_m1 - expected_m1) / num_games_per_match
        )
        new_rating_m2 = (
            ratings[model2_name]
            + K_FACTOR * (score_m2 - expected_m2) / num_games_per_match
        )

        match_results.append(
            {
                "model1": model1_name,
                "model2": model2_name,
                "wins1": wins_m1,
                "wins2": wins_m2,
                "draws": draws,
                "rating1_old": ratings[model1_name],
                "rating2_old": ratings[model2_name],
                "rating1_new": new_rating_m1,
                "rating2_new": new_rating_m2,
            }
        )

        ratings[model1_name] = new_rating_m1
        ratings[model2_name] = new_rating_m2

    # 最終結果をレーティングでソート
    sorted_ratings = sorted(
        ratings.items(), key=lambda item: item[1], reverse=True
    )

    print("\n--- Tournament Finished ---")
    print("Final Rankings:")
    for i, (model, rating) in enumerate(sorted_ratings):
        print(f"{i+1}. {model}: {rating:.2f}")

    # 結果をJSONファイルに保存
    final_results = {
        "final_ratings": dict(sorted_ratings),
        "match_history": match_results,
    }
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "tournament_results.json")

    with open(log_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"\nTournament results saved to {log_path}")


if __name__ == "__main__":
    fire.Fire(run_tournament)
