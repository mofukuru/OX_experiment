import json
import os

import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt

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


def calculate_expected_score(rating_a, rating_b):
    """Eloレーティングにおける期待勝率を計算する"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def get_agent_from_config(model_path, override_noised=None, override_distance=None):
    """
    モデルパスから設定を読み込み、エージェントを初期化する。

    Args:
        model_path: モデルファイルのパス
        override_noised: noised設定を上書きする値（None以外の場合）
        override_distance: distance設定を上書きする値（None以外の場合）
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
        # noised と distance の設定を取得（上書き値があればそれを使用）
        noised = override_noised if override_noised is not None else config.get("noised")
        distance = override_distance if override_distance is not None else config.get("distance")

        agent = CQCAgent_network(
            embedding_type=config.get("embedding_type"),
            ansatz_type=config.get("ansatz_type"),
            nn_network=config.get("nn_network"),
            network_model=config.get("network_model"),
            n_qubits=config.get("n_qubits"),
            feature_map_reps=config.get("feature_map_reps"),
            ansatz_reps=config.get("ansatz_reps"),
            noised=noised,
            distance=distance,
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


def run_evaluation(
    model_path1: str,
    model_path2: str,
    num_games: int = 100,
    track_rating: bool = False,
    rating_interval: int = 100,
    override_noised1: bool = None,
    override_distance1: float = None,
    override_noised2: bool = None,
    override_distance2: float = None
):
    """
    2つのモデルを対戦させ、結果を評価・保存する。

    Args:
        model_path1: プレイヤー1のモデルへのパス。
        model_path2: プレイヤー2のモデルへのパス ("random"も可)。
        num_games: 対戦するゲーム数。
        track_rating: レーティングの推移を追跡するかどうか。
        rating_interval: レーティングを計算する間隔（ゲーム数）。
        override_noised1: プレイヤー1のnoisedパラメータを上書き（for_networkモデルのみ）。
        override_distance1: プレイヤー1のdistanceパラメータを上書き（for_networkモデルのみ）。
        override_noised2: プレイヤー2のnoisedパラメータを上書き（for_networkモデルのみ）。
        override_distance2: プレイヤー2のdistanceパラメータを上書き（for_networkモデルのみ）。
    """
    # エージェントの準備
    agent1 = get_agent_from_config(
        model_path1,
        override_noised=override_noised1,
        override_distance=override_distance1
    )
    agent1.eval()

    # Noisedとdistanceの上書き情報を表示
    noised_info1 = f" (override noised={override_noised1}, distance={override_distance1})" if (override_noised1 is not None or override_distance1 is not None) else ""
    print(f"Player 1 Agent loaded from {model_path1}{noised_info1}")

    if model_path2.lower() == "random":
        agent2 = RandomPolicy()
        print("Player 2 is a RandomPolicy agent.")
    else:
        agent2 = get_agent_from_config(
            model_path2,
            override_noised=override_noised2,
            override_distance=override_distance2
        )
        agent2.eval()

        # Noisedとdistanceの上書き情報を表示
        noised_info2 = f" (override noised={override_noised2}, distance={override_distance2})" if (override_noised2 is not None or override_distance2 is not None) else ""
        print(f"Player 2 Agent loaded from {model_path2}{noised_info2}")

    # 結果記録用
    results = {
        "model1": model_path1,
        "model2": model_path2,
        "num_games": num_games,
        "summary": {"wins_p1": 0, "wins_p2": 0, "draws": 0},
        "game_logs": [],
    }

    # レーティング追跡用
    if track_rating:
        results["rating_tracking"] = {
            "intervals": [],
            "rating_p1": [],
            "rating_p2": [],
            "win_rate_p1": [],
            "win_rate_p2": [],
            "draw_rate": []
        }
        # レーティング初期化
        rating_p1 = INITIAL_RATING
        rating_p2 = INITIAL_RATING
        interval_wins_p1 = 0
        interval_wins_p2 = 0
        interval_draws = 0

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
            if track_rating:
                interval_wins_p1 += 1
        elif winner == -1:
            results["summary"]["wins_p2"] += 1
            if track_rating:
                interval_wins_p2 += 1
        else:
            results["summary"]["draws"] += 1
            if track_rating:
                interval_draws += 1

        # レーティングの計算と更新（指定された間隔ごと）
        if track_rating and (i + 1) % rating_interval == 0:
            # この間隔での勝率を計算
            interval_games = interval_wins_p1 + interval_wins_p2 + interval_draws

            # スコアを計算 (win=1, draw=0.5, loss=0)
            score_p1 = interval_wins_p1 + 0.5 * interval_draws
            score_p2 = interval_wins_p2 + 0.5 * interval_draws

            # 期待勝率を計算
            expected_p1 = calculate_expected_score(rating_p1, rating_p2) * interval_games
            expected_p2 = calculate_expected_score(rating_p2, rating_p1) * interval_games

            # レーティングを更新
            rating_p1 = rating_p1 + K_FACTOR * (score_p1 - expected_p1) / interval_games
            rating_p2 = rating_p2 + K_FACTOR * (score_p2 - expected_p2) / interval_games

            # 間隔ごとの勝率を記録
            win_rate_p1 = interval_wins_p1 / interval_games
            win_rate_p2 = interval_wins_p2 / interval_games
            draw_rate = interval_draws / interval_games

            # レーティング推移を記録
            results["rating_tracking"]["intervals"].append(i + 1)
            results["rating_tracking"]["rating_p1"].append(float(rating_p1))
            results["rating_tracking"]["rating_p2"].append(float(rating_p2))
            results["rating_tracking"]["win_rate_p1"].append(float(win_rate_p1))
            results["rating_tracking"]["win_rate_p2"].append(float(win_rate_p2))
            results["rating_tracking"]["draw_rate"].append(float(draw_rate))

            # 次の間隔のためにカウンターをリセット
            interval_wins_p1 = 0
            interval_wins_p2 = 0
            interval_draws = 0

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

    # 上書きパラメータがある場合、ファイル名に追加
    suffix = ""
    if override_noised1 is not None or override_distance1 is not None:
        suffix += f"_p1_noised{override_noised1}_dist{override_distance1}"
    if override_noised2 is not None or override_distance2 is not None and model_path2.lower() != "random":
        suffix += f"_p2_noised{override_noised2}_dist{override_distance2}"

    log_path = os.path.join(log_dir, f"eval_{p1_name}_vs_{p2_name}{suffix}.json")

    with open(log_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nEvaluation results saved to {log_path}")

    # レーティング推移をグラフとして保存
    if track_rating and len(results["rating_tracking"]["intervals"]) > 0:
        plt.figure(figsize=(12, 8))

        # レーティング推移グラフ
        plt.subplot(2, 1, 1)
        plt.plot(
            results["rating_tracking"]["intervals"],
            results["rating_tracking"]["rating_p1"],
            'b-',
            label=f"Player 1 ({os.path.basename(model_path1)})"
        )
        plt.plot(
            results["rating_tracking"]["intervals"],
            results["rating_tracking"]["rating_p2"],
            'r-',
            label=f"Player 2 ({os.path.basename(model_path2)})"
        )
        plt.xlabel("Game Number")
        plt.ylabel("Elo Rating")
        plt.title("Elo Rating Progression")
        plt.legend()
        plt.grid(True)

        # 勝率推移グラフ
        plt.subplot(2, 1, 2)
        plt.plot(
            results["rating_tracking"]["intervals"],
            results["rating_tracking"]["win_rate_p1"],
            'b-',
            label="Player 1 Win Rate"
        )
        plt.plot(
            results["rating_tracking"]["intervals"],
            results["rating_tracking"]["win_rate_p2"],
            'r-',
            label="Player 2 Win Rate"
        )
        plt.plot(
            results["rating_tracking"]["intervals"],
            results["rating_tracking"]["draw_rate"],
            'g-',
            label="Draw Rate"
        )
        plt.xlabel("Game Number")
        plt.ylabel("Rate")
        plt.title(f"Win/Draw Rate per {rating_interval} Games")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # グラフを保存
        graph_path = os.path.join(log_dir, f"eval_{p1_name}_vs_{p2_name}{suffix}_rating.png")
        plt.savefig(graph_path)
        print(f"Rating progression graph saved to {graph_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(run_evaluation)

    # 使用例:
    # 基本的な評価: python evaluate.py model1.pth model2.pth --num_games=1000
    # レーティング追跡: python evaluate.py model1.pth model2.pth --num_games=1000 --track_rating=True --rating_interval=100
    # ノイズ設定の上書き: python evaluate.py model1.pth model2.pth --override_noised1=False --override_distance1=20.0
    # 両方のモデルのノイズ設定を変更: python evaluate.py model1.pth model2.pth --override_noised1=True --override_distance1=10.0 --override_noised2=False --override_distance2=5.0
