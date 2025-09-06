import json
import os
import time
from datetime import datetime

import fire
import torch

from src.agent import (
    CNNAgent,
    CQCAgent,
    CQCAgent_network,
    Human,
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
            strict=False,  # エラー回避のためstrict=Falseを追加
        )
    else:
        print(
            f"Warning: Model file not found at {model_path}. Agent is not loaded."
        )

    return agent


def save_game_record(history, winner, human_player, ai_model_name):
    """
    ゲームの棋譜を保存する関数
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/human_games/game_{timestamp}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # プレイヤー情報を記録
    player1 = "Human" if human_player == 1 else ai_model_name
    player2 = "Human" if human_player == -1 else ai_model_name

    # 勝者を記録
    if winner == 0:
        winner_name = "Draw"
    elif winner == 1:
        winner_name = player1
    else:
        winner_name = player2

    # 棋譜データを作成
    game_data = {
        "timestamp": timestamp,
        "player1": player1,
        "player2": player2,
        "winner": winner_name,
        "moves": [
            {"position": [move[0], move[1]], "player": 1 if i % 2 == 0 else -1}
            for i, move in enumerate(history)
        ],
    }

    # ファイルに保存
    with open(filename, "w") as f:
        json.dump(game_data, f, indent=2)

    print(f"Game record saved to {filename}")

    return filename


def play_vs_human(model_path, human_player=1):
    """
    ヒューマンプレイヤーとAIエージェントが対戦する関数
    human_player: 1ならヒューマンが先手(O)、-1なら後手(X)
    """
    # AIエージェントをロード
    ai_agent = get_agent_from_config(model_path)
    ai_agent.eval()  # 評価モード

    # ヒューマンプレイヤーを作成
    human_agent = Human(player=human_player)

    # 対戦の設定
    if human_player == 1:
        # ヒューマンが先手
        env = Environment(human_agent, ai_agent)
        player1_name = "Human"
        player2_name = os.path.basename(model_path)
    else:
        # AIが先手
        env = Environment(ai_agent, human_agent)
        player1_name = os.path.basename(model_path)
        player2_name = "Human"

    print(f"--- Starting Game: {player1_name} (O) vs {player2_name} (X) ---")
    print("Board positions are entered as 'row col' (e.g. '0 0' for top-left)")
    print()

    # ゲームを実行
    winner, history = env.play(visualize=True)

    # 結果の表示
    print("--- Game Finished ---")

    # 最終状態のボードを表示
    print("Final board state:")
    env.tictactoe.display_board(sleep_secs=0)

    if winner == 1:
        print(f"Winner: Player 1 ({player1_name})")
    elif winner == -1:
        print(f"Winner: Player 2 ({player2_name})")
    else:
        print("Draw")

    # 棋譜を保存
    save_game_record(
        history, winner, human_player, os.path.basename(model_path)
    )

    # 棋譜の再生
    print("\n--- Replaying Game ---")
    replay_board = TicTacToe()
    replay_board.display_board(sleep_secs=0)
    player_turn = 1
    for move in history:
        replay_board.place(move, player_turn)
        replay_board.display_board(sleep_secs=0.5)
        player_turn *= -1

    print("--- Replay Finished ---")

    return winner


if __name__ == "__main__":
    fire.Fire(
        {
            "play": play_vs_human,
        }
    )
