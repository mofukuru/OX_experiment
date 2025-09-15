import numpy as np
import torch
import tqdm

from src.tictactoe import TicTacToe


class Environment:
    def __init__(self, agent1, agent2, tictactoe=TicTacToe()):
        self.agent1 = agent1
        self.agent2 = agent2
        self.tictactoe = tictactoe

    def train(self, episodes, train=True, visualize=False):
        loss_p1_list = []
        loss_p2_list = []
        reward_p1_list = []
        reward_p2_list = []

        for i in range(episodes):
            if i % 2 == 1:
                p1, p2 = self.agent2, self.agent1
            else:
                p1, p2 = self.agent1, self.agent2

            winner, loss1, loss2, reward1, reward2 = self.game_exec(
                p1, p2, train, visualize
            )

            if i % 2 == 1:
                loss_p1_list.append(loss2)
                loss_p2_list.append(loss1)
                reward_p1_list.append(reward2)
                reward_p2_list.append(reward1)
            else:
                loss_p1_list.append(loss1)
                loss_p2_list.append(loss2)
                reward_p1_list.append(reward1)
                reward_p2_list.append(reward2)

        return loss_p1_list, loss_p2_list, reward_p1_list, reward_p2_list

    def play(self, visualize=False):
        winner, _, _, _, _ = self.game_exec(
            self.agent1, self.agent2, train=False, visualize=visualize
        )
        return winner, self.tictactoe.get_board_history()

    def game_exec(self, p1_agent, p2_agent, train=False, visualize=False):
        self.tictactoe.reset_board()
        s1, a1, s2, a2 = None, None, None, None
        loss1_hist, loss2_hist = [], []

        # 初期ボードを表示
        if visualize:
            print("Initial board:")
            self.tictactoe.display_board(sleep_secs=0.5)

        while not self.tictactoe.gameover():
            # Player 1's Turn
            print("Player 1's turn (O)") if visualize else None
            s1 = self.tictactoe.get_state(player=1)
            if train and s2 is not None:
                if hasattr(p2_agent, "learn"):
                    p2_agent.memorize(s2, a2, s1, 0, False)
                    loss = p2_agent.learn()
                    if loss:
                        loss2_hist.append(loss)
                elif hasattr(p2_agent, "update"):
                    loss = p2_agent.update(self.tictactoe, s2, a2, 0)
                    if loss:
                        loss2_hist.append(loss)

            a1 = p1_agent.action(self.tictactoe)
            if a1 is None:
                break
            self.tictactoe.place(a1, player=1)

            # Player 1の手を表示
            if visualize:
                print(f"Player 1 placed at: {a1}")
                self.tictactoe.display_board(sleep_secs=0.5)

            if self.tictactoe.gameover():
                break

            # Player 2's Turn
            print("Player 2's turn (X)") if visualize else None
            s2 = self.tictactoe.get_state(player=-1)
            if train:
                if hasattr(p1_agent, "learn"):
                    p1_agent.memorize(s1, a1, s2, 0, False)
                    loss = p1_agent.learn()
                    if loss:
                        loss1_hist.append(loss)
                elif hasattr(p1_agent, "update"):
                    loss = p1_agent.update(self.tictactoe, s1, a1, 0)
                    if loss:
                        loss1_hist.append(loss)

            a2 = p2_agent.action(self.tictactoe)
            if a2 is None:
                break
            self.tictactoe.place(a2, player=-1)

            # Player 2の手を表示
            if visualize:
                print(f"Player 2 placed at: {a2}")
                self.tictactoe.display_board(sleep_secs=0.5)

        winner = self.tictactoe.checkwinner()
        reward1 = self._get_reward(1, winner)
        reward2 = self._get_reward(-1, winner)

        if train:
            if a1 is not None:
                if hasattr(p1_agent, "learn"):
                    p1_agent.memorize(s1, a1, None, reward1, True)
                    loss = p1_agent.learn()
                    if loss:
                        loss1_hist.append(loss)
                elif hasattr(p1_agent, "update"):
                    loss = p1_agent.update(self.tictactoe, s1, a1, reward1)
                    if loss:
                        loss1_hist.append(loss)
            if a2 is not None:
                if hasattr(p2_agent, "learn"):
                    p2_agent.memorize(s2, a2, None, reward2, True)
                    loss = p2_agent.learn()
                    if loss:
                        loss2_hist.append(loss)
                elif hasattr(p2_agent, "update"):
                    loss = p2_agent.update(self.tictactoe, s2, a2, reward2)
                    if loss:
                        loss2_hist.append(loss)

        avg_loss1 = np.mean(loss1_hist) if loss1_hist else 0
        avg_loss2 = np.mean(loss2_hist) if loss2_hist else 0
        return winner, avg_loss1, avg_loss2, reward1, reward2

    def _get_reward(self, player_id, winner=None):
        if winner is None:
            winner = self.tictactoe.checkwinner()

        if winner == 0:
            return 0
        if winner == player_id:
            return 1
        else:
            return -1
