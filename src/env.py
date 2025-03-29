import tqdm
from typing import Final
import matplotlib.pyplot as plt
import numpy as np

from src.tictactoe import TicTacToe

SEED: Final[int] = 42

def set_seed(seed: int=42):
    """
    Setting seed
    """
    np.random.seed(seed)

# If you do not want to set seed, you can comment out the below set_seed() function.
set_seed(SEED)

class Env:
    def __init__(self, agent1, agent2, tictactoc=TicTacToe(), elorate=1500):
        self.agent1 = agent2
        self.agent2 = agent1
        self.tictactoc = tictactoc
        self.loss_f1 = []
        self.loss_f2 = []
        self.P1 = []
        self.P2 = []
        self.Draw = []
        self.rate =  [elorate]
        self.elorate = 1500

    def _get_reward_for_agent1(self):
        winner = self.tictactoc.checkwinner()
        if winner == 1:
            return 1
        elif winner == -1:
            return -1
        else:
            return 0

    def _get_reward_for_agent2(self):
        winner = self.tictactoc.checkwinner()
        if winner == 1:
            return -1
        elif winner == -1:
            return 1
        else:
            return 0

    # def plot_loss(self, train):
    #     if not self.agent1.stop and train:
    #         plt.plot(range(len(self.loss_f1)), self.loss_f1, label="player1loss")
    #     if not self.agent2.stop and train:
    #         plt.plot(range(len(self.loss_f2)), self.loss_f2, label="player2loss")
    #     if ((not self.agent1.stop) or (not self.agent2.stop)) and train:
    #         plt.xlabel("# of episodes")
    #         plt.ylabel("Average loss value")
    #         plt.title("Training loss")
    #         plt.legend()
    #         plt.show()
    #     self.loss_f1.clear()
    #     self.loss_f2.clear()

    def collect_winrate(self, record):
        winrate = record[1] / (record[0]+record[1]+record[-1])
        self.P1.append(winrate)

        winrate = record[-1] / (record[0]+record[1]+record[-1])
        self.P2.append(winrate)

        winrate = record[0] / (record[0]+record[1]+record[-1])
        self.Draw.append(winrate)

    # def plot_winrate(self):
    #     plt.plot(range(1,len(self.P1)+1), self.P1, label="player1wins")
    #     plt.plot(range(1,len(self.P2)+1), self.P2, label="player2wins")
    #     plt.plot(range(1,len(self.Draw)+1), self.Draw, label="draw")
    #     plt.xlabel("# of episodes")
    #     plt.ylabel("rate")
    #     plt.title("Rate of player1 wins, player2 wins and draw")
    #     plt.legend()
    #     plt.show()
    #     self.P1.clear()
    #     self.P2.clear()
    #     self.Draw.clear()

    def collect_elorate(self, win_count, draw_count, iters, rate_update):
        K = 0.35
        iters //= rate_update
        W = 1/(10**((1500-self.rate[iters])/400) + 1)
        # new_rate = self.rate[iters] + K*(win_count - (rate_update-draw_count)*W)
        new_rate = self.rate[iters] + K*(win_count + 0.5*draw_count - rate_update*W)
        self.rate.append(new_rate)

    # def plot_elorate(self, rate_update):
    #     plt.plot(range(0,len(self.P1)+1, rate_update), self.rate, label="rate")
    #     plt.xlabel("# of games")
    #     plt.ylabel("elorate")
    #     plt.legend()
    #     plt.show()
    #     self.elorate=self.rate[-1]
    #     self.rate.clear()

    def train(self, episodes, train=True, visualize=False):
        record = {0: 0, 1: 0, -1: 0}
        draw_count = 0
        win_count = 0
        ### change
        rate_update = 100
        for i in tqdm.tqdm(range(episodes)):
            # Swap the first and second move after each session.
            tmp = self.agent1
            self.agent1 = self.agent2
            self.agent2 = tmp
            winner = self.game_exec(train, visualize)
            if i%2 == 0:
                record[winner] += 1
                if winner == 1:
                    win_count += 1
                if winner == 0:
                    draw_count += 1
            else:
                record[-winner] += 1
                if -winner == 1:
                    win_count += 1
                if winner == 0:
                    draw_count += 1
            if (i+1)%rate_update == 0:
                self.collect_elorate(win_count, draw_count, i, rate_update)
                win_count = 0
                draw_count = 0
            self.collect_winrate(record)
        print("result:")
        print("Player1   Draw   Player2")
        print(f"  {record[1]}      {record[0]}      {record[-1]}  ")
        self.plot_loss(train)
        ### change
        with open("test.txt", "a") as f:
            # print(self.P1, file=f)
            # f.write("\n")
            print(self.rate, file=f)
            f.write("\n")
        print(self.P1)
        print(self.rate)
        r = self.rate.copy()
        self.plot_elorate(rate_update)
        self.plot_winrate()
        return r

    def game_exec(self, train=False, visualize=True):
        self.tictactoc.reset_board()
        self.loss_f1t = []
        self.loss_f2t = []
        self.steps = 0
        if visualize:
            self.tictactoc.display_board()
        while not self.tictactoc.gameover():
            state = self.tictactoc.get_state(player=1)
            while True:
                self.agent1.player = 1
                action = self.agent1.action(self.tictactoc)
                if self.tictactoc.is_valid_action(action, player=1):
                    self.tictactoc.place(action, player=1)
                    self.steps += 1
                    if train and not self.agent1.stop:
                        reward = 0
                        if self.tictactoc.gameover():
                            reward = self._get_reward_for_agent1()
                        if action is not None:
                            self.loss_num = self.agent1.update(self.tictactoc, state, action, reward)
                            self.loss_f1t.append(self.loss_num)
                    break
                else:
                    print(f"Invalid action was provided: ({action})")
            if self.tictactoc.gameover():
                if visualize:
                    self.tictactoc.display_board()
                break
            if visualize:
                print("Black's turn:")
                self.tictactoc.display_board()

            state = self.tictactoc.get_state(player=-1)
            while True:
                self.agent2.player = -1
                action = self.agent2.action(self.tictactoc)
                if self.tictactoc.is_valid_action(action, player=-1):
                    self.tictactoc.place(action, player=-1)
                    self.steps += 1
                    if train and not self.agent2.stop:
                        reward = 0
                        if self.tictactoc.gameover():
                            reward = self._get_reward_for_agent2()
                        if action is not None:
                            self.loss_num = self.agent2.update(self.tictactoc, state, action, reward)
                            self.loss_f2t.append(self.loss_num)
                    break
                else:
                    print(f"Invalid action was provided: ({action})")
            if self.tictactoc.gameover():
                if visualize:
                    self.tictactoc.display_board()
                break
            if visualize:
                print("White's turn")
                self.tictactoc.display_board()

        winner = self.tictactoc.checkwinner()
        if not self.agent1.stop:
            self.loss_f1.append(np.mean(np.array(self.loss_f1t)))
        if not self.agent2.stop:
            self.loss_f2.append(np.mean(np.array(self.loss_f2t)))
            # print(f"avg {np.mean(np.array(self.loss_f1t))} and {np.mean(np.array(self.loss_f2t))}")
            # print(f"sum {sum(self.loss_f1t)} and {sum(self.loss_f2t)}")
        # print(self.steps)

        if visualize:
            if winner == 0:
                print("Draw")
            elif winner == 1:
                print("Player1 wins!!")
            else:
                print("Player2 wins!!")
        return winner

class Environment:
    def __init__(self, agent1, agent2, tictactoc=TicTacToe()):
        self.agent1 = agent1
        self.agent2 = agent2
        self.tictactoc = tictactoc
        self.loss_f1 = []
        self.loss_f2 = []
        self.P1 = []
        self.P2 = []
        self.Draw = []

    def _get_reward_for_agent1(self):
        winner = self.tictactoc.checkwinner()
        if winner == 1:
            return 1
        elif winner == -1:
            return -1
        else:
            return 0

    def _get_reward_for_agent2(self):
        winner = self.tictactoc.checkwinner()
        if winner == 1:
            return -1
        elif winner == -1:
            return 1
        else:
            return 0

    # def plot_loss(self, train):
    #     if not self.agent1.stop and train:
    #         plt.plot(range(len(self.loss_f1)), self.loss_f1, label="player1loss")
    #     if not self.agent2.stop and train:
    #         plt.plot(range(len(self.loss_f2)), self.loss_f2, label="player2loss")
    #     if ((not self.agent1.stop) or (not self.agent2.stop)) and train:
    #         plt.xlabel("# of episodes")
    #         plt.ylabel("Average loss value")
    #         plt.title("Training loss")
    #         plt.legend()
    #         plt.show()
    #     self.loss_f1.clear()
    #     self.loss_f2.clear()

    def collect_winrate(self, record):
        winrate = record[1] / (record[0]+record[1]+record[-1])
        self.P1.append(winrate)

        winrate = record[-1] / (record[0]+record[1]+record[-1])
        self.P2.append(winrate)

        winrate = record[0] / (record[0]+record[1]+record[-1])
        self.Draw.append(winrate)

    # def plot_winrate(self):
    #     plt.plot(range(len(self.P1)), self.P1, label="player1wins")
    #     plt.plot(range(len(self.P2)), self.P2, label="player2wins")
    #     plt.plot(range(len(self.Draw)), self.Draw, label="draw")
    #     plt.xlabel("# of episodes")
    #     plt.ylabel("rate")
    #     plt.title("Rate of player1 wins, player2 wins and draw")
    #     plt.legend()
    #     plt.show()
    #     self.P1.clear()
    #     self.P2.clear()
    #     self.Draw.clear()

    def train(self, episodes, train=True, visualize=False):
        record = {0: 0, 1: 0, -1: 0}
        for _ in tqdm.tqdm(range(episodes)):
            winner = self.game_exec(train, visualize)
            record[winner] += 1
            self.collect_winrate(record)
        print("result:")
        print("Player1   Draw   Player2")
        print(f"  {record[1]}      {record[0]}      {record[-1]}  ")
        self.plot_loss(train)
        self.plot_winrate()

    def game_exec(self, train=False, visualize=True):
        self.tictactoc.reset_board()
        self.loss_f1t = []
        self.loss_f2t = []
        self.steps = 0
        if visualize:
            self.tictactoc.display_board()
        while not self.tictactoc.gameover():
            state = self.tictactoc.get_state(player=1)
            while True:
                self.agent1.player = 1
                action = self.agent1.action(self.tictactoc)
                if self.tictactoc.is_valid_action(action, player=1):
                    self.tictactoc.place(action, player=1)
                    self.steps += 1
                    if train and not self.agent1.stop:
                        reward = 0
                        if self.tictactoc.gameover():
                            reward = self._get_reward_for_agent1()
                        if action is not None:
                            self.loss_num = self.agent1.update(self.tictactoc, state, action, reward)
                            self.loss_f1t.append(self.loss_num)
                    break
                else:
                    print(f"Invalid action was provided: ({action})")
            if self.tictactoc.gameover():
                if visualize:
                    self.tictactoc.display_board()
                break
            if visualize:
                print("Black's turn:")
                self.tictactoc.display_board()

            state = self.tictactoc.get_state(player=-1)
            while True:
                self.agent2.player = -1
                action = self.agent2.action(self.tictactoc)
                if self.tictactoc.is_valid_action(action, player=-1):
                    self.tictactoc.place(action, player=-1)
                    self.steps += 1
                    if train and not self.agent2.stop:
                        reward = 0
                        if self.tictactoc.gameover():
                            reward = self._get_reward_for_agent2()
                        if action is not None:
                            self.loss_num = self.agent2.update(self.tictactoc, state, action, reward)
                            self.loss_f2t.append(self.loss_num)
                    break
                else:
                    print(f"Invalid action was provided: ({action})")
            if self.tictactoc.gameover():
                if visualize:
                    self.tictactoc.display_board()
                break
            if visualize:
                print("White's turn")
                self.tictactoc.display_board()

        winner = self.tictactoc.checkwinner()
        if not self.agent1.stop:
            self.loss_f1.append(np.mean(np.array(self.loss_f1t)))
        if not self.agent2.stop:
            self.loss_f2.append(np.mean(np.array(self.loss_f2t)))
            # print(f"avg {np.mean(np.array(self.loss_f1t))} and {np.mean(np.array(self.loss_f2t))}")
            # print(f"sum {sum(self.loss_f1t)} and {sum(self.loss_f2t)}")
        # print(self.steps)

        if visualize:
            if winner == 0:
                print("Draw")
            elif winner == 1:
                print("Player1 wins!!")
            else:
                print("Player2 wins!!")
        return winner
