# OX_experiment

This is the source code for [Quantitative Evaluation of Quantum/Classical Neural Network Using a Game Solver Metric](https://arxiv.org/abs/2503.21514)

## Usage

This project uses reinforcement learning to train agents through self-play in a Tic-Tac-Toe environment.

### Training

The `train.py` script is used to train the agents. It supports several agent types (CNN, QCNN, etc.) and handles the entire training process.

During training, the script will:
- Use an epsilon-greedy strategy with a decaying epsilon, meaning the agent explores less as it learns more.
- For DQN-based agents, use an Experience Replay buffer to stabilize learning.
- Measure the total training time.
- Save the fully trained model weights as a `.pth` file to the `./models/train/` directory.
- Save detailed training logs (rewards, losses, epsilon value per episode) and the final configuration to a JSON file in the `./logs/train/` directory.

**How to Run:**

The script uses the `fire` library to handle command-line arguments. You specify the agent type to train and its specific parameters.

```bash
# Example: Train a CNN agent for 5000 episodes
python train.py cnn --network=CCNN --total_episodes=5000
```

### Evaluation (Head-to-Head)

The `evaluate.py` script is used to evaluate two trained agents by pitting them against each other.

The script will:
- Load two specified agents (or one agent and a random opponent).
- Run a specified number of games between them, alternating the starting player for fairness.
- Calculate and display the win/loss/draw statistics and win rates.
- Save a detailed report, including game-by-game logs, to a JSON file in the `./logs/eval/` directory.

**How to Run:**

You need to provide paths to the two models you want to evaluate. To play against a random agent, use the keyword `random` as the second model path.

```bash
# Example: Evaluate model1.pth against model2.pth for 100 games
python evaluate.py models/train/model1.pth models/train/model2.pth --num_games=100

# Example: Evaluate model1.pth against a random agent for 100 games
python evaluate.py models/train/model1.pth random --num_games=100
```

### Tournament (Round-Robin Evaluation)

The `tournament.py` script provides a powerful way to evaluate and rank all of your trained models.

The script will:
- Automatically discover all trained models in the `./models/train/` directory.
- Conduct a round-robin tournament where every model plays against every other model.
- Use the Elo rating system to assign a numerical strength rating to each model, starting from 1500.
- Update ratings after each match based on the outcome.
- After all matches are complete, it will print a final ranking of all models.
- Save the final leaderboard and detailed match histories to `./logs/tournament_results.json`.

**How to Run:**

Simply run the script from your terminal. You can optionally specify the number of games to play for each match-up.

```bash
# Run a tournament with default settings (100 games per match)
python tournament.py

# Run a tournament with 200 games per match
python tournament.py --num_games_per_match=200
```

### Play and Replay a Game

The `play_game.py` script allows you to watch a single game between two agents and then replay the moves one by one.

**How to Run:**

Provide the paths to the two models you want to see play.

```bash
# Example: Play a game between model1.pth and model2.pth
python play_game.py models/train/model1.pth models/train/model2.pth
```

### Play Against a Trained Agent

The `play_vs_human.py` script allows you to play Tic-Tac-Toe against any trained agent. This interactive mode lets you test the strength of different agents yourself and see their decision-making in real time.

The script will:
- Let you play as either the first player (O) or the second player (X)
- Save a record of the game, including all moves, to a JSON file in `./logs/human_games/`
- Replay the full game at the end of the match
- Automatically handle loading the agent model and displaying the board state

**How to Run:**

```bash
# Play against a model as the first player (O)
python play_vs_human.py play --model_path=models/train/model1.pth --human_player=1

# Play against a model as the second player (X)
python play_vs_human.py play --model_path=models/train/model1.pth --human_player=-1
```

During the game, enter your moves as "row col" (e.g., "0 0" for the top-left position).

JSON file and pth file name is not edited, and some of the content may be different.
