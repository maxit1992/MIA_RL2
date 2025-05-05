# TP1

## Description

This project implements a Tetris game with a graphical user interface (GUI) using Pygame. It also includes reinforcement
learning (RL) models, such as Double Deep Q-Network (DDQN) and Dueling DQN, to train an AI agent to play the game. The
project allows users to play the game manually or watch a trained model play.

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`
-

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/maxit1992/MIA_RL2.git
    cd MIA_RL2/tp1
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Game

To start the game, run the following command:

```sh
python -m src.main play [--model-file=<model-file> --model=<model_type>] [--human]
```

- If --model-file together with --model is provided, the game will be played by the trained model.
- If no model is provided, the game will be controlled by the user via keyboard inputs.

### Keyboard Controls

- Arrow Left: Move the block left.
- Arrow Right: Move the block right.
- Arrow Down: Move the block down faster.
- Arrow Up: Rotate the block.

## Training a Model

To train a model, run the following command:

```sh
python -m src.main train --model=<model-type>
```

The model can be either `ddqn` or `dueling-dqn`. The training process will run for a fixed number of
episodes, and the model will be saved to a file.

Already trained models are available in the `resources` folder.

## Code Quality

All code contains documentation. No vulnerabilities or code smells were detected by SonarQube analysis.

