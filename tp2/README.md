# TP2

## Description

In this TP, students are required to solve a problem using advanced RL algorithms studied in class. In my solution, I
continued the work done in previous TP and implemented a PPO agent for the Tetris game.
A [report](resources/??) with the analysis and conclusions of the work done is available.

## Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/maxit1992/MIA_RL2.git
    cd MIA_RL2/tp2
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Game

This version only allows `stable_baselines3` models to play. To start the game, run the following command:

```sh
python -m src.main play --model-file=<model-file>
```

## Training a Model

To train a model, run the following command:

```sh
python -m src.main train
```

The only model available for training is the PPO model. This can be changed in `trainer.py` by modifying the `model`
variable.

Already trained models are available in the `resources` folder.

## Code Quality

All code contains documentation. No vulnerabilities or code smells were detected by SonarQube analysis.

