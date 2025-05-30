"""
This script allows you to train a PPO model to play Tetris or to run the game with a trained model.

Usage:
    main.py play --model-file=<model-file>
    main.py train

Options:
    -h --help                       Show this screen.
    --model-file=<model-file>       Specify the model used to play the game.
"""
import os
import sys

from docopt import docopt
from stable_baselines3 import PPO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from .tetris.trainer import TrainerPPO
from .tetris.game_ui import GameUI


def play(args: dict):
    """
    Play Tetris game with the specified model.

    Args:
        args (dict): Command line arguments.
    """
    if args['--model-file']:
        model = PPO.load(args['--model-file'])
        ui = GameUI()
        ui.play(model)
    else:
        raise RuntimeError('Invalid run mode')


def train():
    """
    Train the PPO RL model.
    """
    trainer = TrainerPPO()
    trainer.train()


def main():
    """
    Main function to parse command line arguments and start the game.
    """
    args = docopt(__doc__)
    if args['play']:
        play(args)
    elif args['train']:
        train()
    else:
        print("Invalid command. Use --help for usage information.")


if __name__ == "__main__":
    main()
