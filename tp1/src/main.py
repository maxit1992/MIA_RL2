"""
This script allows you to play Tetris game or to train a DQN model to play it and then run the game.

Usage:
    main.py play [--model-file=<model-file> --model=<model_type>] [--human]
    main.py train --model=<model-type>

Options:
    -h --help                       Show this screen.
    --human                         Play tetris using keyboard.
    --model-file=<model-file>       Specify the model used to play the game.
    --model=<model_type>            The type of model to play/train. Valid options are: ddqn or dueling-dqn.
"""
from docopt import docopt
import torch

from src.model.trainer import TrainerDDQN
from .tetris.game_ui import GameUI
from .model.model import DDQN, DuelingDQN


def play(args: dict):
    """
    Play Tetris game with the specified player.

    Args:
        args (dict): Command line arguments.
    """
    if args['--model-file']:
        if args['--model'] == 'ddqn':
            model = DDQN(output_dim=5)
        elif args['--model'] == 'dueling-dqn':
            model = DuelingDQN(output_dim=5)
        else:
            raise RuntimeError('Invalid model type')
        model.load_state_dict(torch.load(args['--model-file']))
        model.eval()
        ui = GameUI()
        ui.play(model)
    elif args['--human']:
        ui = GameUI()
        ui.play()
    else:
        raise RuntimeError('Invalid run mode')


def train(args: dict):
    """
    Train the DQN RL model.

    Args:
        args (dict): Command line arguments.
    """
    if args['--model'] == 'ddqn':
        model = DDQN(output_dim=5)
    elif args['--model'] == 'dueling-dqn':
        model = DuelingDQN(output_dim=5)
    else:
        raise RuntimeError('Invalid model type')
    trainer = TrainerDDQN(model=model)
    trainer.train()

def main():
    """
    Main function to parse command line arguments and start the game.
    """
    args = docopt(__doc__)
    if args['play']:
        play(args)
    elif args['train']:
        train(args)
    else:
        print("Invalid command. Use --help for usage information.")


if __name__ == "__main__":
    main()