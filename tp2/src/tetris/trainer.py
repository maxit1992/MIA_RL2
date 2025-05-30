import os
import pickle as pkl
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env

from .gym_env import TetrisGymWrapper


class _RewardLogging(BaseCallback):
    """
    Custom callback for logging the reward per episode.
    """

    def __init__(self, verbose=0):
        super(_RewardLogging, self).__init__(verbose)
        self.reward_history = []

    def _on_training_start(self) -> None:
        self.reward_history = []

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            self.reward_history.append(self.training_env.env_method("get_last_score")[0])
        return True


class TrainerPPO:
    """
    Trainer class for training a PPO model on the Tetris game.
    This class initializes the environment, sets up the model, and starts the training process.
    """

    def __init__(self):
        # No initialization needed for this trainer
        pass

    def train(self, total_timesteps: int = 10_000_000, save_freq: int = 100_000):
        """
        Train the PPO model on the Tetris game.

        Args:
            total_timesteps (int): Total number of timesteps for training.
            save_freq (int): Frequency of saving the model.
        """
        # Initialize the environment
        vec_env = make_vec_env(TetrisGymWrapper, n_envs=1)

        # Initialize the PPO model
        model = PPO("CnnPolicy", vec_env, verbose=1)

        # Train the model
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path="./resources/snapshot/",
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        reward_callback = _RewardLogging()
        callbacks = CallbackList([checkpoint_callback, reward_callback])
        model.learn(total_timesteps=total_timesteps, callback=callbacks)

        # Save the trained model
        episode_rewards_history = reward_callback.reward_history
        self.save_model(model, "./resources/model/", "ppo_")
        self.save_metric(episode_rewards_history, "./resources/model/", "rewards_")
        self.plot_rewards(episode_rewards_history)

    @staticmethod
    def save_model(model, base_path, prefix):
        """
        Save the model's parameters to a file.

        Args:
            model (PPO): The model to be saved.
            base_path (str): The base directory for saving the model.
            prefix (str): The prefix for the saved file name.
        """
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        model.save(base_path + prefix + datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + ".zip")

    @staticmethod
    def save_metric(metric, base_path, prefix):
        """
        Save a metric (e.g., rewards) to a file.

        Args:
            metric (list): The metric data to be saved.
            base_path (str): The base directory for saving the metric.
            prefix (str): The prefix for the saved file name.
        """
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        file_name = base_path + prefix + datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + ".pkl"
        with open(file_name, "wb") as file:
            pkl.dump(metric, file)

    @staticmethod
    def plot_rewards(rewards):
        """
        Plot the rewards over episodes.

        Args:
            rewards (list): A list of rewards per episode.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(rewards) + 1), rewards, label="Reward per episode", alpha=0.4)
        if len(rewards) >= 50:
            rewards_smoothed = np.convolve(rewards, np.ones(50) / 50, mode="valid")
            plt.plot(range(50, len(rewards) + 1), rewards_smoothed, label="Moving average over 50 episodes",
                     color="red", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("PPO Playing Tetris")
        plt.legend()
        plt.grid(True)
        plt.show()
