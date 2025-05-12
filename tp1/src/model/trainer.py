import collections
import copy
import os
import pickle as pkl
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..tetris.game import Game


class TrainerDDQN:
    """
    A trainer class for training a Deep Q-Network (DQN) model to play Tetris.

    This class handles the training process, including environment interaction,
    experience replay, and model optimization.
    """
    TRANSITION = collections.namedtuple("Transition",
                                        ("state", "action", "reward", "next_state"))

    def __init__(self, model: torch.nn.Module):
        """
        Initialize the TrainerDDQN class.

        Args:
            model (torch.nn.Module): The DQN model to be trained.
        """
        self.eval_network = model
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon_start = 1
        self.epsilon_end = 0.001
        self.epsilon_decay_episodes = 600
        self.buffer_size = 10_000
        self.batch_size = 64
        self.target_update_freq = 100

    def train(self, episodes: int = 1000, max_steps: int = 1000, refresh_rate: int = 50):
        """
        Train the DQN model.

        Args:
            episodes (int): The total number of episodes for training.
            max_steps (int): The maximum number of steps per episode.
            refresh_rate (int): The frequency (in episodes) to log training progress.
        """
        # Initialize the environment
        game = Game()

        # Initialize the networks
        target_network = copy.deepcopy(self.eval_network)
        target_network.load_state_dict(self.eval_network.state_dict())
        optimizer = torch.optim.Adam(self.eval_network.parameters(), lr=self.learning_rate, weight_decay=0)
        loss_fn = torch.nn.MSELoss()
        replay_buffer = collections.deque(maxlen=self.buffer_size)

        # Initialize training parameters
        epsilon = self.epsilon_start
        update_steps = 0
        episode_rewards_history = []
        loss_history = []

        for episode in range(1, episodes + 1):
            game.reset()
            episode_steps = 0

            for _ in range(max_steps):
                # Play a step in the game
                state, action, reward, next_state = self.play_step(game, epsilon)

                # Store transition in replay buffer
                replay_buffer.append(self.TRANSITION(state, action, reward, next_state))

                # Execute a model training step
                train_loss = self.train_step(self.eval_network, target_network, optimizer, loss_fn, replay_buffer)
                loss_history.append(train_loss)

                # End of episode step
                episode_steps += 1
                update_steps += 1
                if update_steps % self.target_update_freq == 0:
                    target_network.load_state_dict(self.eval_network.state_dict())

                if game.game_over:
                    break

            # End of episode
            epsilon = max(self.epsilon_end, self.epsilon_start -
                          (episode / self.epsilon_decay_episodes) * (self.epsilon_start - self.epsilon_end))
            episode_rewards_history.append(game.score)
            if episode % refresh_rate == 0:
                avg_reward = np.mean(episode_rewards_history[-refresh_rate:])
                avg_loss = np.mean(loss_history[-refresh_rate:])
                self.save_model(self.eval_network, "./resources/model_snapshot/", f"tetris_{episode}_")
                print(
                    f"Episode: {episode}/{episodes} | Steps: {episode_steps} | "
                    f"Avg reward: {avg_reward:.2f} | Avg loss: {avg_loss:.3f} | Epsilon: {epsilon:.3f}")

        # End training
        self.save_model(self.eval_network, "./resources/model/", "tetris_")
        self.save_metric(episode_rewards_history, "./resources/model/", "rewards_")
        self.plot_rewards(episode_rewards_history)

    def play_step(self, game: Game, epsilon: float):
        """
        Perform a single step in the game using the epsilon-greedy policy.

        Args:
            game (Game): The Tetris game environment.
            epsilon (float): The current epsilon value for exploration.

        Returns:
            tuple: A tuple containing the current state, action, reward, and next state.
        """
        # Current state
        state = torch.tensor(game.get_state(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        current_score = game.score

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(list(game.ACTION_SPACE.keys()))
        else:
            with torch.no_grad():
                q_values = self.eval_network(state)
                action = torch.argmax(q_values).item()
        game.ACTION_SPACE[action]()
        game.move_continue()

        # Next state
        action = torch.tensor([[action]], dtype=torch.long)
        game_over = game.game_over
        if game_over:
            next_state = None
            reward = -5
        else:
            next_state = torch.tensor(game.get_state(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            reward = game.score - current_score
        reward = torch.tensor([reward], dtype=torch.float32)

        return state, action, reward, next_state

    def train_step(self, eval_network: torch.nn.Module, target_network: torch.nn.Module,
                   optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, replay_buffer: collections.deque):
        """
        Perform a single training step using a batch of transitions from the replay buffer.

        Args:
            eval_network (torch.nn.Module): The evaluation network.
            target_network (torch.nn.Module): The target network.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            loss_fn (torch.nn.Module): The loss function.
            replay_buffer (collections.deque): The replay buffer storing transitions.

        Returns:
            float: The training loss for the current step.
        """
        if len(replay_buffer) < self.batch_size:
            return

        # Sample a batch of transitions
        batch = random.sample(replay_buffer, self.batch_size)
        batch = self.TRANSITION(*zip(*batch))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Forward step
        optimizer.zero_grad()
        state_action_values = eval_network(state_batch).gather(1, action_batch).squeeze(1)

        # Expected value
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            eval_next_q_values = eval_network(non_final_next_states)
            eval_best_next_actions = eval_next_q_values.max(1)[1].unsqueeze(1)
            target_next_q_values = target_network(non_final_next_states)
            selected_target_next_q_values = target_next_q_values.gather(1, eval_best_next_actions).squeeze(1)
            next_state_values[non_final_mask] = selected_target_next_q_values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        # Loss
        loss = loss_fn(state_action_values, expected_state_action_values)

        # Backward step
        loss.backward()
        optimizer.step()
        return loss.item()

    @staticmethod
    def save_model(model, base_path, prefix):
        """
        Save the model's parameters to a file.

        Args:
            model (torch.nn.Module): The model to be saved.
            base_path (str): The base directory for saving the model.
            prefix (str): The prefix for the saved file name.
        """
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        file_name = base_path + prefix + datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + ".pkl"
        torch.save(model.state_dict(), file_name)

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
        plt.title("DQN Playing Tetris")
        plt.legend()
        plt.grid(True)
        plt.show()
