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
from .model import DDQN


class TrainerDDQN:
    """
    Simulate the Tetris game for training an agent model.
    """
    TRANSITION = collections.namedtuple("Transition",
                                        ("state", "action", "reward", "next_state"))

    def __init__(self, model: torch.nn.Module):
        self.eval_network = model
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.001
        self.epsilon_decay_episodes = 5000
        self.buffer_size = 10_000
        self.batch_size = 64
        self.target_update_freq = 200

    def train(self, episodes: int = 10000, max_steps: int = 1000, refresh_rate: int = 100):
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

        for episode in range(1, episodes + 1):
            game.reset()
            episode_steps = 0

            for _ in range(max_steps):
                # Play a step in the game
                state, action, reward, next_state = self.play_step(game, epsilon)

                # Store transition in replay buffer
                replay_buffer.append(self.TRANSITION(state, action, reward, next_state))

                # Execute a model training step
                self.train_step(self.eval_network, target_network, optimizer, loss_fn, replay_buffer)

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
                self.save_model(self.eval_network, "./resources/model_snapshot/", f"tetris_{episode}_")
                print(
                    f"Episode: {episode}/{episodes} | Steps: {episode_steps} | "
                    f"Avg reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

        # End training
        self.save_model(self.eval_network, "./resources/model/", "tetris_")
        self.save_metric(episode_rewards_history, "./resources/model/", "rewards_")
        self.plot_rewards(episode_rewards_history)

    def play_step(self, game, epsilon):
        # Current state
        state = torch.tensor(game.get_state(), dtype=torch.float32).unsqueeze(0)
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
            reward = -1000
        else:
            next_state = torch.tensor(game.get_state(), dtype=torch.float32).unsqueeze(0)
            reward = game.score - current_score
        reward = torch.tensor([reward - self.get_rows_penalty(game)], dtype=torch.float32)

        return state, action, reward, next_state

    def train_step(self, eval_network, target_network, optimizer, loss_fn, replay_buffer):

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
            eval_best_next_actions = eval_next_q_values.max(1)[1].unsqueeze(1)  # Indices (argmax)
            target_next_q_values = target_network(non_final_next_states)
            selected_target_next_q_values = target_next_q_values.gather(1, eval_best_next_actions).squeeze(1)
            next_state_values[non_final_mask] = selected_target_next_q_values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        # Loss
        loss = loss_fn(state_action_values, expected_state_action_values)

        # Backward step
        loss.backward()
        optimizer.step()

    @staticmethod
    def get_rows_penalty(game):
        """
        Calculate the penalty for the number of rows with blocks in the game.
        """
        non_zero_rows = np.nonzero(np.sum(game.grid.grid, axis=1))[0]
        if non_zero_rows.size > 0:
            return (game.grid.num_rows - non_zero_rows[0]) * 100
        else:
            return 0

    @staticmethod
    def save_model(model, base_path, prefix):
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        file_name = base_path + prefix + datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + ".pkl"
        torch.save(model.state_dict(), file_name)

    @staticmethod
    def save_metric(metric, base_path, prefix):
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        file_name = base_path + prefix + datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + ".npy"
        with open(file_name, "wb") as file:
            pkl.dump(metric, file)

    @staticmethod
    def plot_rewards(rewards):
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


dqn = DDQN(input_dim=201, output_dim=5)
trainer = TrainerDDQN(model=dqn)
trainer.train()
