import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.tetris.game import Game


class TetrisGymWrapper(gym.Env):
    """
    Custom Tetris environment wrapper that follows gym interface.
    """

    def __init__(self):
        super(TetrisGymWrapper, self).__init__()
        self.game = Game()
        self.action_space = spaces.Discrete(len(self.game.ACTION_SPACE))
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 100, 50), dtype=np.uint8)
        self.last_score = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.game.reset()
        state = self.get_state()
        return state, {}

    def step(self, action):
        """
        Perform a step in the Tetris game.

        Args:
            action (int): The action to be performed.

        Returns:
            tuple: A tuple containing the next state, reward, done flag, and additional info.
        """
        current_score = self.get_reward()
        self.game.ACTION_SPACE[action]()
        self.game.move_continue()
        done = self.game.game_over
        if done:
            reward = 0
            self.last_score = self.game.score
        else:
            reward = self.get_reward() - current_score + 0.1
        next_state = self.get_state()
        truncated = False
        return next_state, reward, done, truncated, {}

    def render(self):
        # Rendering is implemented in the GameUI class, not here
        pass

    def close(self):
        # No specific cleanup needed for this environment
        pass

    def is_done(self):
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.game.game_over

    def get_state(self):
        """
        Get the current state of the game.

        Returns:
            np.ndarray: The current state of the game.
        """
        state = self.game.get_state().astype(np.uint8)
        state = np.repeat(np.repeat(state, 5, axis=0), 5, axis=1)
        state = state[np.newaxis, :, :]
        return state

    def get_game(self):
        """
        Get the current game instance.

        Returns:
            Game: The current Tetris game instance.
        """
        return self.game

    def get_last_score(self):
        """
        Get the current score of the game.

        Returns:
            int: The current score of the game.
        """
        return self.last_score

    def get_reward(self):
        """
        Get the current reward of the game. Reward is calculated based on the number of lines cleared, the aggregated
        height of the grid, the number of top-covered holes, and the bumpiness.

        Returns:
            int: The current reward of the game.
        """
        return 100 * self.game.lines_cleared \
            - 0.4 * self.get_max_height() \
            - 0.7 * self.get_holes()

    def get_max_height(self):
        """
        Calculate the max height of the blocks in the game.

        Returns:
            int: The max height of the blocks.
        """
        heights = []
        if self.game.blocks_locked > 0:
            for column in range(self.game.grid.num_cols):
                if np.any(self.game.grid.matrix[:, column] != 0):
                    heights.append(self.game.grid.num_rows - np.argmax(self.game.grid.matrix[:, column] != 0))
                else:
                    heights.append(0)
            return max(heights)
        else:
            return 0

    def get_aggregated_height(self):
        """
        Calculate the aggregated height of the blocks in the game.

        Returns:
            int: The aggregated height of the blocks.
        """
        height = 0
        if self.game.blocks_locked > 0:
            for column in range(self.game.grid.num_cols):
                if np.any(self.game.grid.matrix[:, column] != 0):
                    height += self.game.grid.num_rows - np.argmax(self.game.grid.matrix[:, column] != 0)
            return height / self.game.blocks_locked
        else:
            return 0

    def get_holes(self):
        """
        Calculate the number of top-covered holes in the game.

        Returns:
            int: The number of holes in the game.
        """
        holes = 0
        for column in range(self.game.grid.num_cols):
            for row in range(self.game.grid.num_rows):
                if self.game.grid.matrix[row][column] == 0:
                    if any(self.game.grid.matrix[r][column] != 0 for r in range(0, row)):
                        holes += 1
        return holes

    def get_bumpiness(self):
        """
        Calculate the bumpiness of the game.

        Returns:
            int: The bumpiness of the game.
        """
        bumpiness = 0
        heights = []
        if self.game.blocks_locked > 0:
            for column in range(self.game.grid.num_cols):
                if np.any(self.game.grid.matrix[:, column] != 0):
                    heights.append(self.game.grid.num_rows - np.argmax(self.game.grid.matrix[:, column] != 0))
                else:
                    heights.append(0)
            for i in range(len(heights) - 1):
                bumpiness += abs(heights[i] - heights[i + 1])
            return bumpiness / self.game.blocks_locked
        else:
            return 0
