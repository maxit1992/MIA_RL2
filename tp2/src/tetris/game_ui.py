import sys

import pygame
import pygame.surfarray
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from .colors import Colors
from .game import Game
from .gym_env import TetrisGymWrapper


class GameUI:
    """
    A class to handle the graphical user interface (GUI) for the Tetris game.
    """

    def __init__(self):
        """
        Initialize the GameUI class, setting up the display and fonts.
        """
        pygame.init()

        self.title_font = pygame.font.Font(None, 40)
        self.score_surface = self.title_font.render("Score", True, Colors.white)
        self.next_surface = self.title_font.render("Next", True, Colors.white)
        self.game_over_surface = self.title_font.render("GAME OVER", True, Colors.white)

        self.score_rect = pygame.Rect(320, 55, 170, 60)
        self.next_rect = pygame.Rect(320, 215, 170, 180)

        self.screen = pygame.display.set_mode((500, 620))
        pygame.display.set_caption("Python Tetris")

        self.clock = pygame.time.Clock()

    def play(self, model_player: PPO):
        """
        Start the game loop, allowing a trained model to play.

        Args:
            model_player (PPO): A trained PPO model to play the game.
        """
        game_wrapper = make_vec_env(TetrisGymWrapper, n_envs=1)
        state = game_wrapper.reset()
        done = False
        game_update = pygame.USEREVENT
        pygame.time.set_timer(game_update, 50)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == game_update:
                    if done:
                        state = game_wrapper.reset()
                        done = False
                    else:
                        action, _ = model_player.predict(state, deterministic=True)
                        state, _, done, _ = game_wrapper.step(action)
            self.draw(game_wrapper.env_method("get_game")[0])

    def draw(self, game):
        """
        Render the game screen, including the grid, score, and next block.

        Args:
            game (Game): The Tetris game instance.
        """
        score_value_surface = self.title_font.render(str(game.score), True, Colors.white)

        self.screen.fill(Colors.dark_blue)
        self.screen.blit(self.score_surface, (365, 20, 50, 50))
        self.screen.blit(self.next_surface, (375, 180, 50, 50))

        if game.game_over:
            self.screen.blit(self.game_over_surface, (320, 450, 50, 50))

        pygame.draw.rect(self.screen, Colors.light_blue, self.score_rect, 0, 10)
        self.screen.blit(score_value_surface, score_value_surface.get_rect(centerx=self.score_rect.centerx,
                                                                           centery=self.score_rect.centery))
        pygame.draw.rect(self.screen, Colors.light_blue, self.next_rect, 0, 10)
        game.draw(self.screen)
        pygame.display.update()
        self.clock.tick(60)
