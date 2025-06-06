import sys

import pygame
import pygame.surfarray
import torch

from .colors import Colors
from .game import Game


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

    def play(self, model_player: torch.nn.Module = None):
        """
        Start the game loop, allowing either user input or a trained model to play.

        Args:
            model_player (torch.nn.Module, optional): A trained PyTorch model to play the game.
                If None, the game will be controlled by user input.
        """
        game = Game()
        game.reset()
        lock_step = False
        game_update = pygame.USEREVENT
        pygame.time.set_timer(game_update, 200)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and model_player is None:
                    self.handle_keyboard(event, game)
                elif event.type == game_update and game.game_over == False:
                    game.move_continue()
                    lock_step = False
            if model_player:
                lock_step = self.handle_model_step(model_player, game, lock_step)
            self.draw(game)

    @staticmethod
    def handle_keyboard(event, game):
        """
        Handle keyboard input for controlling the game.

        Args:
            event (pygame.event.Event): The Pygame event representing a key press.
            game (Game): The Tetris game instance.
        """
        if game.game_over:
            game.reset()
        if event.key == pygame.K_LEFT and not game.game_over:
            game.move_left()
        if event.key == pygame.K_RIGHT and not game.game_over:
            game.move_right()
        if event.key == pygame.K_DOWN and not game.game_over:
            game.move_down()
            game.update_score(0, 1)
        if event.key == pygame.K_UP and not game.game_over:
            game.rotate()

    @staticmethod
    def handle_model_step(model_player: torch.nn.Module, game: Game, lock_step: bool):
        """
        Handle a single step of the game using a trained model.

        Args:
            model_player (torch.nn.Module): The trained PyTorch model.
            game (Game): The Tetris game instance.
            lock_step (bool): Whether the model is currently locked from taking another step.

        Returns:
            bool: The updated lock_step status.
        """
        if game.game_over:
            game.reset()
            lock_step = False
        elif not lock_step:
            state = torch.tensor(game.get_state(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_values = model_player(state)
                action = torch.argmax(q_values).item()
            game.ACTION_SPACE.get(action)()
            lock_step = True
        return lock_step

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
