import random

import numpy as np

from .blocks import IBlock, JBlock, LBlock, OBlock, SBlock, TBlock, ZBlock
from .grid import Grid


class Game:
    ACTION_SPACE = {}

    def __init__(self):
        self.grid = Grid()
        self.blocks = [IBlock(), JBlock(), LBlock(), OBlock(), SBlock(), TBlock(), ZBlock()]
        self.current_block = self.get_random_block()
        self.next_block = self.get_random_block()
        self.game_over = False
        self.score = 0
        self.lines_cleared = 0
        self.blocks_locked = 0
        self.ACTION_SPACE = {0: self.move_nothing, 1: self.move_down, 2: self.move_left, 3: self.move_right,
                             4: self.rotate}

    def update_score(self, new_lines_cleared, move_down_points):
        if new_lines_cleared == 1:
            self.score += 100
        elif new_lines_cleared == 2:
            self.score += 300
        elif new_lines_cleared == 3:
            self.score += 500
        self.score += move_down_points
        self.lines_cleared += new_lines_cleared

    def get_random_block(self):
        if len(self.blocks) == 0:
            self.blocks = [IBlock(), JBlock(), LBlock(), OBlock(), SBlock(), TBlock(), ZBlock()]
        block = random.choice(self.blocks)
        self.blocks.remove(block)
        return block

    def move_left(self):
        self.current_block.move(0, -1)
        if self.block_inside() == False or self.block_fits() == False:
            self.current_block.move(0, 1)

    def move_right(self):
        self.current_block.move(0, 1)
        if self.block_inside() == False or self.block_fits() == False:
            self.current_block.move(0, -1)

    def move_down(self):
        self.move_continue()
        self.update_score(0, 1)

    def move_continue(self):
        self.current_block.move(1, 0)
        if self.block_inside() == False or self.block_fits() == False:
            self.current_block.move(-1, 0)
            self.lock_block()

    def move_nothing(self):
        # Let the game continue without moving the block
        pass

    def lock_block(self):
        tiles = self.current_block.get_cell_positions()
        for position in tiles:
            self.grid.matrix[position.row][position.column] = self.current_block.id
        self.current_block = self.next_block
        self.next_block = self.get_random_block()
        self.blocks_locked += 1
        rows_cleared = self.grid.clear_full_rows()
        if rows_cleared > 0:
            self.update_score(rows_cleared, 0)
        if not self.block_fits():
            self.game_over = True

    def reset(self):
        self.grid.reset()
        self.blocks = [IBlock(), JBlock(), LBlock(), OBlock(), SBlock(), TBlock(), ZBlock()]
        self.current_block = self.get_random_block()
        self.next_block = self.get_random_block()
        self.score = 0
        self.lines_cleared = 0
        self.blocks_locked = 0
        self.game_over = False

    def block_fits(self):
        tiles = self.current_block.get_cell_positions()
        for tile in tiles:
            if not self.grid.is_empty(tile.row, tile.column):
                return False
        return True

    def rotate(self):
        self.current_block.rotate()
        if self.block_inside() == False or self.block_fits() == False:
            self.current_block.undo_rotation()

    def block_inside(self):
        tiles = self.current_block.get_cell_positions()
        for tile in tiles:
            if not self.grid.is_inside(tile.row, tile.column):
                return False
        return True

    def draw(self, screen):
        self.grid.draw(screen)
        self.current_block.draw(screen, 11, 11)

        if self.next_block.id == 3:
            self.next_block.draw(screen, 255, 290)
        elif self.next_block.id == 4:
            self.next_block.draw(screen, 255, 280)
        else:
            self.next_block.draw(screen, 270, 270)

    def get_state(self):
        """
        Returns the current state of the game as a binary matrix.

        Returns:
            np.ndarray: A binary matrix representing the current state of the game.
        """
        state = self.grid.matrix.copy()
        tiles = self.current_block.get_cell_positions()
        for position in tiles:
            state[position.row][position.column] = self.current_block.id
        state = (state != 0) * 255
        return state
