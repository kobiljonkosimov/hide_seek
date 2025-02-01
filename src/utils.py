import numpy as np
from src.game import *


def calc(state, seeker_pos, grid):
    hider_x, hider_y = state
    seeker_x, seeker_y = seeker_pos

    # Move away from the seeker if close
    if abs(hider_x - seeker_x) + abs(hider_y - seeker_y) <= 7:
        if hider_x < seeker_x and hider_x > 0 and grid[hider_y][hider_x - 1] == 0:
            return 2  # Move left
        elif hider_x > seeker_x and hider_x < GRID_SIZE - 1 and grid[hider_y][hider_x + 1] == 0:
            return 3  # Move right
        elif hider_y < seeker_y and hider_y > 0 and grid[hider_y - 1][hider_x] == 0:
            return 0  # Move up
        elif hider_y > seeker_y and hider_y < GRID_SIZE - 1 and grid[hider_y + 1][hider_x] == 0:
            return 1  # Move down

    # Avoid traps (e.g., corners)
    if (hider_x == 0 and hider_y == 0) or (hider_x == GRID_SIZE - 1 and hider_y == 0) or \
       (hider_x == 0 and hider_y == GRID_SIZE - 1) or (hider_x == GRID_SIZE - 1 and hider_y == GRID_SIZE - 1):
        if hider_x > 0 and grid[hider_y][hider_x - 1] == 0:
            return 2  # Move left
        elif hider_x < GRID_SIZE - 1 and grid[hider_y][hider_x + 1] == 0:
            return 3  # Move right
        elif hider_y > 0 and grid[hider_y - 1][hider_x] == 0:
            return 0  # Move up
        elif hider_y < GRID_SIZE - 1 and grid[hider_y + 1][hider_x] == 0:
            return 1  # Move down

    return None  # No override, use the Q-learning action