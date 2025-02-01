import cv2
import numpy as np
import random
import time
from src.game import *
from src.utils import calc
from src.ai import QLearningHider

def main():
    global grid

    hider_agent = QLearningHider(GRID_SIZE)
    hider_agent.load_model('quick_model.pkl')

    while True:
        player_pos, seeker_pos = initialize_positions(grid)

        while True:
            img = draw_grid(grid, player_pos, seeker_pos)
            cv2.imshow('Hide & Seek', img)

            if is_cell_in_fov(seeker_pos, player_pos, grid):
                print("Seeker found the Hider! Resetting positions...")
                break

            state = player_pos
            best_move = calc(state, seeker_pos, grid)
            action = best_move if best_move is not None else hider_agent.get_action(state)

            if action == 0:
                new_pos = (player_pos[0], player_pos[1] - 1)
            elif action == 1:
                new_pos = (player_pos[0], player_pos[1] + 1)
            elif action == 2:
                new_pos = (player_pos[0] - 1, player_pos[1])
            elif action == 3:
                new_pos = (player_pos[0] + 1, player_pos[1])

            if is_valid_position(new_pos, grid):
                player_pos = new_pos

            seeker_pos = seeker_move(player_pos, seeker_pos, grid)

            time.sleep(0.15)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    grid = random.choice(list(MAPS.values()))
    main()