import time
import cv2
import pandas as pd
import numpy as np
import random
import heapq
import pickle
from maps import MAPS
from ai import QLearningHider
from collections import deque

# Screen dimensions
CELL_SIZE = 50  # Cell size for the grid
GRID_SIZE = 13  # Grid size (13x13)

# Colors in BGR format for OpenCV
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

grid = random.choice(list(MAPS.values()))  # Select a random map
hider_agent = QLearningHider(GRID_SIZE, initial_epsilon=0.9, epsilon_decay=0.999)

def draw_grid(grid, player_pos, seeker_pos):
    """ Draw the grid using OpenCV. """
    img = np.zeros((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE, 3), dtype=np.uint8)
    
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] == 1:
                cv2.rectangle(img, (x * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), BLACK, -1)  # Draw obstacles
            elif is_cell_in_fov(seeker_pos, (x, y), grid):  # Calculate FOV dynamically
                cv2.rectangle(img, (x * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), YELLOW, -1)  # Cells in Seeker's FOV
            else:
                cv2.rectangle(img, (x * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), WHITE, -1)  # Empty cells
            
            # Draw grid borders
            cv2.rectangle(img, (x * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), BLACK, 1)

    # Draw the players
    cv2.rectangle(img, (seeker_pos[0] * CELL_SIZE, seeker_pos[1] * CELL_SIZE), 
                  ((seeker_pos[0] + 1) * CELL_SIZE, (seeker_pos[1] + 1) * CELL_SIZE), RED, -1)  # Seeker
    cv2.rectangle(img, (player_pos[0] * CELL_SIZE, player_pos[1] * CELL_SIZE), 
                  ((player_pos[0] + 1) * CELL_SIZE, (player_pos[1] + 1) * CELL_SIZE), GREEN, -1)  # Hider

    return img

def random_move(position, grid):
    """ Move the hider randomly in one of the four directions. """
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Down, Up, Right, Left
    random_direction = random.choice(directions)
    new_pos = (position[0] + random_direction[0], position[1] + random_direction[1])
    
    # Check if the new position is within bounds and not an obstacle
    if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE and grid[new_pos[1]][new_pos[0]] == 0:
        return new_pos
    else:
        return position  # Stay in the same position if the move is invalid

def is_cell_in_fov(seeker_pos, cell_pos, grid):
    """ Check if a cell is in the field of view of the seeker. """
    px, py = seeker_pos
    cx, cy = cell_pos
    # Restrict the FOV range to 4
    if abs(px - cx) > 4 or abs(py - cy) > 4:
        return False

    # Line of sight algorithm (Bresenham's line algorithm)
    x0, y0 = px, py
    x1, y1 = cx, cy
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if grid[y0][x0] == 1:  # Obstacle blocks the view
            return False
        if (x0, y0) == (x1, y1):  # Reached the target cell
            return True
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

def heuristic(a, b):
    """ Use Manhattan distance as heuristic """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal, grid):
    """ A* algorithm to find the shortest path from start to goal """
    open_list = []
    closed_list = set()
    came_from = {}

    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    heapq.heappush(open_list, (f_score[start], start))

    while open_list:
        current = heapq.heappop(open_list)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return reversed path

        closed_list.add(current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Left, Right, Up, Down
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE and grid[neighbor[1]][neighbor[0]] == 0:
                if neighbor in closed_list:
                    continue

                tentative_g_score = g_score.get(current, float('inf')) + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    if neighbor not in [i[1] for i in open_list]:
                        heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return []  # Return an empty path if no path is found

def seeker_move(player_pos, seeker_pos, grid):
    """ Seeker chooses the shortest path to the hider using A* """
    path = a_star(seeker_pos, player_pos, grid)
    if path:  # If a path is found, move to the next step in the path
        return path[0]  # Move to the next position in the path
    else:
        # If no path is found, move randomly (fallback)
        return random_move(seeker_pos, grid)

def is_valid_position(pos, grid):
    """ Check if the position is valid (not on an obstacle) """
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[y][x] == 0

def initialize_positions(grid):
    """ Initialize the positions of the hider and seeker randomly. """
    while True:
        hider_pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if is_valid_position(hider_pos, grid):
            break

    while True:
        seeker_pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if is_valid_position(seeker_pos, grid) and manhattan_distance(hider_pos, seeker_pos) >= 5:
            break

    return hider_pos, seeker_pos

def manhattan_distance(pos1, pos2):
    """ Calculate the Manhattan distance between two positions. """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

MAX_VISITED_POSITIONS = 5  # Keep track of the last 5 positions

visited_positions = deque(maxlen=MAX_VISITED_POSITIONS)

def is_repeated_position(pos):
    if pos in visited_positions:
        return True
    visited_positions.append(pos)
    return False


def main():
    global grid

    rewards_data = []  # Initialize list to store rewards and episode numbers

    try:
        # Initialize positions
        player_pos, seeker_pos = initialize_positions(grid)

        # Training parameters
        num_episodes = 50000  # Number of training episodes
        for episode in range(num_episodes):
            player_pos, seeker_pos = initialize_positions(grid)  # Reset positions for each episode
            total_reward = 0  # Track total reward for the episode
            if episode % 100 == 0:
                grid = random.choice(list(MAPS.values()))  # Change the map every 100 episodes

            start_time = time.time()  # Start the timer for the round

            while True:
                elapsed_time = time.time() - start_time  # Calculate elapsed time

                # If the round has exceeded 30 seconds, skip to the next round
                if elapsed_time > 30:
                    print(f"Episode {episode + 1} timed out after 30 seconds.")
                    break  # Skip to the next round

                # Draw the grid and players
                img = draw_grid(grid, player_pos, seeker_pos)

                # Display the image using OpenCV
                #cv2.imshow('Hide & Seek', img)
                # Check if the seeker finds the hider within its FOV
                if is_cell_in_fov(seeker_pos, player_pos, grid):
                    print("Seeker found the Hider!")
                    total_reward -= 10  # Negative reward if the seeker finds the hider
                    break  # End the episode

                # Get the hider's action from the Q-learning agent
                state = player_pos
                action = hider_agent.get_action(state)

                # Map action to movement
                if action == 0:  # Up
                    new_pos = (player_pos[0], player_pos[1] - 1)
                elif action == 1:  # Down
                    new_pos = (player_pos[0], player_pos[1] + 1)
                elif action == 2:  # Left
                    new_pos = (player_pos[0] - 1, player_pos[1])
                elif action == 3:  # Right
                    new_pos = (player_pos[0] + 1, player_pos[1])

                # Check if the new position is valid
                if is_valid_position(new_pos, grid):
                    player_pos = new_pos  # Update hider's position if valid

                # Move the seeker towards the player using A*
                seeker_pos = seeker_move(player_pos, seeker_pos, grid)

                 # Reward mechanism
                if is_cell_in_fov(seeker_pos, player_pos, grid):
                    reward = -10  # Negative reward if the seeker finds the hider
                elif len(visited_positions) >= MAX_VISITED_POSITIONS:
                    reward = -5  # Penalize revisiting the same positions
                elif manhattan_distance(player_pos, seeker_pos) > 5:
                    reward = 1 * manhattan_distance(player_pos, seeker_pos)  # Positive reward for moving away from the seeker
                elif new_pos == player_pos:
                    reward = -5  # Negative reward for staying in the same position
                else:
                    reward = 1  # Positive reward for staying hidden

                total_reward += reward  # Accumulate total reward

                # Update Q-values
                next_state = player_pos
                hider_agent.update_q_value(state, action, reward, next_state)

                # Check for quitting the game
                if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to quit
                    break

            # Append the total reward and episode number to the rewards_data list
            rewards_data.append([episode + 1, total_reward])

            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")
    finally:
        # Convert rewards_data to a DataFrame and save it to a CSV file
        rewards_df = pd.DataFrame(rewards_data, columns=['Episode', 'Total Reward'])
        rewards_df.to_csv(f'rewards_{episode}.csv', index=False)

        # Save the model at the end of training
        hider_agent.save_model(f'q_learning_model_{episode}.pkl')
        print(f"Model saved as 'q_learning_model{episode}.pkl'")

# Run the main function
main()
