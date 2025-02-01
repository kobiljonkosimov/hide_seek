import cv2
import numpy as np
import random
import heapq
from collections import deque
from src.maps import MAPS

CELL_SIZE = 50  
GRID_SIZE = 13  

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

def draw_grid(grid, player_pos, seeker_pos):
    img = np.zeros((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE, 3), dtype=np.uint8)
    
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] == 1:
                cv2.rectangle(img, (x * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), BLACK, -1)  
            elif is_cell_in_fov(seeker_pos, (x, y), grid): 
                cv2.rectangle(img, (x * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), YELLOW, -1)  
            else:
                cv2.rectangle(img, (x * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), WHITE, -1)  
            
            cv2.rectangle(img, (x * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE), BLACK, 1)

    cv2.rectangle(img, (seeker_pos[0] * CELL_SIZE, seeker_pos[1] * CELL_SIZE), 
                  ((seeker_pos[0] + 1) * CELL_SIZE, (seeker_pos[1] + 1) * CELL_SIZE), RED, -1)  
    cv2.rectangle(img, (player_pos[0] * CELL_SIZE, player_pos[1] * CELL_SIZE), 
                  ((player_pos[0] + 1) * CELL_SIZE, (player_pos[1] + 1) * CELL_SIZE), GREEN, -1)  

    return img

def is_cell_in_fov(seeker_pos, cell_pos, grid):
    px, py = seeker_pos
    cx, cy = cell_pos
    if abs(px - cx) > 4  or abs(py - cy) > 4:
        return False

    x0, y0 = px, py
    x1, y1 = cx, cy
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if grid[y0][x0] == 1:  
            return False
        if (x0, y0) == (x1, y1): 
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
            return path[::-1]  
        closed_list.add(current)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  
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

    return []  
def seeker_move(player_pos, seeker_pos, grid):
    path = a_star(seeker_pos, player_pos, grid)
    if path:  
        return path[0]  
    else:
        return random_move(seeker_pos, grid)

def random_move(position, grid):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  
    random_direction = random.choice(directions)
    new_pos = (position[0] + random_direction[0], position[1] + random_direction[1])
    
    if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE and grid[new_pos[1]][new_pos[0]] == 0:
        return new_pos
    else:
        return position  
def is_valid_position(pos, grid):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[y][x] == 0

def initialize_positions(grid):
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
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

MAX_VISITED_POSITIONS = 5  

visited_positions = deque(maxlen=MAX_VISITED_POSITIONS)

def is_repeated_position(pos):
    if pos in visited_positions:
        return True
    visited_positions.append(pos)
    return False