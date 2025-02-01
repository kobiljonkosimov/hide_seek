import time
import pandas as pd
from src.game import *
from src.ai import QLearningHider

def main():
    global grid

    rewards_data = [] 
    try:
        player_pos, seeker_pos = initialize_positions(grid)

        num_episodes = 50000 
        for episode in range(num_episodes):
            player_pos, seeker_pos = initialize_positions(grid)  
            total_reward = 0  
            if episode % 100 == 0:
                grid = random.choice(list(MAPS.values()))  

            start_time = time.time()  

            while True:
                elapsed_time = time.time() - start_time  

                if elapsed_time > 10:
                    print(f"Episode {episode + 1} timed out after 10 seconds.")
                    break  

                img = draw_grid(grid, player_pos, seeker_pos)

                cv2.imshow('Hide & Seek', img)

                if is_cell_in_fov(seeker_pos, player_pos, grid):
                    print("Seeker found the Hider!")
                    total_reward -= 10  
                    break  

                state = player_pos
                action = hider_agent.get_action(state)

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

                if is_cell_in_fov(seeker_pos, player_pos, grid):
                    reward = -10 
                elif len(visited_positions) >= MAX_VISITED_POSITIONS:
                    reward = -5  
                elif manhattan_distance(player_pos, seeker_pos) > 5:
                    reward = 1 * manhattan_distance(player_pos, seeker_pos)  
                elif new_pos == player_pos:
                    reward = -5  
                else:
                    reward = 1  

                total_reward += reward  
                next_state = player_pos
                hider_agent.update_q_value(state, action, reward, next_state)

                if cv2.waitKey(100) & 0xFF == ord('q'):  
                    break

            rewards_data.append([episode + 1, total_reward])

            print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward}")
    finally:
        rewards_df = pd.DataFrame(rewards_data, columns=['Episode', 'Total Reward'])
        rewards_df.to_csv(f'rewards_{episode}.csv', index=False)

        hider_agent.save_model(f'q_learning_model_{episode}.pkl')
        print(f"Model saved as 'q_learning_model{episode}.pkl'")

if __name__ == "__main__":
    grid = random.choice(list(MAPS.values()))  # Select a random map
    hider_agent = QLearningHider(GRID_SIZE, initial_epsilon=0.9, epsilon_decay=0.999)
    main()