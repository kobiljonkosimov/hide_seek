import numpy as np
import random
import pickle

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return random.sample(self.buffer, len(self.buffer))
        else:
            return random.sample(self.buffer, batch_size)

class QLearningHider:
    def __init__(self, grid_size, learning_rate=0.1, discount_factor=0.9, initial_epsilon=0.9, epsilon_decay=0.999, replay_buffer_size=10000):
        self.grid_size = grid_size
        self.q_table = np.zeros((grid_size, grid_size, 4))  # 4 actions: up, down, left, right
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Explore a random action
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploit the best-known action

    def update_q_value(self, state, action, reward, next_state):
        buffer_size = len(self.replay_buffer.buffer)
        if buffer_size >= 32:
            batch = self.replay_buffer.sample(32)
        else:
            batch = self.replay_buffer.sample(buffer_size)

        for s, a, r, ns in batch:
            best_next_action = np.argmax(self.q_table[ns[0], ns[1]])
            td_target = r + self.discount_factor * self.q_table[ns[0], ns[1], best_next_action]
            td_error = td_target - self.q_table[s[0], s[1], a]
            self.q_table[s[0], s[1], a] += self.learning_rate * td_error

        self.replay_buffer.add(state, action, reward, next_state)
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def save_model(self, filename):
        """Save Q-table to file."""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filename):
        """Load Q-table from file."""
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
