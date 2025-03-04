import gym
import numpy as np

# Initialize the FrozenLake environment (non-slippery version for simplicity)
env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize Q-table with zeros
q_table = np.zeros((n_states, n_actions))

# Hyperparameters
num_episodes = 1000
max_steps = 100
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1  # Exploration rate

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    for step in range(max_steps):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])
        
        new_state, reward, done, _ = env.step(action)
        
        # Update Q-table using the Bellman equation
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
        )
        
        state = new_state
        if done:
            break

print("Trained Q-Table:")
print(q_table)
