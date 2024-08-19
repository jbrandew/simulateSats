import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import pdb 

# Create LunarLander environment
env = gym.make('LunarLander-v2', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# DQN parameters
alpha = 0.001  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 1000  # Number of episodes

# Build DQN model
model = Sequential([
    Dense(16, input_dim=state_size, activation='relu'),
    Dense(16, activation='relu'),
    Dense(action_size, activation='linear')
])
model.compile(loss='mse', optimizer=Adam(learning_rate=alpha))

# DQN algorithm
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state[0], [1, state_size])
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()  # Explore
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])  # Exploit
        
        # Perform action and observe next state and reward
        next_state, reward, done, trunc, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        
        # Update Q-value
        target = reward + gamma * np.max(model.predict(next_state, verbose = 0))
        q_values = model.predict(state, verbose = 0)
        q_values[0][action] = target
        model.fit(state, q_values, epochs=1, verbose=0)
        
        total_reward += reward
        state = next_state
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Print episode stats
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

env.close()
