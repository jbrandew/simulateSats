import gym
import numpy as np

# Create the FrozenLake-v0 environment
env = gym.make("FrozenLake-v1", map_name="8x8", render_mode="human")

# Initialize Q-table with zeros
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# Hyperparameters
alpha = 0.5  # learning rate
gamma = 0.9  # discount factor
epsilon = 1  # exploration rate
num_episodes = 1000
epsilon_decay = 0.001  

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        # Render the environment

        
        # Choose action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit
        
        # Take action and observe next state and reward
        next_state, reward, done, truncated, _ = env.step(action)
  
        
        # Update Q-value using Q-learning update rule
        best_next_action = np.argmax(q_table[next_state])
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * q_table[next_state, best_next_action] - q_table[state, action])
        
        state = next_state
    
    epsilon = max(epsilon - epsilon_decay, 0)

    # Print total reward of the episode
    print("Episode {}: Total Reward = {}".format(episode+1, reward))
    
    # Close the rendering window after each episode
    #env.close()

# Evaluate learned policy
total_rewards = 0
num_episodes_eval = 100
for _ in range(num_episodes_eval):
    state = env.reset()
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)
        env.render() 
        total_rewards += reward
        state = next_state

# Print average reward over evaluation episodes
avg_reward_eval = total_rewards / num_episodes_eval
print("Average Reward over {} Evaluation Episodes: {:.2f}".format(num_episodes_eval, avg_reward_eval))

# Close the environment
env.close()
