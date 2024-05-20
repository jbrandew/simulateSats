import gymnasium as gym

import numpy as np
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import pdb

# Define the DQNAgent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.state_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=.01))
        return model

    def act(self, state):
        state_batched = state[np.newaxis, :]
        q_values = self.model.predict(state_batched, verbose = 0)
        return np.argmax(q_values[0])  # Greedy action selection

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            #so, for the next target for our network that predicts the value of each transition (state/transition pair)
            #make the target the reward you got + what we think the future rewards will be
            target = reward + np.amax(self.model.predict(next_state, verbose = 0)[0])
        target_f = self.model.predict(state, verbose = 0)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

# Initialize the environment
env = gym.make(
    "LunarLander-v2",
    render_mode="human",
    continuous = False,
    gravity = -10.0,
    enable_wind = False,
    wind_power = 15.0,
    turbulence_power = 1.5,
)

#state size in this case is the shape of the vector 
state_size = np.shape(env.observation_space)[0]
action_size = env.action_space.n

# Initialize the DQN agent
agent = DQNAgent(state_size, action_size)

# Training loop
episodes = 100
for episode in range(episodes):
    state, _ = env.reset()
    env.render()
    total_reward = 0
    done = False
    while not done:
        # Perform action and observe environment
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        env.render()
        #then add the batch dimension for each
        batchedNextState = np.expand_dims(next_state, axis=0)
        batchedCurrentState = np.expand_dims(state, axis=0)
        reward = reward if not done else -10  # Punish falling into a hole
        # Train agent
        agent.train(batchedCurrentState, action, reward, batchedNextState, done)
        state = next_state
        total_reward += reward
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

def evaluate_policy(agent, env, episodes=100):
    total_rewards = []
    for _ in range(episodes):
        state, _  = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    average_reward = np.mean(total_rewards)
    print("Average reward over {} episodes: {:.2f}".format(episodes, average_reward))
    return average_reward

# Evaluate the policy
average_reward = evaluate_policy(agent, env)
