import numpy as np
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

import pdb 

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="8x8")#, render_mode="human")


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Instantiate the agent
state_size = env.observation_space.n
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

batch_size = 32
episodes = 1000
for e in range(episodes):
    print(e)
    state, _ = env.reset()
    #one hot encode 
    state = np.eye(state_size)[state]

    for time in range(500): # Set a maximum episode length
        action = agent.act(state)
        next_state, reward, done, truncate, _ = env.step(action)
        reward = reward if not done else -10
        next_state =  np.eye(state_size)[next_state]
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, episodes, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            pdb.set_trace()
            agent.replay(batch_size)

def evaluate_policy(agent, episodes=100):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, truncate, _ = env.step(action)
            env.render() 
            episode_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
        total_rewards.append(episode_reward)
    average_reward = np.mean(total_rewards)
    print("Average reward over {} episodes: {:.2f}".format(episodes, average_reward))
    return average_reward

# Evaluate the policy
average_reward = evaluate_policy(agent)
