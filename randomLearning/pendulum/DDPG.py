import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import pdb

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, states_, terminal

def build_actor_network(input_dims, n_actions):
    inputs = layers.Input(shape=input_dims)
    net = layers.Dense(256, activation='relu')(inputs)
    net = layers.Dense(128, activation='relu')(net)
    outputs = layers.Dense(n_actions, activation='tanh')(net)
    outputs = layers.Lambda(lambda x: x * 2)(outputs)
    return models.Model(inputs=inputs, outputs=outputs)

def build_critic_network(input_dims, n_actions):
    input_state = layers.Input(shape=input_dims)
    state_net = layers.Dense(256, activation='relu')(input_state)
    state_net = layers.Dense(128, activation='relu')(state_net)

    input_action = layers.Input(shape=(n_actions,))
    action_net = layers.Dense(128, activation='relu')(input_action)

    net = layers.Add()([state_net, action_net])
    net = layers.Activation('relu')(net)
    outputs = layers.Dense(1)(net)
    return models.Model(inputs=[input_state, input_action], outputs=outputs)

class Agent:
    def __init__(self, alpha=0.001, beta=0.002, input_dims=[3], n_actions=1,
                 max_size=1000000, tau=0.005, env=None, batch_size=256, 
                 noise=0.1, gamma=0.99):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        pdb.set_trace()
        self.batch_size = batch_size
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.actor = build_actor_network(input_dims, n_actions)
        self.critic = build_critic_network(input_dims, n_actions)
        self.target_actor = build_actor_network(input_dims, n_actions)
        self.target_critic = build_critic_network(input_dims, n_actions)

        self.actor.compile(optimizer=optimizers.Adam(learning_rate=alpha))
        self.critic.compile(optimizer=optimizers.Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=optimizers.Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=optimizers.Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load

# Environment
env = gym.make('Pendulum-v1', render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

# Training
agent = Agent(env = env)
episodes = 100
for episode in range(episodes):
    state = env.reset()
    env.render() 
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        env.render() 
        agent.train(np.array([state]), np.array([action]), np.array([reward]), np.array([next_state]), np.array([done]))
        state = next_state
        total_reward += reward
    print("Episode:", episode + 1, "Total Reward:", total_reward)
env.close()