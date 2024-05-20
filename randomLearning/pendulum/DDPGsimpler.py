import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import gym

import pdb 

# Actor model
# so, this learns the policy/mapping from 
class Actor(models.Model):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(32, activation='relu')
        self.out = layers.Dense(action_dim, activation='tanh')
        self.action_bound = action_bound

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.out(x)
        return x * self.action_bound

# Critic model
class Critic(models.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(32, activation='relu')
        self.out = layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

# Ornstein-Uhlenbeck Noise for exploration
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
            self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound):
        self.actor = Actor(state_dim, action_dim, action_bound)
        self.critic = Critic(state_dim, action_dim)
        self.actor_optimizer = optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = optimizers.Adam(learning_rate=0.002)
        self.action_bound = action_bound

    def get_action(self, state, noise_object):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        sampled_action = self.actor(state).numpy()[0]
        noise = noise_object()
        sampled_action += noise
        return np.clip(sampled_action, -self.action_bound, self.action_bound)

    def train(self, states, actions, rewards, next_states, done):
        with tf.GradientTape() as tape:
            target_actions = self.actor(next_states)
            critic_value = self.critic(next_states, target_actions)
            target_value = rewards + (1 - done) * critic_value
            current_value = self.critic(states, actions)
            critic_loss = tf.math.reduce_mean(tf.math.square(target_value - current_value))
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            critic_value = self.critic(states, actions)
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

# Main training loop
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

agent = DDPGAgent(state_dim, action_dim, action_bound)
noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))

pdb.set_trace() 

episodes = 1000
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.get_action(state, noise)
        next_state, reward, done, _ = env.step(action)
        agent.train(np.expand_dims(state, axis=0), np.expand_dims(action, axis=0), np.expand_dims(reward, axis=0), np.expand_dims(next_state, axis=0), np.expand_dims(done, axis=0))
        state = next_state
        episode_reward += reward
    print("Episode:", episode+1, " Reward:", episode_reward)
