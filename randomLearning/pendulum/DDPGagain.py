import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym

import pdb 

# Environment
env = gym.make('Pendulum-v1', render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

# Actor model
# actor learns the "policy," mapping the states to actions 
def create_actor_model():
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(action_dim, activation='tanh')(x)
    outputs = outputs * action_bound
    model = tf.keras.Model(inputs, outputs)
    return model

# Critic model
# critic lears the "values," of each action/state pair 
def create_critic_model():
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    
    state_x = layers.Dense(256, activation='relu')(state_input)
    action_x = layers.Dense(256, activation='relu')(action_input)
    x = layers.Concatenate()([state_x, action_x])
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

# DDPG agent
class DDPGAgent:
    def __init__(self):
        self.actor_model = create_actor_model()
        self.critic_model = create_critic_model()
        self.target_actor = create_actor_model()
        self.target_critic = create_critic_model()
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def get_action(self, state):
        state = np.reshape(state[0], [1, state_dim])
        return self.actor_model.predict(state)

    def train(self, states, actions, rewards, next_states, done):
        # Update critic
        next_actions = self.target_actor.predict(next_states)
        q_values = self.target_critic.predict([next_states, np.array(next_actions)])
        q_values = rewards + 0.99 * q_values * (1 - done)
        with tf.GradientTape() as tape:
            critic_value = self.critic_model([states, actions])
            critic_loss = tf.math.reduce_mean(tf.math.square(q_values - critic_value))
        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        pdb.set_trace() 

        # Update actor
        with tf.GradientTape() as tape:
            actions = self.actor_model(states)
            critic_value = self.critic_model([states, np.array(actions)])
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        # Update target networks
        self.update_target_networks()

    def update_target_networks(self):
        actor_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = 0.99 * actor_target_weights[i] + 0.01 * actor_weights[i]
        self.target_actor.set_weights(actor_target_weights)

        critic_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = 0.99 * critic_target_weights[i] + 0.01 * critic_weights[i]
        self.target_critic.set_weights(critic_target_weights)

# Training
agent = DDPGAgent()
episodes = 100
for episode in range(episodes):
    state = env.reset()
    #env.render() 
    done = False
    total_reward = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, trunc, _ = env.step(action[0])
        
        #I get something weird appended to the state returned 
        agent.train(np.array([state[0]]), np.array([action]), np.array([reward]), np.array([next_state]), np.array([done]))
        state = next_state
        total_reward += reward
    print("Episode:", episode + 1, "Total Reward:", total_reward)
env.close()
