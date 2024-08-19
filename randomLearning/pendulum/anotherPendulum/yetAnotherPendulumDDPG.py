import pdb 

import random
import pandas as pd
import numpy as np
from PIL import Image
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten,Activation,Concatenate
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.models import Model,load_model, model_from_json
import tensorflow as tf
import gym
import math
import pygame, sys
from tensorflow import keras
from collections import deque
import math
#enable eager execution in tensorflow
tf.config.run_functions_eagerly(True)


env = gym.make('Pendulum-v1')

input_shape = (3,)
num_actions = 1


def actor_network(input_shape=(3,)):
        model = Sequential()
        model.add(Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.2),input_shape=input_shape))
        model.add(Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
        model.add(Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
        model.add(Dense(num_actions,activation='tanh'))
        return model

actor, target_actor = actor_network(),actor_network()
optimizer_actor = Adam(learning_rate=0.001)


def critic_network(state_dim, action_dim):
    # Define the input layers
 
    state_input = Input(shape=(state_dim,1),dtype=tf.float64)
    action_input = Input(shape=(action_dim,1),dtype=tf.float64)

    
    state_h1 = Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(state_input)
    state_h2 = Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(state_h1)

    action_h1 = Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(action_input)
    action_h2 = Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(action_h1)
    
    concat = Concatenate(axis = 1)([state_h2, action_h2])

    # Define the output layer
    dense1 = Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(concat)
    dense2 = Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.2))(dense1)
    output = Dense(1, activation='linear')(dense2)
    
    model = Model(inputs=[state_input, action_input], outputs=output)
    return model
    
critic, target_critic = critic_network(3,1),critic_network(3,1)
optimizer_critic = Adam(learning_rate=0.001)


def update_target_networks(actor_model, critic_model, target_actor_model, target_critic_model):
    tau = 0.05
    # Update the target actor model
    actor_weights = actor_model.get_weights()
    target_actor_weights = target_actor_model.get_weights()
    for i in range(len(actor_weights)):
        target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]
    target_actor_model.set_weights(target_actor_weights)

    # Update the target critic model
    critic_weights = critic_model.get_weights()
    target_critic_weights = target_critic_model.get_weights()
    for i in range(len(critic_weights)):
        target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]
    target_critic_model.set_weights(target_critic_weights)
    return target_critic_model, target_actor_model


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta, dt, size):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.size = size
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.size) * self.mu
    
    def __call__(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.state = x + dx
        return self.state

def ddpg_add_exploration_noise(exploration_noise, action, noise_scale):
    noise = noise_scale * exploration_noise()

    action = np.clip(action + noise, -2.0, 2.0)
    
    return action

# Usage example:
action_dim = 1  # Dimensionality of the action space
noise_mu = 0.0
noise_sigma = 0.2
noise_theta = 0.15
noise_dt = 0.01

# Create an instance of the OrnsteinUhlenbeckActionNoise class
exploration_noise = OrnsteinUhlenbeckActionNoise(noise_mu, noise_sigma, noise_theta, noise_dt, size=action_dim)


gamma = tf.cast(tf.constant(0.95),tf.float64)
num_episodes = 1000
maxlen = 1000
batch = 128
replay = deque(maxlen=maxlen)
epoch = 0
count=0
max_loss = math.inf
count = 0

#lets start with a pretty high epsilon 
epsilon = .5 

for episode in range(num_episodes):
    ep_len = 0
    state = env.reset()
    print('epsilon getting updated',epsilon)
    epsilon*=0.99
    # Run the episode
    while True:
        count+=1
        ep_len+=1
    
        pdb.set_trace() 
        action = 2*actor.predict(np.array(state).reshape(-1,3),verbose=0)[0]
        action = ddpg_add_exploration_noise(exploration_noise, action, noise_scale=0.1)
        
        next_state, reward, done, _ = env.step(action)
        done = 1 if done else 0
        
        print('reward and status',reward,state)
        state = state.reshape(3)
            
        replay.append((np.array(state),action,reward,np.array(next_state),done))
        state = next_state

        if done:
            break
    
        if count>batch:
            count = 0
            batch_ = random.sample(replay,batch)
            current_state = tf.convert_to_tensor([x[0] for x in batch_])
            next_state = tf.convert_to_tensor([x[3] for x in batch_])
            reward = tf.convert_to_tensor([x[2] for x in batch_])
            done =   tf.convert_to_tensor([x[4] for x in batch_])
            actions =   tf.convert_to_tensor([x[1] for x in batch_])
            other_actions = [[1,0] for x in range(batch)]
            
            q_actions = target_actor(next_state) 
            target_q = tf.cast(reward,tf.float64) + (tf.cast(tf.constant(1.0),tf.float64)-tf.cast(done,tf.float64))*gamma*tf.cast(target_critic([next_state,q_actions]),tf.float64)

            with tf.GradientTape() as tape:
                current_q_value = critic([current_state,actions])
                critic_loss = tf.reduce_mean(tf.math.pow(target_q-tf.cast(current_q_value,tf.float64),2))
    
            grads_critic = tape.gradient(critic_loss, critic.trainable_variables)
            optimizer_critic.apply_gradients(zip(grads_critic, critic.trainable_variables))
                
            with tf.GradientTape() as tape:
                actions = actor(current_state,training=True)                
                current_q_value = critic([current_state,actions],training=True)
                actor_loss = -tf.reduce_mean(current_q_value)
                
            grads_actor = tape.gradient(actor_loss, actor.trainable_variables)
            optimizer_actor.apply_gradients(zip(grads_actor, actor.trainable_variables))

            print('Epoch {} done with loss actor={} , critic={} !!!!!!'.format(epoch,actor_loss,critic_loss))
            if epoch%10==0:
                    actor.save('pendulum/actor/')
                    critic.save('pendulum/critic/')
            
            if epoch%5==0:
                    target_critic, target_actor = update_target_networks(actor,critic,target_actor,target_critic)
            epoch+=1

