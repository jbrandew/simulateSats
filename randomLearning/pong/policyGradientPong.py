import gym
import gymnasium
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

# Define hyperparameters
learning_rate = 0.0001
gamma = 0.99
num_epochs = 5
batch_size = 128

# Define the PPO agent's policy network
class PPOModel(Model):
    def __init__(self, num_actions):
        super(PPOModel, self).__init__()
        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.policy_logits = Dense(num_actions, activation=None)
        self.value = Dense(1, activation=None)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value

# Initialize environment and policy network
env = gymnasium.make("ALE/Pong-v5")
num_actions = env.action_space.n
model = PPOModel(num_actions)
optimizer = Adam(learning_rate)

# Define loss function
def compute_loss(model, states, actions, advantages, returns, old_probs):
    logits, values = model(states)
    values = tf.squeeze(values, axis=-1)
    
    new_probs = tf.nn.softmax(logits)
    probs = tf.gather(new_probs, actions, batch_dims=1)
    ratio = new_probs / (old_probs + 1e-10)
    
    # PPO clip loss
    unclipped_loss = advantages * ratio
    clipped_loss = advantages * tf.clip_by_value(ratio, 1-0.2, 1+0.2)
    policy_loss = -tf.reduce_mean(tf.minimum(unclipped_loss, clipped_loss))
    
    value_loss = tf.reduce_mean(tf.square(returns - values))
    
    return policy_loss + 0.5 * value_loss

# Training loop
for epoch in range(num_epochs):
    observations = []
    actions = []
    rewards = []
    values = []
    old_probs = []

    observation = env.reset()
    done = False
    while not done:
        observation = tf.image.rgb_to_grayscale(observation)
        observation = tf.image.resize(observation, (84, 84))
        observation = tf.squeeze(observation)
        observations.append(observation)

        logits, value = model(np.expand_dims(observation, axis=0))
        action = tf.random.categorical(logits, 1)[0, 0]
        actions.append(action)
        values.append(value)

        old_probs.append(tf.nn.softmax(logits)[0, action])

        observation, reward, done, _ = env.step(action)
        rewards.append(reward)
    
    # Compute returns and advantages
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        returns[t] = R
        advantages[t] = R - values[t]

    # Normalize advantages
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

    # Convert lists to numpy arrays
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    advantages = np.array(advantages, dtype=np.float32)
    returns = np.array(returns, dtype=np.float32)
    old_probs = np.array(old_probs, dtype=np.float32)

    # Train policy network
    with tf.GradientTape() as tape:
        loss = compute_loss(model, observations, actions, advantages, returns, old_probs)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Print epoch stats
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.numpy()}")

env.close()
