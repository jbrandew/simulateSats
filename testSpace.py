import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from tianshou.env import SubprocVectorEnv
from tianshou.data import Collector, ReplayBuffer
from tianshou.policy import A2CPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

# Create the CartPole environment
env_name = "CartPole-v1"
train_envs = SubprocVectorEnv([lambda: gym.make(env_name) for _ in range(8)], wrapper_class=gym.wrappers)
test_envs = SubprocVectorEnv([lambda: gym.make(env_name) for _ in range(100)], wrapper_class=gym.wrappers)

# Define the network for actor and critic
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

net = Net(state_shape, hidden_sizes=[128, 128], activation=nn.ReLU, device="cpu")
actor = Actor(net, action_shape, softmax_output=False, device="cpu").to("cpu")
critic = Critic(net, device="cpu").to("cpu")
actor_critic = ActorCritic(actor, critic)

# Define the optimizer
optim = Adam(actor_critic.parameters(), lr=3e-4)

# Define the policy using A2C (which is an actor-critic method)
policy = A2CPolicy(
    actor,
    critic,
    optim,
    dist_fn=torch.distributions.Categorical,
    discount_factor=0.99,
    gae_lambda=0.95,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
)

# Define the buffer and collectors
train_collector = Collector(policy, train_envs, ReplayBuffer(20000))
test_collector = Collector(policy, test_envs)

# Train the policy
result = onpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    max_epoch=10,
    step_per_epoch=10000,
    repeat_per_collect=4,
    episode_per_test=10,
    batch_size=64,
)

# Test the trained policy
policy.eval()
test_collector.reset()
result = test_collector.collect(n_episode=10, render=0.1)
print(f"Final reward: {result['rews'].mean()}, length: {result['lens'].mean()}")
