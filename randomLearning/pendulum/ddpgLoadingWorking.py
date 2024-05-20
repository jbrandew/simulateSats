import gymnasium as gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import matplotlib.pyplot as plt
from IPython import display

import pdb 

def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render())
    #plt.title("%s | Step: %d %s" % (env.spec.id,step, info))
    plt.axis('off')
    
    plt.show(block=False)
    plt.close() 
    display.clear_output(wait=True)
    display.display(plt.gcf())

env = gym.make("Pendulum-v1", render_mode="rgb_array")

# # The noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
# model.learn(total_timesteps=10000, log_interval=10)
# model.save("ddpg_pendulum")


# del model # remove to demonstrate saving and loading
# pdb.set_trace() 
model = DDPG.load("ddpg_pendulum")
#vec_env = model.get_env()
obs = env.reset()[0]

while(True): 
    action, _states = model.predict(obs)
    obs, rewards, dones, trunc, info = env.step(action)
    print(rewards)
    #env.render()
    show_state(env.env)


