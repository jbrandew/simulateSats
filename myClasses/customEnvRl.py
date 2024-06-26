#Hello! This class is for the RL agent. 
#we are defining everything in terms of our personal satellite :))
#very high coupling between this RL agent and the satellite, as many variables are basically the same 

#first, get the satellite class 
import myClasses.Player as Player

#make imports for the gym environment and plots 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time

#make imports for processing and models 
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#create an enviroment, based on our satellite's view
#this satellite env inherits from the normal env class 
class SatelliteWorldview(Env):
    def __init__(self, satellite):
        """
        Function for initializing the satellite worldview 
        """

        #so, initialize the worldview 
        super(SatelliteWorldview, self).__init__()
        
        #store the satellite
        self.satellite = satellite

        #initialize observation space 
        #this "observation space" will be dependent on how many other satellites there are, etc. 
        #should encompass both queing delay and propagation delay
        self.observation_shape = np.shape(self.satellite.adjMatrix)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.inf*np.ones(self.observation_shape),
                                            dtype = np.float16)

        # Define an action space describing how many satellites we are connected to, for routing
        self.action_space = spaces.Discrete(len(self.satellite.connectedToPlayers),)
                        

    def reset(self):
        """
        Function for resetting components.
        Please note, most enviroment specific resetting will be done by the upper "satellite grid" env. 
        This "env" is really just for my own satellites view, which can be affected by the upper part  
        """
        #reset episode reward
        self.ep_return  = 0

    def step(self, action): 
        """
        A step function is where you interact with the environment, and get back information about your interaction.
        In this case, you dont get back any information, as everything is done externally. 

        That is: 
        The state is periodically updated from information from neighbors.
        The reward is retroactively assigned. That is, the reward is given/assigned after the packet is recieved, which
        is farther down the line. 
        The "done" factor is returned within the discrete event stack, when all packets are delivered or we time out
        We just set the "packetNum" to this interaction, to label the tuple of information. 
        """
        
        #for now, just randomly sample action, while we dont have agent set up 
        action = self.action_space.sample()


        packetNum = 0 
        return self.state, None, False, packetNum 


    def close(self):
        
        return     
    
    

#transition consists of state, action, next state, reward
#just a tuple of information 
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#create replay object 
class ReplayMemory(object):

    #create basic object, dequer for actual memory 
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    #put a transition/experience on the buffer 
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    #randomly sample from our past memory for an amount = batch size 
    #sample without replacement, but doesnt actually remove from the buffer/sampling 
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    #get the length of the buffer. not sure if this is calculated every time 
    def __len__(self):
        return len(self.memory)
    

#create DQN network 
class DQN(nn.Module):

    #create the initial DQN network
    #just 3 layers, with the input being the observations, and the actions being the output
    def __init__(self, n_observations, n_actions):
        #initialize network 
        super(DQN, self).__init__()

        #create layers 
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    #forward pass through the network, using the basic input 
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent: 
    
    def __init__(self): 

        #set up hardware:
        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #set up environment
        #will do this with custom registered environment 
        self.env = gym.make()#"CartPole-v1", render_mode="rgb_array")

        #setup hyperparameters; used for training 
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4

        # Get number of actions from gym action space
        n_actions = self.env.action_space.n
        # Get the number of state observations
        state, info = self.env.reset()
        n_observations = len(state)

        #policy net = network we use to make our decisions. its the one that we use forward passes to interact with the environment
        #target net = network we use to train upon i.e. the network that generates the target that we use to update the policy net 
        policy_net = DQN(n_observations, n_actions).to(self.device)
        target_net = DQN(n_observations, n_actions).to(self.device)
        target_net.load_state_dict(policy_net.state_dict())

        #create optimizer and buffer 
        optimizer = optim.AdamW(policy_net.parameters(), lr=self.LR, amsgrad=True)
        memory = ReplayMemory(10000)


    #create function for selecting action based on state 
    def select_action(self, state):
        #for now, take random action 
        #can optimize later... ('state', 'action', 'next_state', 'reward'))
        
        #select random action 
        action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long) 
        
        #create experience and push it 
        self.memory.push(self.env.state, action, self.env.state, None)
        return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)


    #need "optimize method here"
    #and then, need training loop, probably outside...maybe....
