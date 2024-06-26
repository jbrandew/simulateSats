#Hello! This class is for the RL agent. 
#we are defining everything in terms of our personal satellite :))
#very high coupling between this RL agent and the satellite, as many variables are basically the same 

#first, get the satellite class 
import myClasses.Player as Player

#make imports for the gym environment and plots 
import numpy as np 
 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time
import copy

#make imports for processing and models 
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb 

import math


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

        self.n_actions = n_actions
        self.n_observations = n_observations

        #create layers 
        self.layer1 = nn.Linear(self.n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, self.n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    #forward pass through the network, using the basic input 
    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
#
class DQNAgentRouting: 
    
    def __init__(self, satellite): 

        #set up hardware:
        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #store the satellite this agent gives routing info to 
        self.satellite = satellite

        #setup hyperparameters; used for training 
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4

        #initialize networks when we first select the action, as by that point, the topology will be set up 
        self.initializedNetworks = False

    def initializeNetworks(self): 
        #from the satellite, get the number of possible actions and the shape of the observation space
        #get number of actions from connected players
        n_actions = len(self.satellite.connectedToPlayers)
        #get size of observation space from the adjMatrix and QLengths (should be equal)
        #TODO: look at observation compression. Only in nearby vicinity most likely matters 
        n_observations = np.size(self.satellite.adjMatrix)
        n_observations+= np.size(self.satellite.QFinishTimes)

        #policy net = network we use to make our decisions. its the one that we use forward passes to interact with the environment
        #target net = network we use to train upon i.e. the network that generates the target that we use to update the policy net 
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        #create optimizer and buffer 
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

    #create function for selecting action based on state 
    def select_action(self, packet):
        """
        Function to select action based on policy using an input packet and the current environment state. 

        Inputs:
        packet: meta data on packet 

        Outputs: 
        action: what satellite to route the packet towards 

        """

        #if we have not initialized the networks, then do that. 
        if(not self.initializedNetworks): 
            self.initializedNetworks = True
            self.initializeNetworks() 
        
        #create experience from adjMatrix and QLengths
        adjMatrixState = np.ravel(copy.deepcopy(self.satellite.adjMatrix))
        queueLengthState = copy.deepcopy(self.satellite.getQLengths())

        #format the network input data 
        overallState = np.concatenate([adjMatrixState, queueLengthState])
        overallState = [float(i) for i in overallState]
        overallState = torch.tensor(overallState)

        #create predicted state (simple for now)
        predictedState = overallState
                
        #use epsilon-greedy exploration
        sample = random.random()
        eps_threshold = 1

        if sample > eps_threshold:
            #with torch.no_grad():

            #so, get the policy net output
            actionOutput = self.policy_net(overallState).argmax().item()
        
        else: 
            actionOutput = np.random.randint(self.policy_net.n_actions)

        #then, convert it to viable satellite ind 
        indexableSats = sorted(self.satellite.connectedToPlayers)
        satIndToForwardTo = indexableSats[actionOutput].adjMatPersonalIndex

        #create experience and push it 
        self.memory.push(overallState, satIndToForwardTo, predictedState, None)

        # print("Action")
        # print(actionOutput)
        # print("Satellite index to send to")
        # print(satIndToForwardTo)
        
        # pdb.set_trace() 

        #return the viable satellite index now :) 
        return satIndToForwardTo 
        
        #torch.tensor([[self.policy_net.forward(overallState)]], device=self.device, dtype=torch.long)
    
    def optimize(self):
        """
        Optimize the current network with respect to experiences in buffer. 

        Please note, currently this assumes that the state of the topology when we are making experiences is the
        same as the state of the topology when we are training. 

        """
        
        
        
        return 