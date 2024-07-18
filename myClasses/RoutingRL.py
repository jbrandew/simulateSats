#Hello! This class is for the RL agent. 
#we are defining everything in terms of our personal satellite :))
#very high coupling between this RL agent and the satellite, as many variables are basically the same 

#first, get the satellite class 
import myClasses.Player as Player

#make imports for the gym environment and plots 
import numpy as np 
 
import matplotlib.pyplot as plt
import PIL.Image as Image
import random

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
                        ('state', 'action', 'next_state', 'propDelay'))

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
    """
    Basic DQN with slight modifications class 
    """
    #create the initial DQN network
    #just 3 layers, with the input being the observations, and the actions being the output
    def __init__(self, n_observations, n_actions, networkType = "RNN"):
        #initialize network 
        super(DQN, self).__init__()

        self.n_actions = n_actions
        self.n_observations = n_observations
        self.networkType = networkType

        if(networkType == "FF"): 
            self.initializeBasicFFNetwork() 
        
        if(networkType == "RNN"): 
            self.initializeRNNNetwork()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    #forward pass through the network, using the basic input 
    def forward(self, x):

        #convert to proper data type 
        x = x.to(torch.float32)

        if(self.networkType == "FF"):
            x = F.relu(self.layer1(x.unsqueeze(0)))
            x = F.relu(self.layer2(x))
            #reduce dimensionality
            return self.layer3(x)[0]

        if(self.networkType == "RNN"): 
            #pass through RNN, formatting dimensionality and only taking one of the outputs 
            x = F.relu(self.layer1(x.unsqueeze(0))[0])
            #pass the last hidden layer output to the feed forward net 
            x = F.relu(self.layer2(x))
            #forward again 
            #x = F.relu(self.layer3(x))
            #reduce dimensionality 
            return self.layer3(x)[0]
        
    
    def initializeBasicFFNetwork(self): 
        #create layers 
        self.layer1 = nn.Linear(self.n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, self.n_actions)
        
    def initializeRNNNetwork(self): 
        
        #set up one RNN layer 
        self.layer1 = nn.RNN(self.n_observations, 5, 6)
        #then, set up feed forward layers
        self.layer2 = nn.Linear(5, 32)
        self.layer3 = nn.Linear(32, self.n_actions)
        
#
class DQNAgentRouting: 
    """
    Agent that uses DQN for routing decisions 
    """

    def __init__(self, 
                 satellite, 
                 trainingPolicy, 
                 trainingManager = None): 


        #set up hardware:
        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #store the satellite this agent gives routing info to 
        self.satellite = satellite

        #setup hyperparameters; used for training 
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        #used to be .05 
        self.EPS_END = 0.5
        #lower "decay" value actually increases rate we go to the "eps_end" value 
        self.EPS_DECAY = 3000
        #used to be .005
        #make Tau = 1 to disable target network concept 
        self.TAU = 0.5
        self.LR = 1e-3

        self.discountFactor = 0.9 

        #initialize the # of steps we have completed 
        self.steps_done = 0 

        #initialize storage across episodes
        self.crossEpsAction = []
        self.crossEpsLoss = []
        self.crossEpsReward = []

        #store info 
        self.trainingPolicy = trainingPolicy

        #if we actually have a manager, then use centralized training over distributed
        #so, store the training manager
        self.trainingManager = trainingManager
        
    def initializeNetworks(self): 
        #from the satellite, get the number of possible actions and the shape of the observation space
        #get number of actions from connected players
        n_actions = len(self.satellite.connectedToPlayers)

        #get size of observation space from the adjMatrix and QLengths
        #TODO: look at observation compression, as most likely only nearest info matters that much 
        n_observations = np.sum(self.satellite.adjMatrix != np.inf)

        #policy net = network we use to make our decisions. its the one that we use forward passes to interact with the environment
        #target net = network we use to train upon i.e. the network that generates the target that we use to update the policy net 
        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        #create optimizer and buffer 
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.fullExperienceMemory = ReplayMemory(10000)
        self.nonRewardMemory = {}

        #past rewards
        self.epsRewards = [] 
        self.epsLoss = []
        self.epsActions = []

    def resetData(self): 
        #then reset the episode rewards for the agent
        self.epsRewards = []
        self.epsLoss = []
        self.epsActions = [] 
    
    def episodeAnalysis(self): 
        
        #print the past average reward and loss  
        print("Average Reward")
        print(np.average(self.epsRewards))

        print("Average Loss")
        print(np.average(self.epsLoss))
        self.crossEpsLoss = self.crossEpsLoss + [np.average(self.epsLoss)]

        #also, print the stats of the selected actions 
        print("Action Stats")
        actionStats = np.unique(self.epsActions, return_counts=True)
        self.crossEpsAction = self.crossEpsAction + [actionStats]

        self.crossEpsReward = self.crossEpsReward + [np.average(self.epsRewards)]

        print(actionStats)


    #create function for selecting action based on state 
    def select_action(self, packet):
        """
        
        Function to: 
        1. get and format state 
        2. get an action based on that state
        3. update the target network 

        Kind of convoluted, because at different times we need to index by adjacency matrix
        in 2d format, vs 1d for the network input

        Inputs:
        packet: meta data on packet 

        Outputs: 
        action: what satellite to route the packet towards 

        """
        
        #create experience from adjMatrix and QLengths
        adjMatrixState = copy.deepcopy(self.satellite.adjMatrix)

        #get the queue lengths 
        queueLengthState = copy.deepcopy(self.satellite.getQLengths())

        #then, combine the adjMatrixState and the qeue lengths 
        #need the 2d format for this 
        for ind, value in enumerate(queueLengthState): 
            
            adjMatrixState[ind] +=value/2
            adjMatrixState[:,ind] +=value/2
            adjMatrixState[ind,ind] -=value/2

        #then, get the non-inf values 
        #non-inf means there exists a valid connection between the two 
        #this should never change shape/size, because the topology remains the same 
        #note, this also automatically reshapes into input shape 
        overallState = torch.from_numpy(adjMatrixState[adjMatrixState != np.inf])

        #use epsilon-greedy exploration
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done+=1

        if sample > eps_threshold:
            with torch.no_grad():

                #so, get the policy net output
                actionOutput = self.policy_net(overallState).argmax().item()
        
        else: 
            actionOutput = np.random.randint(self.policy_net.n_actions)

        #then, convert it to viable satellite ind 
        #we need to do this because indexableSats != connectedToPlayers
        indexableSats = sorted(self.satellite.connectedToPlayers)
        satIndToForwardTo = indexableSats[actionOutput].adjMatPersonalIndex

        #create predicted state (simple for now)
        #to create the predicted state, get the timing for processing a packet    
        timeToProcess = indexableSats[actionOutput].generateProcessingOnePacketTime() 

        #then, increase the associated queueLength state 
        #so first, get basic adjMatrixState info 
        predictedState = copy.deepcopy(adjMatrixState)

        #then, get the predicted state with modifications using time to process 
        predictedState[satIndToForwardTo]+=timeToProcess/2
        predictedState[:,satIndToForwardTo]+=timeToProcess/2
        predictedState[satIndToForwardTo, satIndToForwardTo]-=timeToProcess/2

        #then, reindex the new state 
        predictedState = torch.from_numpy(predictedState[predictedState != np.inf])

        #if we are doing distributed training, store experience and optimize your self 
        if(self.trainingPolicy == "distributed"): 
            #create experience and push it 
            self.nonRewardMemory[packet.packetIndex] = [overallState, actionOutput, predictedState]
            self.optimize() 

        #if we are doing centralized training
        else: 
            #then just send experience to central network
            self.trainingManager.storeExperience([self.satellite.adjMatPersonalIndex,
                                                packet.packetIndex,
                                                overallState,
                                                predictedState])
                
            #there is no optimize method here, as thats done in the central node training
            

        #target networks != incompatible with centralized training 
        #afterwards, update the target network
        #so get the dictionaries 
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        #then, slowly update the new network 
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        #store the action to episode buffer 
        self.epsActions = self.epsActions + [satIndToForwardTo]

        return satIndToForwardTo 
        
    def optimize(self):
        """
        Optimize the current network with respect to experiences in buffer. 

        Please note, currently this assumes that the state of the topology when we are making experiences is the
        same as the state of the topology when we are training. 

        """
        #this only works with experiences that have their reward
        #so, read in a value from the buffer: 

        smallBatchSize = 2

        #small batch size seems better in general 
        # 2 gave better performance....
        #im not sure why that is. maybe coupling samples in training introduces temporal dependence, which in this case might
        #be good 

        #this is useful: torch.cat(batch.state).shape[0]

        if len( self.fullExperienceMemory ) < smallBatchSize:
            return
        
        transitions = self.fullExperienceMemory.sample(smallBatchSize)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        #so, get necessary elements from the batch 
        state_batch = torch.cat(batch.state)
        #dont need to cat the action, as the dimensions of each element are matching 
        #same with propDelay
        action_batch = torch.tensor(batch.action)
        next_state_batch = torch.cat(batch.next_state)

        #reshape the state and next state 
        #first get numElementsPerSet 
        numElementsPerSet = int(state_batch.size()[0] / smallBatchSize)
        
        #then, reshape 
        state_batch = state_batch.reshape([smallBatchSize, numElementsPerSet])
        next_state_batch = next_state_batch.reshape([smallBatchSize, numElementsPerSet])

        #reward is generated as the inverse of the propagation delay 
        reward_batch = torch.tensor([1/i for i in batch.propDelay])

        #store the reward for that batch 
        self.epsRewards = self.epsRewards + [np.average(reward_batch)]

        #then, get the current state values
        #use the action that we actually executed beforehand
        #alternatively, this could just be the max operatior as well...  
        #we need to match the dimensions of indexer vs data, which is why we do the squeeze 
        #we do this with gradients, because we will optimize with them in a second 
        state_action_values = self.policy_net(state_batch).gather(1,action_batch.unsqueeze(1))

        #.gather(1, action_batch), either use the action for indexing, or just index by action

        #get the next state values 
        #should be a list here...
        #will need to make modifications for the approach when i use batching instead of single values
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1).values

        #then get the values for next state actions using the reward  
        target_state_action_values = (next_state_values * self.GAMMA) + self.discountFactor * reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, target_state_action_values.unsqueeze(1))

        self.epsLoss = self.epsLoss + [np.average(loss.detach().numpy())]

        # Optimize the model
        #zero out gradients, as we arent using memory here over batches 
        self.optimizer.zero_grad()
        #back propagate with respect to the generated loss 
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return 
    
    def retroactiveRewardCreation(self, packet): 
        """
        Here, retroactively create reward based on packet properties 
        Currently, its simply the inverse of propagation delay of the packet through the network 
        """

        #first get the time it took for the packet to go through the network 
        packetPropDelay = packet.packetArriveTime - packet.packetSendTime
        #then, create and push the experience 
        self.fullExperienceMemory.push(*self.nonRewardMemory[packet.packetIndex], packetPropDelay)


    def plotTrainingInfo(self):
        """
        Just plot the info over the episode. This includes both action distribution and model loss over time.
        
        """

        #after the episodes, examine the data 
        crossEpsLoss = self.crossEpsLoss
        crossEpsAction = self.crossEpsAction

        #store the action distribution 
        #shape of numEpisodes by numPossibleActions 
        numPossibleActions = len(crossEpsAction[0][0])
        num_episodes = len(crossEpsLoss)
        actionDistributionStorage = np.ones([num_episodes, numPossibleActions])

        #get action distribution for going to each satellite
        #so, for each action set 
        for actionSetInd, actionSet in enumerate(crossEpsAction): 
            #get the num of each action 
            numTimesSentTo1 = actionSet[1][0]
            numTimesSentTo3 = actionSet[1][1]

            actionDistributionStorage[actionSetInd, 0] = numTimesSentTo1
            actionDistributionStorage[actionSetInd, 1] = numTimesSentTo3


        #then, after we have the stored distribution, plot it over the episode number 
        fig, ax = plt.subplots()
        action1, = ax.plot(np.arange(num_episodes), actionDistributionStorage[:,0], label = '# Packets Directed to Server 2')
        action2, = ax.plot(np.arange(num_episodes), actionDistributionStorage[:,1], label = '# Packets Directed to Server 1')
        ax.set(xlabel='Episode #', ylabel='# of Packets Sent to one direction',
            title='Action Distribution over Episode. Each episode involves transmission of 500 packets.')
        ax.legend(handles=[action1, action2])
        plt.show()

        #simply plot the loss over time
        fig, ax = plt.subplots()
        ax.plot(np.arange(num_episodes), crossEpsLoss)
        ax.set(xlabel='Episode #', ylabel='Loss',
            title='Loss over Episode')
        plt.show()


        #simply plot the loss over time
        fig, ax = plt.subplots()
        ax.plot(np.arange(num_episodes), crossEpsAction)
        ax.set(xlabel='Episode #', ylabel='Loss',
            title='Loss over Episode')
        plt.show()