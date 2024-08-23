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
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import math

#transition consists of state, action, next state, reward
#just a tuple of information 
Transition = namedtuple('Transition',
                        ('dest','state', 'action', 'next_state', 'reward'))

# class Transition:
#     def __init__(self, state, action, next_state, propDelay):
#         self.state = state
#         self.action = action
#         self.next_state = next_state
#         self.propDelay = propDelay


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
    def sample(self, numSamples):
        return random.sample(self.memory, numSamples)

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
    def __init__(self, 
                 n_observations, 
                 n_actions, 
                 initialAdjMatrix = None, 
                 device = "cpu",
                 networkType = "RNN",
                 numEmbeddings = None,
                 ):
        """
        DQN initialization 

        n_observations: how many observations do we see (non inf values in adj matrix)
        n_actions: how many different actions can we take 
        initialAdjMatrix: used for structural creation of GCNN network 
        device: type of device we are using. helps with GPU compatibility in training 
        networkType: what architecture are we using 
        numEmbeddings: how many different embeddings we can have. here, its just the # of satellites in our grid 
        """

        #initialize network 
        super(DQN, self).__init__()

        self.n_actions = n_actions
        self.n_observations = n_observations
        self.initialAdjMatrix = initialAdjMatrix
        self.device = device 
                
        self.networkType = networkType
        self.numEmbeddings = numEmbeddings

        if(networkType == "FF"): 
            self.initializeBasicFFNetwork() 
        
        if(networkType == "RNN"): 
            self.initializeRNNNetwork()

        if(networkType == "EmbedRNN"): 
            self.initializeEmbedRNNNetwork()

        if(networkType == "GCNN"): 
            self.initializeGCNNNetwork()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    #forward pass through the network, using the basic input 
    def forward(self, 
                rawAdjMat, 
                dest = None):
        """
        This function has different functionality, based on the type of network we are using. 
    
        inputs: 
        rawAdjMat: unprocessed adj Matrix 
        dest: packet destination

        output:
        batched output/action values given 

        """

        #if we are not working with batched data
        if(rawAdjMat.dim() == 2): 
            #create batch dimension
            rawAdjMat = rawAdjMat.unsqueeze(0)
            
            #create batch dimension for dest if necessary 
            if(dest is not None): 
                dest = dest.unsqueeze(0)
 
        #convert to proper data type for each input  
        rawAdjMat = rawAdjMat.to(torch.float32)

        #put input tensor on device. aids in GPU compatability 
        rawAdjMat = rawAdjMat.to(self.device)

        #get non inf nor 0 vals over batched input
        validVals = [smallRawAdjMat[(smallRawAdjMat != np.inf) & (smallRawAdjMat != 0)] for smallRawAdjMat in rawAdjMat]
        
        #format data
        validVals = torch.tensor(np.array(validVals))

        if(self.networkType == "FF"):
            
            x = F.relu(self.layer1(validVals))
            x = F.relu(self.layer2(x))
            #reduce dimensionality
            return self.layer3(x)

        if(self.networkType == "RNN"): 
            #pass through RNN, formatting dimensionality and only taking one of the outputs 
            x = F.relu(self.layer1(validVals))
            #pass the last hidden layer output to the feed forward net 
            x = F.relu(self.layer2(x))
            #forward again 
            #x = F.relu(self.layer3(x))
    
            return self.layer3(x)
        
        if(self.networkType == "EmbedRNN"):

            #format data
            dest = torch.tensor(dest)
            
            #get embedding of destination
            embedded = self.embeddingLayer(dest)

            #get adj mat processed 
            x = F.relu(self.layer1(validVals))

            #get the combined version data output 
            #concat along non batch dimension
            x = torch.cat([embedded, x], dim = 1) 

            #then, input to next layer 
            x = F.relu(self.layer2(x))

            #then get output 
            return self.layer3(x)
        
        if(self.networkType == "GCNN"):

            """
            Dest processing 
            """
            #format data
            dest = torch.tensor(dest)
            
            #get embedding of destination. remove batch dimension, as GCNConv doesnt work with that
            embedded = self.embeddingLayer(dest)[0]

            """
            Adj Mat processing to get valid indices  
            """

            #get indices of valid edges within the graph             
            #edge_index = torch.tensor(np.array(np.nonzero(np.triu(rawAdjMat))), dtype=torch.long)
            
            #first, get upper triangular to eliminate repeats
            #will need to do these operations across many batch entries....
            edge_index = np.triu(rawAdjMat)

            #then, get proper mask 
            edge_index = (edge_index != 0) & np.isfinite(edge_index)

            #then, get nonzero values, getting indices of the valid mask entries 
            edge_index =  torch.tensor(np.array(np.nonzero(edge_index)), dtype=torch.long)
            #this removes the batch dimension 
            edge_index = edge_index[1:3]

            #then, get the values of the edges 
            #then, get the rows and cols for getting the proper weights 
            #use 1 and 2 because we are skipping the batch dimension 
            rows = edge_index[0].numpy()
            cols = edge_index[1].numpy()            

            #get the edge weights using the rows and cols 
            edge_weights = rawAdjMat[:, rows, cols]

            #for now just unsqueeze
            edge_weights = edge_weights.squeeze(0)
            
            #create batch dimension
            #edge_index = edge_index.unsqueeze(0)

            #create filler for node values. shape is batch dimension * # nodes * single feature 
            node_values = torch.tensor(np.ones([len(rawAdjMat[0]),1]), dtype=torch.float32)
            
            #then, go through networks 
            #pdb.set_trace()
            adjMatProc = self.gconv1(node_values, edge_index, edge_weights) 
            adjMatProc = self.gconv2(adjMatProc, edge_index, edge_weights)
            #remove channel dimension
            adjMatProc = adjMatProc[:,0]
            
            """
            Combined processing 
            """

            adjMatProc = torch.cat([embedded, adjMatProc])

            #then, input to next layer 
            adjMatProc = F.relu(self.layer2(adjMatProc))

            #then get output 
            return self.layer3(adjMatProc)
                
        if(self.networkType == "AttentionRNN"):
            
            raise Exception("Parameters not yet initialized")

            #format data
            dest = torch.tensor(dest)
            
            #get embedding of destination 
            embedded = self.embeddingLayer(dest)

            #get adj mat processed 
            adjMatProc = F.relu(self.layer1(rawAdjMat.unsqueeze(0))[0])

            #get the combined version data output 
            #concat along dimension thats dependent on if we are batched or not 
            rawAdjMat = torch.cat([embedded, adjMatProc], dim= adjMatProc.dim() - 1)

            #then, input to next layer 
            rawAdjMat = F.relu(self.layer2(rawAdjMat))

            rawAdjMat = F.relu(self.layer3(rawAdjMat))

            #then get output 
            return self.layer4(rawAdjMat)
        
    def initializeBasicFFNetwork(self): 
        #create layers 
        self.layer1 = nn.Linear(self.n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, self.n_actions)
        
    def initializeRNNNetwork(self): 
        
        #set up one RNN layer 
        self.layer1 = nn.RNN(self.n_observations, 5, 3)

        #then, set up feed forward layers
        self.layer2 = nn.Linear(5, 16)
        self.layer3 = nn.Linear(16, self.n_actions)

    def initializeEmbedRNNNetwork(self):
        #first, create an embedding layer for the input data involving the destination of the packet 
        self.embeddingLayer = nn.Embedding(num_embeddings=self.numEmbeddings, embedding_dim=10)

        #then, create a linear layer for spatial relationships
        self.layer1 = nn.Linear(self.n_observations, 16)

        #then, create a linear layer for combining them  
        self.layer2 = nn.Linear(16 + 10, 16)

        #then, create a final output layer 
        self.layer3 = nn.Linear(16, self.n_actions)

    def initializeGCNNNetwork(self):

        #first, create an embedding layer for the input data involving the destination of the packet 
        self.embeddingLayer = nn.Embedding(num_embeddings=self.numEmbeddings, embedding_dim=10)

        #then, create two GCNNs for the input adj matrix 
        self.gconv1 = GCNConv(1, 16)
        self.gconv2 = GCNConv(16, 1)

        #then, create a linear layer for combining them  
        self.layer2 = nn.Linear(self.n_observations + self.embeddingLayer.embedding_dim, 16)

        #then, create a final output layer 
        self.layer3 = nn.Linear(16, self.n_actions)
 
    def initializeAttentionNetwork(self):


        #first, create an embedding layer for the input data involving the destination of the packet 
        self.embeddingLayer = nn.Embedding(num_embeddings=self.numEmbeddings, embedding_dim=10)
        
        #then, create a linear layer for spatial relationships
        self.layer1 = nn.Linear(self.n_observations, 16)

        self.layer2 = nn.MultiheadAttention(26, 1)

        #then, create a linear layer for combining them  
        self.layer3 = nn.Linear(16 + 10, 16)

        #then, create a final output layer 
        self.layer4 = nn.Linear(16, self.n_actions)


class DQNAgentRouting: 
    """
    Agent that uses DQN for routing decisions 
    
    satellite: the satellite this agent operates 
    trainingPolicy: centralized vs distributed
    rewardType: retroactive or immediate. 
    retroactive is computed as the propagation delay of the packets you are involved in sending
    immediate is the difference in distance between the place you got it from and the one you are sending towards
    
    trainingManager: if doing centralized training, this is the manager for giving gradients/coordinating 
    satelliteGridSize: need to know how many satellites there are for me to possibly forward packets in the direction of
    """

    def __init__(self, 
                 satellite, 
                 trainingPolicy, 
                 trainingManager = None,
                 rewardType = "immediate",
                 satelliteGridSize = None): 

        #set up hardware:
        # if GPU is to be used
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"


        #store the satellite this agent gives routing info to 
        self.satellite = satellite

        #store size
        self.satelliteGridSize = satelliteGridSize
    
        #setup hyperparameters; used for training 
        self.BATCH_SIZE = 32
        self.GAMMA = 0.98 
        self.EPS_START = 0.9
        #used to be .05 
        #so regardless of poliy we learned, we always go random at least 15% of the time. 
        self.EPS_END = 0.15
        #lower "decay" value actually increases rate we go to the "eps_end" value 
        self.EPS_DECAY = 10000
        #used to be .005
        #make Tau = 1 to disable target network concept 
        self.TAU = 0.9
        self.LR = 1e-3

        #initialize the # of steps we have completed 
        self.steps_done = 0 

        #initialize storage across episodes
        self.crossEpsAction = []
        self.crossEpsLoss = []
        self.crossEpsReward = []

        #store info 
        self.trainingPolicy = trainingPolicy
        self.rewardType = rewardType

        #if we actually have a manager, then use centralized training over distributed
        #so, store the training manager
        self.trainingManager = trainingManager
        
    def initializeNetworks(self): 
        #from the satellite, get the number of possible actions and the shape of the observation space
        #get number of actions from connected players
        n_actions = len(self.satellite.connectedToPlayers)

        #get size of observation space from the adjMatrix and QLengths
        #TODO: look at observation compression, as most likely only nearest info matters that much 
        n_observations = np.sum((self.satellite.adjMatrix != np.inf) & (self.satellite.adjMatrix != 0))

        #policy net = network we use to make our decisions. its the one that we use forward passes to interact with the environment
        #target net = network we use to train upon i.e. the network that generates the target that we use to update the policy net 
        self.policy_net = DQN(n_observations, 
                              n_actions, 
                              self.satellite.adjMatrix,
                              self.device, 
                              "FF", 
                              self.satelliteGridSize).to(self.device)
        
        self.target_net = DQN(n_observations, 
                              n_actions, 
                              self.satellite.adjMatrix,
                              self.device, 
                              "FF", 
                              self.satelliteGridSize).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())

        #create optimizer and buffer 
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)

        #if we are doing retroactive rewards, create pollinated and non pollinated memory 
        if(self.rewardType == "retroactive"): 
            self.fullExperienceMemory = ReplayMemory(10000)
            self.nonRewardMemory = {}
        
        #if we are doing immediate rewards, just create a basic memory   
        elif(self.rewardType == "immediate"): 
            self.memory = ReplayMemory(1000)
            
        #past rewards
        self.epsRewards = [] 
        self.epsLoss = []
        self.epsActions = []

        self.crossPerformance = []

    def resetData(self):

        #then reset the episode rewards for the agent
        self.epsRewards = []
        self.epsLoss = []
        self.epsActions = [] 
    
    def episodeAnalysis(self): 
        
    
        #print the past average reward and loss  
        print("Average Reward, for the following satellite")
        print(self.satellite.adjMatPersonalIndex)
        print(np.average(self.epsRewards))

        # print("Average Loss")
        # print(np.average(self.epsLoss))
        # self.crossEpsLoss = self.crossEpsLoss + [np.average(self.epsLoss)]

        # #also, print the stats of the selected actions 
        # #print("Action Stats")
        # #print(self.epsActions)
        # actionStats = np.unique(self.epsActions, return_counts=True)
        # self.crossEpsAction = self.crossEpsAction + [actionStats]

        # self.crossEpsReward = self.crossEpsReward + [np.average(self.epsRewards)]

        # print(actionStats)

    def getOverallState(self):
        """
        Function to get the state of the network. 
        Index using only non-inf values for overallState. adjMatrixState incorporates q lengths. 

        Inputs:

        Outputs:
        adjMatrixState: adjMatrix incorporating q lengths and prop delay. 
        overallState: adjMatrix properly indexed 

        """

        #create experience from adjMatrix and QLengths
        adjMatrixState = copy.deepcopy(self.satellite.adjMatrix)

        if(not np.all(np.diag(adjMatrixState) == 0)):
            #something bad with adjMatrix generation 
            pdb.set_trace() 
        
        #get the queue lengths 
        queueLengthState = copy.deepcopy(self.satellite.getQLengths())

        #then, combine the adjMatrixState and the qeue lengths 
        #need the 2d format for this 
        for ind, value in enumerate(queueLengthState): 
            
            adjMatrixState[ind] +=value/2
            adjMatrixState[:,ind] +=value/2
            #0 along diagonal for adjMatrix 
            adjMatrixState[ind,ind] -=value

        #data conversion          
        overallState = torch.from_numpy(adjMatrixState) 

        return adjMatrixState, overallState

        
    def getEpsThreshold(self):
        return self.EPS_END + (self.EPS_START - self.EPS_END) * \
                math.exp(-1. * self.steps_done / self.EPS_DECAY)

    #create function for selecting action based on state 
    def select_action(self, 
                      packet,
                      explorationType = "eps",
                      selectiveActions = True):
        """
        
        Function to: 
        1. get and format state 
        2. get an action based on that state
        3. update the target network 

        Kind of convoluted, because at different times we need to index by adjacency matrix
        in 2d format, vs 1d for the network input

        Inputs:
        packet: meta data on packet 
        selectiveActions: do we restrict our action space based on what the packet has seen?
        
        Outputs: 
        action: what satellite to route the packet towards 

        """
        
        #get the state stuff 
        adjMatrixState, overallState = self.getOverallState()
        qLengths = self.satellite.getQLengths() 
                
        #first, get the fullActionOutput 
        if(explorationType == "eps"):
            #use epsilon-greedy exploration
            sample = random.random()
            eps_threshold = self.getEpsThreshold() 
            self.steps_done+=1

            #then, if we are using our policy 
            if sample > eps_threshold:
                with torch.no_grad():
                    #first, get the policy net output
                    #get only the first entry, because we are not working with batching
                    fullActionOutput = self.policy_net(overallState, torch.tensor(packet.endSat))[0]
            else: 
                #generate a random policy net output
                fullActionOutput = torch.rand(self.policy_net.n_actions, device = self.device)
        else: 
            raise Exception("This exploration type hasnt been implemented.")

        #get the indexable version of the satellites we are connected to
        indexableSats = sorted(self.satellite.connectedToPlayers)

        #if we arent doing selective action space restriction
        if(not selectiveActions): 
            #then just get the max 
            actionOutput = fullActionOutput.argmax().item()
            satIndToForwardTo = indexableSats[actionOutput].adjMatPersonalIndex

        else:

            #then, get the argsort for the policy net output
            actionPreferenceList = torch.argsort(fullActionOutput, descending=True)

            #so then, get the satellites in the order that we prefer them
            satPreferenceList = [indexableSats[idx] for idx in actionPreferenceList]

            #then, get the adjMatIndices for each 
            satIndexPreferenceList = [sat.adjMatPersonalIndex for sat in satPreferenceList]

            #after getting the preference list, then get the nodes we cant send to 
            nodesToNotSendTo = packet.playersInvolvedInSending

            #then, get the first item in satPreferenceList that doesnt appear in nodesToNotSendTo
            try: 
                satIndToForwardTo, index = next((item, idx) for idx, item in enumerate(satIndexPreferenceList) if item not in nodesToNotSendTo)
            #if there are none present (like in a 2ISL case), then get the first viable edge and use that
            except StopIteration: 
                satIndToForwardTo, index = next((item, idx) for idx, item in enumerate(satIndexPreferenceList) if item in nodesToNotSendTo)

            #then, get the action output for the chosen index 
            actionOutput = actionPreferenceList[index]

        #create predicted state (simple for now)
        #to create the predicted state, get the timing for processing a packet    
        timeToProcess = indexableSats[actionOutput].generateProcessingOnePacketTime() 

        #then, increase the associated queueLength state 
        #so first, get basic adjMatrixState info 
        predictedState = copy.deepcopy(adjMatrixState)

        #then, get the predicted state with modifications using time to process 
        predictedState[satIndToForwardTo]+=timeToProcess/2
        predictedState[:,satIndToForwardTo]+=timeToProcess/2
        #subtract out entirely from origin node
        predictedState[satIndToForwardTo, satIndToForwardTo]-=timeToProcess

        #then, reindex the new state 
        predictedState = torch.from_numpy(predictedState) #[predictedState != np.inf])

        #if we are doing distributed training, store experience and optimize your self 
        if(self.trainingPolicy == "distributed"): 

            #if we are doing retroactive rewards 
            if(self.rewardType == "retroactive"): 
                #create experience and push it 
                self.nonRewardMemory[packet.packetIndex] = [overallState, actionOutput, predictedState]

            #if we are doing immediate rewards 
            elif(self.rewardType == "immediate"): 
                #create more basic experience
                #get the distance covered by using our satellite's visibility 
                #essentially the time distance covered 
                reward = self.satellite.getTimeDistanceDiff(actionOutput, packet.endSat)
                
                #set reward to just be the inverse of the estimated residual prop delay  
                #reward = -self.satellite.adjMatrix[actionOutput, packet.endSat]

                #rewardWithCounterfactual = timeDistanceCovered

                # #create initial sum from counter factuals 
                # counterfactualSum = 0

                # #then, get the counterfactual based on that 
                # #so, for each possible action you could take 
                # for possibleActionOutput in range(self.policy_net.n_actions): 
                #     #subtract out the relative advantage of taking that action 
                #     counterfactualSum = counterfactualSum - (self.satellite.getTimeDistanceDiff(possibleActionOutput, packet.endSat) - timeDistanceCovered)
                
                # #then, final modifications for normalization
                # counterfactualSum = counterfactualSum / (self.policy_net.n_actions - 1)
                # rewardWithCounterfactual = timeDistanceCovered - counterfactualSum

                #then, normalize with respect to rewards already computed
                if(len(self.epsRewards) > 2): 
                    #pdb.set_trace()
                    reward = 10 * (reward - np.average(self.epsRewards))/(np.std(self.epsRewards))
                    #print(rewardWithCounterfactual)
                    #pdb.set_trace() 

                #then, push the experience 
                self.memory.push(packet.endSat, overallState, actionOutput, predictedState, reward)

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

        #small batch size seems better in general 
        # 2 gave better performance....
        #im not sure why that is. maybe coupling samples in training introduces temporal dependence, which in this case might
        #be good 

        #this is useful: torch.cat(batch.state).shape[0]

        #get data depending on the reward type 
        if(self.rewardType == "retroactive"): 
            if len( self.fullExperienceMemory ) < self.BATCH_SIZE:
                return
            
            transitions = self.fullExperienceMemory.sample(self.BATCH_SIZE)

        elif(self.rewardType == "immediate"): 
            if len( self.memory ) < self.BATCH_SIZE:
                return
            
            transitions = self.memory.sample(self.BATCH_SIZE)

        #batch it up 
        batch = Transition(*zip(*transitions))

        #data formatting for each component
        #special formatting for 2D state 
        dest_batch = torch.tensor(batch.dest, device = self.device)
        state_batch = torch.tensor(np.array(batch.state))
        action_batch = torch.tensor(batch.action, device = self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state))
        reward_batch = torch.tensor(batch.reward, device = self.device)

        #store episode specific data 
        self.epsRewards = self.epsRewards + [np.average(reward_batch)]

        #reshape the state and next state 
        #first get numElementsPerSet 
        #numElementsPerSet = int(state_batch.size()[0] / self.BATCH_SIZE)
        
        #then, reshape 
        #pdb.set_trace()
        #state_batch = state_batch.reshape([self.BATCH_SIZE, numElementsPerSet])
        #next_state_batch = next_state_batch.reshape([self.BATCH_SIZE, numElementsPerSet])

        #store reward 


        #self.epsPerformance = self.epsPerformance + [np.average(batch.propDelay)]

        #then, get the current state values
        #use the action that we actually executed beforehand
        #alternatively, this could just be the max operatior as well...  
        #we do this with gradients, because we will optimize with them in a second 
        #reshape to match indexing
        #  
        action_batch = action_batch.view(1, self.BATCH_SIZE)

        state_action_values = torch.gather(self.policy_net(state_batch, dest_batch), 1, action_batch)

        #get the next state values 
        #should be a list here...
        #will need to make modifications for the approach when i use batching instead of single values
        with torch.no_grad():
            next_state_values = self.target_net( next_state_batch, dest_batch).max(1).values

        #then get the values for next state actions using the reward  
        target_state_action_values = (next_state_values * self.GAMMA) + reward_batch

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

        #only go through with this if we are doing retroactive rewards
        if(self.rewardType == "retroactive"): 
            #first get the time it took for the packet to go through the network 
            packetDelay = packet.packetArriveTime - packet.packetSendTime
            #then, create and push the experience 
            try: 
                #use 1/packetDelay for the reward 
                self.fullExperienceMemory.push(*self.nonRewardMemory[packet.packetIndex], 1/packetDelay)
            except Exception as e: 
                raise Exception("Couldnt properly push memory")

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