#Hello :)
#this is the class for the central training network
#it deals with combining experiences and sending gradients to sub nodes for further training
#it has some restrictions, such as the mixing network currently being strictly monotonic for all directions


#make imports for the gym environment and plots 
import numpy as np 
 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

import time
import copy

#make imports for processing and models 
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb 

import math

import copy
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules.models.multiagent import QMixer

#transition consists of state, action, next state, reward
#just a tuple of information 
Transition = namedtuple('Experience',
                        ('agentInd', 'actionArray', 'nextActionArray', 'propDelay'))

#create replay object 
class CentralizedReplayMemory(object):

    #TOUSE
    class PacketTransmitExperience:
        def __init__(self,
                     initialGlobalState):
            """
            This 
             
            """
            
            #upon initial creation of an experience
            #initially numpy array...
            self.initialGlobalState = torch.from_numpy(initialGlobalState[initialGlobalState != np.inf]).float()

            #create storage for information that will be appended to over the packet transmission 
            self.agentInds = []  # List
            self.agentCurrStates = []  # List
            self.agentPredictedStates = []  # List

            #create storage for the information that will come from packet reaching final destination
            self.finalGlobalState = None  
            self.packetDelay = None 
            self.pathLength = None

        def appendAgentChoice(self, 
                              agentInd, 
                              agentCurrState,
                              agentPredictedState): 
            
            #first convert info that will be used as inputs to the network
            agentCurrState = agentCurrState.float()
            agentPredictedState = agentPredictedState.float() 

            #Store the relevant info
            self.agentInds.append(agentInd)
            self.agentCurrStates.append(agentCurrState)
            self.agentPredictedStates.append(agentPredictedState)

        def appendFinalData(self,
                            finalGlobalState,
                            packet):
            """
            This appends the final data to the packet experience, based on interactions with the packet. 

            There are two options. 
            1. use the final global state given to us
            2. use the 
            """


            #convert info that will be used as network inputs (or used for backpropagation)
            finalGlobalState = finalGlobalState.float()
            #packet delay was within numpy operations, so it defaults to float64. pytorch defaults to float32, so need
            #to convert 

            #get packet delay
            packetDelay = packet.packetArriveTime - packet.packetSendTime
            packetDelay = packetDelay.astype(np.float32)

            self.packet = packet
            self.finalGlobalState = finalGlobalState[finalGlobalState != np.inf]
            self.packetDelay = packetDelay

        def unique_indices(self, lst):
            unique_elements = set(lst)
            indices = {element: lst.index(element) for element in unique_elements}
            return [*indices.values()]
        
        def getUniqueHops(self):
            """
            Function to get the unique hops from a packets traversal through network.
            """

            #store the path length for myself 
            self.pathLength = len(self.agentInds)

            #if(self.pathLength < 2):
            #print(self.packet.startSat)
            #pdb.set_trace() 

            #first get the unique agents 
            indicesToUse = self.unique_indices(self.agentInds)

            #then, get a subset of each 
            self.agentInds = [self.agentInds[agentInd] for agentInd in indicesToUse]
            self.agentCurrStates = [self.agentCurrStates[agentInd] for agentInd in indicesToUse]
            self.agentPredictedStates = [self.agentPredictedStates[agentInd] for agentInd in indicesToUse]

    #create basic object, dequer for actual memory 
    #create dictionary for experiences that are in the process of being created 
    #might need a better custom data structure for this, as doing alot of appending 
    def __init__(self, capacity):
        """
        Initialize the completed memory and in progress memory.

        inProgressMemory are the experiences that have yet to be completed. 
        completedMemory are the experiences that are fully completed, outlining an entire path experience. 

        """

        self.completedMemory = deque([], maxlen=capacity)
        self.inProgressMemory = {}

    def sample(self, batch_size):
        """
        Just get a number of experiences equal to the batch_size 

        Inputs:
        batch_size: how many samples we are training on 

        Outputs: 
        A # of completed experiences  
        """

        return random.sample(self.completedMemory, batch_size)
    

    def startExperience(self,
                         packet,
                         initialGlobalState):
        """
        Create path transmit experience
        """

        self.inProgressMemory[packet.packetIndex] = self.PacketTransmitExperience(initialGlobalState)


    def assignPropDelay(self,
                        packet,
                        finalGlobalState): 
        """
        Assign a propagation delay to a packet transmission experience

        Please note, the path for this experience wont include the final destination satellite 
        
        Input:
        packet object
        """

        #get the path experience 
        finishedPathExperience = self.inProgressMemory.pop(packet.packetIndex)

        #then, format global state

        #for this version,  we are adding to the queues based on what nodes were actually used in sending 
        # for satInd in packet.playersInvolvedInSending: 
        #     finalGlobalState[satInd] +=.01/2
        #     finalGlobalState[:,satInd] +=.01/2
        #     finalGlobalState[satInd, satInd] -=.01/2

        #only take non-infinity values 
        finalGlobalState = torch.from_numpy(finalGlobalState[finalGlobalState != np.inf])

        #then, add the data
        finishedPathExperience.appendFinalData(finalGlobalState, packet)

        #then, add experience to completed memory 
        self.completedMemory.append(finishedPathExperience)

#        pdb.set_trace()

    def push(self, 
             agentInd, 
             packetInd, 
             currState, 
             predictedState):
        
        """
        Experience creation and modification for routing utilizing current buffers 
        
        Inputs:
        agentInd: index of the agent that has made this choice
        packetInd: what index does the packet have for this path?
        currState: the state of the environment when the agent made the decision
        predictedState: the state that the agent predicts the environment will be in when it takes an action 

        Outputs:

        """
        self.inProgressMemory[packetInd].appendAgentChoice(agentInd,
                                                           currState,
                                                           predictedState)
        
        
#create DQN network 
class JointActionValueNetwork(nn.Module):
    """
    Centralized training structure that utilizes the "module list" to pass gradients to sub networks.
    In the future, this should use "hypernetworks" to ensure we are optimizing in a stable/eventually convergent way. 
    """
    #create the initial DQN network
    #just 3 layers, with the input being the observations, and the actions being the output
    def __init__(self, 
                 sats, 
                 centralTrainingMethod = "VDN"):
        """
        Create network that combines data outputs from multiple sub agents. 

        sats: satellite objects we are working mith
        centralTrainingMethod: how we are training our agents to work together
        
        """

        #initialize network 
        super(JointActionValueNetwork, self).__init__()

        #store the agents 
        self.sats = sats

        #instantiate the central training algorithm
        self.centralTrainingMethod = centralTrainingMethod
        
        #then, have the first layer be a module list from the agent's policy networks 
        self.subnets = nn.ModuleList([self.sats[i].agent.policy_net for i in range(len(self.sats))]) 
        
        #TODO: make the state shape modular...

        #now, we make the data formatter and mixer based on the requested training method 
        #just match "state_shape" to the debug output shape....
        if(centralTrainingMethod == "QMix"):
            self.dataFormatter = self.QMixFormatter

            self.mixer = TensorDictModule(
                module=QMixer(
                    state_shape=(55,),
                    mixing_embed_dim=32,
                    n_agents=len(self.sats),
                    device="cpu",
                ),
                in_keys=[("agents", "chosen_action_value"), "state"],
                out_keys=["chosen_action_value"],
            )

        if(centralTrainingMethod == "VDN"): 

            self.dataFormatter = self.VDNFormatter

            self.mixer = self.VDNMixer

    def resetData(self):
        #for each subAgent  
        for subAgent in [self.sats[i].agent for i in range(len(self.sats))]: 
            #reset the respective data/statistics 
            subAgent.resetData() 
            
            #print(subAgent.getEpsThreshold())
        

    def VDNFormatter(self,
                     actionVals,
                     globalState,
                     batchSize): 
        
        #sum along the dimension that corresponds to agents 
        return actionVals

    def QMixFormatter(self,
                      actionVals,
                      globalState,
                      batchSize): 

        qMixInput = TensorDict({
            "agents": TensorDict({
                "chosen_action_value": actionVals
            }),
            "state": torch.tensor(globalState)
        }, [batchSize])

        return qMixInput

    def VDNMixer(self,
                 vals): 
        return torch.sum(vals, dim = 1)
    
    def parentForward(self, subnet_outputs, globalState, batchSize):
        """
        Just get output with respect to parent network 
        ...
        """
        
        return 

    def childForward(self, x): 
        """
        Just get output with respect to children networks 
        """

        subnet_outputs = [subnet(x) for subnet in self.subnets]
        return subnet_outputs
    
    def selectiveChildForward(self, 
                              experienceBatch, 
                              batchSize,
                              timeReference,
                              withGrad): 
        """
        Full pass through our networks. Properly formats based on batchSize. 
        1. get the action values from the actions we chose
        2. format and get the corresponding state 
        3. send inputs to mixers 

        Inputs:
        experienceBatch: batch of experiences
        batchSize: how many experiences
        timeReference: is this propagating with respect to initial or post experience 
        withGrad: are creating gradients for this? 

        Output: 
        joint action value 

        """

        #create storage for subnet output
        chosen_action_values = torch.zeros(batchSize, len(self.sats))

        #iterate through the experiences in the batch 
        for subExperienceInd in range(batchSize): 

            # Pass input through each sub-network and collect outputs
            for agentInd, currObs, futureObs in zip(experienceBatch[subExperienceInd].agentInds, 
                                                    experienceBatch[subExperienceInd].agentCurrStates,
                                                    experienceBatch[subExperienceInd].agentPredictedStates):

                if(timeReference == "initial"): 
                    #store the max value of the considered actions (so the chosen actions) 
                    #forward propagate through child network
                    chosen_action_values[subExperienceInd, agentInd] = torch.max(self.subnets[agentInd](currObs))

                elif(timeReference == "final"): 
                    #store the max value of future action 
                    #forward propagate through child network
                    chosen_action_values[subExperienceInd, agentInd] = torch.max(self.subnets[agentInd](futureObs))

                else: 
                    raise Exception("Bad Time ref value")
                
            #add dimensionality, first for batch, and second for action input 
            #chosen_action_values = chosen_action_values.unsqueeze(0)
        
        chosen_action_values = chosen_action_values.unsqueeze(2)
        
        #set up storage for global state 
        globalState = torch.zeros(batchSize, len(experienceBatch[0].initialGlobalState))


        #there is a notion of how do we define "current/initial state" and "next state"
        #for a central trainer

        #currently, current state = the state of the network when the packet is first sent from the ground 
        #next state = the first next state of the initial agent....this characterization is debateable 

        #TODO: explore possibly just adding wrt the path utilized through the adjacency matrix 

        #store the global state, depending on the type of prop we are doing 
        if(timeReference == "initial"): 
            for ind, subExperience in enumerate(experienceBatch): 
                globalState[ind] = subExperience.initialGlobalState
             
        elif(timeReference == "final"): 
            for ind, subExperience in enumerate(experienceBatch): 
                
                #globalState[ind] = subExperience.finalGlobalState

                globalState[ind] = subExperience.agentPredictedStates[0]

                #stacked_tensor = torch.stack(subExperience.agentPredictedStates)
                #globalState[ind] = torch.max(stacked_tensor, dim=0)[0]

        else: 
            raise Exception("Bad Time ref value")

        #create qMixInput
        # Then, create dictionary for input to central node 
        mixerInput = self.dataFormatter(chosen_action_values,
                                        globalState,
                                        batchSize)

        #after creating dictionary, pass through qMix and get output
        #with torch.no_grad(): 
        if(withGrad): 
            if(self.centralTrainingMethod == "QMix"): 
                mixerOutput = self.mixer(mixerInput)['chosen_action_value']
            else: 
                mixerOutput = self.mixer(mixerInput)
        
        else: 
            with torch.no_grad(): 
                if(self.centralTrainingMethod == "QMix"): 
                    mixerOutput = self.mixer(mixerInput)['chosen_action_value']
                else:
                    mixerOutput = self.mixer(mixerInput)

        return mixerOutput
    
class QMixerAgent(nn.Module): 
    """
    This class is used for centralized training of sub agents 
    """

    def __init__(self):
        """
        Here we just set up hyperparameters and personal variables 
        """

        #initialize network 
        super(QMixerAgent, self).__init__()

        #setup hyperparameters for training 
        self.BATCH_SIZE = 64

        #self.GAMMA = 0.99
        #self.EPS_START = 0.9
        #self.EPS_END = 0.05
        #lower "decay" value actually increases rate we go to the "eps_end" value. might be total # steps required to get to min
        ##need a very high amount for MARL as the # total experiences per agent in the total # episodes may go down
        #self.EPS_DECAY = 100000
        #used to be .005
        #make Tau = 1 to disable target network concept 
        #tau not used with this stuff 
        #self.TAU = 0.1
        
        self.LR = 1e-3
        self.discountFactor = 0.95

        #past rewards
        self.epsRewards = [] 
        self.epsLoss = []
        self.epsPathLengths = [] 

        self.epsPerformance = [] 

    def resetData(self, displayData):
        
        if(displayData):
            print("Eps")
            
            self.episodeAnalysis()  

        #then reset my episode rewards and joint network  
        self.epsRewards = []
        self.epsLoss = []
        self.jointActionValueNetwork.resetData()

    
    def episodeAnalysis(self): 
        
        #print the past average reward and loss  
        print("Average Reward")
        print(np.average(self.epsRewards))

        print("Average Loss")
        print(np.average(self.epsLoss))
        
        print("Average Path Length")
        print(np.average(self.epsPathLengths))

    def assignPropDelay(self, packet, finalGlobalState):
        #after pushing to the memory, optimize over all agents
        self.memory.assignPropDelay(packet, finalGlobalState)

        #optimize over batches of size 10 
        if(len(self.memory.completedMemory) >= self.BATCH_SIZE):
            self.optimizeMultiAgent()

    def startExperience(self, packet, initialGlobalState): 
        self.memory.startExperience(packet, initialGlobalState)

    def initializeNetworks(self, sats):

        self.sats = sats

        #create experience buffer pair 
        #max size of buffer is batch size for now (just use fresh data)
        self.memory = CentralizedReplayMemory(self.BATCH_SIZE) 

        #create network to evaluate the joint action set's value  
        self.jointActionValueNetwork = JointActionValueNetwork(self.sats)        
        
        # Define loss function and optimizer
        #these are default
        #paper uses MSE 
        self.criterion = nn.MSELoss()
        #self.optimizer = optim.SGD(self.jointActionValueNetwork.parameters(), lr=0.001)
        
        #change optimizer based on online 
        self.optimizer = optim.Adam(self.jointActionValueNetwork.parameters(), self.LR)
                
    def storeExperience(self, args): 
        """
        Store sats' index, 
        packet index
        value output for overall state
        value output for predicted state
        """
        
        self.memory.push(*args)

    def optimizeMultiAgent(self):
        """
        Optimize using the QMix network. Doesnt need any inputs because it pulls from the memory.
        Please note, this selectively backpropagates with respect to the agents that were used to generate a path. 

        """
        #first, get an experience batch to use 
        experienceBatch = random.sample(self.memory.completedMemory, self.BATCH_SIZE)

        #then, get the unique hops for each sub experience 
        # & remove the experiences you trained on (so only train on data once)
        #I read that multi agent needs very fresh data 
        for subExperience in experienceBatch:

            subExperience.getUniqueHops()
            self.memory.completedMemory.remove(subExperience)
        
        jointActionValue = self.jointActionValueNetwork.selectiveChildForward(experienceBatch, 
                                                                              self.BATCH_SIZE,
                                                                              "initial",
                                                                              True)

        #get joint action value using next state. dont store gradients when doing it
        jointActionValueNext = self.jointActionValueNetwork.selectiveChildForward(experienceBatch, 
                                                                                  self.BATCH_SIZE,
                                                                                  "final",
                                                                                  False)

        #so get the packet delay, and then add a dimension 
        batchPacketDelay = torch.tensor([subExperience.packetDelay for subExperience in experienceBatch]).unsqueeze(1)

        #get the experience path length
        batchPathLength = torch.tensor([subExperience.pathLength for subExperience in experienceBatch]).unsqueeze(1)

   
        #create the target from jointActionValue, using the propagation delay and discount factor 
        #please note, that reward here = 1/packetDelay for the experience 
        #also, possibly normalizing reward based on previous delay 
        
        #normalize reward based on sliding window
        #so, if we have more than 5

        # if( len(self.epsPerformance) > 5): 
        #     #then, first get the average across batches 
        #     batchAvg = torch.mean(batchPacketDelay[-5:], dim = 1, keepdim= True)
            
        #     #then, get reward based on this 
        #     jointActionValueTarget = self.discountFactor * jointActionValueNext +  batchAvg / batchPacketDelay 

        # else: 

            #if we dont have enough batches, do it normally 
            
        jointActionValueTarget = self.discountFactor * jointActionValueNext + (self.discountFactor/batchPacketDelay)

        #so, then store into performance 
        self.epsPerformance.append(batchPacketDelay)

        #then, compute the loss based on this joint target
        #TODO: consider other loss functions 
        loss = self.criterion(jointActionValue, jointActionValueTarget)

        #remove gradients 
        self.optimizer.zero_grad()

        #go backwards on loss 
        loss.backward() 

        #then, clip the gradients to prevent exploding gradients 
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.01)  # Adjust max_norm as needed

        #then, have the optimizer step in the next direction 
        self.optimizer.step() 
        
        #store stats for loss, reward, and average path length 
        self.epsLoss.append(np.average(loss.detach().numpy()))
        self.epsRewards.append(np.average(self.discountFactor/batchPacketDelay))
        self.epsPathLengths.append(np.average(batchPathLength.detach().numpy()))

































