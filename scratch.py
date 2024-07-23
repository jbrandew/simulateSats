        #if we are doing iterative action space restriction, 
        if(True): 
            #then, if we are using our policy 
            if sample > eps_threshold:
                with torch.no_grad():
                    #first, get the policy net output
                    actionOutput = self.policy_net(overallState)

            #if we are not using our policy
            else: 
                #generate a random policy net output
                fullActionOutput = torch.rand(self.policy_net.n_actions)

            #then, get the argsort for the policy net output
            actionPreferenceList = torch.argsort(fullActionOutput, descending=True)

            #after getting the actionPreferenceList, then....convert the actionPreferenceList into 
            #satelliteInds to send to 
            indexableSats = sorted(self.satellite.connectedToPlayers)
            satPreferenceList = [indexableSats[idx] for idx in actionPreferenceList]
            satIndexPreferenceList = [sat.adjMatPersonalIndex for sat in satPreferenceList]

            #after getting the preference list, then get the nodes we cant send to 
            nodesToNotSendTo = packet.playersInvolvedInSending

            #then, get the first item in satPreferenceList that doesnt appear in nodesToNotSendTo
            satIndToForwardTo = next(item for item in satIndexPreferenceList if item not in nodesToNotSendTo)

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
            # #so, take the packet and for the path used
            # for satUsed in packet.playersInvolvedInSending:
            #     #get the process time 
            #     processTime = satUsed.generateProcessingOnePacketTime()
            #     #then, add it to the rows and cols of adj mat
            #     ind = satUsed.adjMatPersonalIndex()

            #     #create copy of initial state 
            #     self.finalGlobalState = copy.deepcopy(self.initialGlobalState)
            #     #then, add it in to respective values 
            #     self.finalGlobalState[ind]+=processTime/2
            #     self.finalGlobalState[:,ind]+=processTime/2
            #     self.finalGlobalState[ind,ind]-=processTime/2
                
            #     return

if(ind == 1 and False): 

                        #analyze the difference in routing table 
                        QLengths = np.maximum(player.QFinishTimes, player.currTime)
                        QLengths = QLengths - player.currTime

                        #get traffic aware table
                        trafficAwareRoutingTable = myMath.dijkstraWithNodeValuesAllInitialHops(player.adjMatrix, player.adjMatPersonalIndex, QLengths)

                        #get non traffic aware routing table
                        nonTrafficAwareRoutingTable = myMath.dijkstraWithNodeValuesAllInitialHops(player.adjMatrix, player.adjMatPersonalIndex)

                 

        # if(routingPolicy == 'OSPF'): 
        #     for broadcastAdjMatInd in range(200): 
        #         #so create time and events 
        #         updateTime = fullyFlushedNetworkETA*broadcastAdjMatInd/100
        #         queueEvent = Event(updateTime,
        #                             "updateAdjMats",
        #                             {})
        #         eventQueue.push(queueEvent)

        # if(routingPolicy == 'mixedSingleAgentRLRestOSPF'): 
        #     for broadcastAdjMatInd in range(200): 
        #         #so create time and events 
        #         updateTime = fullyFlushedNetworkETA*broadcastAdjMatInd/200
        #         queueEvent = Event(updateTime,
        #                             "updateAdjMats",
        #                             {})
        #         eventQueue.push(queueEvent)

        # if(routingPolicy == 'RL'): 
        #     for broadcastAdjMatInd in range(200): 
        #         #so create time and events 
        #         updateTime = fullyFlushedNetworkETA*broadcastAdjMatInd/200
        #         queueEvent = Event(updateTime,
        #                             "updateAdjMats",
        #                             {})
        #         eventQueue.push(queueEvent)
            # print(self.packet.packetSendTime)
            # print(self.packetDelay)
            # print(self.packet.routingMetadata)
            # pdb.set_trace()
        
    def forwardBad(self, 
                childObservations, 
                globalState, 
                batchSize):
        
        """
        Full pass through our networks

        Inputs:
        childObservations: states that the children see of the world around
        globalState: state of the world, as the central node sees it
        batchSize: batching size of the data :D

        Output: 
        joint action value 
        """

        # Pass input through each sub-network and collect outputs
        subnet_outputs = [subnet(childObservations) for subnet in self.subnets]
        
        #get chosen action values
        chosen_action_values = [torch.max(subOutput) for subOutput in subnet_outputs]

        #create qMixInput
        # Then, create dictionary for input to central node 
        qMixInput = TensorDict({
            "agents": TensorDict({
                "chosen_action_value": chosen_action_values
            }),
            "state": globalState
        }, [batchSize])

        #after creating dictionary, pass through qMix and get output
        with torch.no_grad(): 
            qMixOutput = self.qMix(qMixInput)['chosen_action_value'].unsqueeze(1)
        
        return qMixOutput

    def getUniqueHopsBad(self, jointExperienceToFormat):
        #we want the structure of: [0] = agentInds. [1] = currStates. [2] = nextStates. [3] = propDelay. 
        
        #create storage for reformatted experiences 
        reformatted = [[] for _ in range(5)]
        
        #for each agent index 
        for subAgentExperienceInd, agentInd in enumerate(jointExperienceToFormat[0]):
            
            #if we have not seen the agent before in this path experience:
            if len(reformatted) == 0 or agentInd not in reformatted[0]: 
                #then, append necessary data 
                reformatted[0].append(agentInd)
                reformatted[1].append(jointExperienceToFormat[1][subAgentExperienceInd])
                reformatted[2].append(jointExperienceToFormat[2][subAgentExperienceInd])

        #then, append the global state and propagation delay
        reformatted[3].append(jointExperienceToFormat[-2])
        reformatted[4].append(jointExperienceToFormat[-1])

        return reformatted



            globalState = [subExperience.finalGlobalState for subExperience in experienceBatch]

            globalState = [subExperience.initialGlobalState for subExperience in experienceBatch]

def selectiveChildForward(self, 
                              childAgentInds, 
                              childObservations,
                              globalState, 
                              batchSize,
                              withGrad = True): 
     

jointActionValueBatch = [self.jointActionValueNetwork.selectiveChildForward(subExperience.agentInds,
                                                                                subExperience.agentCurrStates,
                                                                                subExperience.initialGlobalState,
                                                                                self.BATCH_SIZE) for subExperience in experience]

        # jointActionValue = self.jointActionValueNetwork.selectiveChildForward(experience.agentInds,
        #                                                                         experience.agentCurrStates,
        #                                                                         experience.initialGlobalState,
        #                                                                         self.BATCH_SIZE)

        jointActionValueNextBatch = [self.jointActionValueNetwork.selectiveChildForward(subExperience.agentInds,
                                                                                subExperience.agentPredictedStates,
                                                                                subExperience.initialGlobalState,
                                                                                self.BATCH_SIZE) for subExperience in experience]

        # #get joint action value using next state. dont store gradients when doing it
        # jointActionValueNext = self.jointActionValueNetwork.selectiveChildForward(experience.agentInds,
        #                                                                             experience.agentPredictedStates,
        #                                                                             experience.finalGlobalState,
        #                                                                             1,
        #                                                                             self.BATCH_SIZE)

    def getUniqueHopsDeprecated(self,experience):

        #duplicate prop delay for easier processing
        experience[-1] = [experience[-1]]*len(experience[0])

        # Extract unique entries based on the first element of each sublist
        unique_entries = []
        seen_entries = set()

        #iterate through list of entries
        for ind, entry in enumerate(experience[0]):
            key = entry  # Convert list to tuple to use as a hashable key
            if key not in seen_entries:
                unique_entries.append([item[ind] for item in experience])
                seen_entries.add(key)

        reformatted = []

        #reformat
        for itemInd in range(len(experience)):
            reformatted.append( [item[itemInd] for item in unique_entries] )

        return reformatted

    def optimizeMultiAgentDeprecated(self): 
        """
        Here, we optimize in our multi agent case, in which we will need to selectively backpropagate, etc. 
            
        """

        #first, get experience 
        #please note, this does not remove it from the buffer 
        #only using one sample, so not batching currently
        experience = random.sample(self.memory.completedMemory, 1)[0]

        #reformat experience to only take unique hops
        experience = self.getUniqueHops(experience)

        #create storage for action values 
        #this is based on the base structure of the child network outputs
        fullActionValues = copy.deepcopy(self.jointActionValueNetwork.childOutputStructure)
       
        #put in known values, only for agents that are actually used 
        #        fullActionValues[experience[0]] = experience[1]
        for index, value in zip(experience[0], experience[1]):
            fullActionValues[index] = value

        #create storage for next action values 
        #this is based on the base structure of the child network outputs
        fullActionValuesNext = copy.deepcopy(self.jointActionValueNetwork.childOutputStructure)

        #put in known values, only for agents that are actually used 
        #fullActionValuesNext[experience[0]] = experience[2]
        for index, value in zip(experience[0], experience[2]):
            fullActionValuesNext[index] = value

        #get parent output for joint value 
        jointValue = self.jointActionValueNetwork.parentForward(fullActionValues)

        #get parent output for joint value for next  
        jointValueNext = self.jointActionValueNetwork.parentForward(fullActionValuesNext)
        
        #then, get the target. use the prop delay of the experience 
        jointValueTarget = jointValueNext + self.discountFactor*experience[3][0]

        #then, compute the loss based on this joint target
        loss = self.criterion(jointValue, jointValueTarget)

        #zero out gradients
        self.optimizer.zero_grad() 

        #then, go backwards
        #please note, this only generates gradients for the agents that actually computed values 
        loss.backward() 

        #then, have the optimizer step in the next direction 
        self.optimizer.step() 

        #store stats for loss and  
        self.epsLoss = self.epsLoss + [np.average(loss.detach().numpy())]
        self.epsRewards = self.epsRewards + []


        #yeet and leave :D 
        return 


#what do we seek to do here? 
#store gradient for final layer, and then use that to backprop against multiple layers 
#to do that, need a basic class for testing: 
class BigSoloNet(nn.Module):
    def __init__(self):
        super(BigSoloNet, self).__init__()
        # Define layers based on the specified sizes
        self.fc1 = nn.Linear(4, 10)   # First fully connected layer: 4 inputs -> 10 outputs
        self.fc2 = nn.Linear(10, 4)   # Second fully connected layer: 10 inputs -> 4 outputs
        self.fc3 = nn.Linear(4, 10)   # Third fully connected layer: 4 inputs -> 10 outputs
        self.fc4 = nn.Linear(10, 1)   # Fourth fully connected layer: 10 inputs -> 1 output

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation to the second layer
        x = F.relu(self.fc3(x))  # Apply ReLU activation to the third layer
        x = self.fc4(x)          # Output of the fourth layer
        return x

        # #format the network input data 
        # dont need to do this with modified state now. 
        # overallState = np.concatenate([adjMatrixState, queueLengthState])
        # overallState = [float(i) for i in overallState]
        # overallState = torch.tensor(overallState) 


# class PacketState: 

#     """
#     Describes what the packet is actually doing 
#     """
#     def __init__(self, initState): 
#         self.allowedValues = {'Dormant', 
#                                 'InTransmit', 
#                                 'InProcessQueue',
#                                 'InTransmitQueue',
#                                 'Finished'}
#         if initState not in self.allowedValues: 
#             raise ValueError("Bad Packet State Value") 
# class Packet(): 
#     """
#     This class is separate functionally from the "Player" set, but 
#     this seemed like the best place to put it. It really just 
#     represents the lifecycle of a packet. 

#     """
#     def __init__(self, **kwargs): 
#         """
#         Initialization of packet

#         Inputs: 

#         startLocation: xyz of the packet for when it is sent out 
#         endLocation: xyz of location for packet to arrive at 
#         scheduledAwakeTime: when the packet is supposed to be sent/start its path
#         pathToTake: the series of terminals it travels over to get to its 
#         destination. This is currently assigned when the packet reaches the time
#         of "awake," so its non adaptable  
#         """

#         for key, value in kwargs.items():
#             setattr(self, key, value)

#         #packet is dormant first 
#         self.currentPhase = PacketState('Dormant')

#         return 
    
#player class describes any operator on or above the earth 
#probably going to use rho and phi more often than purely longitude and latitude 
#nah lets just always use x,y,z...maybe easier in polar all the time...worry later   

def executeSimulation(self,
                          numPackets, 
                          numPacketsPerPerson = 1): 
        """
        Main function for executing simulation of a set # of packets. 
        
        Inputs: 
        numPackets: how many packet transmits we are simulating
        numPacketsPerPerson: how dense the packets are relative to the senders 

        Outputs: 
        relevant stats about data, including: 
        # hops per packet
        avg latency per packet
        overall time to send all 
        """
        
        #Phase 1: initialization
        #connect the needed topology: 
        self.manager.executeSimulation(numPackets,
                                       numPacketsPerPerson) 

        #get the locations of all packets that are sent out. 
        #myMath.generate_points_on_sphere_mostly_uniform(numPackets, )

        
        return  

def executeSimulation(self, 
                          numPeople,
                          numPacketsPerPerson,
                          simulationTime, 
                          timeStepLength,
                          processRateForSatellites): 
        
        """
        Main function for executing simulation of a set # of packets. 
        Please note, if you want to extend this simulation much further, the 
        initialization step could be broken up into chunks

        Not done...
        
        Inputs: 
        numPeople: how many users we are simulating
        numPacketsPerPerson: how dense the packets are relative to the senders 
        simulationTime: how long the simulation should run for 
        timeStepLength: how often are we checking for events 
        processRateForSatellites: how fast do satellites process packets (poisson)
        
        Outputs: 
        relevant stats about data, including: 
        # hops per packet
        avg latency per packet
        overall time to send all 
        """
        
        #Phase 1: initialization
        #connect the needed topology: 
        self.connect2ISL() 

        #setup the adjacency matrix based on given topology  
        self.manager.currAdjMat = self.manager.generateAdjacencyMatrix() 

        #create packet set to operate over 
        packetsToUse = self.manager.createPacketSet(numPeople,
                                            numPacketsPerPerson,
                                            simulationTime)

        #Phase 2: create event loop 
        #so for each time step 
        for timeStepInd in range(int(simulationTime/timeStepLength)): 
            #check for events that have occured  
            
            
            return 

        return 
    
    def executeChainSimulationNoCollision(self,
                               numPeople,
                               numPacketsPerPerson,
                               simulationTime):
        """
        Function for simulation of chained events. 
        
        Please note, this pre assigns a path upon the sending of a packet. 
        No dynamic routing is used/accounting for current state of servers. 

        TODO: Please note, this does not account for the case where the satellite 
        closest to end user and start user is the same satellite. 
        """

        #get satellite locations
        satLocs = self.manager.getSatLocations() 
        #reshape to cut out orbital plane data
        #ex. : [20,18,3] -> [360,3]
        satLocs = np.reshape(satLocs, [np.shape(satLocs)[0]*np.shape(satLocs)[1], np.shape(satLocs)[2]])
        raveledSats = np.ravel(self.sats)

        #generate adj mat 
        self.manager.generateAdjacencyMatrix()

        #first, create a priority queue for events  
        #create pQueue 
        eventQueue = PQueue()

        #then initialize a set of packets
        #so first, get locations of all the packets, spread
        #across the globe. xyz is in km btw. 
        startLocations = myMath.generate_points_on_sphere_mostly_uniform(numPeople,
                                                                         self.earthRadius)

        endLocations = myMath.generate_points_on_sphere_mostly_uniform(numPeople,
                                                                         self.earthRadius)
        
        #next, get random times for sending the packets out a
        packetSendTimes = np.random.uniform(0, 
                                            simulationTime, 
                                            (numPeople*numPacketsPerPerson,))
        
        packetArriveTimes = -1*np.ones(numPeople*numPacketsPerPerson)

        #then, add the "packetSent" events to the stack 
        for packetInd in range(len(packetSendTimes)): 
            #create arguments and queueEvent
            kargs = {"startLocation":startLocations[packetInd],
                    "endLocation":endLocations[packetInd],
                    "packetInd":packetInd}
            queueEvent = Event(packetSendTimes[packetInd],
                               "packetSent",
                               kargs)
            eventQueue.push(queueEvent)

        #can add the "update topology"/"update satellite position" here 
        
        #while we arent empty in the eventQueue
        while not eventQueue.is_empty(): 
        
            #get the next event 
            event = eventQueue.pop()

            #then, based on the event type change behavior.
            #if the eventType is "packetSent"  
            if event.eventType  == "packetSent":
                #first, get the satellite closest to start and end 
                #note, assuming you must use satellite for start and end 
                #(no terrestrial networks)  
                closestSatIndToStart, _ = myMath.closest_point(satLocs, 
                                                            event.kargs["startLocation"])
                 
                closestSatIndToEnd, _ = myMath.closest_point(satLocs,
                                                          event.kargs["endLocation"])
                #then, get the path: 
                #use adjacency matrix to get the path for each 
                
                if(routingPolicy == "basic"): 
                    pathToTake , _ = myMath.dijkstraWithPath(self.currAdjMat, 
                                                        closestSatIndToStart,
                                                        closestSatIndToEnd)
                    
                    if len(pathToTake) == 1 and closestSatIndToStart!=closestSatIndToEnd:
                        raise Exception("couldnt find path betweent start and end node")

                    #after we create a path, create an event type of "arriveAtConstellation"
                    #first, get the time of arriving at that first satellite 
                    #the indexing may possibly be wrong for raveled satellites
                    #but get the eventEndTime by accounting for initial propagation 
                    timeOfOccurence = event.timeOfOccurence + myMath.dist3d(event.kargs["startLocation"],  
                                                            raveledSats[closestSatIndToStart].getCoords())/(3e8)


                    event.kargs["pathToTake"] = pathToTake
                    event.kargs["lastSatInd"] = closestSatIndToEnd
                

                queueEvent = Event(timeOfOccurence, 
                                   "packetArriveAtConstellation",
                                   event.kargs)
                
                #then, add the new event on the pQ
                eventQueue.push(queueEvent)
            #if the event type is "packetArriveAtConstellation"
            if event.eventType == "packetArriveAtConstellation": 
                #need to first handle the case of the from and to satellite being the same 
                #so if we only have one in the path
                if(len(event.kargs["pathToTake"]) == 1): 
                    #then, get the time of occurence of landing at the dest 
                    timeOfOccurence = event.timeOfOccurence + myMath.dist3d(event.kargs["endLocation"],  
                                                        raveledSats[event.kargs["lastSatInd"]].getCoords())/(3e8)
                    #create and push accordingly 
                    queueEvent = Event(timeOfOccurence,
                                       "packetArriveAtDestination",
                                       event.kargs)
                    
                    eventQueue.push(queueEvent)

                    continue 

                #then, first, get the traversal time for the next link 
                fromPlayer = event.kargs["pathToTake"][0]
                toPlayer = event.kargs["pathToTake"][1] 

                propTime = self.currAdjMat[fromPlayer, toPlayer]
                
                #create corresponding time 
                timeOfOccurence = event.timeOfOccurence + propTime 

                #store the args 
                event.kargs["currentIndexInPath"] = 0
                
                #create event 
                queueEvent = Event(timeOfOccurence, 
                                   "packetArriveAtNextPlayer",
                                   event.kargs)
               
                #then, add the new event on the pQ
                eventQueue.push(queueEvent)

            #if our event type is arriving at next player,
            if event.eventType == "packetArriveAtNextPlayer": 

                #then, first check if we are at the end 
                if(event.kargs["currentIndexInPath"] is len(event.kargs["pathToTake"])): 
                    #if we are, then create end event. So, first get the last arrival time
                    timeOfOccurence = event.timeOfOccurence + myMath.dist3d(event.kargs["endLocation"],  
                                                        raveledSats[event.kargs["lastSatInd"]].getCoords())/(3e8)
                    
                    queueEvent = Event(timeOfOccurence,
                                       "packetArriveAtDestination",
                                       event.kargs)
                    
                    eventQueue.push(queueEvent)

                    continue 

                #if we are not at the end, then proceed to the next player 
                #so get the next arrival time 
                fromPlayer = event.kargs["pathToTake"][event.kargs["currentIndexInPath"]]
                toPlayer = event.kargs["pathToTake"][event.kargs["currentIndexInPath"]+1]
                timeOfOccurence = event.timeOfOccurence + self.currAdjMat[fromPlayer, toPlayer]
                
                #and create next packet 
                #store the args 
                event.kargs["currentIndexInPath"] = event.kargs["currentIndexInPath"]+1

                #create event 
                queueEvent = Event(timeOfOccurence, 
                                   "packetArriveAtNextPlayer",
                                   event.kargs)  

                #then, add the new event on the pQ
                eventQueue.push(queueEvent)     

            #if the packet is arriving at the destination
            if event.eventType == "packetArriveAtDestination": 
                #then, store the packet end time 
                #prrint(event.kargs["packetInd"]  )
                packetArriveTimes[event.kargs["packetInd"]] = event.timeOfOccurence

        #then, finally just return the difference between the two. 
        return packetArriveTimes - packetSendTimes

    def executeChainSimulationWithCollisions(self,
                               numPeople,
                               numPacketsPerPerson,
                               simulationTime,
                               packetCollisionEnabled = True):
        """
        simulationTime: how long we are sending packetsOverTheTime

        Function for simulation of chained events. 
        
        Please note, this pre assigns a path upon the sending of a packet. 
        No dynamic routing is used/accounting for current state of servers. 

        """

        #get satellite locations
        satLocs = self.manager.getSatLocations() 
        #reshape to cut out orbital plane data
        #ex. : [20,18,3] -> [360,3]
        satLocs = np.reshape(satLocs, [np.shape(satLocs)[0]*np.shape(satLocs)[1], np.shape(satLocs)[2]])
        raveledPlayers =  np.concatenate([np.ravel(self.sats), self.baseStations])

        #generate adj mat 
        self.manager.generateAdjacencyMatrix()

        #first, create a priority queue for events  
        #create pQueue 
        eventQueue = PQueue()

        #then initialize a set of packets
        #so first, get locations of all the packets, semi evenly spread
        #across the globe. xyz is in km btw. 
        startLocations = myMath.generate_points_on_sphere_mostly_uniform(numPeople,
                                                                         self.earthRadius)

        endLocations = myMath.generate_points_on_sphere_mostly_uniform(numPeople,
                                                                         self.earthRadius)
        
        #next, get random times for sending the packets out 
        packetSendTimes = np.random.uniform(0, 
                                            simulationTime, 
                                            (numPeople*numPacketsPerPerson,))
        
        packetArriveTimes = -1*np.ones(numPeople*numPacketsPerPerson)

        #create the packets 
        for personInd in range(numPeople): 
            for smallPacketInd in range(numPacketsPerPerson): 
                kargs = {"startLocation":startLocations[personInd],
                        "endLocation":endLocations[personInd],
                        "packetInd":personInd*numPacketsPerPerson + smallPacketInd}
                queueEvent = Event(packetSendTimes[personInd*numPacketsPerPerson + smallPacketInd],
                                "packetSent",
                                kargs)
                eventQueue.push(queueEvent)

        #while we arent empty in the eventQueue
        while not eventQueue.is_empty(): 

            #get the next event 
            event = eventQueue.pop()


            #then, based on the event type change behavior.
            #if the eventType is "packetSent"  
            if event.eventType  == "packetSent":
                #first, get the satellite closest to start and end 
                #note, assuming you must use satellite for start and end 
                #(no terrestrial networks)  
                closestSatIndToStart, _ = myMath.closest_point(satLocs, 
                                                            event.kargs["startLocation"])
                 
                closestSatIndToEnd, _ = myMath.closest_point(satLocs,
                                                          event.kargs["endLocation"])
                #then, get the path: 
                #use adjacency matrix to get the path for each 
                pathToTake , _ = myMath.dijkstraWithPath(self.currAdjMat, 
                                                     closestSatIndToStart,
                                                     closestSatIndToEnd)
                
                if len(pathToTake) == 1 and closestSatIndToStart!=closestSatIndToEnd:
                    raise Exception("couldnt find path betweent start and end node")

                #after we create a path, create an event type of "arriveAtConstellation"
                #first, get the time of arriving at that first satellite 
                #the indexing may possibly be wrong for raveled satellites
                #but get the eventEndTime by accounting for initial propagation 
                timeOfOccurence = event.timeOfOccurence + myMath.dist3d(event.kargs["startLocation"],  
                                                        raveledPlayers[closestSatIndToStart].getCoords())/(3e8)

                event.kargs["pathToTake"] = pathToTake
                event.kargs["lastSatInd"] = closestSatIndToEnd
                event.kargs["currentIndexInPath"] = 0 

                queueEvent = Event(timeOfOccurence, 
                                   "packetArriveAtNextPlayer",
                                   event.kargs)
                
                #then, add the new event on the pQ
                eventQueue.push(queueEvent)

            #if our event type is arriving at next player,
            if event.eventType == "packetArriveAtNextPlayer": 

                #if we are, then create end event. So, first get the process time
                #how to do this? first, generate how long it takes to process one packet
                satelliteIndWeAreAt = event.kargs["pathToTake"][event.kargs["currentIndexInPath"]]
                satelliteWeAreAt = raveledPlayers[satelliteIndWeAreAt]

                #get the finish processing time 
                endProcessTime = satelliteWeAreAt.generateProcessingOneMorePacketTime(event.timeOfOccurence, packetCollisionEnabled) 

                #then, create the time of occurence based on this process time
                timeOfOccurence = endProcessTime 

                #create event to queue
                queueEvent = Event(timeOfOccurence,
                                   "packetFinishProcessing",
                                   event.kargs)
                
                #push the event 
                eventQueue.push(queueEvent)

                continue 

            #if the packet is done waiting at queue of corresponding satellite 
            if event.eventType == "packetFinishProcessing":

                #if we only had to wait at one to begin with  
                #or if we are at the end of path 
                if(len(event.kargs["pathToTake"]) == 1 or event.kargs["currentIndexInPath"] is len(event.kargs["pathToTake"])-1): 
                    #then, get the time of occurence of landing at the dest 
                    timeOfOccurence = event.timeOfOccurence + myMath.dist3d(event.kargs["endLocation"],  
                                                        raveledPlayers[event.kargs["lastSatInd"]].getCoords())/(3e8)
                    
                    #then, store the data for when the final arrival of the packet happened 
                    packetArriveTimes[event.kargs["packetInd"]] = timeOfOccurence
                    
                    continue 
                
                #otherwise, first, get the traversal time for the next link 
                fromPlayer = event.kargs["pathToTake"][event.kargs["currentIndexInPath"]]
                toPlayer = event.kargs["pathToTake"][event.kargs["currentIndexInPath"]+1]
                propTime = self.currAdjMat[fromPlayer, toPlayer]
                
                #create corresponding time 
                timeOfOccurence = event.timeOfOccurence + propTime 

                #store the args 
                event.kargs["currentIndexInPath"] = event.kargs["currentIndexInPath"]+1
                
                #create event 
                queueEvent = Event(timeOfOccurence, 
                                   "packetArriveAtNextPlayer",
                                   event.kargs)
               
                #then, add the new event on the pQ
                eventQueue.push(queueEvent)


        #then, finally just return the difference between the two. 
 
        return packetArriveTimes - packetSendTimes



    def connectModifiedSpiralTopology(self): 
        """

        THIS DOESNT WORK, AS SHOWN BY GRAPHIC. i think the error has to do with mirrored
        effect... yea
        errors have to do with the fact that when we iterate through, we do both cases

        Similar topology to original spiral, but this time after connecting to 
        3 satellites, iterating through you "hop down" one link iteration 
        so that each spiral remains at roughly the same latitude 

        Effect: 
        Conect all sats in modified spiral topology 
        """
        #first connect in the basic 2ISL case 
        self.connect2ISL() 

        #then, connect to adjacent planes 
        #iterate through a certain plane, and then satellites 
        for smallSatInd in range(self.numSatPerPlane):  
            indJump = 0 

            for planeInd in range(self.numPlanes):   
                #uhhh...connect satellite to adjacent ones 

                if(indJump == 1 or indJump == 2): 
                    #for forward case:  
                    satOfPrevPlane = self.sats[(planeInd - 1 ) % self.numPlanes, smallSatInd]
                    self.sats[planeInd, smallSatInd].connectToSat(satOfPrevPlane, True)
                    
                    #for back case: 
                    satOfNextPlane = self.sats[(planeInd + 1) % self.numPlanes, smallSatInd]
                    self.sats[planeInd, smallSatInd].connectToSat(satOfNextPlane, True)  

                #in case where we are at the "bottom corner" of the spiral 
                if(indJump == 0): 
                    #for forward case:  
                    satOfPrevPlane = self.sats[(planeInd - 1 ) % self.numPlanes, (smallSatInd + 1) % self.numSatPerPlane]
                    self.sats[planeInd, smallSatInd].connectToSat(satOfPrevPlane, True)
                    
                    #for back case: 
                    satOfNextPlane = self.sats[(planeInd + 1) % self.numPlanes, smallSatInd]
                    self.sats[planeInd, smallSatInd].connectToSat(satOfNextPlane, True)  

                #in case where we are at the final sat of the spiral line 
                if(indJump == 3): 
                    #for forward case:  
                    satOfPrevPlane = self.sats[(planeInd - 1 ) % self.numPlanes, smallSatInd]
                    self.sats[planeInd, smallSatInd].connectToSat(satOfPrevPlane, True)
                    
                    #for back case: 
                    satOfNextPlane = self.sats[(planeInd + 1) % self.numPlanes, (smallSatInd - 1) % self.numSatPerPlane]
                    self.sats[planeInd, smallSatInd].connectToSat(satOfNextPlane, True)                

                #inc and reset appropriately 
                indJump+=1 
                indJump = indJump % 4 


    # def connectToSat(self, satToConnectTo, polarRegionRestriction = True): 
    #     """
    #     Just connect to a given satellite using your internals

    #     Inputs: 
    #     satToConnectTo: ... 
    #     polarRegionRestriction: do not allow connection if satellite to connect
    #     is above 70N or below 70S 

    #     Outputs: 
    #     boolean designating if we made the connection or not 
    #     """
    #     #if we have the restriction, enable it 
        
    #     if(polarRegionRestriction): 
    #         lats = myMath.cartesian_to_geodetic(*satToConnectTo.getCoords(), 6371) 
            
    #         if(lats[0] > 70 or lats[0] < -70): 
    #             return False 

    #         selfLats = myMath.cartesian_to_geodetic(*self.getCoords(), 6371) 
            
    #         if(selfLats[0] > 70 or selfLats[0] < -70): 
    #             return False 
            
        
    #     self.connectedSats = self.connectedSats + [satToConnectTo]
    #     return True 

    # def connectToBaseStation(self, baseStationToConnectTo): 
    #     """
    #     Just connect to a given basesattion using your internals

    #     baseStationToConnectTo: ... 
    #     """
    #     self.connectedBaseStations = self.connectedBaseStations + [baseStationToConnectTo]  


#simulate effect of satellite failure on path failure 
#rate = simmer.simulatePathFailureFast(25, 100, 100)
#print("Hello :D")
#print(rate)
#pdb.set_trace()


#this gives me the # rows that have a non inf value 
#non_inf_rows = np.sum(~np.isinf(adjMat).any(axis=1))

#print("Number of rows wit non-infinite values:", non_inf_rows)

#plotUsageAnalysis() 

# avgLength, avgTime, maxTime = simmer.simulateTransmits(1000)
# print("Avg # of links used:" + str(avgLength))
# print("Time in ms: "+ str(avgTime))
# print("Max Time: "+str(maxTime))




# numTrials = 1000
# #2 = num base stations
# setOftoAndFro = np.random.randint(0,numLEOs + 2, size =(numTrials, 2))
# totalTime = 0
# ind = 0 
# maxTime = 0
# totalLength = 0 
# for set in setOftoAndFro: 
    
#     ind+=1
#     time = myMath.dijkstra(adjMat, set[0], set[1])
#     totalTime+=time
#     print(ind)
#     print(time)
#     if(maxTime < time): 
#         maxTime = time

#     #totalLength+=len(path)
#     #print(totalTime)

# avgTime = totalTime/numTrials 
# print("Time in ms: "+ str(avgTime*1000))
# print("Max Time: "+str(maxTime))
#print("Average Length: "+str(totalLength/numTrials))


#pseudoCapacity = lemmeManage.getPseudoCapacity(1000)

# with open('pseudoCapacity.npy', 'rb') as f:

#     a = np.load(f)


# ///
#create manager 
# lemmeManage = Manager.Manager(constellationType="walkerDelta",
#                                 constellationConfig=constellationConfig,
#                                 baseStationLocations = configData['BaseStationLocations'], 
#                                 fieldOfViewAngle = configData['BaseStationViewAngle'], 
#                                 phasingParameter = configData['phasingParameter'],
#                                 sunExclusionAngle= configData['SunExclusionAngle'],
#                                 sunLocation=configData['SunLocation'])




        #store the xyz position of the packet upon inception 
        self.position = position 

        #store when you are supposed to awake
        self.scheduledAwakeTime = scheduledAwakeTime

        #currently, path is set upon initialization
        #TODO: make paths adaptable/assign them at the date of awaking
        #with the adaptable routing table 
        self.pathToTake = pathToTake

        #packet is dormant first 
        self.currentPhase = PacketState('Dormant')

                #if we are not at the end, then proceed to the next player  
                #so get the next arrival time 
                timeOfOccurence = event.timeOfOccurence + self.currAdjMat[event.kargs["currentIndexInPath"],
                                                                       event.kargs["currentIndexInPath"]+1]
                
                #and create next packet 
                #store the args 
                event.kargs["currentIndexInPath"] = event.kargs["currentIndexInPath"]+1

                #create event 
                queueEvent = Event(timeOfOccurence, 
                                   "packetArriveAtNextPlayer",
                                   event.kargs)  

                #then, add the new event on the pQ
                eventQueue.push(queueEvent)     

    def connectDisjointSpiralTopologyaoeuaoeuaoeu(self): 
        """
        DEPRECATED (use simple version instead)
        Similar to spiral topology implementation. However, we will
        instead use every other. 

        Effect: disjoint spiral connections
        """

        self.connect2ISL() 

        #then, connect to adjacent planes 
        #iterate through planes and then sats within a plane 
        for smallSatInd in range(self.numSatPerPlane):  
            for planeInd in range(self.numPlanes): 
                #uhhh...connect satellite to adjacent ones 
                #for forward case:  
                if(planeInd % 2 == 0): 
                    satOfPrevPlane = self.sats[(planeInd + 1 ) % self.numPlanes, smallSatInd]
                    self.sats[planeInd, smallSatInd].connectToPlayer(satOfPrevPlane, True, self.sunExclusionAngle, self.sunLocation)
                
                #for back case: 
                else: 
                    satOfNextPlane = self.sats[(planeInd - 1) % self.numPlanes, smallSatInd]
                    self.sats[planeInd, smallSatInd].connectToPlayer(satOfNextPlane, True, self.sunExclusionAngle, self.sunLocation)  

