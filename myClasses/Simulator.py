import math 
import myClasses.myPlots as myPlots
import myPackages.myMath as myMath 
import numpy as np 
import pdb 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq
from myClasses.Player import * 
from myClasses.Manager import * 
import time

import threading

class Simulator(): 
    """
    This class serves to just hold and execute different 
    simulation objectives/functions 

    Now functions as the controller in the model, view, controller architecture
    Provides the main interface with main execution 
    
    """
    def __init__(self, managerData): 
        
        #create manager 
        self.manager = Manager(**managerData)
        
        #create figure and plot to disploy data 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') 
        self.view = myPlots.GraphicsView(self.manager, fig, ax)

    # def executeChainSimulation(self,
    #                            numPeople,
    #                            numPacketsPerPerson,
    #                            simulationTime):
        
    #     self.manager.executeChainSimulation(numPeople,
    #                            numPacketsPerPerson,
    #                            simulationTime)

    def visualizeSimulation(self, 
                           kargs):
        """
        Wrapper for visualizing simulation in real time. 

        Inputs:
        kargs: refer to executeGeneralSimulation

        Outputs:
        times and visualized processing 
        """

        #create lock for multithreading 
        lock = threading.Lock()

        #then, append the lock information for executing the simulation 
        kargs["visualizerOn"] = lock

        #create threads for processing processing
        simulationThread = threading.Thread(target=self.executeGeneralSimulation, kwargs = kargs)

        # Start computation thread
        simulationThread.start()
    
        #have main thread be the visualization
        #self.showVisualization(lock)


    def plotCurrentState(self,
                         plotQueueSizes = False,
                         accessLock = None): 
        """
        Getting statistics to pass on to view function for graphing. 
        This has some computations, so it belongs in simulator, and not in myPlots

        

        Inputs: multithreading lock, as the main thread needs to access data that the computation thread may change. 
        Outputs: None
        Effect: information passed to view 
        """

        #get basic stats about environment
        satelliteLocations = self.manager.getSatLocations()
        baseStationLocations = self.manager.getBaseStationLocations()
        links, numLinks = self.manager.getXYZofLinks(6) 
        pointSizes = []

        #get the queue sizes 
        if(plotQueueSizes):
            #so, then compute the relative size of each queue
            queueFinishTimes = self.manager.queueFinishTimes
            relativeFinish = queueFinishTimes - self.currentTime

            #then, get the normalized factor 
            pointSizes = (relativeFinish - min(relativeFinish)) / (max(relativeFinish) - min(relativeFinish))

        self.view.multiplot(0,
                            satelliteLocations,
                            baseStationLocations,
                            links,
                            numLinks,
                            pointSizes)
        

    def showVisualization(self,
                          accessLock): 
        """
        Just visualiving current state using FuncAnimation
        """
        hold = FuncAnimation(self.view.fig, 
                             self.updateGraphicsMore, 
                             frames=100, 
                             interval=0.1)       
        plt.show() 

        return 



    def executeGeneralSimulation(self,
                                 initialTopology = "IPO",
                                 routingPolicy = None, 
                                 topologyPolicy = None, 
                                 
                                 numPeople = 100,
                                 numPacketsPerPerson = 1,
                                 simulationTime = .0001,                            

                                 queingDelaysEnabled = "False", 
                                 weatherEnabled = "False", 
                                 adjMatrixUpdateInterval = -100, 
                                 outageFrequency = None, 
                                 
                                 visualizerOn = None
                                 ): 
        
        """
        This method is the modularized version of executing a simulation. 

        Inputs: 
        initialTopology: how the satellites are initially connected. Possible answers include: IPO, etc. 
        routingPolicy: if i have a packet at a satellite, how do I as a manager (or satellite @ distributed) decide on how to route that packet that I have 
        topologyPolicy: if i have some observation of the current environment, how do I update the topology of my graph/satellite network? :D 

        numPeople: how many people requesting packets sent 
        numPacketsPerPerson: how many packets each person is requesting to transmit
        simulationTime: how long we are running the simulation for 

        queingDelaysEnabled: do I account for how long it takes to process packets at a server? Or am I just concerned with the propagation delay? 
        weatherEnabled: do i have stuff like....uh idk i need to implement this later. Might include like rainstorms or higher radiation from the sun 
        adjMatrixUpdateInterval: how often we update the postition of satellites in the constellation. Mostly used as you would think. 
        outageFrequency: how often we have satellites that break.     

        visualizerOn: if we want to visualize the state of the satellites queues and the links, then we get a lock for resources 

        Outputs: 
        deliveryTimes: how long it took to send each packet that we initially created. 

        """
        
        #this is the initialization section 

        #get satellite locations
        satLocs = self.manager.getSatLocations() 
        #reshape to cut out orbital plane data
        #ex. : [20,18,3] -> [360,3]
        satLocs = np.reshape(satLocs, [np.shape(satLocs)[0]*np.shape(satLocs)[1], np.shape(satLocs)[2]])
        raveledPlayers =  np.concatenate([np.ravel(self.manager.sats), self.manager.baseStations])

        #generate adj mat 
        self.manager.generateAdjacencyMatrix()

        #first, create a priority queue for events  
        #create pQueue 
        eventQueue = PriorityQueue()

        #then initialize a set of packets
        #so first, get locations of all the packets, semi evenly spread
        #across the globe. xyz is in km btw. 
        startLocations = myMath.generate_points_on_sphere_mostly_uniform(numPeople,
                                                                         self.manager.earthRadius)

        endLocations = myMath.generate_points_on_sphere_mostly_uniform(numPeople,
                                                                         self.manager.earthRadius)        

        #next, get random times for sending the packets out 
        packetSendTimes = np.random.uniform(0, 
                                            simulationTime, 
                                            (numPeople*numPacketsPerPerson,))
        
        #get storage for when the packets arrive 
        packetArriveTimes = -1*np.ones(numPeople*numPacketsPerPerson)

        #create the packet send events
        for personInd in range(numPeople): 
            for smallPacketInd in range(numPacketsPerPerson): 
                kargs = {"startLocation":startLocations[personInd],
                        "endLocation":endLocations[personInd],
                        "packetInd":personInd*numPacketsPerPerson + smallPacketInd}
                queueEvent = Event(packetSendTimes[personInd*numPacketsPerPerson + smallPacketInd],
                                "packetSent",
                                kargs)
                eventQueue.push(queueEvent)

        #first, create the reference time for when to update environment parameters 
        updateReferenceTime = 0 
        
        #while we arent empty in the eventQueue
        while not eventQueue.is_empty(): 

            #get the next event 
            event = eventQueue.pop()

            #coordinate current time
            self.currentTime = event.timeOfOccurence

            #if our time is outside this interval, then update correspondingly 
            if(event.timeOfOccurence > updateReferenceTime + adjMatrixUpdateInterval): 
                #this updates at least as often as necessary
                #this is because it updates when the time constraint is violated, and then updates to the timing that created the violation
                self.manager.updatePathData(updateReferenceTime, event.timeOfOccurence)
                updateReferenceTime = event.timeOfOccurence 

            #then, iterate through the event types
            if event.eventType  == "packetSent":
                #first, get the satellite closest to start and end 
                #note, assuming you must use satellite for start and end 
                #note, this is a non-dynamic path that doesnt account for the slight change in constellation within the path sending. If a change in topology occurs while packet is sent, youre toast. 
                #(no terrestrial networks)  
                closestSatIndToStart, _ = myMath.closest_point(satLocs, 
                                                            event.kargs["startLocation"])
                 
                closestSatIndToEnd, _ = myMath.closest_point(satLocs,
                                                          event.kargs["endLocation"])
                
                #then, get the waitTimes for each using our currentTime 
                #TODO: you really dont need to compute this every time, could approximate somehow 
                waitTimes = np.maximum(self.manager.queueFinishTimes - self.currentTime, 0)

                #use adjacency matrix to get the path for each 
                pathToTake , _ = myMath.dijkstraWithNodeValuesAndPath(self.manager.currAdjMat, 
                                                                      waitTimes,
                                                                      closestSatIndToStart,
                                                                      closestSatIndToEnd)

                # pathToTake , _ = myMath.dijkstraWithPath(self.manager.currAdjMat, 
                #                                          closestSatIndToStart,
                #                                          closestSatIndToEnd)
                               

                if len(pathToTake) == 1 and closestSatIndToStart!=closestSatIndToEnd:
                    pdb.set_trace()
                    raise Exception("couldnt find path between start and end node")

                #create event for arriving at next player. (so arriving at constellation)
                #first, get the time of arriving at that first satellite 
                #the indexing may possibly be wrong for raveled satellites
                #but get the eventEndTime by accounting for initial propagation 
                timeOfOccurence = event.timeOfOccurence + myMath.dist3d(event.kargs["startLocation"],  
                                                        raveledPlayers[closestSatIndToStart].getCoords())/(3e5)

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
                endProcessTime = satelliteWeAreAt.generateProcessingOneMorePacketTime(event.timeOfOccurence, queingDelaysEnabled) 

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
                                                        raveledPlayers[event.kargs["lastSatInd"]].getCoords())/(3e5)
                    
                    #then, store the data for when the final arrival of the packet happened 
                    packetArriveTimes[event.kargs["packetInd"]] = timeOfOccurence
                    
                    continue 
                
                #otherwise, first, get the traversal time for the next link 
                fromPlayer = event.kargs["pathToTake"][event.kargs["currentIndexInPath"]]
                toPlayer = event.kargs["pathToTake"][event.kargs["currentIndexInPath"]+1]
                propTime = self.manager.currAdjMat[fromPlayer, toPlayer]
                
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

    def update(self, 
               frame, 
               satTimePerFrame, 
               satPolicy = "Closest",
               baseStationPolicy = "inView"): 
        """
        update function for frame data 

        Inputs:
        frame: frame # we are on in the animation 
        satTimePerFrame: how much time within a frame that the satellite moves
        satPolicy: how to determine what connections to keep/change overtime
        baseStationPolicy: how to connect to players as a base station 

        Outputs: 

        Effect: 
        updated state and graphics

        """
        #update the position of each satellite 
        self.manager.updateConstellationPosition(satTimePerFrame) 
        #update the connections for each satellite 
        self.manager.updateTopology(satPolicy, baseStationPolicy)
        #plot current state 
        self.plotCurrentState() 


    def timeFrameSequencing(self, timeRatio, FPS, animationDuration): 
        """
        This code starts the time frame varying animation.

        Its debateable for this to be located within Simulator and not in view/plots,
        but as the animation also functionally holds processing as well, we are using
        it here. 
        
        Inputs: 
        timeRatio: ratio of time of animation to real time. Please note that currently an orbital period is ~6300 seconds, 
        so some ratio is needed( satTime/realTime)
        FPS: frames per second 
        animationDuration: how long you want animation to run (can stop early)

        Output: 
        time varying plot of satellites and corresponding ISLs over time 

        """

        #get intermediate variables 
        numFrames = int(FPS*animationDuration)
        realTimePerFrame = int(1000*animationDuration/numFrames)
        satTimePerFrame = realTimePerFrame*timeRatio

        #this "func animation" works with both updating positions and plotting 
        #each frame  

        #so, we pass in: the figure to use for updating, the function to call each frame/time, # frames / updates, time per frame, and the args within the function
        hold = FuncAnimation(self.view.fig, 
                      self.update, 
                      frames=numFrames, 
                      interval=realTimePerFrame, 
                      fargs = (satTimePerFrame,))
    
        plt.show() 

        return 

    def getSatsAndPlanesInViewOfBaseStation(self):
        """
        Just a function to get the number of orbital planes and num satellites in view
        of a base station. 

        Input: none
        Output: array of tuples of sats in view, planes in view, with length = # base stations

        """ 

        #create storage for num in view 
        numInView = [0]*len(self.manager.baseStations) 

        #create storage for all coordinates 
        allSatCoords  = self.manager.getSatLocations()   

        #then, for each base station 
        for ind, baseStation in enumerate(self.manager.baseStations): 
            numInView[ind] = baseStation.numSatellitesInView(allSatCoords)

        return numInView

    def simulatePathFailureFast(self, 
                                satFailuresPerGroup, 
                                numGroupsOfSatFailures, 
                                numPathsEvaluated): 
        """
        This function is incredibly slow due to the computational complexity (~500 000 shortest paths in a graph computed)
        It examines how many paths the failure of a group of satellites causes to fail. 
        Currently, it uses the floyd warshall algorithm, returning the shortest path from every node
        to every other node, and then for the paths that intersect a group of satellites we say "fail,"
        we say that path has failed. 

        Inputs: 
        satFailuresPerGroup: how many satellites fail in each group 
        numGroupsOfSatFailures: how many group of satellites failing we examine 
        numPathsEvaluated: how many pairs of nodes/paths we examine 

        Outputs: 
        overall path failure rate (paths failed/paths examined)

        TODO: I dont think at actually fully works currently. Something is possibly wrong with the 
        floyd warshall/reconstruct path functions, although FW was tested previously, so im not sure.
        Seems like we are running into inf loop somewhere. Was getting inf loop with -1 in adjmat

        Not getting correct results for at least IPO...
        -could check # of failures from path disconnects from actual structure or sat. failure.
        
        """

        #first, get our storage for groups of satellites that will fail 
        createStorageForGroups = [set()]*numGroupsOfSatFailures
        #then, iterate through and generate indices of satellites that will fail 
        for i in range(len(createStorageForGroups)): 
            createStorageForGroups[i] = np.random.choice(self.manager.numLEOs, size=satFailuresPerGroup)

        totalPathsFailed = 0 
        
        nextNodeMatrix = myMath.floyd_warshall(self.manager.currAdjMat)

        #now, for each group of sats to fail 
        for ind, group in enumerate(createStorageForGroups): 
            print(str(ind) + "/" + str(len(createStorageForGroups)))         

            #first, generate # pairs corresponding to the # of paths we examine 
            nodePairs = np.random.choice(self.manager.numLEOs, size=[numPathsEvaluated,2])
            
            #get the optimal paths for the node sets 
            optimalPaths = [0]*len(nodePairs)
            for ind, pair in enumerate(nodePairs): 
                optimalPaths[ind] = myMath.reconstruct_path(pair[0],pair[1],nextNodeMatrix)

            #get the set of satellites that fails 
            failedSatellites = createStorageForGroups[ind]

            #failedPaths = 0 
            #then, get the number of paths that have a satellite in the failed satellites
            #so, iterating through our optimal paths: 
            for optimalPath in optimalPaths:
                #check if we have any overlap: 
                totalPathsFailed+=bool(set(optimalPath) & set(failedSatellites)) or optimalPath==-1

        return totalPathsFailed

    #what are we doing? implementing the simulation type with path failures 
    def simulatePathFailureFromSatFailure(self, 
                                          satFailuresPerGroup, 
                                          numGroupsOfSatFailures, 
                                          numPathsEvaluated): 
        """
        This function examines the effect a satellite failure has path failures. 
        It does this by first: 
        1. generating "numGroupsOfSatFailures" groups of "satFailuresPerGroup" satellites that 
        we will effectively disable 
        2. for sat failure group, get "numPathsEvaluated" 
        sat pairs and their associated optimal paths in normal network 
        3. evaluate the "rate" or proportion of these paths that have a sat from the 
        failure group in them 
        4. average over all sat failure groups (so from steps 2 and 3) to get the average rate 

        Inputs: 
        satFailuresPerGroup: how many satellties fail in each trial 
        numGroupsOfSatFailures: how many trials to conduct 
        numPathsEvaluated: how many paths in each trial to examine to see path failure rate 
        
        """
        #first, flatten sats out for easier processing
        #flatSats = np.ravel(self.sats)

        #first, get our storage for groups of satellites that will fail 
        createStorageForGroups = [set()]*numGroupsOfSatFailures
        #then, iterate through and generate indices of satellites that will fail 
        for i in range(len(createStorageForGroups)): 
            createStorageForGroups[i] = np.random.choice(self.manager.numLEOs, size=satFailuresPerGroup)

        totalPathsFailed = 0 
        #now, for each group of sats to fail 
        
        for ind, group in enumerate(createStorageForGroups): 
            print(str(ind) + "/" + str(len(createStorageForGroups))) 
            #first, generate # pairs corresponding to the # of paths we examine 
            nodePairs = np.random.choice(self.manager.numLEOs, size=[numPathsEvaluated,2])
            
            #for each node pair: 
            for nodePair in nodePairs: 
                #get the optimal path 
                path, _ = myMath.dijkstraWithPath(self.manager.currAdjMat, nodePair[0],nodePair[1])
                #then, examine if that path contains a node from the group we are looking at 
                for nodeInPath in path: 
                    #only add once if you found one in the group tho
                    if(nodeInPath in group): 
                        totalPathsFailed+=1
                        break 

        #compute the avg rate of failure 
        #uhhh...this should work. it makes sense. 
        avgRateOfFailure = totalPathsFailed/(numPathsEvaluated*numGroupsOfSatFailures)
        return avgRateOfFailure


    def simulateTransmits(self, numTrials): 
        """
        For a given topology, computes the mean tx time, max tx time, and 
        mean # of transmits/links used for a given transmission (transmission
        being defined as a source and destination, and only looking at 
        propagation delay)
        
        Input: 
        numTrials: how many trials we are executing 

        Output: 
        avgLength: average number of links required for a transmission 
        avgTime: average time of going through path in seconds 
        maxTime: max time of going through any path in seconds   
        """

        #get adjacency matrix for satellites 
        #please note...hmmm...it should have already been init with that topology...
        #i think the manager should initialize the topology and base stations in its own init
        adjMat = self.manager.generateAdjacencyMatrix() 

        #size 2 as we have a destination and a start
        #we want to go between any sat and any BS, so sum those for the max ind 
        setOftoAndFro = np.random.randint(0, self.manager.numLEOs + len(self.manager.baseStations), size =(numTrials, 2))

        totalTime = 0
        totalLength = 0
        ind = 0 
        maxTime = 0
        totalLength = 0 
        for set in setOftoAndFro: 
            
            ind+=1
            path, time = myMath.dijkstraWithPath(adjMat, set[0], set[1])
            
            totalTime+=time
            totalLength+=len(path) 
            if(maxTime < time): 
                maxTime = time 

            print(ind)

        avgTime = totalTime/numTrials
        avgLength = totalLength/numTrials

        return avgLength, avgTime, maxTime

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
        eventQueue = PriorityQueue()

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
                                                        raveledSats[closestSatIndToStart].getCoords())/(3e5)


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
                                                        raveledSats[event.kargs["lastSatInd"]].getCoords())/(3e5)
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
                                                        raveledSats[event.kargs["lastSatInd"]].getCoords())/(3e5)
                    
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
                print(event.kargs["packetInd"]  )
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
        eventQueue = PriorityQueue()

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
                                                        raveledPlayers[closestSatIndToStart].getCoords())/(3e5)

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
                                                        raveledPlayers[event.kargs["lastSatInd"]].getCoords())/(3e5)
                    
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