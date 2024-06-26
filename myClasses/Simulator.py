#custom classes import 
import myClasses.myPlots as myPlots
import myPackages.myMath as myMath 
from myClasses.Player import * 
from myClasses.Manager import *
import myClasses.Packet as Packet

#processing classes 
import math 
import numpy as np 
import heapq
import time
import copy 

#plotting classes 
import pdb 
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import tkinter

class Simulator(): 
    """
    This class serves to just hold and execute different 
    simulation objectives/functions 

    Now functions as the controller in the model, view, controller architecture
    Provides the main interface with main execution. 

    It also manages objects like packets, weather, etc. that are particular to the 
    
    """
    def __init__(self, managerData): 
        
        #create manager from config 
        self.manager = Manager(**managerData)
        
        #create figure and plot to disploy data 
        self.fig = plt.figure(figsize = (10, 7),facecolor= 'coral')
        self.ax = self.fig.add_subplot(111, projection='3d') 
        self.view = myPlots.GraphicsView(self.manager, self.fig, self.ax)

    class SimulationSnapshot(): 
        """
        This class just gives a snapshot of the environment we are in.
        Used for visualization purposes later on. 
        Probably will be used for RL training down the line as well. 
        """

        def __init__(self, manager, view, currentTime):
            #first store the manager and the view 
            self.manager = manager
            self.view = view 
            self.currentTime = currentTime
            #then, store the initial state of the data 
            self.snapshot()  
            return 
        
        def snapshot(self): 
            """
            Just take a snapshot of the manager state. 
            So, any info taken here will be used for visualization purposes

            Do slight amount of processing so we can visualize queue lengths.

            This method is used with both the visualizer, and likely the RL method later on to save experiences

            """

            #first, get a deep copy of the entire state of our simulation 
            self.satelliteLocations = copy.deepcopy(self.manager.getSatLocations())
            self.baseStationLocations = copy.deepcopy(self.manager.getBaseStationLocations())
            links, numLinks = self.manager.getXYZofLinks(6) 
            self.links = copy.deepcopy(links)
            self.numLinks = copy.deepcopy(numLinks) 
            self.queueFinishTimes = copy.deepcopy(self.manager.queueFinishTimes[0:self.manager.numLEOs])

            #here, we are doing a slight amount of processing for better visualization.
            #please keep in mind that if the currentTime > queueFinish time, then the queue is done 
            self.relativeFinish = np.maximum(self.queueFinishTimes - self.currentTime, 0)

            #if we are in the initial state/no packets in there, then just give normal 
            if( max(self.relativeFinish) == min(self.relativeFinish) ):
                self.sphereColors = np.ones(self.manager.numLEOs)
                #self.sphereColors = 10*np.ones(self.manager.numLEOs)
            else: 

                #normalization doesnt really matter cause color map plots regardless, only important part is inverting so that its properly plotted (dimmer color => less traffic to be serviced)
                #but need mapping to a space of [0,1]
                self.sphereColors = ((self.relativeFinish - min(self.relativeFinish)) / (max(self.relativeFinish) - min(self.relativeFinish)))
            
            print("Hello") 
            print(self.relativeFinish[np.nonzero(self.relativeFinish)])
            
        
        def selfPlot(self): 
            self.view.multiplot(0,
                    self.satelliteLocations,
                    self.baseStationLocations,
                    self.links,
                    self.numLinks,
                    self.sphereColors)                     
                 
    def plotMySnapshotSet(self):
        """
        This function uses the existing set of snapshots to provide visualization of the average queue length over time
        
        Inputs: 
        none
        
        Outputs: 
        matplotlib plot of average queue length over time 

        """

        #also, get the average.....hm.....
        #get the max queue length average 
        # total = 0 
        # for snapshotInd, snapshot in enumerate(self.snapshotStorage): 
        #    #maxed = max(np.average(snapshot.relativeFinish), maxed) 
        #     total+= np.average(snapshot.relativeFinish)
        # print("Average")
        # print(total/snapshotInd) 

        #create storage for average queue length 
        averageQueueLength = np.ones(len(self.snapshotStorage)) 
        #create storage for corresponding time 
        timeOfSnapshots = np.ones(len(self.snapshotStorage))

        #iterating through our snapshots
        for snapshotInd, snapshot in enumerate(self.snapshotStorage): 
            #for each, store the average queue length 
            averageQueueLength[snapshotInd] = np.average(snapshot.relativeFinish)
            #store the time associated with the snapshot ind 
            timeOfSnapshots[snapshotInd] = np.average(snapshot.currentTime)
        
        #then, plot the data: 
        myPlots.plotXY(timeOfSnapshots, averageQueueLength, "Time Point in Simulation (seconds)", "Average Queue Length (seconds)")

        return 

    def simulateWithVisualizer(self, 
                               simulationArgs,
                               visualizerArgs):
        """
        function to simulate and visualize the results

        simulationArgs: refer to executeGeneralSimulation
        visualizerArgs: refer to showVisualization
        """
         
        #if we are going to visualize, we need snapshots within the event stack 
        simulationArgs["takeSnapshots"] = visualizerArgs["visualizerOn"]

        #if we are enabling a visualizer 
        if(visualizerArgs["visualizerOn"]): 
        
            #first get how much in simulation time the frames should be spaced between each other
            #so get the period that we are sending packets over 
            packetSendTimeFrame = simulationArgs["packetSendTimeFrame"]
        
            #then get how many frames we have 
            numFrames = visualizerArgs["visualizeTime"] * visualizerArgs["FPS"]

            #then, we have an "adjustment factor" to account for visualizing the environment while the packets are being processed/going through the network
            simulationTimeBetweenFrames = packetSendTimeFrame*simulationArgs["timeFactor"]/(numFrames)
            simulationArgs["simulationTimeBetweenSnapshots"] = simulationTimeBetweenFrames
            simulationArgs["numEnvironmentSnapshots"] = numFrames
            
            #create variable for expected time of fully flushed network
            simulationArgs["fullyFlushedNetworkETA"] = packetSendTimeFrame*simulationArgs["timeFactor"]
            simulationArgs["routingPolicy"] = simulationArgs["routingPolicy"]

            #set up storage for the snapshots by creating list, with one entry allotted for one frame  
            self.snapshotStorage = [0]*numFrames

        #after getting that, do the computation in the event stack
        holdData = self.executeGeneralSimulation(**simulationArgs)
    
        #have main thread be the visualization
        self.showVisualization(**visualizerArgs)


    def showVisualization(self,
                          visualizerOn,
                          visualizeTime,
                          FPS):
        """
        Function to plot the ending results of the simulation, using snapshots from the actual simulation execution. 

        Inputs: 
        visualizerOn: enabling the visualization process
        visualizeTime: how long we want to take to display results 
        FPS: frames per second 
        
        Outputs: 

        Effect: 
        Plots in 3D the traffic, position of satellites, etc. in time varying capacity 
        """ 

        #please note, that even tho it looks like "hold" isnt used, its auto erased from memory if not assigned. So keep it there. 
        #also note, this auto gives a "frame" variable to the plotSnapshotFromStorage function 
        #also note, interval is in milliseconds 
        hold = FuncAnimation(self.view.fig, 
                      self.plotSnapshotFromStorage, 
                      frames=FPS*visualizeTime, 
                      interval=1000/FPS,
                      repeat = False)       

        plt.show() 

    def plotCurrentState(self): 
        #first take snapshot 
        snapshot = self.SimulationSnapshot(self.manager, self.view, 0)
        #then plot it for the current time 
        snapshot.selfPlot()    
     
    def plotSnapshotFromStorage(self, frame): 
        print(frame)
        #the "frame" 
        #plot the aleadry stored snapshot 
        self.snapshotStorage[frame].selfPlot()
    
    def generatePackets(self, 
                        numPeople,
                        numPacketsPerPerson,
                        startLocationDistribution,
                        sendTimeDistribution, 
                        sendTimeFrameLength,
                        precomputePath): 
        """
        Generate a set of packet objects 
        
        Inputs: 
        numPacketsPerPerson: how many packets per person to create
        numPeople: how many people are sending packets 
        startLocationDistribution: how the origin locations of packets is distributed
        sendTimeDistribution: how the timing of the send time of packets is distributed
        sendTimeFrameLength: over what period of time to send packets over 
        precomputePath: do we compute the path for all the packets beforehand? This is part of the "basic routing policy" 
        
        Output: 
        packets: set of packet objects that have corresponding properties         
        """
        satLocs = self.manager.getSatLocations()
        satLocs = np.reshape(satLocs, [np.shape(satLocs)[0]*np.shape(satLocs)[1], np.shape(satLocs)[2]])

        packets = [0]*numPacketsPerPerson*numPeople
        satStarts = [0]*len(packets)
        satEnds = [0]*len(packets)
        routingMetaData = [dict() for x in range(len(packets))]

        if(startLocationDistribution == "PseudoUniform"):
            startLocations = myMath.generate_points_on_sphere_mostly_uniform(numPeople,
                                                                            self.manager.earthRadius)
            endLocations = myMath.generate_points_on_sphere_mostly_uniform(numPeople*numPacketsPerPerson,
                                                                            self.manager.earthRadius)        
        
        #distribute start and finish of packets to just use one person, and have it be the farthest distance possible
        elif(startLocationDistribution == "SingleFar"):
                                                                            
            startLocations = np.array([[0,0,6000000]])
            endLocations = startLocations*-1   

        elif(startLocationDistribution == "simpleForSingleRL"): 

            startLocations = np.array([[-2000,-2000,0]])
            endLocations = startLocations*-1

        if(sendTimeDistribution == "Uniform"):
            #next, get random times for sending the packets out 
            packetSendTimes = np.random.uniform(0, 
                                                sendTimeFrameLength, 
                                                (numPeople*numPacketsPerPerson,))

        #store the indices of the start and end satellites 
        for packetInd in range(len(packets)): 
            satStarts[packetInd], _ = myMath.closest_point(satLocs, 
                                                            startLocations[int(packetInd/numPacketsPerPerson)])
            satEnds[packetInd], _ = myMath.closest_point(satLocs,
                                                            endLocations[int(packetInd/numPacketsPerPerson)])        
        #if we are supposed to precompute the path 
        if(precomputePath):

            #then, for each packet, use the start and end satellites to get the path
            for packetInd in range(len(packets)):                 
                #use adjacency matrix to get the path for each 
                routingMetaData[packetInd]["path"] , _ = myMath.dijkstraWithPath(self.manager.currAdjMat, 
                                                                                 satStarts[packetInd],
                                                                                 satEnds[packetInd])
                
                #handle exception of not being able to find a path 
                if len(routingMetaData[packetInd]["path"]) == 1 and satStarts[packetInd]!=satEnds[packetInd]:
                    raise Exception("couldnt find path between start and end") 
        
        #then, initialize all the necessary packets 
        for packetInd in range(numPacketsPerPerson*numPeople): 
            packets[packetInd] = Packet.Packet(startLocations[int(packetInd/numPacketsPerPerson)],
                                               endLocations[int(packetInd/numPacketsPerPerson)],
                                               satStarts[packetInd],
                                               satEnds[packetInd],
                                               None,
                                               packetSendTimes[packetInd],
                                               routingMetaData[packetInd],
                                               None)
        
        return packets
        
    def executeGeneralSimulation(self,
                                 
                                 numPeople = 100,
                                 numPacketsPerPerson = 1,
                                 packetSendTimeFrame = .0001,                            
                                 personDistribution = "PseudoUniform", 

                                 queingDelaysEnabled = "False", 
                                 weatherEnabled = "False", 
                                 environmentUpdateInterval = None, 
                                 outageFrequency = None, 
                                 dynamicLocation = True, 

                                 takeSnapshots = False, 
                                 simulationTimeBetweenSnapshots = None, 
                                 numEnvironmentSnapshots = None,
                                 timeFactor = None,

                                 fullyFlushedNetworkETA = None,
                                 routingPolicy = None
                                 ): 
        
        """
        This method is the modularized version of executing a simulation. There are multiple phases to this. 

        Create Simulation environment: 
        -adds all people/transmitters to environment

        Initial stack: 
        -adds all initial events to the stack. This includes predefined events such as packet send times, enviroment update 
        times, taking snapshot times, updating routing tables etc. 
        -please note that the success of the visualization and general routing algorithm is highly dependent on snapshot resolution, 
        environment update resolution, etc. 

        Process stack: 
        -iteratively process the stack until empty. This program works by iteratively spawning new events until there are no more 
        to process. An example of this would be packet send -> packet arrive at constellation/first satellite-> packet finish processing -> 
        packet arrive at next satellite, and so on.... 

        Inputs: 
        initialTopology: how the satellites are initially connected. Possible answers include: IPO, etc. 
        routingPolicy: if i have a packet at a satellite, how do I as a manager (or satellite @ distributed) decide on how to route that packet that I have 
        topologyPolicy: if i have some observation of the current environment, how do I update the topology of my graph/satellite network? :D 

        numPeople: how many people requesting packets sent 
        numPacketsPerPerson: how many packets each person is requesting to transmit
        packetSendTimeFrame: over how long of period do we send packets

        queingDelaysEnabled: do I account for how long it takes to process packets at a server? Or am I just concerned with the propagation delay? 
        weatherEnabled: do i have stuff like....uh idk i need to implement this later. Might include like rainstorms or higher radiation from the sun 
        environmentUpdateInterval: how often we update the postition of satellites in the constellation. Mostly used as you would think. 
        outageFrequency: how often we have satellites that break.     

        takeSnapshots: are we taking snapshots of the simulation environment over time? 
        simulationTimeBetweenSnapshots: how long to wait between taking snapshots 
        numEnvironmentSnapshots: how many snapshots to take 

        fulllyFlushedNetworkETA: how long do we expect it to take to fully flush the network 
        routingPolicy: how our servers route packets. need this for specifying what events we are creating. 
        
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
        
        #initialize routing tables. This is really only applicable for OSPF :) 
        self.manager.initializeSatelliteRoutingTables()

        #first, create a priority queue for events  
        #create pQueue 
        eventQueue = PQueue()

        #create the packets 
        #TODO: change input config here 
        packets = self.generatePackets(numPeople,
                                       numPacketsPerPerson,
                                       personDistribution,
                                       "Uniform",
                                       packetSendTimeFrame,
                                       "True")

        #create the events fro the packets being sent 
        for packetInd in range(len(packets)):
            kargs = {"packet": packets[packetInd]} 
            queueEvent = Event(packets[packetInd].packetSendTime,
                               "packetSent",
                               kargs)
            eventQueue.push(queueEvent)

        #section for updating routing information. in OSPF, this could be creating MST, but in RL, could be general routing table update 
        #logically, probably have the update MST method occur less often than the update adj mat method 
        
        #will probably have much more adjMat updates/broadcasts than updateRoutingTable methods 
        #just because computationally one is way more than the other 

        if(routingPolicy == 'OSPF'): 
            for updateRoutingTableInd in range(3): 
                #so create time and events 
                updateTime = fullyFlushedNetworkETA*updateRoutingTableInd/3
                queueEvent = Event(updateTime,
                                    "updateRoutingTable",
                                    {})
                
                #then push the events 
                eventQueue.push(queueEvent)
        
        if(routingPolicy == 'OSPF'): 
            for broadcastAdjMatInd in range(100): 
                #so create time and events 
                updateTime = fullyFlushedNetworkETA*broadcastAdjMatInd/100
                queueEvent = Event(updateTime,
                                    "updateAdjMats",
                                    {})
                eventQueue.push(queueEvent)

        #otherwise, compute path at every recieval instance 
        #then, create the snapshot events if we are supposed to 
        if(takeSnapshots):
            #create the time stamps for all 
            snapshotTimes = np.arange(0, numEnvironmentSnapshots*simulationTimeBetweenSnapshots, simulationTimeBetweenSnapshots)

            #for each snapshot
            for snapshotInd in range(numEnvironmentSnapshots): 

                kargs = {"snapshotInd":snapshotInd}
                queueEvent = Event(snapshotTimes[snapshotInd],
                                   "takeSnapshot",
                                   kargs)
                eventQueue.push(queueEvent)

        #first, create the reference time for when to update environment parameters 
        updateReferenceTime = 0 
        
        updatedAlready = False 
        #while we arent empty in the eventQueue
        while not eventQueue.is_empty(): 

            #get the next event 
            event = eventQueue.pop()

            #coordinate current time
            self.currentTime = event.timeOfOccurence

            #if our time is outside this interval, then update correspondingly 
            if(event.timeOfOccurence > updateReferenceTime + environmentUpdateInterval): 
                #this updates at least as often as necessary
                #this is because it updates when the time constraint is violated, and then updates to the timing that created the violation
                print(":)")
                print(event.timeOfOccurence)

                self.manager.updateSatelliteStates(updateReferenceTime, event.timeOfOccurence)
                updateReferenceTime = event.timeOfOccurence 

            #if its to take the snapshot 
            if(event.eventType == "takeSnapshot"):
                #store it in corresponding place 
                self.snapshotStorage[event.kargs["snapshotInd"]] = self.SimulationSnapshot(self.manager, self.view, self.currentTime)

            #the next two methods only apply to the OSPF routing policy 
            #if the event is to update the adjacency matrix 
            if event.eventType == "updateAdjMats": 
                #update all personal times first 
                for player in raveledPlayers: 
                    #first update the personal times 
                    player.updateUsingPersonalInfo(event.timeOfOccurence) 
                
                for player in raveledPlayers: 
                    #then, update the adj matrix 
                    player.updateAdjMatrixFromNeighbors()
                    
            #if the event is to update the routing table 
            if event.eventType == "updateRoutingTable": 
                #just update the respective routing tables 
                for ind, player in enumerate(raveledPlayers):
                    if(ind == 1 and False): 

                        #analyze the difference in routing table 
                        QLengths = np.maximum(player.QFinishTimes, player.currTime)
                        QLengths = QLengths - player.currTime

                        #get traffic aware table
                        trafficAwareRoutingTable = myMath.dijkstraWithNodeValuesAllInitialHops(player.adjMatrix, player.adjMatPersonalIndex, QLengths)

                        #get non traffic aware routing table
                        nonTrafficAwareRoutingTable = myMath.dijkstraWithNodeValuesAllInitialHops(player.adjMatrix, player.adjMatPersonalIndex)

                        pdb.set_trace() 
                    player.updateRoutingTable() 


            #if its for sending a packet :D 
            if event.eventType  == "packetSent":
                
                #create event for arriving at next player. (so arriving at constellation)
                #first, get the time of arriving at that first satellite 
                #the indexing may possibly be wrong for raveled satellites
                #but get the eventEndTime by accounting for initial propagation 
                timeOfOccurence = event.timeOfOccurence + myMath.dist3d(event.kargs["packet"].startLocation,  
                                                                        raveledPlayers[event.kargs["packet"].startSat].getCoords())/(3e8)
                
                #store the currSat 
                event.kargs["packet"].currSat = event.kargs["packet"].startSat

                #create the event
                queueEvent = Event(timeOfOccurence, 
                                   "packetArriveAtNextPlayer",
                                   event.kargs)
                
                #then, add the new event on the pQ
                eventQueue.push(queueEvent)
                
            #if our event type is arriving at next player,
            if event.eventType == "packetArriveAtNextPlayer": 
                
                #get our current satellite 
                satelliteIndWeAreAt = event.kargs["packet"].currSat
                
                #get the finish processing time
                print(satelliteIndWeAreAt) 
                endProcessTime = raveledPlayers[satelliteIndWeAreAt].generateProcessingOneMorePacketTime(event.timeOfOccurence, queingDelaysEnabled) 
                
                #create event to queue, based on when we finish processing 
                queueEvent = Event(endProcessTime,
                                   "packetFinishProcessing",
                                   event.kargs)
                
                #push the event 
                eventQueue.push(queueEvent)                

                continue 

            #if the packet is done waiting at queue of corresponding satellite 
            if event.eventType == "packetFinishProcessing":

                #different conditions for each routingPolicy  
                
                #for basic, its if we only had to wait at one to begin with  
                #or if we are at the end of path
                if((routingPolicy == "basic" and (len(event.kargs["packet"].routingMetadata["path"]) == 1 or event.kargs["packet"].reachedEnd()))  or
                    #for OSPF, its if the current satellite is the last satellite in the routing path 
                   (routingPolicy == "OSPF"  and (event.kargs["packet"].reachedEnd())) or
                   (routingPolicy == "mixedSingleAgentRLRestOSPF"  and (event.kargs["packet"].reachedEnd()))):
                     
                    #so then, get the time of occurence of landing at the dest 
                    timeOfOccurence = event.timeOfOccurence + myMath.dist3d(event.kargs["packet"].endLocation,  
                                                                            raveledPlayers[event.kargs["packet"].endSat].getCoords())/(3e8)
                    
                    #then, store the data for when the final arrival of the packet happened 
                    event.kargs["packet"].packetArriveTime = timeOfOccurence

                    print("Made it! :D")

                    continue 
                
                #get the current and next players 
                fromPlayerInd = event.kargs["packet"].currSat
                fromPlayer = raveledPlayers[fromPlayerInd]
                toPlayerInd = fromPlayer.getNextHopAndUpdatePacket(event.kargs["packet"])

                #adjMat stores propagation delays 
                propTime = self.manager.currAdjMat[fromPlayerInd, toPlayerInd]
                
                #create corresponding time 
                timeOfOccurence = event.timeOfOccurence + propTime 

                #create event 
                queueEvent = Event(timeOfOccurence, 
                                   "packetArriveAtNextPlayer",
                                   event.kargs)
                               
                #then, add the new event on the pQ
                eventQueue.push(queueEvent)

        #then, finally just return the difference between the two. 
        #create storage for latency 
        latencyTimes = np.zeros(len(packets))
        #then, iterating through the packets
        for packetInd in range(len(packets)): 
            latencyTimes[packetInd] = packets[packetInd].packetArriveTime - packets[packetInd].packetSendTime

        self.plotMySnapshotSet()

        #print average latency
        print("Average latency")
        print(np.average(latencyTimes))
        
        #then, return it
        return latencyTimes


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

        #so, use their version going forward 
        #1. plot initial figure. data is still saved in self.view.fig 
        #self.update(0,satTimePerFrame)
        #holding = self.ax. 

        #2. we have the update function already for a frame 
        #test = self.fig aoeu 
 
        #3. we pass in: the figure to use for updating, the function to call each frame/time, # frames / updates, time per frame, and the args within the function
        hold = FuncAnimation(self.fig, 
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
        
        #eh bad
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

        avgTime = totalTime/numTrials
        avgLength = totalLength/numTrials

        return avgLength, avgTime, maxTime

   