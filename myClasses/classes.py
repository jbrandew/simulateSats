import math 
import myPackages.myPlots as myPlots
import myPackages.myMath as myMath 
import numpy as np 
import pdb 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Simulator(): 
    """
    This class serves to just hold and execute different 
    simulation objectives/functions 
    
    """
    def __init__(self, manager): 
        self.manager = manager
        
        #create figure and plot to disploy data 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') 
        self.view = myPlots.GraphicsView(self.manager, fig, ax)


    def update(self, frame, satTimePerFrame): 
        """
        update function for frame data 
        """
        self.manager.updateConstellationPosition(satTimePerFrame) 
        self.view.update_graphics() 


    def timeFrameSequencing(self, timeRatio, FPS, animationDuration): 
        """
        This code starts the time frame varying animation 
        
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
        #pdb.set_trace() 
        hold = FuncAnimation(self.view.fig, 
                      self.update, 
                      frames=numFrames, 
                      interval=realTimePerFrame, 
                      fargs = (satTimePerFrame,))
    
        plt.show() 

        return 

    def multiPlot(self): 
        """
        Just plot current satellites, base stations, and ISLs

        Input: none
        Output: plot posted to GUI 
        
        """
        myPlots.multiPlot(0,
                          self.manager.getSatLocations(), 
                          self.manager.getBaseStationLocations(), 
                          self.manager.getXYZofLinks(maxNumLinksPerSat=6))


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
        #pdb.set_trace()  
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


#general manager/scheduler of processes   
class Manager(): 
    """
    to coordinate interactions between different players. This includes creating ISLs with 
    diff topologies, and may include dealing with time frame changes, 
    interactions with stochastic elements, etc. 
    
    """
    
    def __init__(self, 
                 constellationType, 
                 constellationConfig, 
                 baseStationLocations, 
                 fieldOfViewAngle, 
                 phasingParemeter): 
        """
        This does the initialization step for internals, as well as generating satellites 
        and base stations from respective locations 

        walkerPoints: x y z of all walker satellites
        constellationType: type of constellation we want to use 
        constellationConfig: configuration variables needed for creating that constellation
        baseStationLocations: x y z of all relay/baseStations :) 
        phasingParameter: the angular offset/revolution difference between adjacent planes
        """

        if(constellationType == "walkerDelta"): 
            walkerPoints = myMath.generateWalkerStarConstellationPoints(*constellationConfig)
        else:
            raise ValueError("Not valid constellation type")

        #calculate logical parameters based on point dimensions 
        self.numPlanes = np.shape(walkerPoints)[0]
        self.numSatPerPlane = np.shape(walkerPoints)[1]
        self.numLEOs = np.shape(walkerPoints)[0] * np.shape(walkerPoints)[1]
        self.phasingParameter = phasingParemeter

        #init storage for satellites, base stations, and links   
        self.sats = np.tile(LEO(), [self.numPlanes, self.numSatPerPlane]) 

        #call generate satellites function, which initializes our structure 
        self.generateSatellites(walkerPoints)
        #first, format the baseStationInputs 
        self.generateBaseStations(baseStationLocations, fieldOfViewAngle)

        #this could be wrong lol 
        self.numBaseStations = len(self.baseStations)

        #set up storage for adjacency matrix 
        self.currentAdjacencyMatrix =  np.tile(np.Infinity, [self.numLEOs + len(self.baseStations), self.numLEOs + len(self.baseStations)])


    def updateConstellationPosition(self, timeDiff): 
        """
        this updates the position of all satellites in our constellation 

        Inputs: 
        timeDiff: difference in time between frames

        Outputs: 
        none, just changed constellation shape/state 
        """
        for satArr in self.sats: 
            for sat in satArr:
                sat.updatePosition(timeDiff)

    def getSatLocations(self): 
        #create storage for all coordinates 
        allSatCoords  = np.empty((np.shape(self.sats)+(3,)))

        #first, store the coords of all satellites  
        for ind1, row in enumerate(self.sats): 
            for ind2, sat in enumerate(row):
                allSatCoords[ind1, ind2] = sat.getCoords() 

        return allSatCoords        

    def getBaseStationLocations(self):
        """
        Return list of x y z locations of base stations

        Input: 
        Output: list of BS locations
        """
        hold = [0]*len(self.baseStations)
        for ind, baseStation in enumerate(self.baseStations): 
            hold[ind] = baseStation.getCoords()
        
        return hold

    def averageDistToBaseStation(self): 
        """
        Get average distance of all satellites to base station 

        Input: None
        Output: avg dist 

        """
        
        storeAvgPropToBS = np.ones(len(np.ravel(self.sats)))

        #iterate through our satellites 
        for sat in np.ravel(self.sats): 
            #for each satellite, get prop delay 
            sum = 0
            for baseStation in self.baseStations: 
                sum+=self.propDelayBetween(sat, baseStation) 

            storeAvgPropToBS[sat.index] = sum 

        return storeAvgPropToBS/np.size(self.baseStations)

    def getClosestSatelliteToCoordinate(self, x, y, z): 
        """
        Return satellite closest to coord in x y z
        Please note, could make the consistent usage of this function much
        more efficient with partitioning the original set of coordinates 

        in: x, y, z of point 
        out: closest satellite 
        """

        #create storage         
        minDist = np.Infinity
        closePlaneInd = 0
        closeSatInd = 0 

        #iterate through satellites 
        for planeInd in range(self.numPlanes): 
            for satInd in range(self.numSatPerPlane):
                #if we are the min dist, then store the inds  
                if(minDist > myMath.dist3d([x,y,z], self.sats[closePlaneInd][closeSatInd].getCoords())):
                    closePlaneInd = planeInd
                    closeSatInd = satInd 

        #return whats needed 
        return planeInd, satInd 

    def createISL(self, sat1, sat2, polarRegionRestriction): 
        """
        Create ISL between satellites 

        Inputs: 
        sat1: first satellite 
        sat2: second satellite 
        polarRegionRestriction: are we only allowing connections between satellites
        that are between -70 and 70 latitude 

        Output:
        nothing, just changed connected satellite arrays 
        """
        
        if(polarRegionRestriction): 
            lats1 = myMath.cartesian_to_geodetic(*sat1.getCoords(), 6371) 
             
            if(lats1[0] > 70 or lats1[0] < -70): 
                return False 

            lats2 = myMath.cartesian_to_geodetic(*sat2.getCoords(), 6371) 
            
            if(lats2[0] > 70 or lats2[0] < -70): 
                return False 

        sat1.connectedSats = sat1.connectedSats + [sat2]
        sat2.connectedSats = sat2.connectedSats + [sat1]

        return 


    def getPseudoCapacity(self, numberPackets): 
        """
        This function gets the "pseudo capacity" for each satellite. Pseudo capacity meaning
        how much time does one of the satellites get used. It does this by: 
        a. getting # of packet arrivals in the given amount of time
        b. get path (start, nodes between, end) for each packet arrival
        c. for each path and each link used, add prop delay to each end of the link. 
        d. get the total "use time" for average satellite and base station. Also plot
        for use time vs proximity to base station 

        Inputs: 
        #rateOfArrivalForSingleSatellite: ... 
        #amountOfTime: amount of time we are simulating over 
        numberPackets: number of packets we are simulating over 
        

        Output: 
        Array with total "use time" for each satellite and base station 
        """

        #create storage for the paths
        listOfPaths = [0]*numberPackets 

        #create random start and end of each
        numPlayers = self.numLEOs + len(self.baseStations)
        #2 for start and end. 
        random_integers = np.random.randint(0, numPlayers, size=[numberPackets, 2])
        
        #resample to prevent same to and fro
        for ind in range(numberPackets): 
            if(random_integers[ind, 0] == random_integers[ind,1]):
                random_integers[ind,1] = np.random.randint(0, numPlayers)

        #for each packet, get the path 
        
        for packetInd in range(numberPackets): 
            print(str(packetInd)+"/"+str(numberPackets))
        
            listOfPaths[packetInd] = myMath.dijkstraWithPath(self.currAdjMat, 
                                                        random_integers[packetInd,0], 
                                                        random_integers[packetInd,1])[0]

        #create storage for use time of each player 
        useTimeForEachPlayer = np.zeros(numPlayers) 

        #iterate through paths
        for path in listOfPaths:
                  
            #iterate through start nodes  
            for linkStartNodeInd in range(len(path) - 1): 
                #add the prop delay to each the start node and end node 
                
                startNode = path[linkStartNodeInd]
                endNode = path[linkStartNodeInd+1]

                useTimeForEachPlayer[startNode]+=self.currAdjMat[startNode, endNode]
                useTimeForEachPlayer[endNode]+=self.currAdjMat[startNode, endNode]

        return useTimeForEachPlayer

    def propDelayBetween(self, player1, player2): 
        """
        Return the prop delay between two players

        Input: 
        player1
        player2

        Output: 
        prop delay 
        """

        #for each connected satellite, set corresponding adj matrix entry to propagation delay
        point1 = player1.getCoords()
        point2 = player2.getCoords()
        
        #working within km so 10e5
        if( myMath.dist3d(point1, point2)/(3e5) == np.inf ): 
            pdb.set_trace() 
        return myMath.dist3d(point1, point2)/(3e5)    

    def getXYZofLinks(self, maxNumLinksPerSat=6): 
        """
        Just output array of XYZ positions of all links 

        Input: 
        Output: 
        array of 2 coord links 
        """

        #each sat in this case has max of 4 links, so make an array that big 
        #it will be a shape = numLeos*4 x 3 x 2 
        # max Num Links x xyz x to or fro
        holdLinks = np.zeros([self.numLEOs*maxNumLinksPerSat, 2, 3])

        ind = 0 
        #iterate through our satellites 
        for fromSat in np.ravel(self.sats): 
            #iterate through each satellites connections
            for toSat in fromSat.connectedSats: 
                #create link 
                holdLinks[ind] = np.concatenate([[fromSat.getCoords()], [toSat.getCoords()]], axis = 0)  
                ind+=1 
            for toBaseStation in fromSat.connectedBaseStations: 
                holdLinks[ind] = np.concatenate([[fromSat.getCoords()], [toBaseStation.getCoords()]], axis = 0)
                ind+=1               

        return holdLinks
    
    def generateBaseStations(self, baseStationLocations, fieldOfViewAngle): 
        """
        Just generate the baseStation objects. 

        Input: 
        baseStationLocations: x y z of the baseStations 
        fieldOfViewAngle: angle that a satellite must be above with respect to horizon line
        to be considered in view
        
        Effect/Output: 
        Created base station objects 
        """
        #iterate through the locations (if we have locations at all)
        if(baseStationLocations is not None): 
            #create storage for base stations 
            self.baseStations = np.tile(baseStation(), [np.shape(baseStationLocations)[0]])
            for ind in range(np.shape(baseStationLocations)[0]): 
                #convert properly 
                formattedLocation = myMath.geodetic_to_cartesian(*baseStationLocations[ind])
                #for each location create the associated object
                self.baseStations[ind] = baseStation(*formattedLocation, fieldOfViewAngle)
        #otherwise set as place holder 
        else: 
            self.baseStations = [] 

    def generateSatellites(self, walkerPoints): 
        """
        Just storing satellites when given walker constellation points

        Input: walker points (constellation points)
        
        Effect: sets up our satellite internals using walkerPoints 
        """
        #iterate through planes and then sats within a plane 
        for planeInd in range(self.numPlanes): 
            for smallSatInd in range(self.numSatPerPlane):
                #initialize a satellite each time  
                self.sats[planeInd, smallSatInd] = LEO(*(walkerPoints[planeInd,smallSatInd]),
                                                        planeInd, 
                                                        smallSatInd) 
                
    def connectBaseStationsToSatellites(self): 
        """
        Just connects base station to all satellites within each of their field of views

        Input: 
        Output: 
        """
        #for each base station
        for baseStation in self.baseStations: 
            #for each satellite 
            #(the ravel function just compresses into one dimension)
            for satellite in np.ravel(self.sats): 
                #check if we are in field of view 
                
                if(baseStation.isSatelliteInView(satellite.getCoords())): 
                    #if we are, connect them both 
                    baseStation.connectToSat(satellite)
                    satellite.connectToBaseStation(baseStation)

            for baseStationTo in self.baseStations:  
                if(baseStation != baseStationTo): 
                    baseStation.connectedBaseStations = baseStation.connectedBaseStations + [baseStationTo]

    def stochasticFailure(self, numSatellitesFail): 
        """
        Disable a number of satellites 

        numSatellitesFail: the number of satellites that will randomly fail 
        """                                              
        
        #choose without repeat of satellites  
        satIndFails = np.random.choice(np.arange(self.numLEOs), numSatellitesFail, replace=False)

#for all topologies in this section there is an important concept: 
#when we complete a full revolution, the planeIndex can reset to 0.
#however, the satellite index within a plane must change. this will be 
#equal to: # satellites per rev. = # planes * phasing parameter / ( 360 / # satellites per plane)
#which can be thought of as: 
# # satellites = number of degree increase per revolution / degree increase per satelline in plane
            
#in all these topologies, thats the # "5"
    def connect2ISL(self): 
        """
        This function connects our satellites in a formation where we have 
        2 ISLs per satellite, just connecting to those in the same plane. 
        PLEASE NOTE, this type of function only connects satellites to 
        other satellites. It does not connect base stations to 
        satellites in its field of view. 

        No inputs needed, just make the necessary topology connections.
        Please be careful of the "count of ISLs" != the sum of connections. 
        (it should be sum of connections / 2)
        """

        #iterate through planes and then sats within a plane 
        for planeInd in range(self.numPlanes): 
            for smallSatInd in range(self.numSatPerPlane):    
                #uhhh...connect satellite to adjacent ones 
                #PLEASE NOTE: only in this topology, can we traditionally enable 
                #the connections that are above/below polar boundaries
                #not sure why, but thats what it is in research paper
                #might have something to do with increased EM interference in those
                #regions             
                self.createISL(self.sats[planeInd, smallSatInd], self.sats[planeInd, (smallSatInd + 1) % self.numSatPerPlane], False)


    def checkSpiralResetOfIndex(self, planeIndex, satIndex, originalPlaneIndex, originalSatIndex, goLeft):
        vertDistanceInSatsPerRev = self.numPlanes * self.phasingParameter / ( 360 / self.numSatPerPlane)

        if(vertDistanceInSatsPerRev - int(vertDistanceInSatsPerRev) != 0): 
            raise Exception("Behavior not defined for non-integer vertical distance for revolution")

        vertDistanceInSatsPerRev = int(vertDistanceInSatsPerRev)

        #if we are at the original plane 
        if(planeIndex == originalPlaneIndex): 
            
            #then add or subtract the satellite index by the proper amount
            #this amount is based on the amount in the vertical direction per revolution
            if(goLeft): 
                satIndex -= vertDistanceInSatsPerRev
            else: 
                satIndex += vertDistanceInSatsPerRev
            
        
        return planeIndex % self.numPlanes, satIndex            

    def connectVNTopology(self, startSatInd = 3): 
        """
        What is the VN topology? overlapping partial ladders centered around a certain latitude. 
        In total, we have 8 partial disjoint spiral topologies. 
        4 of which are on the "downward side". The two sets of 4 overlap heavily though. 
        
        Inputs: 
        startSatInd: what satellite within a plane we will start at 
        Outputs: 
        Effect: VN connected topology in our satellites :) 
        """
        self.connect2ISL() 

        #upward set 
        for partialSpiralInd in range(4):
            self.connectSpiralTopologySimple(2, 5, 5*partialSpiralInd, startSatInd-partialSpiralInd, True)
            
        #downward set, so offset the sat ind by half a revolution 
        for partialSpiralInd in range(4):
            self.connectSpiralTopologySimple(2, 5, 5*partialSpiralInd, startSatInd-partialSpiralInd+10, True)
            
        return 

    def connectLadderTopology(self): 
        """
        This topology creates a kind of "sideways ladder" topology by incrementing 
        both the plane ind and sat ind by one each time 

        Input: None
        Output:
        Effect: Ladder connected satellite constellation 
        """

        self.connect2ISL()
        
        #in total, in this topology there are 40 connections 
        #there are 2 sets of 20. 

        #offset by 2 as there is not perfect alignment  
        planeInd1 = 0 
        satInd1 = 0

        planeInd2 = 1
        satInd2 = 0 

        #this # 40 is kind of random, just a parameter so each plane has 2 connections
        #to each adjacent plane (ensures even in a polar region orbit, they stay connected)

        for connectInd in range(self.numPlanes*2): 
            
            self.createISL(self.sats[planeInd1, satInd1], self.sats[planeInd2, satInd2], False) 

            #index appropriately. this indexing is what differentiates it from
            #the normal spiral topology (indexing sat ind as well)
            planeInd1+=1
            satInd1+=1 

            #if we reach end of plane index 
            if(planeInd1 >= self.numPlanes): 
                #change sat index appropriately 
                planeInd1 = 0
                #(see comments above topology section to explain this number "5")
                satInd1+=5 

            #index appropriately 
            planeInd2+=1
            satInd2+=1 

            #if we reach end of plane index 
            if(planeInd2 >= self.numPlanes): 
                #change sat index appropriately (proper reset in other direction) 
                #something wrong with this reindexing....
                planeInd2 = 0
                satInd2+=5       

            #reset satellite indices well 
            satInd1 = satInd1%self.numSatPerPlane
            satInd2 = satInd2%self.numSatPerPlane

        return 

    def connectZigzagTopology(self): 
        """
        Connect satellites in zig zag formation. What does that mean? In this base case, 
        that means a continuous zig zag revolving around the earth. Crossing 4 planes
        results in one increment of a satellite index in one of the planes. 

        Please note that the inclining down for the plane index is just when we 
        are on the "other side" for the plane position. 

        This type of implementation may only work in this walker delta constellation. 
        i.e, 20 planes is evenly divided by the sat ind/latitude period of 4 in a zig zag. 
        
        Could pretty easily expand this to ladders that still mesh correctly with this topology
        
        Input: none 
        Output: zig zag connected satellites, of zig zag period = 4 satellites 
        """

        self.connect2ISL() 

        #for each zig zag 
        for zigZagInd in range(self.numSatPerPlane):
            #the satellite we start with is at plane 0, and this index 
            startSat = self.sats[0,zigZagInd]
            
            #initialize our indices 
            planeInd1 = 0
            satInd1 = zigZagInd

            planeInd2 = 1 
            satInd2 = zigZagInd

            #while we havent returned to our original plane  
            while(planeInd2 != 0 ): 
                #create connection 
                self.createISL(self.sats[planeInd1,satInd1], self.sats[planeInd2, satInd2], True)

                #index planes and modulus accordingly 
                planeInd1+=1 
                if(planeInd1%4 == 0): 
                    satInd1-=1 
                
                planeInd2+=1
                if(planeInd2%4 == 0): 
                    satInd2-=1 

                planeInd1 = planeInd1%20 
                planeInd2 = planeInd2%20 

            #add 5 as thats the coefficient of satellite offset per revolution in this
            #(can see comments above the topology section to explain this)
            satInd2+=5
            self.createISL(self.sats[planeInd1,satInd1], self.sats[planeInd2, satInd2], True)
           
        return 

    def connectSpiralTopologySimple(self, satsBetweenConnections= 1, numberConnections = -1, startPlaneInd = 0, startSatInd = 0, ISL2Done = False): 
        """
        Connect all satellites in a spiral fashion. 
        This method uses the fact of periodicity in modulus operator connecting
        num planes and satellites in each plane. Seems kind of similar to some notion
        of finite fields (periodicity in modulus)
        This implementation only works for certain implementations of walker star, I believe. 

        Input: 
        numberConnections: this is effectively the length of the spiral (not accounting for the 
        satsBetweenConnections parameter). If we just use default case, then its 
        self.numPlanes * self.numSatPerPlane, as thats the # connections with 1 for each 
        startSatInd, startPlaneInd: designates the satellite that the spiral starts at. For the
        default case of full spiral, it doesnt matter where it starts, as it will return
        to that satellite 
        ISL2done: if we have already connected the 2ISL topology (if we have already, dont do it again)

        satsBetweenConnections: number of satellites between connections. Just 1
        in normal spiral, but in something like disjoint, it will be 2.
        TO DO / check: might need to make how numberConnections is used be dependent on this parameter.
        As in, for default case, if we have this param = 2, then numConnections in default could be
        halved  

        Output:
        Effect: all satellites connected in spiral fashion 
        """

        #first make in plane connections, if we havent already
        if(not ISL2Done): 
            self.connect2ISL() 

        #if we use default case 
        if(numberConnections == -1): 
            numberConnections = self.numPlanes * self.numSatPerPlane

        planeInd1 = startPlaneInd
        satInd1 = startSatInd

        #for spiral, focus on indexing by plane while mostly keeping the sat ind constant
        planeInd2 = startPlaneInd + 1
        satInd2 = startSatInd

        #there will be num connections = numPlanes*numSatPerPlane  
        for ind in range(numberConnections): 
            
            #create ISL connection 
            try: 
                self.createISL(self.sats[planeInd1,satInd1], self.sats[planeInd2,satInd2], True)            
            except Exception as e: 
                print("wasnt able to connect two satellites")

            #index appropriately 
            planeInd1+=satsBetweenConnections

            #if we reach end of plane index 
            if(planeInd1 >= self.numPlanes): 
                #change sat index appropriately 
                planeInd1 = planeInd1 - self.numPlanes
                #(see comments above topology section to explain this number "5")
                satInd1+=5 

            #index appropriately 
            planeInd2+=satsBetweenConnections

            #if we reach end of plane index 
            if(planeInd2 >= self.numPlanes): 
                #change sat index appropriately (proper reset in other direction)
                planeInd2 = planeInd2 - self.numPlanes
                satInd2+=5             

            #reset satellite indices well 
            satInd1 = satInd1%self.numSatPerPlane
            satInd2 = satInd2%self.numSatPerPlane

        return 
                       
    def connectDisjointSpiralTopologySimple(self): 
        self.connectSpiralTopologySimple(2)
        return 

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
                    self.sats[planeInd, smallSatInd].connectToSat(satOfPrevPlane)
                
                #for back case: 
                else: 
                    satOfNextPlane = self.sats[(planeInd - 1) % self.numPlanes, smallSatInd]
                    self.sats[planeInd, smallSatInd].connectToSat(satOfNextPlane)  



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


    def generateAdjacencyMatrix(self): 
        """
        Hello :D 
        This function returns an adajcency matrix based upon the current satellite positions
        as well as the actual connected satellites (with ISLs) 
        Please note, this assumes each one has a line of path sight to each connected satellite. 

        TO DO: set diagonal = 0 
        
        Output: adjacency matrix where dist = prop delay 
        -note: should have 0s on diagonal, and should be symmetric about the diagonal 
        """
        #alright. How to do this? 
        #hmm... iterate through the satellites, and for each pair, set the matrix entry to 1 
        #iterate through planes and then sats within a plane 
        
        #set up adjacency matrix storage 
        #lets have row be "from" and col be "to"
        adjMat = np.tile(np.Infinity, [self.numLEOs + len(self.baseStations), self.numLEOs + len(self.baseStations)])

        #then concatenate to get one array of all players 
        allPlayers = np.concatenate([np.ravel(self.sats), self.baseStations], axis = 0)
        
        #iterate through players and assign index 
        for playerInd in range(len(allPlayers)): 
            allPlayers[playerInd].index = playerInd 


        for playerFrom in allPlayers:
            #iterate through the players satellites and base stations  
            for satTo in playerFrom.connectedSats: 
            
                adjMat[playerFrom.index, satTo.index] = self.propDelayBetween(playerFrom, satTo)
            for baseStationTo in playerFrom.connectedBaseStations: 
                adjMat[playerFrom.index, baseStationTo.index] = self.propDelayBetween(playerFrom, baseStationTo)

        self.currAdjMat = adjMat

        return adjMat


#player class describes any operator on or above the earth 
#probably going to use rho and phi more often than purely longitude and latitude 
#nah lets just always use x,y,z...maybe easier in polar all the time...worry later
class Player(): 

    altitude, long, lat = 0,0,0
    x,y,z = 0,0,0
    
    theta, phi = 0,0 

    def __init__(self, xIn, yIn, zIn): 
        self.x, self.y, self.z = xIn, yIn, zIn 

    def getCoords(self): 
        return self.x, self.y, self.z 

#LEO (low earth orbit) describes the lowest orbiting set of satellites 
class LEO(Player): 

    def __init__(self, xIn = 0, yIn = 0, zIn = 0, planeIndex = 0, subSatIndex = 0):
        """
        Init function for LEO.
        Just pass off coordinates to parent "Player" 

        xIn, yIn, zIn: coordinates in 3d space 
        planeIndex: number plane we are in in walker star 
        subSatIndex: number satellite we are in in a single plane 
        
        """
        super().__init__(xIn, yIn, zIn)

        #store indices 
        self.planeIndex = planeIndex
        
        self.subSatIndex = subSatIndex

        #number links we are allowed to have
        #not sure if we will use this  
        self.numberISLlinksAllowed = 2  

        #satellites we are connected to 
        self.connectedSats = [] 

        #base stations we are connected to
        #base station itself will probably do the connecting/disconnecting  
        self.connectedBaseStations = []

    def updatePosition(self, timeDiff): 
        """
        Based on current position and time diff,
        set your next position

        Inputs:
        timeDiff = satTimeDiff between positions

        Effect: 
        updates our position
        
        """
        self.x, self.y, self.z = myMath.calculate_next_position([self.x, self.y, self.z], timeDiff)

    def connectToSat(self, satToConnectTo, polarRegionRestriction = True): 
        """
        Just connect to a given satellite using your internals

        Inputs: 
        satToConnectTo: ... 
        polarRegionRestriction: do not allow connection if satellite to connect
        is above 70N or below 70S 

        Outputs: 
        boolean designating if we made the connection or not 
        """
        #if we have the restriction, enable it 
        
        if(polarRegionRestriction): 
            lats = myMath.cartesian_to_geodetic(*satToConnectTo.getCoords(), 6371) 
            
            if(lats[0] > 70 or lats[0] < -70): 
                return False 

            selfLats = myMath.cartesian_to_geodetic(*self.getCoords(), 6371) 
            
            if(selfLats[0] > 70 or selfLats[0] < -70): 
                return False 
            
        
        self.connectedSats = self.connectedSats + [satToConnectTo]
        return True 

    def connectToBaseStation(self, baseStationToConnectTo): 
        """
        Just connect to a given basesattion using your internals

        baseStationToConnectTo: ... 
        """
        self.connectedBaseStations = self.connectedBaseStations + [baseStationToConnectTo]        


#base station class desribes the players on the ground that arent moving and act as forwarders 
class baseStation(Player): 

    def __init__(self, xIn = 0, yIn = 0, zIn = 0, minElevationAngle = 0): 
        """
        Init function for base station. 
        Pass off coords to parent "Player" and set your own angle 

        xIn, yIn, zIn: coords in 3d space 
        minElevationAngle: min coordinate you need to be above to be considered in view 
        """
        super().__init__(xIn, yIn, zIn)
        self.minElevationAngle = minElevationAngle 

        #can connect to any amount of satellites 
        self.numberLinksAllowed = float('inf')

        #store base stations we are connected to 
        #these will be the satellites within our field of view/above a certain elevation angle 
        self.connectedSats = [] 

        self.connectedBaseStations = []

    def connectToSat(self, satToConnectTo): 
        """
        Just connect to a given satellite using your internals

        satToConnectTo: ... 
        """
        self.connectedSats = self.connectedSats + [satToConnectTo]    

    def isSatelliteInView(self, satelliteCoords): 
        """
        Function to return if a satellite is above a certain elevation angle 
        with respect to this base station 

        satelliteCoords: xyz of the satellite 
        minElevationAngle: min angle above horizon line that the satellite must be 
        to be considered "in view" 
        """

        #the max angle of the 3 angles formed by the 3 points: 
        #earth center, satellite center, base station location
        #is the angle with the base station as the apex. 
        #this angle is always obtuse, and angle - 90 = angle above horizon 

        #please note that [0,0,0] represents the center of the earth 
        #so first get the three angles 
        angles = myMath.compute_triangle_angles(satelliteCoords, [self.x, self.y, self.z], [0,0,0])
        #get the horizon angle 
        angleAboveHorizon = max(angles) - 90
        #return if its above the horizon 
        return angleAboveHorizon > self.minElevationAngle
    
    def numSatellitesInView(self, satelliteCoordsToCheck): 
        """"
        Function to check how many satellites are in view of our base station
        Only functional for walker star constellation for LEO orbit currently 

        Inputs: 
        satelliteCoordsToCheck: all xyz of satellites that we want to see
        ###numOrbitalPlane: how many planes of orbit we have 
        minElevationAngle: minimum angle you need to be above to be considered in view 
        
        Outputs: 
        num satellites in view
        num disjoint orbits in view (probably only applicable to LEOs only)
        """
        
        numInView = 0
    
        #now, i need to also get the number of disjoint orbit planes i have access to 
        #how to do this? 
        #could either change to two loops and index, or...hmm yea 

        numDisjointOrbitPlanesInView = 0

        #iterate through num orbital planes 
        for orbitalPlaneInd in range( np.shape(satelliteCoordsToCheck)[0]): 
            addedPlane = False 

            #iterate through num satellites in each plane 
            for satIndInPlane in range(np.shape(satelliteCoordsToCheck)[1]): 
                #please realize that walkerPoints is in the shape of (plane num, sat num, 3 (xyz))
                #so, in order to index it using the satind:
                #planeNum = satInd/numSatPerPlane 
                #satNumWithinPlane = natInd%numSatPerPlane 
                #and then min elevation angle is normal 
                
                #pdb.set_trace() 
                inView = self.isSatelliteInView(
                    satelliteCoords=satelliteCoordsToCheck[orbitalPlaneInd, satIndInPlane])
                
                
                #if this is the first time we are viewing this orbital plane, act accordingly
                if(inView): 
                    numDisjointOrbitPlanesInView+=(not addedPlane) 
                    addedPlane = True
                
                numInView+=inView 

        return numInView, numDisjointOrbitPlanesInView