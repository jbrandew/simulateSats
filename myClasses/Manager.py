import math 
import myClasses.myPlots as myPlots
import myPackages.myMath as myMath 
import numpy as np 
import pdb 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq
from myClasses.Player import * 
from myClasses.Event import Event

#general manager/scheduler, model of processes   
class Manager(): 
    """
    to coordinate interactions between different players. This includes creating ISLs with 
    diff topologies, and may include dealing with time frame changes, 
    interactions with stochastic elements, etc. 

    Acts as the model in the model, view, controller architecture. Provides
    main calculations and monitoring of states of components 
    
    """
    
    def __init__(self, 
                 constellationType, 
                 constellationConfig, 
                 baseStationLocations, 
                 fieldOfViewAngle, 
                 phasingParameter, 
                 sunExclusionAngle,
                 sunLocation,
                 earthRadius): 
        """
        This does the initialization step for internals, as well as generating satellites 
        and base stations from respective locations 

        walkerPoints: x y z of all walker satellites
        constellationType: type of constellation we want to use 
        constellationConfig: configuration variables needed for creating that constellation
        baseStationLocations: x y z of all relay/baseStations :) 
        phasingParameter: the angular offset/revolution difference between adjacent planes
        sunExclusionAngle: min angle between line of sight and sun path to ensure stable connection
        """

        if(constellationType == "walkerDelta"): 
            walkerPoints, normVecs = myMath.generateWalkerStarConstellationPoints(*constellationConfig) 
        else:
            raise ValueError("Not valid constellation type")

        #take in input data 
        self.earthRadius = earthRadius

        self.sunExclusionAngle = sunExclusionAngle

        self.sunLocation = sunLocation

        #calculate logical parameters based on point dimensions 
        self.numPlanes = np.shape(walkerPoints)[0]
        self.numSatPerPlane = np.shape(walkerPoints)[1]
        self.numLEOs = np.shape(walkerPoints)[0] * np.shape(walkerPoints)[1]
        self.phasingParameter = phasingParameter

        #init storage for satellites, base stations, and links    
        self.sats = np.tile(LEO(), [self.numPlanes, self.numSatPerPlane]) 

        #call generate satellites function, which initializes our structure 
        self.generateSatellites(walkerPoints, normVecs)
        #first, format the baseStationInputs 
        self.generateBaseStations(baseStationLocations, fieldOfViewAngle)

        #this could be wrong lol 
        self.numBaseStations = len(self.baseStations)

        #set up storage for adjacency matrix 
        self.currAdjMat =  np.tile(np.Infinity, [self.numLEOs + len(self.baseStations), self.numLEOs + len(self.baseStations)])

        #set up storage for anticipated waiting times at each location 
        self.queueFinishTimes = np.zeros(self.numLEOs + len(self.baseStations))

    def updateEnvironmentAndPathData(self,
                        oldTime, 
                        newTime): 
        """
        Update our adjacency matrix for satellites based on the new reference time. Assumes that the simulation starts at time 0. 

        Inputs: 
        oldTime: time of the last update 
        newTime: current time/what we are "updating to" 

        Outputs:
        Effect: 
        properly updated state and adj matrix 
        """
        
        #update the constellation position
        self.updateConstellationPosition(newTime - oldTime)
        #after that, update the adjacency matrix
        self.currAdjMat = self.generateAdjacencyMatrix()
        #update the waiting times for each of the servers
        self.updateQueueFinishTimes()


    def updateQueueFinishTimes(self): 
        """
        This just maps the stored "anticipated" emptying out of queue of each player to the waitTimes here 

        Inputs: None
        Outputs: None
        Effect: updates the "anticipated times" 
        """

        #so for each satellite, update the finish time 
        #indexing through all players, so ravel the satellites...
        for satInd, checkSat in enumerate(np.ravel(self.sats)):
            self.queueFinishTimes[satInd] = checkSat.finishProcessingTime
        
        #then, do the same thing with base stations 
        for baseStationInd, baseStation in enumerate(self.baseStations): 
            self.queueFinishTimes[satInd + baseStationInd] = baseStation.finishProcessingTime 



    def createPacketSet(self,
                        numPeople,
                        numPacketsPerPerson,
                        simulationTime):
        """
        This function creates a set of packets uniformly over the simulation time,
        mostly uniformly across the earth, and uniformly across the people we are
        operating over

        Inputs: 
        numPeople: number of people that are sending packets 
        numPacketsPerPerson: number of packets each person is sending 
        simulationTime: how long the simulation is operating for 

        Outputs: 
        set of initialized packets 
        """ 
        #then initialize a set of packets
        #so first, get locations of all the packets, spread
        #across the globe. xyz is in km btw. 
        startLocations = myMath.generate_points_on_sphere_mostly_uniform(self.earthRadius,
                                                                    numPeople)

        endLocations = myMath.generate_points_on_sphere_mostly_uniform(self.earthRadius,
                                                                    numPeople)
        
        #next, get random times for sending the packets out 
        packetSendTimes = np.random.uniform(0, 
                                            simulationTime, 
                                            (numPeople*numPacketsPerPerson,))
        
        #next, get the path each packet will take 
        #please note, this assumes hardware limitations of the user on the earth
        #and as such, will connect to the nearest satellite instead of the one 
        #that results in the shortest path 

        #get satellite locations
        satLocs = self.getSatLocations() 

        #initialize the set of packets 
        packets = [0]*(numPacketsPerPerson*numPeople)

        #for each person 
        for personInd in range(numPeople):
            #for each packet 
            for packetInd in range(numPacketsPerPerson): 
                #get the packetInd
                overallPacketInd = personInd*numPacketsPerPerson + packetInd
                #then, get the closest satellite to the start 
                closestSatIndToStart = myMath.closest_point(satLocs, startLocations[overallPacketInd])
                #then, get the closest satellite to the end 
                closestSatIndToEnd = myMath.closest_point(satLocs, endLocations[overallPacketInd])
                #then, use adjacency matrix to get the path for each 
                pathToTake = myMath.dijkstraWithNodeValuesAndPath(self.currAdjMat,
                                                                  self.queueFinishTimes, 
                                                                  closestSatIndToStart,
                                                                  closestSatIndToEnd)

                #create dict for packet parameters 
                packetParams = {'startLocation' : startLocations[overallPacketInd],
                                'endLocation' : endLocations[overallPacketInd], 
                                'scheduledAwakeTime' : packetSendTimes[overallPacketInd],
                                'pathToTake' : pathToTake}
                
                #store a new packet 
                packets[overallPacketInd] = Packet(packetParams)
        
        return packets 



    def find_closest_sats(self, m):
        """
        Function to get the closest m satellites to all other satellites 

        input: # of satellites we want
        output: the m closest satellites to a satellite, repeated for all satellites 
        
        """
        result = {}
        raveledSats = np.ravel(self.sats)

        for current_sat in raveledSats:
            # Initialize a min heap to store the m closest points
            min_heap = []

            for other_sat in raveledSats:
                if current_sat != other_sat:
                    dist = myMath.distance(current_sat.getCoords(), other_sat.getCoords())
                    
                    heapq.heappush(min_heap, (dist, other_sat))

            # Get the m closest points by popping from the min heap
            result[current_sat] = [sat for _, sat in heapq.nsmallest(m, min_heap)]

        return result


    def updateTopology(self, 
                       satPolicy, 
                       baseStationPolicy): 
        """
        Updates the topology based on state 
        """

        if(satPolicy == "Closest"): 
            #call that update
            self.connectSatellitesToClosest() 
            #need to also connect each base station to all sats in view
        if(satPolicy == "Retain"):  
            x = 1   
        if(baseStationPolicy == "InView"): 
            self.connectBaseStationsToSatellites()  


    def connectSatellitesToClosest(self, maxISLNum = 2): 
        """
        Connect each satellite to "maxISLNum" closest satellites 

        Inputs:
        maxISLNum: max # ISLs we can connect to 

        Outputs: 

        Effect: 
        Satellites are now connected to the closest players 
        """
        #first, get the set of locations for each satellite 
        #locs = self.getSatLocations
        #then, get closest to each 
        closest = self.find_closest_sats(maxISLNum)
        #then, for each, clear out current connections and connect to that many
        
        #for each satKey in our array 
        for satKey in closest.keys():
            #reset the connections 
            satKey.resetConnections()
            #and then connect to the closest "maxISLNum" satellites 
            #convert to set as thats what connectedToPlayers works with 
            satKey.connectedToPlayers = set(closest[satKey]) 

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
        #the 3 is from # dimensions in xyz 
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
    
        sat1.connectToPlayer(sat2, polarRegionRestriction, 0, self.sunLocation)
        sat2.connectToPlayer(sat1, polarRegionRestriction, 0, self.sunLocation)

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
            #print(str(packetInd)+"/"+str(numberPackets))
        
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
        holdLinks = np.zeros([self.numLEOs*maxNumLinksPerSat + len(self.baseStations)*self.numLEOs, 2, 3])
        #o also, get the index for adding links 
        ind = 0 
        
        #iterate through our satellites 
        for fromSat in np.ravel(self.sats): 
            #iterate through each satellites connections
            for player in fromSat.connectedToPlayers: 
                #output link if valid  
                holdLinks[ind] = np.concatenate([[fromSat.getCoords()], [player.getCoords()]], axis = 0)  
                ind+=1 
        
        for fromBaseStation in self.baseStations:
            for toPlayer in fromBaseStation.connectedToPlayers: 
                holdLinks[ind] = np.concatenate([[fromBaseStation.getCoords()], [toPlayer.getCoords()]], axis = 0)
                ind+=1               
        
        return holdLinks, ind 
    
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

    def generateSatellites(self, walkerPoints, normVecs): 
        """
        Just storing satellites when given walker constellation points

        Input: 
        walkerPoints: constellation points
        normVecs: normal vectors for each satellite. each satellite will have the same 
        normal vector as all those in its plane, as one normal vector determines a 
        circular path  
        
        Effect: sets up our satellite internals using walkerPoints 
        """
        #iterate through planes and then sats within a plane 
        for planeInd in range(self.numPlanes): 
            for smallSatInd in range(self.numSatPerPlane):
                #initialize a satellite each time  
                self.sats[planeInd, smallSatInd] = LEO(*(walkerPoints[planeInd,smallSatInd]),
                                                        planeInd, 
                                                        smallSatInd,
                                                        normVecs[planeInd]) 
                
    def connectBaseStationsToSatellites(self): 
        """
        Just connects base station to all satellites within each of their field of views

        Input: 
        Output: 
        """
        #for each base station
        ind = 0
        for baseStation in self.baseStations: 
            #for each satellite 
            #(the ravel function just compresses into one dimension)
            for satellite in np.ravel(self.sats): 
                #check if we are in field of view 
                if(baseStation.isSatelliteInView(satellite.getCoords())): 
                                   
                    #if we are, connect them both.
                    #ignore the polar constraint for this type of connection 
                    baseStation.connectToPlayer(satellite, False, self.sunExclusionAngle, self.sunLocation)
                    satellite.connectToPlayer(baseStation, False, self.sunExclusionAngle, self.sunLocation)
            
            
        #for satellite in np.ravel(self.sats): 
        #    if(len(satellite.connectedToPlayers) != 0): 
        #        print(len(satellite.connectedToPlayers))
            #not connecting base stations to each other currently. 

            #for baseStationTo in self.baseStations:  
            #    if(baseStation != baseStationTo): 
            #        baseStation.connectToPlayer(baseStation, False, self.sunExclusionAngle, self.sunLocation)
                    #baseStation.connectedBaseStations = baseStation.connectedBaseStations + [baseStationTo]
        


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
                #self.createISL(self.sats[planeInd, smallSatInd], self.sats[planeInd, (smallSatInd + 1) % self.numSatPerPlane], False)
                
                self.sats[planeInd, smallSatInd].connectToPlayer(self.sats[planeInd, (smallSatInd + 1) % self.numSatPerPlane], False)
                self.sats[planeInd,  (smallSatInd + 1) % self.numSatPerPlane].connectToPlayer(self.sats[planeInd, smallSatInd], False)

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
                    self.sats[planeInd, smallSatInd].connectToPlayer(satOfPrevPlane, True, self.sunExclusionAngle, self.sunLocation)
                
                #for back case: 
                else: 
                    satOfNextPlane = self.sats[(planeInd - 1) % self.numPlanes, smallSatInd]
                    self.sats[planeInd, smallSatInd].connectToPlayer(satOfNextPlane, True, self.sunExclusionAngle, self.sunLocation)  

    def generateAdjacencyMatrix(self): 
        """
        Hello :D 
        This function returns an adajcency matrix based upon the current satellite positions
        as well as the actual connected satellites (with ISLs) 
        Please note, this assumes each one has a line of path sight to each connected satellite. 

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
            for playerTo in playerFrom.connectedToPlayers:
                #only use the connections if its valid  
                adjMat[playerFrom.index, playerTo.index] = self.propDelayBetween(playerFrom, playerTo)
        
        #set diagonal to 0
        for playerInd in range(len(adjMat)): 
            adjMat[playerInd, playerInd] = 0

        self.currAdjMat = adjMat

        return adjMat

