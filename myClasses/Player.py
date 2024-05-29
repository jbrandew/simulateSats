import math 
import myClasses.myPlots as myPlots
import myPackages.myMath as myMath 
import numpy as np 
import pdb 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import heapq

import random

class PriorityQueue:
    """
    could make this more efficent using the min heap implementation
    below. need to test that against this since that is more likely 
    to be wrong due to increased complexity 
    """
    def __init__(self):
        self.queue = []

    def push(self, item):
        self.queue.append(item)
        self.queue.sort(key=lambda x: x.timeOfOccurence)

    def pop(self):
        if not self.is_empty():
            return self.queue.pop(0)
        else:
            print("Priority queue is empty.")

    def is_empty(self):
        return len(self.queue) == 0

    def peek(self):
        if not self.is_empty():
            return self.queue[0]
        else:
            print("Priority queue is empty.")

    def size(self):
        return len(self.queue)


class PacketState: 

    """
    Describes what the packet is actually doing 
    """
    def __init__(self, initState): 
        self.allowedValues = {'Dormant', 
                                'InTransmit', 
                                'InProcessQueue',
                                'InTransmitQueue',
                                'Finished'}
        if initState not in self.allowedValues: 
            raise ValueError("Bad Packet State Value") 
class Packet(): 
    """
    This class is separate functionally from the "Player" set, but 
    this seemed like the best place to put it. It really just 
    represents the lifecycle of a packet. 

    """
    def __init__(self, **kwargs): 
        """
        Initialization of packet

        Inputs: 

        startLocation: xyz of the packet for when it is sent out 
        endLocation: xyz of location for packet to arrive at 
        scheduledAwakeTime: when the packet is supposed to be sent/start its path
        pathToTake: the series of terminals it travels over to get to its 
        destination. This is currently assigned when the packet reaches the time
        of "awake," so its non adaptable  
        """

        for key, value in kwargs.items():
            setattr(self, key, value)

        #packet is dormant first 
        self.currentPhase = PacketState('Dormant')

        return 
    
#player class describes any operator on or above the earth 
#probably going to use rho and phi more often than purely longitude and latitude 
#nah lets just always use x,y,z...maybe easier in polar all the time...worry later

#possible error from removing () after "Player"
class Player: 
    """
    This class describes any operator that interacts via ISLs (ISLs also include satellite to 
    base station connections)

    """
    
    def __init__(self, xIn = 0, yIn = 0, zIn = 0): 
        self.x, self.y, self.z = xIn, yIn, zIn 
        #create storage for who we are connected to 
        self.connectedToPlayers = set() 

        #this represents when the server will be done processing
        #all packets. Note: may be in the past  
        self.finishProcessingTime= -1  
        
        #this represents the processing rate. 
        self.processRate = 100

    def generateProcessingOneMorePacketTime(self, timeRequested, packetCollsionEnabled = True): 
        """
        What is this function doing? Its adding one more packet to the
        queue of this server, as well as generating the time when it 
        will be done with the requested packet. 

        Inputs: 
        timeRequested: when the finish processing time was requested
        packetCollision: if we are having packets create collision at satellites 
        
        Outputs: 
        timeFinished: when the player/server will finish the processing
        of the given packet 
        """

        #first, get the basic case of how long it takes to process
        #assuming exp. distributed interarrival time 
        #(arrival meaning processing this packet) 
        
        #first, generate point on the uniform distribution 
        # Generate a random number between 0 and 1
        random_probability = random.random()

        #then using the reverse solved CDF of the exp. interarr. time
        interArrivalTime = -np.log(1 - random_probability)/(self.processRate)


        #in no collision case, just return normal int. time 
        if not packetCollsionEnabled: 
            self.finishProcessingTime = interArrivalTime + timeRequested

        else: 
            #two cases: 
            #case one: server is not servicing any packets: 
            if(timeRequested > self.finishProcessingTime): 
                self.finishProcessingTime = interArrivalTime + timeRequested

            #case two: server is servicing packets: 
            if(timeRequested <= self.finishProcessingTime): 
                self.finishProcessingTime = interArrivalTime + self.finishProcessingTime

        #after updating our personal finish processing time, return it 
        #for the packet events use :) 
        return self.finishProcessingTime

    def getCoords(self): 
        return self.x, self.y, self.z 
    
    def resetConnections(self): 
        self.connectedToPlayers = set() 

    def connectToPlayer(self, 
                        playerToConnectTo, 
                        polarRegionRestriction = True, 
                        sunExclusionAngle = 0,
                        sunLocation = [0,0,0]):
        """
        Function for connecting to a different player. This is one directional. 

        Inputs: 
        playerToConnectTo: who you are forming the comm link with 
        polarRegionRestriction: are we restricting what region we can form this connection with
        sunExclusionAngle: what min angle needed between LOS path and sun to connect 
        
        Output: 
        boolean saying if the connection was formed or not

        Effect: 
        Connection formed between players 
        """ 
        
        #if we have regional restriction for connection 
        #if(False): 
        if(polarRegionRestriction): 
            lats = myMath.cartesian_to_geodetic(*playerToConnectTo.getCoords(), 6371) 
            
            if(lats[0] > 70 or lats[0] < -70): 
                return False 

            selfLats = myMath.cartesian_to_geodetic(*self.getCoords(), 6371) 
            
            if(selfLats[0] > 70 or selfLats[0] < -70): 
                return False 

        #first, get the vector of this vector to sun 
        toSunVec = np.subtract(np.array(self.getCoords()), np.array(sunLocation))
        toSatVec = np.subtract(np.array(self.getCoords()), np.array(playerToConnectTo.getCoords()))

        #get difference in angle: 
        angle = myMath.angle_between_vectors(toSunVec, toSatVec) 
        #if we are below that angle 
        if(angle < sunExclusionAngle):
            #then dont make connection
            #note, in this system where we have the polar region restriction usually, 
            #you dont really interfere with this all that often   
            return False 

        self.connectedToPlayers.add(playerToConnectTo)# = self.connectedToPlayers + [playerToConnectTo]
        #playerToConnectTo.connectedToPlayers.add(self)

        return True

#LEO (low earth orbit) describes the lowest orbiting set of satellites 
class LEO(Player): 

    def __init__(self, xIn = 0, yIn = 0, zIn = 0, planeIndex = 0, subSatIndex = 0, normal_vector = []):
        """
        Init function for LEO.
        Just pass off coordinates to parent "Player" 

        xIn, yIn, zIn: coordinates in 3d space 
        planeIndex: number plane we are in in walker star 
        subSatIndex: number satellite we are in in a single plane 
        normal_vector: vector determining the circular path of the satellite around the earth 
        
        """
        super().__init__(xIn, yIn, zIn)

        #store indices 
        self.planeIndex = planeIndex
        
        self.subSatIndex = subSatIndex

        #number links we are allowed to have
        #not sure if we will use this  
        self.numberISLlinksAllowed = 2  

        #satellites we are connected to 
        #self.connectedSats = [] 

        #base stations we are connected to
        #base station itself will probably do the connecting/disconnecting  
        #self.connectedBaseStations = []

        #store the normal vector determining our circular path 
        self.normVec = normal_vector

        #calculate orbital period (just to store, as it wont change over time)
        self.orbitPeriod = myMath.satelliteOrbitalPeriod(self.x, self.y, self.z)

    def distance(self, other):
        # Your distance calculation logic
        selfCoords = self.getCoords()
        otherCoords = other.getCoords() 
        return myMath.distance(selfCoords, otherCoords)

    def __lt__(self, other):
        # Implement less-than comparison based on some criterion (e.g., distance)
        return self.distance(other) < other.distance(self)
    

    def updatePosition(self, timeDiff): 
        """
        Based on current position and time diff,
        set your next position

        Inputs:
        timeDiff = satTimeDiff between positions

        Effect: 
        updates our position
        
        """
        #self.x, self.y, self.z = myMath.calculate_next_position([self.x, self.y, self.z], timeDiff)
        
        #get angle diff from two positions 
        angleRadDiff = 2*np.pi*timeDiff/self.orbitPeriod

        #calculate new position
        self.x, self.y, self.z = myMath.calculate_new_position(self.normVec, 
                                                               [self.x, self.y, self.z], 
                                                               angleRadDiff)

      


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
        #self.connectedSats = [] 

        #self.connectedBaseStations = []

    # def connectToSat(self, satToConnectTo): 
    #     """
    #     Just connect to a given satellite using your internals

    #     satToConnectTo: ... 
    #     """
    #     self.connectedSats = self.connectedSats + [satToConnectTo]    

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
    
class PriorityQueue:
    #use heaps/heapify here because we are usually working with a nearly sorted array 
    def __init__(self):
        self.queue = []

    def push(self, item):
        self.queue.append(item)
        self._heapify_up(len(self.queue) - 1)

    def pop(self):
        if not self.is_empty():
            if len(self.queue) == 1:
                return self.queue.pop()
            else:
                min_item = self.queue[0]
                self.queue[0] = self.queue.pop()
                self._heapify_down(0)
                return min_item
        else:
            print("Priority queue is empty.")

    def is_empty(self):
        return len(self.queue) == 0

    def peek(self):
        if not self.is_empty():
            return self.queue[0]
        else:
            print("Priority queue is empty.")

    def size(self):
        return len(self.queue)

    def _heapify_up(self, index):
        parent_index = (index - 1) // 2
        if parent_index >= 0 and self.queue[parent_index].timeOfOccurence > self.queue[index].timeOfOccurence:
            self.queue[parent_index], self.queue[index] = self.queue[index], self.queue[parent_index]
            self._heapify_up(parent_index)

    def _heapify_down(self, index):
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        smallest = index

        if left_child_index < len(self.queue) and self.queue[left_child_index].timeOfOccurence < self.queue[smallest].timeOfOccurence:
            smallest = left_child_index

        if right_child_index < len(self.queue) and self.queue[right_child_index].timeOfOccurence < self.queue[smallest].timeOfOccurence:
            smallest = right_child_index

        if smallest != index:
            self.queue[index], self.queue[smallest] = self.queue[smallest], self.queue[index]
            self._heapify_down(smallest)
