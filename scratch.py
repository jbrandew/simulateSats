

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

