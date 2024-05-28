#Hello :D 

import yaml 
import myClasses.myPlots as myPlots
import myPackages.myMath as myMath 
import myPackages.myRandom as myRandom 
import pdb 
import myClasses.Simulator as Simulator
import myClasses.Manager as Manager

#just need a little bit of computation...just a little bit 
import numpy as np 

#read in the config file 
with open("environmentConfig.yaml", "r") as stream: 
    #attempt to read out 
    try: 
        configData = yaml.safe_load(stream)
    except yaml.YAMLError as exc: 
        print(exc) 

#now that we have the coordinates for all the satellites, lets proceed. 
#first verify by using the two relays set up and looking at # sats in view per relay
#to do that, first create two relays 

#format config data for generating constellation
constellationConfig = [
    configData['NumberLEOSatellites'],
    configData['Inclination'],
    configData['OrbitalPlanes'],
    configData['phasingParameter'], 
    configData['orbitAltitudeLEO'] +  configData['earthRadius']    
]

#format data for getting manager 
managerData = {
    "constellationType": "walkerDelta",
    "constellationConfig": constellationConfig,
    "baseStationLocations": configData['BaseStationLocations'],
    "fieldOfViewAngle": configData['BaseStationViewAngle'],
    "phasingParameter": configData['phasingParameter'],
    "sunExclusionAngle": configData['SunExclusionAngle'],
    "sunLocation": configData['SunLocation'],
    "earthRadius": configData['earthRadius']
}

#get simulator 
simmer = Simulator.Simulator(managerData)

#connect satellites 
simmer.manager.connectSpiralTopologySimple(ISL2Done=False) 

#connect base stations to satellites in view 
#simmer.manager.connectBaseStationsToSatellites() 

#optionally plot the configuration for debugging 
#simmer.plotCurrentState() 

#simmer.manager.updateTopology("Closest", "None")
simmer.timeFrameSequencing(15, 10, 20)

#pdb.set_trace() 

#so then, test the generalSimulationMethod
#hold = simmer.executeGeneralSimulation()

kargs = {"initialTopology": "IPO",
"routingPolicy": None,
"topologyPolicy": None,

"numPeople": 100,
"numPacketsPerPerson": 1,
"simulationTime": 0.0001,

"queingDelaysEnabled": "False",
"weatherEnabled": "False",
"adjMatrixUpdateInterval": -100,
"outageFrequency": None,

"visualizerOn": None
}

#hold = simmer.visualizeSimulation(kargs)


#debug to examine validity
#print(hold)













#simmer.manager.updateTopology("Retain", "InView")

#avLength, avgTime, _ = simmer.simulateTransmits(100)
#print("Non Chained Path Sim Time")
#print(avgTime)
#pdb.set_trace() 

#lemmeManage.connect2ISL()
simmer.manager.updateTopology("Closest", "None")#could be InView not None
#simmer.multiPlot() 
#exit() ''[] 

#hold = simmer.manager.executeChainSimulationWithCollisions(100,1,0,True)
#print("Chained Simulation Time")
#print(np.average(hold))
#pdb.set_trace() 
#exit() 
simmer.timeFrameSequencing(15, 10, 20)

#now, examining those in view 
#view = simmer.getSatsAndPlanesInViewOfBaseStation()

#also, plot the connections and so forth 
#simmer.multiPlot() 

#get adjacency matrix 
#adjMat = lemmeManage.generateAdjacencyMatrix() 



#should probably be in the plots stuff...
#apparently utils = personal math stuff...
def plotUsageAnalysis():

    with open('pseudoCapacity.npy', 'rb') as f:

        a = np.load(f)   
    #load in usage times, for just the satellites  
    usageTimes = a[0:360]
    #then examine the dist of each satellite to the two base stations 
    avgPropTimes = lemmeManage.averageDistToBaseStation()
    #examine correlation between proximity to base station and usage of link 
    corrcoef = np.corrcoef(usageTimes[0:360], avgPropTimes) 

    plt.scatter(avgPropTimes, usageTimes) 
    plt.xlabel('Average Propagation Delay to BS')
    plt.ylabel('Pseudo Measure of Usage')
    plt.title('Simulated 1000 Transmissions')

hold = simmer.manager.sats
total = 0 
for sat in np.ravel(hold):
    if(len(sat.connectedToPlayers) != 2): 
        print(len(sat.connectedToPlayers))
    total+=len(sat.connectedToPlayers)

check, _ = simmer.manager.getXYZofLinks()
pdb.set_trace() 