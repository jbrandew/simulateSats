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
    "earthRadius": configData['earthRadius'],
    "initialTopology": configData['initialTopology'],
    "routingPolicy": configData['routingPolicy'],
    "topologyPolicy": configData['topologyPolicy'],
}

#simulator args 
simulationArgs = {
    "numPeople": configData['numPeople'],
    "numPacketsPerPerson": configData['numPacketsPerPerson'],
    "packetSendTimeFrame": configData['packetSendTimeFrame'],
    "queingDelaysEnabled": configData['queingDelaysEnabled'],
    "weatherEnabled": configData['weatherEnabled'],
    "environmentUpdateInterval": configData['environmentUpdateInterval'],
    "outageFrequency": configData['outageFrequency'],
    "timeFactor": configData['timeFactor']
}

#visualizer args 
visualizerArgs = {
    "visualizerOn": configData['visualizerOn'],
    "visualizeTime": configData['visualizeTime'],
    "FPS": configData['FPS'],

}

#get simulator 
simmer = Simulator.Simulator(managerData)

#connect satellites 
simmer.manager.connectSpiralTopologySimple(ISL2Done=False) 

#then enact simulation and visualize the data 
hold = simmer.simulateWithVisualizer(simulationArgs, visualizerArgs)

quit()


#connect base stations to satellites in view 
#simmer.manager.connectBaseStationsToSatellites() 

#optionally plot the configuration for debugging 
#simmer.plotCurrentState() 

#simmer.manager.updateTopology("Closest", "None")
#simmer.timeFrameSequencing(15, 10, 20)

#pdb.set_trace() 

#so then, test the generalSimulationMethod
#hold = simmer.executeGeneralSimulation()












#simmer.manager.updateTopology("Retain", "InView")

#avLength, avgTime, _ = simmer.simulateTransmits(100)
#print("Non Chained Path Sim Time")
#print(avgTime)
#pdb.set_trace() 

#lemmeManage.connect2ISL()
#simmer.manager.updateTopology("Closest", "None")#could be InView not None
#simmer.multiPlot() 
#exit() ''[] 

#hold = simmer.manager.executeChainSimulationWithCollisions(100,1,0,True)
#print("Chained Simulation Time")
#print(np.average(hold))
#pdb.set_trace() 
#exit() 
#simmer.timeFrameSequencing(15, 10, 20)

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