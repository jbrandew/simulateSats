#Hello :D 

import yaml 
import myClasses.myPlots as myPlots
import myPackages.myMath as myMath 
import myPackages.myRandom as myRandom 
import pdb 
import myClasses.Simulator as Simulator
import myClasses.Manager as Manager

#just for now
import matplotlib.pyplot as plt

#for profiling 
import profile

#just need a little bit of computation...just a little bit 
import numpy as np 

#for setting the seed to be the same across all calls
import random

#read in the config file 
with open("environmentConfig.yaml", "r") as stream: 
    #attempt to read out 
    try: 
        configData = yaml.safe_load(stream)
    except yaml.YAMLError as exc: 
        print(exc) 

#set constant seed
random.seed(1)

#format config data for generating constellation
constellationConfig = [
    configData['NumberLEOSatellites'],
    configData['Inclination'],
    configData['OrbitalPlanes'],
    configData['phasingParameter'], 
    configData['orbitAltitudeLEO'] +  configData['earthRadius']    
]

#format data for getting manager 
#manager manages and creates satellites/base stations 
managerData = {
    "constellationType": configData["LEOConstellation"],
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
    "packetProcessRate": configData['packetProcessRate'],
    "dynamicLocation": configData['dynamicLocation'],
    "RLtraining": configData['RLtraining']
}

#simulator args. so environment and stuff 
#there is some overlap, like the routing policy, because the environment needs to know how to create evnents 
simulationArgs = {
    "numPeople": configData['numPeople'],
    "numPacketsPerPerson": configData['numPacketsPerPerson'],
    "packetSendTimeFrame": configData['packetSendTimeFrame'],
    "personDistribution" : configData['personDistribution'],
    "queingDelaysEnabled": configData['queingDelaysEnabled'],
    "weatherEnabled": configData['weatherEnabled'],
    "environmentUpdateInterval": configData['environmentUpdateInterval'],
    "outageFrequency": configData['outageFrequency'],
    "timeFactor": configData['timeFactor'],
    "routingPolicy": configData['routingPolicy'],
    "sendTimeDistribution": configData['sendTimeDistribution']
}

#visualizer args 
visualizerArgs = {
    "visualizerOn": configData['visualizerOn'],
    "visualizeTime": configData['visualizeTime'],
    "FPS": configData['FPS'],

}

#set up RL config: 
num_episodes = 20

#get simulator object 
simmer = Simulator.Simulator(managerData)
#hold = simmer.simulateWithVisualizer(simulationArgs, visualizerArgs)


#iterate through episodes  
for i_episode in range(num_episodes):

    #get latency each time 
    holdLatencyTimes = simmer.executeGeneralSimulation(**simulationArgs)

    #reset the state of the simulator
    simmer.resetWorldState()

    print("Average latency for packets for this episode:")
    print(np.average(holdLatencyTimes))
    print("Episode # ")
    print(i_episode)

#after the episodes, examine the data 
crossEpsLoss = simmer.manager.sats[0][0].agent.crossEpsLoss
crossEpsAction = simmer.manager.sats[0][0].agent.crossEpsAction

#store the action distribution 
#shape of numEpisodes by numPossibleActions 
numPossibleActions = len(crossEpsAction[0][0])
actionDistributionStorage = np.ones([num_episodes, numPossibleActions])

#get action distribution for going to each satellite
#so, for each action set 
for actionSetInd, actionSet in enumerate(crossEpsAction): 
    #get the num of each action 
    numTimesSentTo1 = actionSet[1][0]
    numTimesSentTo3 = actionSet[1][1]

    actionDistributionStorage[actionSetInd, 0] = numTimesSentTo1
    actionDistributionStorage[actionSetInd, 1] = numTimesSentTo3


#then, after we have the stored distribution, plot it over the episode number 
fig, ax = plt.subplots()
action1, = ax.plot(np.arange(num_episodes), actionDistributionStorage[:,0], label = '# Packets Directed to Server 2')
action2, = ax.plot(np.arange(num_episodes), actionDistributionStorage[:,1], label = '# Packets Directed to Server 1')
ax.set(xlabel='Episode #', ylabel='# of Packets Sent to one direction',
       title='Action Distribution over Episode. Each episode involves transmission of 500 packets.')
ax.legend(handles=[action1, action2])
plt.show()

#simply plot the loss over time
fig, ax = plt.subplots()
ax.plot(np.arange(num_episodes), crossEpsLoss)
ax.set(xlabel='Episode #', ylabel='Loss',
       title='Loss over Episode')
plt.show()

pdb.set_trace() 
simmer.manager.sats[0]



#enact simulation 
#hold = simmer.simulateWithVisualizer(simulationArgs, visualizerArgs)

quit()

#simmer.manager.updateTopology("Closest", "inView")
#simmer.manager.connect2ISL() 


#do simple plot
#simmer.plotCurrentState() 
#simmer.timeFrameSequencing(15, 10, 20)

#profile.run('simmer.simulateWithVisualizer(simulationArgs, visualizerArgs)')

#then enact simulation and visualize the data 




#connect base stations to satellites in view 
#simmer.manager.connectBaseStationsToSatellites() 

#optionally plot the configuration for debugging 
#

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