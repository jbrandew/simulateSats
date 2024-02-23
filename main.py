#Hello :D 

import yaml 
import myClasses.myPlots as myPlots
import myPackages.myMath as myMath 
import myPackages.myRandom as myRandom 
import pdb 
import myClasses.Simulator as Simulator
import myClasses.Manager as Manager

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

#connect a certain topology 
#lemmeManage.connectSpiralTopologySimple() 

#get simulator 
simmer = Simulator.Simulator(managerData)
#simmer.manager.connectSpiralTopologySimple() 
simmer.manager.updateTopology("Retain", "InView")

#avLength, avgTime, _ = simmer.simulateTransmits(50)
#pdb.set_trace()

#lemmeManage.connect2ISL()
#simmer.manager.updateTopology("Closest", "InView")
simmer.multiPlot() 
exit() 

print("here")
simmer.manager.executeChainSimulation(1,1,10)

#simmer.timeFrameSequencing(15, 10, 20)

#now, examining those in view 
#view = simmer.getSatsAndPlanesInViewOfBaseStation()

#also, plot the connections and so forth 
#simmer.multiPlot() 

#get adjacency matrix 
#adjMat = lemmeManage.generateAdjacencyMatrix() 


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