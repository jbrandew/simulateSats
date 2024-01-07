#Hello :D 
#today we are working on replicating the results from the paper Dr. Brinton sent out
#classes we need: super class: player. sub classes: LEO, base station. 

import yaml 
import myPackages.myPlots as myPlots
import myPackages.myMath as myMath 
import myPackages.myRandom as myRandom 
import pdb 
import myClasses.classes as myClasses

import matplotlib.pyplot as plt

#shouldnt import here, main is not for computation 
import numpy as np 

#read in the config file 
with open("environmentConfig.yaml", "r") as stream: 
    #attempt to read out 
    try: 
        configData = yaml.safe_load(stream)
    except yaml.YAMLError as exc: 
        print(exc) 

#after we get the config data, proceed with making a set of LEOs 
#first make an empty one 
numLEOs = configData['NumberLEOSatellites']

#the inclination angle is how far above the equator the orbital planes are 
inclinationAngle = configData['Inclination']

#the numOrbitalPlanes is just how many planes/orbits there are. 
numOrbitalPlanes = configData['OrbitalPlanes']

#so the degree spacing between each plane along the equator is 360/this number
degreeSpacing = numOrbitalPlanes/configData['OrbitalPlanes']

#get the phase parameter between each orbital plane 
phaseParameter = configData['phasingParameter']

#get the altitude of orbit 
altitude = configData['orbitAltitudeLEO']

#get the radius of earth 
earthRadius = configData['earthRadius']

#get the baseStationLocations and field of view 
baseStationLocations = configData['BaseStationLocations']
baseStationFieldofView = configData['BaseStationViewAngle']

#aight, now we get the coordinates for all the satellites 
walkerPoints = myMath.generateWalkerStarConstellationPoints(
    numSatellites=numLEOs,
    inclination=inclinationAngle,
    numPlanes=numOrbitalPlanes,
    phaseParameter=phaseParameter, 
    altitude=altitude + earthRadius
)

#now that we have the coordinates for all the satellites, lets proceed. 
#first verify by using the two relays set up and looking at # sats in view per relay
#to do that, first create two relays 

#use star to convert between array -> list of usable values 
relayA = myClasses.baseStation(*myMath.geodetic_to_cartesian(82.5, -62.3),30)
relayB = myClasses.baseStation(*myMath.geodetic_to_cartesian(70.4, 31.1),30)

#after making the two relays...uhhh
#we need to compute how many satellites are in view of the base station.
#in order to do so, iterate through all satellites in formation 
numInView, numOrbitInView = relayA.numSatellitesInView(walkerPoints)

#print("Num Satellites in View: " + str(numInView)) 
#print("Num Disc. Planes in View: " + str(numOrbitalPlanes - numOrbitInView))


#lets look at the connections of the satellites. how do we do this?
lemmeManage = myClasses.Manager(walkerPoints, 
                                baseStationLocations, 
                                baseStationFieldofView, 
                                phaseParameter)

lemmeManage.connectLadderTopology() 

#in 2ISL with 2 base stations, max # of connection per = 4 
linkCoords = lemmeManage.getXYZofLinks(maxNumLinksPerSat=6) 

#divide by 100 just so we dont see the sphere of the globe for now
#myPlots.multiPlot(earthRadius/100, walkerPoints, lemmeManage.getBaseStationLocations(), linkCoords)

adjMat = lemmeManage.generateAdjacencyMatrix() 

#this gives me the # rows that have a non inf value 
non_inf_rows = np.sum(~np.isinf(adjMat).any(axis=1))

#print("Number of rows wit non-infinite values:", non_inf_rows)

simmer = myClasses.Simulator(lemmeManage)

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

#plotUsageAnalysis() 

# avgLength, avgTime, maxTime = simmer.simulateTransmits(1000)
# print("Avg # of links used:" + str(avgLength))
# print("Time in ms: "+ str(avgTime))
# print("Max Time: "+str(maxTime))

rate = simmer.simulatePathFailureFromSatFailure(25, 100, 100)
print("Hello :D")
print(rate)


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



