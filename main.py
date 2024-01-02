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

print("Num Satellites in View: " + str(numInView)) 
print("Num Disc. Planes in View: " + str(numOrbitalPlanes - numOrbitInView))

#now, need to pass in the list of locations in x y z of base stations 
BSLocHold = np.ones([2, 3])
BSLocHold[0] = myMath.geodetic_to_cartesian(82.5, -62.3)
BSLocHold[1] = myMath.geodetic_to_cartesian(70.4, 31.1)

#lets look at the connections of the satellites. how do we do this?
lemmeManage = myClasses.Manager(walkerPoints, BSLocHold, 10, phaseParameter)
lemmeManage.connect2ISL() 
lemmeManage.connectBaseStationsToSatellites() 

#in 2ISL with 2 base stations, max # of connection per = 4 
linkCoords = lemmeManage.getXYZofLinks(6) 
#pdb.set_trace() 
#myPlots.plotWalkerStar(walkerPoints) 
#myPlots.multiPlot(earthRadius/100, walkerPoints, BSLocHold, linkCoords)

#exit() 

adjMat = lemmeManage.generateAdjacencyMatrix() 
#pdb.set_trace() 

#after we get adjacency matrix, analyze use time: 
#for satellites vs base stations 
#for satellites closer vs farther to base station 

#plt.show()

#pdb.set_trace()

#now that we have adj matrix computed, get mean latency with monte carlo
#sampled base station and users 
#in order to do that, get 2 random num in length of numLeos

numTrials = 1000
#2 = num base stations
setOftoAndFro = np.random.randint(0,numLEOs + 2, size =(numTrials, 2))
totalTime = 0
ind = 0 
maxTime = 0
totalLength = 0 
for set in setOftoAndFro: 
    
    ind+=1
    time = myMath.dijkstra(adjMat, set[0], set[1])
    totalTime+=time
    print(ind)
    print(time)
    if(maxTime < time): 
        maxTime = time

    #totalLength+=len(path)
    #print(totalTime)

avgTime = totalTime/numTrials 
print("Time in ms: "+ str(avgTime*1000))
print("Max Time: "+str(maxTime))
#print("Average Length: "+str(totalLength/numTrials))


#pseudoCapacity = lemmeManage.getPseudoCapacity(1000)

# with open('pseudoCapacity.npy', 'rb') as f:

#     a = np.load(f)

def plotUsageAnalysis(): 
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


