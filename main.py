#Hello :D 

import yaml 
import myPackages.myPlots as myPlots
import myPackages.myMath as myMath 
import myPackages.myRandom as myRandom 
import pdb 
import myClasses.classes as myClasses

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

#create manager 
lemmeManage = myClasses.Manager(constellationType="walkerDelta",
                                constellationConfig=constellationConfig,
                                baseStationLocations = configData['BaseStationLocations'], 
                                fieldOfViewAngle = configData['BaseStationViewAngle'], 
                                phasingParemeter = configData['phasingParameter'])

#connect a certain topology 
lemmeManage.connect2ISL() 

#get simulator 
simmer = myClasses.Simulator(lemmeManage)

#now, examining those in view 
#view = simmer.getSatsAndPlanesInViewOfBaseStation()

#also, plot the connections and so forth 
#simmer.multiPlot() 

#get adjacency matrix 
adjMat = lemmeManage.generateAdjacencyMatrix() 

#simulate effect of satellite failure on path failure 
rate = simmer.simulatePathFailureFast(25, 100, 100)
print("Hello :D")
print(rate)
pdb.set_trace()


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