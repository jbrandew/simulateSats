import numpy as np
import myPackages.myRandom as myRandom 
import math 
import pdb
import heapq
import sys
#hello! this package works with implementing custom math functions i need
#some of the functions (such as the Walker one) may be more implementation 
#focused 

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

def checkBeyondPolarLat(x, y, z, latBound): 
    """
    Return 1 if above the lat 
    Return 0 if between 
    Return -1 if below the lat 
    """
    
    lat, _, _ = cartesian_to_geodetic(x,y,z,6371) 
    
    if(latBound > 70): 
        return 1 
    if(latBound < -70): 
        return -1
    return 0 

def calculate_spherical_differences(point1, point2):
    r1, theta1, phi1 = point1
    r2, theta2, phi2 = point2

    delta_r = r2 - r1
    delta_theta = theta2 - theta1
    delta_phi = phi2 - phi1

    return np.round([delta_r, delta_theta, delta_phi], decimals=2)

def xyz_to_lat_long(x, y, z):
    """
    Function to convert 3D space input to lat long 

    input: x,y,z 
    output: lat, long 

    """
    # Calculate longitude
    longitude = math.atan2(y, x)
    
    # Calculate hypotenuse in the xy plane
    r_xy = math.sqrt(x**2 + y**2)
    
    # Calculate latitude
    latitude = math.atan2(z, r_xy)
    
    # Convert radians to degrees
    latitude = math.degrees(latitude)
    longitude = math.degrees(longitude)
    
    return latitude, longitude

def generatePoissonInterarrivalTimes(numTimes, rate): 
    """
    Basic function to return a number of interarrival times based upon an input rate 

    Inputs: 
    numTimes = number of inter arrival times we are working with 
    rate = the rate of the poisson process 
    """

    #create random probabilities 
    poissonProbs = np.random.uniform(0, 1, numTimes)
    #use inverse cdf function to generate the inter arrival time  
    interArrivalTimes = np.log(1-poissonProbs)/(-rate)

    return interArrivalTimes
    
def rotate_3d_points(points, angles):
    """
    This function takes a set of points in 3d space 
    (row for the point number) and rotates in 3 dimensions 
    by 3 respective angles 

    points: the points we are rotating (still in xyz format) 
    angles: the angles about each axis we are rotating. When we say "about" 
    we really just mean rotating around...range for each is 0 to 2Pi 

    """
    # Convert angles to radians
    angles_rad = np.radians(angles)
    
    # Define rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles_rad[0]), -np.sin(angles_rad[0])],
                   [0, np.sin(angles_rad[0]), np.cos(angles_rad[0])]])

    Ry = np.array([[np.cos(angles_rad[1]), 0, np.sin(angles_rad[1])],
                   [0, 1, 0],
                   [-np.sin(angles_rad[1]), 0, np.cos(angles_rad[1])]])

    Rz = np.array([[np.cos(angles_rad[2]), -np.sin(angles_rad[2]), 0],
                   [np.sin(angles_rad[2]), np.cos(angles_rad[2]), 0],
                   [0, 0, 1]])

    # Apply rotations
    
    rotated_points = np.dot(Rz, np.dot(Ry, np.dot(Rx, points)))

    return rotated_points


def dist3d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    
    return distance

def generateWalkerStarConstellationPoints(
        numSatellites, 
        inclination,
        numPlanes, 
        phaseParameter, 
        altitude): 
    """
    This method generates as many points as needed for the walker star constellation.
    
    It will use the rotate_3d_points function as well. 

    numSatellites: number of satellites we are working with 
    inclination: it really can be with reference to any axis since we are equally 
    distributing the planes, but we will use the x axis 
    numPlanes: number of circles over which we distribute the satellites
    phaseParameter: parameter for calculating the phase difference for 
    planes. You get it from pP*360/t, where t = num satellites  
    altitude: how far above the earth our satellite is  

    Im very confused by phaseParameter. If (pP*360/t)*num planes != 360, I thought
    we had 2 adjacent planes offset by alot -> we dont. Its fine. Because each plane 
    is periodic, adjacent planes will match up with angle period of 360/(t/numplanes)
    """

    #alright, now that we have intro done, lets work with the calculation 
    numSatellitesPerPlane = numSatellites/numPlanes 
    if(int(numSatellitesPerPlane) - numSatellitesPerPlane != 0): 
        raise Exception("Should have int # satellites per plane") 
    
    #convert to integer for future use 
    numSatellitesPerPlane = int(numSatellitesPerPlane)
    
    #after we have the number of satellties per plane we are working with, 
    #first get the base set of points, to do this, we use the
    #numSatellitesPerPlane number with a linearly spaced angle, and spherical
    #coordinates while the z is kept constant
      
    #vary the angle from 0 to 2*Pi for full circle 
    #get the number of points + 1 without the end so that start != end 
    phi = np.linspace(0, 2 * np.pi, numSatellitesPerPlane+1)[:-1] 
    
    #have the distance from core for radius of circle 
    #this is just altitude for now...change it later 
    distFromCore = altitude 

    #calculate the basic set of points using spherical coordinates 
    basePoints = [distFromCore*np.cos(phi), distFromCore*np.sin(phi), 0*phi] 

    #create storage for all points
    # storage will be: numPlanes, numPointsPerPlane ^ 3 
    bigStore = np.ones([numPlanes, numSatellitesPerPlane, 3])

    #in this loop, for each plane we are going to do 3 rotations. 
    # 1. rotate about the z axis for the phasing parameter angle result 
    # 2. rotate about the y axis for the inclination angle 
    # 3. rotate obout z axis again for "general rotation angle" (difference of planes)
    for planeInd in range(numPlanes): 

        #first, get our deep copy of the basePoints set 
        basePointsCopy = myRandom.deepCopy(basePoints) 

        #after we have a deep copy, then follow through with rotations 
        #please note, these rotations do not directly translate to spherical 
        #coordinate system angles 
        
        #first, rotate for phasing parameter
        zRotateForPhasing = planeInd*phaseParameter*360/numSatellites
        basePointsCopy = rotate_3d_points(basePointsCopy, [0,0,zRotateForPhasing])
        
        #then, rotate for inclination and general plane rotation 
        zRotateAngle = planeInd*360/numPlanes
        yRotateAngle = inclination
        xRotateAngle = 0
        basePointsCopy = rotate_3d_points(basePointsCopy, [xRotateAngle,yRotateAngle,zRotateAngle])

        bigStore[planeInd] = basePointsCopy.T

    return bigStore 

def calculate_angle(point1, point2, point3):
    vector1 = [point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2]]
    vector2 = [point3[0] - point1[0], point3[1] - point1[1], point3[2] - point1[2]]

    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a**2 for a in vector1))
    magnitude2 = math.sqrt(sum(a**2 for a in vector2))

    cos_theta = dot_product / (magnitude1 * magnitude2)
    angle_radians = math.acos(cos_theta)
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

def compute_triangle_angles(point1, point2, point3):
    
    angle1 = calculate_angle(point2, point1, point3)
    angle2 = calculate_angle(point1, point2, point3)
    angle3 = calculate_angle(point1, point3, point2)

    return angle1, angle2, angle3

def geodetic_to_cartesian(latitude, longitude, radius=6371.0):
    # Convert latitude and longitude from degrees to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)

    # Calculate Cartesian coordinates
    x = radius * math.cos(lat_rad) * math.cos(lon_rad)
    y = radius * math.cos(lat_rad) * math.sin(lon_rad)
    z = radius * math.sin(lat_rad)

    return x, y, z

def cartesian_to_geodetic(x, y, z, radius):
    # Calculate longitude
    longitude = math.atan2(y, x)

    # Calculate hypotenuse in XY plane
    xy_hypotenuse = math.sqrt(x**2 + y**2)

    # Calculate latitude
    latitude = math.atan2(z, xy_hypotenuse)

    # Calculate altitude
    altitude = math.sqrt(x**2 + y**2 + z**2) - radius

    # Convert latitude and longitude from radians to degrees
    latitude = math.degrees(latitude)
    longitude = math.degrees(longitude)

    return latitude, longitude, altitude



#alright, i just want to plot a cone :D 
#to do that, need to use the parametric equations. 
#x = rcos(theta)
#y = rsin(theta) 
#z = f(r), can be anything. in this case it will be z = r as we use a linear cone 
#please note that z = r corresponds to the minimum elevation angle = 45 

#so how to do this? create 2d grid of theta and r values
#just duplicate z values (z remains same as theta varies) 
def getConeCoords(): 
    #getting bounds for the two inputs to the parametric equations 
    #theta always goes from 0 to 2*pi 
    theta = np.linspace(0, 2*np.pi, 100) 
    #so max height will be z(highest r) probably
    r = np.linspace(0,10,100) 
    
    #create a meshgrid from these two 
    thetaMesh, rMesh = np.meshgrid(theta,r, indexing = 'ij')

    #theta is constant on x
    #r is constant on y

    #create storage for x y z  
    x = np.ones(np.shape(thetaMesh))
    y = np.ones(np.shape(thetaMesh))
    z = np.ones(np.shape(thetaMesh))

    for i in range(np.shape(thetaMesh)[0]): 
        for j in range(np.shape(rMesh)[0]): 
            x[i,j] = rMesh[i,j]*np.cos(thetaMesh[i,j])
            y[i,j]  = rMesh[i,j]*np.sin(thetaMesh[i,j])
            #with z = r we have a 45 degree cone opening 
            z[i,j]  = rMesh[i,j]

    return x, y, z 



def dijkstra(adj_matrix, start, end):
    n = len(adj_matrix)
    visited = [False] * n
    distances = [float('inf')] * n
    distances[start] = 0
    heap = [(0, start)]

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        if visited[current_node]:
            continue

        visited[current_node] = True

        for neighbor, weight in enumerate(adj_matrix[current_node]):
            if not visited[neighbor]:
                new_distance = distances[current_node] + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(heap, (new_distance, neighbor))

    return distances[end]

def dijkstraWithPath(adj_matrix, start, end):
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    distances = [sys.maxsize] * num_nodes
    parent = [-1] * num_nodes

    distances[start] = 0

    for _ in range(num_nodes):
        min_distance = sys.maxsize
        min_index = -1

        for node in range(num_nodes):
            if not visited[node] and distances[node] < min_distance:
                min_distance = distances[node]
                min_index = node

        visited[min_index] = True

        for node in range(num_nodes):
            if not visited[node] and adj_matrix[min_index][node] >= 0:
                new_distance = distances[min_index] + adj_matrix[min_index][node]
                if new_distance < distances[node]:
                    distances[node] = new_distance
                    parent[node] = min_index

    # Reconstruct the path
    path = []
    current_node = end
    while current_node != -1:
        path.insert(0, current_node)
        current_node = parent[current_node]

    return path, distances[end]

def dijkstraAgain(graph, start):
    num_nodes = len(graph)
    distances = np.full(num_nodes, np.inf)
    visited = np.zeros(num_nodes, dtype=bool)
    distances[start] = 0

    for _ in range(num_nodes):
        current_node = np.argmin(distances * ~visited)
        visited[current_node] = True

        for neighbor, weight in enumerate(graph[current_node]):
            if not visited[neighbor] and distances[current_node] + weight < distances[neighbor]:
                distances[neighbor] = distances[current_node] + weight

    return distances




# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph
class Graph():
 
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
 
    def printSolution(self, dist):
        print("Vertex \t Distance from Source")
        for node in range(self.V):
            print(node, "\t\t", dist[node])
 
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, sptSet):
 
        # Initialize minimum distance for next node
        min = 1e7
 
        # Search not nearest vertex not in the
        # shortest path tree
        pdb.set_trace() 
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
 
        return min_index
 
    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):
 
        dist = [1e7] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
 
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)
 
            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[u] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
                if (self.graph[u][v] > 0 and
                   sptSet[v] == False and
                   dist[v] > dist[u] + self.graph[u][v]):
                    dist[v] = dist[u] + self.graph[u][v]
 
        self.printSolution(dist)