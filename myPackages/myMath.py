import numpy as np
import myPackages.myRandom as myRandom 
import math 
import pdb
import heapq
import sys
import random 
import numpy as np
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u
from datetime import datetime, timedelta


#hello! this package works with implementing custom math functions i need
#some of the functions (such as the Walker one) may be more implementation 
#focused 

def closest_point(points, target_point):
    #return index, then actual point value 
    distances = np.linalg.norm(points - target_point, axis=1)  # Compute Euclidean distances
    closest_index = np.argmin(distances)  # Find index of the closest point
    closest_point = points[closest_index]  # Get the closest point
    return closest_index, closest_point

def generate_points_on_sphere_mostly_uniform(num_points, radius):

    #this method creates mostly uniform points across the sphere
    #surface. to have it completely uniform, would need a 
    #more computationally intense projection/monte carlo square method
    
    #note: the "intuitively" uniform method is below and is not actually uniform

    # Generate random angles for latitude and longitude
    theta = np.arccos(2 * np.random.uniform(0, 1, num_points) - 1)  # Latitude angle with cosine distribution
    phi = np.random.uniform(0, 2 * np.pi, num_points)  # Longitude angle

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Return the points as a NumPy array
    return np.column_stack((x, y, z))


def generate_points_on_sphere_not_uniform(num_points, radius):
    # Generate random angles for latitude and longitude
    theta = np.random.uniform(0, np.pi, num_points)  # Latitude angle
    phi = np.random.uniform(0, 2 * np.pi, num_points)  # Longitude angle

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Return the points as a NumPy array
    return np.column_stack((x, y, z))

def calculateCapacity(
        noisePower, 
        receivePower, 
        operatingBandwidth, 
        linkMargin = 1

): 
    """
    Just return capacity 

    noisePower: sigma^2, power of noise 
    receivePower: power of received signal 
    operatingBandwidth: bandwidth we can use 
    linkMargin: divisor to account for min above dB for power 
    """
    return operatingBandwidth*np.log2(1+noisePower/(receivePower/linkMargin)) 

def calculateReceivePowerSimple(
        transmitPower, 
        operatingWavelength, 
        linkDistance
): 
    """
    Calculate some stuff and return receive power 

    Inputs: 
    transmitPower: :D 
    operating wavelength: :D, in nm  
    linkDistance: distance between the two satellites, in km  

    Outputs: 
    receive power 
    """

    #work with path loss 
    Lps = (operatingWavelength/(4*np.pi*linkDistance))**2

    return transmitPower*Lps    

def generatePointsEvenlyAcrossSphere(numPoints): 

    return 

def calculateISLReceivePowerComplex(
        transmitPower, 
        linkDistance, 
        etaR = 1, 
        etaT = 1, 
        divAngle = 0, 
        rxDiam = 1, 
        operatingWavelength = 1550, 
        transmitPointingError = 0, 
        txPointingError = 0, 
        rxPointingError = 0

): 
    """
    Calculate the receive power of a transferred signal in an optical connection.
    We are using a model thats quite similar to the RF model. 

    Not fully implemented :D 

    Using this model: 
    https://arxiv.org/pdf/2204.13177.pdf

    Heavy descriptions of each are as follows 

    Inputs: 
    Pt: transmit power 

    etaT, etaR: optics efficiency of transmit/receive 
    -essentially a ratio for Transmit Optical power vs input power 

    GT: transmitter gain 
    -GT: 16/(divergence angle^2)
    -divergence angle is the "spread" of the outgoing connection

    GR: receiver gain 
    -GR: (Dr*Pi/lamda)^2
    -Dr: receiver telescope diameter in mm 
    -lambda: operating wavelength 

    LT: tx pointing loss
    -LT: exp(-GT(theta T)^2) 
    -GT: transmit gain 
    -theta T: transmitter pointing error in rads
    so this is how far "angle wise" the beam is from its intended path 

    LR: receiver pointing loss 
    -LR: exp(-GR(theta R)^2)
    -GR receive gain 
    -theta R: receive pointing error in rads 
    so, this is how far the receive 
    
    LPS: free space path loss, (lambda/(4*pi*dss)^2), where dss = link distance, km
    -lambda: operating wavelength, in nm 
    -dss: distance between satellites the connection is between, in km      

    Outputs: 
    Pr: receive power 

    """
    GT = 16/(divAngle**2) 
    GR = (rxDiam*np.pi/operatingWavelength)**2
    LT = np.exp(16/(txPointingError**2))
    
    return 

def angle_between_vectors(v, w):
    """
    Range is 0 -> 180 (so mirrors at 180)
    """

    dot_product = np.dot(v, w)
    if(dot_product == 0): 
        return 180 
    magnitude_v = np.linalg.norm(v)
    magnitude_w = np.linalg.norm(w)
    cos_theta = dot_product / (magnitude_v * magnitude_w)
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def calculate_normal_vector(point1, point2):

    # Calculate normal vector
    normal_vector = np.cross(np.array(point1), np.array(point2))

    # Normalize the normal vector
    normal_vector /= np.linalg.norm(normal_vector)

    return normal_vector

def satelliteOrbitalPeriod(x, y, z): 
    #this assumes circular orbit 
    #THIS ASSUMES METERS
    # Constants
    G = 6.67430e-11  # Gravitational constant in m^3 kg^(-1) s^(-2)
    M = 5.972e24     # Mass of the Earth in kg
    #R = 6371e3       # Earth's radius in meters
    #h = 1000       # Altitude in meters

    #combined radius of the orbit (radius of the earth with altitude of satellite) 
    totalRad = np.sqrt(x**2 + y**2 + z**2)


    #h = distance_from_center - R

    # Calculate orbital speed
    #totalRad = 0 in the init step beforehand 
    if(totalRad == 0): 
        v = 0
        T = 0
    else: 
        v = np.sqrt(G * M / (totalRad))
        T = 2 * np.pi * (totalRad) / v
    
    # Calculate orbital period
    return T

def calculate_next_position(current_position, time_difference):

    """
    This function is not great because it assumes a constant normal vector orientation :D 
    """
    # Unpack current position
    x, y, z = current_position
    orbital_period = satelliteOrbitalPeriod(x,y,z)

    # Calculate the angular velocity
    angular_velocity = 2 * np.pi / orbital_period
    
    # Calculate the current angular position
    current_theta = np.arctan2(y, x)
    
    # Calculate the new angular position after the time difference
    new_theta = current_theta + angular_velocity * time_difference
    
    # Calculate the new position in Cartesian coordinates
    new_x = np.cos(new_theta) * np.sqrt(x**2 + y**2)
    new_y = np.sin(new_theta) * np.sqrt(x**2 + y**2)
    new_z = z
    
    return [new_x, new_y, new_z]

def calculate_new_position(normal_vector, current_position, angle_rad):
    """
    This function is heavily used in calculating the next position for each satellite, given
    the satellite is following a circular orbit. 

    Inputs: 
    normal_vector: this determines the orientation of the circular path of the satellite 
    current_position: where the satellite is at currently in 3d space (xyz) 
    angle_rad: in radians, the angle we want to shift the satellite by in 3d space

    Outputs: 
    new_position: xyz of the satellite in 3d space at the new time frame 
    """
    
    # Calculate the rotation matrix using the normal vector

    rotation_matrix = np.array([[np.cos(angle_rad) + normal_vector[0]**2 * (1 - np.cos(angle_rad)),
                                 normal_vector[0] * normal_vector[1] * (1 - np.cos(angle_rad)) - normal_vector[2] * np.sin(angle_rad),
                                 normal_vector[0] * normal_vector[2] * (1 - np.cos(angle_rad)) + normal_vector[1] * np.sin(angle_rad)],
                                [normal_vector[1] * normal_vector[0] * (1 - np.cos(angle_rad)) + normal_vector[2] * np.sin(angle_rad),
                                 np.cos(angle_rad) + normal_vector[1]**2 * (1 - np.cos(angle_rad)),
                                 normal_vector[1] * normal_vector[2] * (1 - np.cos(angle_rad)) - normal_vector[0] * np.sin(angle_rad)],
                                [normal_vector[2] * normal_vector[0] * (1 - np.cos(angle_rad)) - normal_vector[1] * np.sin(angle_rad),
                                 normal_vector[2] * normal_vector[1] * (1 - np.cos(angle_rad)) + normal_vector[0] * np.sin(angle_rad),
                                 np.cos(angle_rad) + normal_vector[2]**2 * (1 - np.cos(angle_rad))]])

    # Calculate the new position using the rotation matrix
    new_position = np.dot(rotation_matrix, np.array(current_position))

    return new_position


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
    
    lat, _, _ = cartesian_to_geodetic(x,y,z,6371e3) 
    
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

    Nvm its fine, phasing parameter has been figured out 
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
    
    #have the distance from core for radius of circle, added with the altitude of the satellite  
    distFromCore = altitude + 6371e3

    #calculate the basic set of points using spherical coordinates 
    basePoints = [distFromCore*np.cos(phi), distFromCore*np.sin(phi), 0*phi] 

    #create storage for all points
    # storage will be: numPlanes, numPointsPerPlane ^ 3 
    walkerPoints = np.ones([numPlanes, numSatellitesPerPlane, 3])
    
    #create storage for normal vectors, one for each plane 
    normVecs = [0]*numPlanes

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

        walkerPoints[planeInd] = basePointsCopy.T

        #also, calculate the normal vector for each respective circular path 
        #then, you can use the same normal vector for each path 
        normVecs[planeInd] = calculate_normal_vector(walkerPoints[planeInd,0], walkerPoints[planeInd,1]) 

    return walkerPoints, normVecs

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

def geodetic_to_cartesian(latitude, longitude, radius=6371e3):
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

def distance(point1, point2):
    sum = 0
    for ind in range(len(point1)):
        sum+= (point1[ind] - point2[ind]) ** 2
                
    return sum ** 0.5

# def satDistance(sat1, sat2):
#     coords1 = sat1.getCoords() 
#     coords2 = sat2.getCoords() 
#     return ((coords1[0] - coords2[0]) ** 2 + (coords1[1] - coords2[1]) ** 2 + (coords1[2] - coords2[2])) ** 0.5


def find_closest_points(points, m):
    """
    This could possibly be optimized greatly as the state of the 
    points is very dependent iteration to iteration
    """
    result = {}

    for i, current_point in enumerate(points):
        # Initialize a min heap to store the m closest points
        min_heap = []

        for j, other_point in enumerate(points):
            if i != j:
                dist = distance(current_point, other_point)
                heapq.heappush(min_heap, (dist, other_point))

        # Get the m closest points by popping from the min heap
        result[i] = [point for _, point in heapq.nsmallest(m, min_heap)]

    return result

def find_closest_satsBad(sats, m):
    """
    This could possibly be optimized greatly as the state of the 
    points is very dependent iteration to iteration
    """
    result = {}

    for current_sat in sats:
        # Initialize a min heap to store the m closest points
        min_heap = []

        for other_sat in sats:
            if current_sat != other_sat:
                dist = satDistance(current_sat, other_sat)
                heapq.heappush(min_heap, (dist, other_sat))

        # Get the m closest points by popping from the min heap
        result[current_sat] = [point for _, point in heapq.nsmallest(m, min_heap)]

    return result






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


def dijkstraWithNodeValuesAndPath(adj_matrix, nodeValues, start, end):
    
    #modify adjacency matrix by adding to row and column, but taking out the overlap 
    for ind, value in enumerate(nodeValues): 
        adj_matrix[ind] +=value
        adj_matrix[:,ind] +=value
        adj_matrix[ind,ind] -=value

    return dijkstraWithPath(adj_matrix, start, end)

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

def floyd_warshall(graph):
    """
    Floyd warshall algorithm for getting all shortest paths. 

    Inputs: 
    graph: adjmat 

    Outputs: 


    """
    #tested successfully against djikstras algorithm 
    
    num_nodes = len(graph)
    
    # Initialize the distance matrix
    distance = [[float('inf') if i != j else 0 for j in range(num_nodes)] for i in range(num_nodes)]
    
    # Initialize the next node matrix for path reconstruction
    next_node = [[-1 for _ in range(num_nodes)] for _ in range(num_nodes)]
    
    # Set initial distances for direct edges and update next_node matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and graph[i][j] != float('inf'):
                distance[i][j] = graph[i][j]
                next_node[i][j] = j
    
    # Floyd-Warshall algorithm
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance[i][k] + distance[k][j] < distance[i][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
                    next_node[i][j] = next_node[i][k]
    
    # Extract paths for selected pairs
    selected_paths = {}
    for pair in []:
        start_node, end_node = pair[0],pair[1]
        path = reconstruct_path(start_node, end_node, next_node)
        selected_paths[tuple(pair)] = path
    
    #now, get the output path dist for each pair 
    lengths = [0]*len(selected_paths.keys())
    for ind, key in enumerate(selected_paths.keys()): 
        lengths[ind] = path_length(selected_paths[key], graph)

    #could return selected paths or lengths, depending on if you want the path
    #or the path length 
    return next_node 

def path_length(path, adj_matrix):
    #not really tested 
    length = 0
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        edge_weight = adj_matrix[start_node][end_node]
        length += edge_weight

    return length

def reconstruct_path(start, end, next_node):
    path = [start]
    while start != end:
        if(start == -1): 
            return [-1]
        start = next_node[start][end]
        path.append(start)
    return path


def generate_random_graph(num_nodes, density=0.3, weight_range=(1, 10)):
    # Initialize an empty adjacency matrix
    graph = [[float('inf') for _ in range(num_nodes)] for _ in range(num_nodes)]

    # Set diagonal elements to 0 (no self-loops)
    for i in range(num_nodes):
        graph[i][i] = 0

    # Fill in the upper triangular part with random edges
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            if random.random() < density:
                weight = random.randint(weight_range[0], weight_range[1])
                graph[i][j] = weight
                graph[j][i] = weight  # Symmetry

    return graph
