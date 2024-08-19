import heapq
import numpy as np
import myPackages.myMath as myMath 

import pdb

def dijkstraWithDistances(adj_matrix, source):
    n = len(adj_matrix)  # Number of nodes
    distances = np.full(n, np.inf)  # Initialize distances with infinity
    distances[source] = 0  # Distance to the source node is 0
    priority_queue = [(0, source)]  # Min-heap priority queue
    
    #pdb.set_trace()

    while priority_queue:
        current_distance, current = heapq.heappop(priority_queue)
        
        if current_distance > distances[current]:
            continue
        
        for neighbor, weight in enumerate(adj_matrix[current]):
            if weight > 0:  # Check if there is an edge
                distance = current_distance + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))
                    
    return distances

# Example usage:
adj_matrix = np.array([
    [0,    7,   9,  np.inf, np.inf, 14, np.inf, np.inf, np.inf, np.inf],
    [7,    0,   10, 15, np.inf, np.inf, 12, np.inf, np.inf, np.inf],
    [9,    10,  0,  11, 16, np.inf,  7, np.inf, np.inf, np.inf],
    [np.inf, 15, 11, 0,   6, np.inf, np.inf,  9, np.inf, np.inf],
    [np.inf, np.inf, 16, 6,   0,  9,  np.inf, np.inf, 20, np.inf],
    [14,   np.inf, np.inf, np.inf, 9,   0,   2, np.inf, np.inf, 18],
    [np.inf, 12,  7, np.inf, np.inf, 2,   0,  5, np.inf, np.inf],
    [np.inf, np.inf, np.inf, 9, np.inf, np.inf, 5,   0,  4, np.inf],
    [np.inf, np.inf, np.inf, np.inf, 20, np.inf, np.inf, 4,   0,  7],
    [np.inf, np.inf, np.inf, np.inf, np.inf, 18, np.inf, np.inf, 7,   0]
])

source = 0
print("New Function Output:")
print(dijkstra(adj_matrix, source))
print("Old Function Output")
print(myMath.dijkstra(adj_matrix, 0, 3))
