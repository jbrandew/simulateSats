#this class examines the different event types for satellite and others
#within the simulation structure 

#event types with packet: 
#packet begin, packet arrives, packet starts processing, packet departs,
#event types with others: 
#update topology, update satellite position, ...

"""
Current Event Types: 

Packet Specific: 
Packet Sent
Packet Arrive at Constellation
Packet Arrive at Next Player
Packet Arrive At Destination
Packet Finish Processing 

Constellation Specific: 
Update adjacency matrix for process times
Update adjacency matrix for changing distance 
Update adjacency matrix for topology 

Environment: 
Weather 
Attack 

Currently I only have the packet specific events implemented. 

"""

class Event: 
    """
    This class represents all possible occurences, which can spawn other events
    within the priority queue event processing stack. 
    """
    
    def __init__(self,
                 timeOfOccurence,
                 eventType, 
                 kargs): 
        """
        timeOfOccurence: when the event is scheduled to start 
        eventType: what the event is (packet arrives, departs, etc.) 
        *args: arguments depending on the eventType
        """

        self.timeOfOccurence = timeOfOccurence
        self.eventType = eventType
        self.kargs = kargs 

