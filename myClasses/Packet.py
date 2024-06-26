import pdb 

class Packet: 
    """
    This class represents a packet.

    Inputs: 
    routingMetadata: depends on the routing method. has 

    """
    
    def __init__(self,
                 startLocation,
                 endLocation,
                 startSat,
                 endSat, 
                 currSat, 
                 packetSendTime,
                 routingMetadata,
                 packetArriveTime,
                 packetIndex = 0 
                 ):
        
        self.startLocation = startLocation
        self.endLocation = endLocation

        self.startSat = startSat
        self.endSat = endSat
        self.currSat = currSat

        self.packetSendTime = packetSendTime
        self.routingMetadata = routingMetadata
        self.packetArriveTime = packetArriveTime

        self.packetIndex = packetIndex
        self.playersInvolvedInSending = set() 

        #this is only used within the "basic" protocol
        self.routingMetadata["currIndInPath"] = 0

    def reachedEnd(self):
        return self.endSat == self.currSat
        
    def returnNextHopBasic(self): 
        return self.routingMetadata["path"][self.routingMetadata["currIndInPath"]+1]

    def updateToNextHopBasic(self): 
        self.routingMetadata["currIndInPath"]+=1 
        self.currSat = self.routingMetadata["path"][self.routingMetadata["currIndInPath"]]

    def __str__(self):
        return vars(self)

        
