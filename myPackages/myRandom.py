import numpy as np

#hi :D. This function works with everything that i didnt know where to put
#as such, it may include file operations or uhhh...idk just see 

def deepCopy(arrToCopy): 
    """
    this method takes an input array, and then copies it fully to a new one     
    
    arrToCopy: array we are going to copy :D 
    """

    #create the array we are going to copy to 
    toCopyTo = np.ones(np.shape(arrToCopy)) 

    #copy over all data 
    toCopyTo[:] = arrToCopy[:]

    return toCopyTo 