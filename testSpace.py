#what am i doing? 
import numpy as np 
import torch

def func(nums):

    sum = 0
    for num in nums: 
        sum+=num
    return sum 

data = np.arange(10)

class Processor:
    def __init__(self):
        self.mixer = torch.sum


processor = Processor() 

print(processor.mixer(torch.from_numpy(data)))
