import numpy as np 
import copy 
import pdb 

class Elephant:
    def __init__(self, name, age, weight):
        self.name = name
        self.age = age
        self.weight = weight

    def describe(self):
        return f"Elephant {self.name} is {self.age} years old and weighs {self.weight} kg."

x = 1 
y = 1
print(x)
y+=1 
print(x)

# # Example usage
# elephant = Elephant("Dumbo", 10, 1200)
# elephantArray = [elephant]*10
# elephantArray[0].name = "Dumbey"

# #pdb.set_trace() 

# for elephant in elephantArray: 
#     print(elephant.describe()) 