# class LimitedValues:
#     ALLOWED_VALUES = {'VALUE1', 'VALUE2', 'VALUE3'}

#     def __init__(self, value):
#         if value not in self.ALLOWED_VALUES:
#             raise ValueError(f"Invalid value: {value}. Allowed values are {', '.join(self.ALLOWED_VALUES)}")
#         self.value = value

# # Example usage
# try:
#     value = LimitedValues('VALUE1')
#     print(value.value)  # Output: VALUE1
#     #value = LimitedValues('INVALID_VALUE')  # This will raise a ValueError
#     value = "aoeu"
# except ValueError as e:

import pdb 

class Hold(): 
    sety = set()
    arrayy = [] 


thing1 = Hold() 
thing2 = Hold() 

thing1.sety.add("boop")
#thing1.arrayy = ["beep"] + thing1.arrayy
pdb.set_trace() 