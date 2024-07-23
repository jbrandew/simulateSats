import numpy as np 

arr1 = np.arange(10)+100
arr2 = np.arange(10)+105

firstDiff, indexOfFirstDiff = next((item, idx) for idx, item in enumerate(arr2) if item not in arr1)

print("Hehe")
print(firstDiff)
print(indexOfFirstDiff)