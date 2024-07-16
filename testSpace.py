import pdb
def unique_indices(lst):
    unique_elements = set(lst)
    indices = {element: lst.index(element) for element in unique_elements}
    return indices

# Example usage:
my_list = [1, 1, 6, 2, 3, 1, 2, 4, 5, 3, 6]
indices = unique_indices(my_list)
print("Indices of unique elements:", indices)

#for key in indices.values():
#    print(key)

#store values 
vals = [*indices.values()]

#after storing values, then get the subset
newList = [my_list[val] for val in vals]

pdb.set_trace() 