import torch
import torch.nn as nn

import pdb

#testing 2D linear layer 
layer1 = nn.Linear((5,5), 64)


# Assuming batch size is 1 and sequence length is 50
batch_size = 1
shape1 = 5
shape2 = 5

# Create a random input tensor with the specified shape
input_tensor = torch.randn(batch_size, (shape1, shape2))

pdb.set_trace() 

randomOutput = layer1(input_tensor)

# Check the shape of the input tensor
print("Input tensor shape:", input_tensor.shape)

# Pass the input tensor through the Conv1d layer to see the expected input shape
output_tensor = layer1(input_tensor)

# Check the shape of the output tensor
print("Output tensor shape:", output_tensor.shape)

# Access the number of input channels (in_channels) from the Conv1d layer
