import torch
import torch.nn as nn

import pdb

# Example Conv1d layer
# Assume input shape is [batch_size, in_channels, sequence_length]
# Here, we will specify input channels (in_channels) and output channels (out_channels)
conv1d_layer = nn.Conv1d(in_channels=5, out_channels=11, kernel_size=3, stride=1, padding=1)

# Assuming batch size is 1 and sequence length is 50
batch_size = 1
sequence_length = 50

# Create a random input tensor with the specified shape
input_tensor = torch.randn(batch_size, 5, sequence_length)

randomOutput = conv1d_layer(input_tensor)

# Check the shape of the input tensor
print("Input tensor shape:", input_tensor.shape)

# Pass the input tensor through the Conv1d layer to see the expected input shape
output_tensor = conv1d_layer(input_tensor)

# Check the shape of the output tensor
print("Output tensor shape:", output_tensor.shape)

# Access the number of input channels (in_channels) from the Conv1d layer
expected_in_channels = conv1d_layer.in_channels
print("Expected number of input channels:", expected_in_channels)

pdb.set_trace() 
conv1d_layer.weight.shape
input_tensor.shape