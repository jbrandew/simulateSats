#import libraries 
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import pdb 


"""
Generate graph connectedness 
"""

# Number of nodes and features
num_nodes = 10
num_features = 5
num_classes = 2  # Binary classification

# Generate a random adjacency matrix
adj_matrix = np.random.randint(2, size=(num_nodes, num_nodes))

# Make the matrix symmetric
adj_matrix = np.triu(adj_matrix)  # Take the upper triangle of the matrix
adj_matrix = adj_matrix + adj_matrix.T  # Mirror it to make it symmetric

# Ensure no self-loops by zeroing out the diagonal
np.fill_diagonal(adj_matrix, 0)

# Convert the adjacency matrix to edge indices (COO format)
# only take upper diagonal, as you will face repeated edges otherwise 
edge_index = torch.tensor(np.array(np.nonzero(np.triu(adj_matrix))), dtype=torch.long)

#then create data for edge attributes (just the weights in this case)
edge_attr = torch.rand(edge_index.shape[1], 1)  # Each edge has a single feature

"""
Generate Targets and Node Data/Features
"""

# Random feature matrix
x = torch.rand((num_nodes, num_features))

# Random labels for nodes (0 or 1)
y = torch.randint(0, num_classes, (num_nodes,))

# Create a PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)

"""
Create class for the model generation 
"""

#create GCN model: 
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # First GCN layer + ReLU
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)

"""
Init proper objects and train 
"""

# Instantiate the model
model = GCN(in_channels=num_features, out_channels=num_classes)

# Set up the loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # Clear gradients

    # Forward pass
    out = model(data)
    
    # Calculate the loss
    loss = criterion(out, data.y)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Print the loss every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Final output after training
model.eval()
_, pred = model(data).max(dim=1)
accuracy = (pred == data.y).sum().item() / num_nodes
print(f'Accuracy: {accuracy * 100:.2f}%')
