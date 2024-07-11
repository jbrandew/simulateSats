import torch
import torch.nn as nn
import torch.optim as optim

class GenericNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GenericNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# Example usage of the generic neural network
input_size = 10
hidden_size = [20, 15]  # Number of neurons in each hidden layer
output_size = 10
model = GenericNet(input_size, hidden_size, output_size)

# Create synthetic data
torch.manual_seed(42)
input_data = torch.randn(10)  # 1 sample with 10 features 
labels = torch.randn(10)       # 1 target with 10 features 

# Create model instance
num_subnetworks = 4

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

outputs = model(input_data)
print("Initial Performance")
print(outputs - labels)

# Training loop
num_episodes = 1000
for episode in range(1, num_episodes + 1):

    input_data = torch.randn(10)  # 1 sample with 10 features   

    # Forward pass
    outputs = model(input_data)
    
    # Compute loss
    loss = criterion(outputs, labels)
    
    #then, do another forward pass to see if it uses the wrong computational graph 
    input_data = torch.randn(10)
    # Forward pass
    fillerOutputs = model(input_data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(loss)

    #look at the gradients for parent net over time 
    #childGrad = model.subnets[0].fc1.weight.grad
    #print(childGrad)


# Evaluate the model after training
model.eval()
predicted = model(input_data)

print("Post Training Performance")
print(predicted - labels)