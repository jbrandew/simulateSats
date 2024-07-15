import torch
import torch.nn as nn
import torch.optim as optim

import pdb 

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

    def partialForward(self,x): 
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Construct Generic Network 
input_size = 10
hidden_size = [20, 15]  # Number of neurons in each hidden layer
output_size = 10
model = GenericNet(input_size, hidden_size, output_size)

# Create synthetic data
torch.manual_seed(42)
input_data1 = torch.randn(10)  # 1 sample with 10 features 
input_data2 = torch.randn(10)  # 1 sample with 10 features 

labels1 = torch.randn(10)       # 1 target with 10 features 
labels2 = torch.randn(10)       # 1 target with 10 features 

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# Get forward pass data 
outputs1 = model(input_data1)

#do 2nd pass
outputs2 = model(input_data2)

# Compute loss
loss1 = criterion(outputs1, labels1)

# Compute loss
loss2 = criterion(outputs1, labels1)

# Have loss go backwards 
loss2.backward(retain_graph = True)

# Have loss go backwards 
loss1.backward()

pdb.set_trace()





# Zero out gradients 
#optimizer.zero_grad()

# Detach tensors 
model = model.requires_grad_(False)

# Have loss go backwards 
loss.backward()

# Then, examine the grads 
print("Grads After")
print("Last layer weight grad")
print(model.fc3.weight.grad)
print("Second to last layer weight grad")
print(model.fc2.weight.grad)

#so, it doesnt prevent its computation recursively. 
#hm. for now, just 

pdb.set_trace() 

optimizer.step()









# check if we set the inputs = -biases, if the gradients = 0
layer1Bias = model.fc1.bias

# Forward pass with opposite of layer1bias
outputs = model.partialForward(-layer1Bias)

# Compute loss
loss = criterion(outputs, labels)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

#then, look at the gradients for layer1 
pdb.set_trace() 

print(loss)


# Training loop
num_episodes = 100
for episode in range(1, num_episodes + 1):
    
    # Forward pass
    outputs = model(input_data)
    
    # Compute loss
    loss = criterion(outputs, labels)
    
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