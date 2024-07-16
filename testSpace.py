import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules.models.multiagent import QMixer
import torch.optim as optim

import torch.nn as nn

import pdb
import numpy as np 
import matplotlib.pyplot as plt

n_agents = 3

# Define QMix architecture wrapped in TensorDictModule
qMix = TensorDictModule(
    module=QMixer(
        state_shape=(10, 10, 2),
        mixing_embed_dim=32,
        n_agents=n_agents,
        device="cpu",
    ),
    in_keys=[("agents", "chosen_action_value"), "state"],
    out_keys=["chosen_action_value"],
)

class FeedForwardNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Example usage:
input_size = 10
hidden_size = 20
output_size = 1  # Output size is 1 for a single output node

batch_size = 64

# Instantiate 3 feed forward nets 
model1 = FeedForwardNet(input_size, hidden_size, output_size)
model2 = FeedForwardNet(input_size, hidden_size, output_size)
model3 = FeedForwardNet(input_size, hidden_size, output_size)

# Example agent input tensor
agent_input_tensor = torch.randn(batch_size, input_size)  # Batch size of 32

# Example state input tensor 
state_input_tensor = torch.randn(batch_size, 10, 10, 2)

# Target value 
qMixTarget = torch.randn(batch_size)

# Create optimizer
optimizer = optim.Adam(qMix.parameters(), lr=0.1)
criterion = nn.MSELoss()

holdError = np.zeros(1000)

# Training loop 
for epoch in range(1000):
    optimizer.zero_grad()

    #create dummy inputs:
    # Example agent input tensor
    agent_input_tensor = torch.randn(batch_size, input_size)  # Batch size of 32

    # Example state input tensor 
    state_input_tensor = torch.randn(batch_size, 10, 10, 2)

    #Create storage for the child network outputs
    childOutput = torch.zeros(batch_size, n_agents, 1)

    # Then, propagate through child networks
    childOutput[:, 0] = model1(agent_input_tensor)
    childOutput[:, 1] = model2(agent_input_tensor)
    childOutput[:, 2] = model3(agent_input_tensor)

    # Then, create dictionary for input to central node 
    qMixInput = TensorDict({
        "agents": TensorDict({
            "chosen_action_value": childOutput
        }),
        "state": state_input_tensor
    }, [batch_size])

    qMixOutput = qMix(qMixInput)['chosen_action_value'].unsqueeze(1)

    #get loss based on single output 
    loss = criterion(qMixTarget, qMixOutput)
    loss.backward()
    
    optimizer.step()

    error = np.average(qMixTarget - qMixOutput.detach()) 
    holdError[epoch] = error

    #print("Prediction Error Batch Average:") 
    #print(np.average(qMixTarget - qMixOutput.detach()))

plt.plot(holdError)
plt.show()

