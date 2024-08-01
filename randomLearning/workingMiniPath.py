import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Grid Environment
class GridEnvironment:
    def __init__(self, grid_size, obstacles, start, goal):
        self.grid_size = grid_size
        self.obstacles = obstacles
        self.start = start
        self.goal = goal
        self.reset()
        
    def reset(self):
        self.position = self.start
        return self.position
    
    def step(self, action):
        if action == 0:  # Move Up
            new_position = (self.position[0] - 1, self.position[1])
        elif action == 1:  # Move Down
            new_position = (self.position[0] + 1, self.position[1])
        elif action == 2:  # Move Left
            new_position = (self.position[0], self.position[1] - 1)
        elif action == 3:  # Move Right
            new_position = (self.position[0], self.position[1] + 1)
        else:
            raise ValueError("Invalid action")
        
        if self.is_valid_position(new_position):
            self.position = new_position
        
        if self.position == self.goal:
            return self.position, 10, True  # Reward for reaching the goal
        else:
            return self.position, -1, False  # Negative reward for each step
    
    def is_valid_position(self, position):
        x, y = position
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and position not in self.obstacles

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, action_dim)
    
    def forward(self, x):
        return self.fc(x)

# Define the Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_dim, action_dim):
        self.q_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(4)  # Random action
        else:
            with torch.no_grad():
                return torch.argmax(self.q_network(torch.tensor(state, dtype=torch.float32))).item()
    
    def update(self, state, action, reward, next_state, done):
        q_values = self.q_network(torch.tensor(state, dtype=torch.float32))
        next_q_values = self.q_network(torch.tensor(next_state, dtype=torch.float32))
        target = reward + (1 - done) * torch.max(next_q_values)
        
        loss = self.loss_fn(q_values[action], target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Training Function
def train_agent(env, agent, episodes=1000, epsilon=0.1, gamma=0.9):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps Taken = {steps}")

# Setup environment and agent
grid_size = (5, 5)
obstacles = [(2, 2), (3, 2)]
start = (0, 0)
goal = (4, 4)

env = GridEnvironment(grid_size, obstacles, start, goal)
agent = QLearningAgent(state_dim=2, action_dim=4)  # 2 for (x, y) and 4 actions

# Train the agent
train_agent(env, agent)
