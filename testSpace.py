import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network for the Actor-Critic model
class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNet, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        common_out = self.common(x)
        logits = self.actor(common_out)
        value = self.critic(common_out)
        return logits, value

# Function to preprocess the state into a one-hot encoded vector
def one_hot_encode(state, state_dim, device):
    state_vector = torch.zeros(state_dim, device=device)
    state_vector[state] = 1.0
    return state_vector.unsqueeze(0)

# Step 3: Create the environment
env = gym.make("FrozenLake-v1", is_slippery=True)
state_dim = env.observation_space.n
action_dim = env.action_space.n

# Step 4: Define the Actor-Critic model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCriticNet(state_dim, action_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
mse_loss = nn.MSELoss()

# Step 5: Implement the training loop with online learning
def train(env, model, optimizer, num_episodes=10000, gamma=0.99):
    for episode in range(num_episodes):
        state = env.reset()[0]

        done = False
        while not done:
            # Prepare the state for input into the model
            state_tensor = one_hot_encode(state, state_dim, device)

            # Forward pass through the model to get logits and value
            logits, value = model(state_tensor)

            # Get the action probabilities and sample an action
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()

            # Log probability of the action taken
            log_prob = torch.log(probs.squeeze(0)[action])

            # Take a step in the environment
            next_state, reward, done, _, _ = env.step(action)

            # Prepare the next state for input into the model
            next_state_tensor = one_hot_encode(next_state, state_dim, device)

            # Get the next value from the critic
            _, next_value = model(next_state_tensor)

            # Calculate the target for the critic (TD target)
            target = reward + (1 - done) * gamma * next_value.item()

            # Compute the critic loss (TD error)
            critic_loss = mse_loss(value, torch.tensor([target], device=device))

            # Compute the actor loss (policy gradient loss)
            advantage = target - value.item()
            actor_loss = -log_prob * advantage

            # Backpropagate the losses and update the model
            optimizer.zero_grad()
            loss = actor_loss + critic_loss
            loss.backward()
            optimizer.step()

            # Move to the next state
            state = next_state

        # Print episode results
        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {loss.item()}, Reward: {reward}")

# Step 6: Train the model
train(env, model, optimizer)

# Step 7: Test the trained model
def test(env, model, num_episodes=10):
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            state_tensor = one_hot_encode(state, state_dim, device)
            logits, _ = model(state_tensor)
            action = torch.argmax(logits, dim=-1).item()
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Test the trained model
test(env, model)
