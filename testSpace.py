import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a simple feedforward neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to measure time for training
def measure_performance(device, model, criterion, optimizer, input_data, target_data, iterations=100):
    model.to(device)
    input_data = input_data.to(device)
    target_data = target_data.to(device)

    # Warm-up
    for _ in range(10):
        outputs = model(input_data)
        loss = criterion(outputs, target_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Measure time
    start_time = time.time()

    for _ in range(iterations):
        outputs = model(input_data)
        loss = criterion(outputs, target_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Ensure all GPU operations are finished
    if device == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()

    return (end_time - start_time) / iterations

def main():
    # Hyperparameters
    input_size = 1000
    hidden_size = 500
    output_size = 10
    num_samples = 10000
    learning_rate = 0.001
    iterations = 100

    # Generate random input and target data
    input_data = torch.randn(num_samples, input_size)
    target_data = torch.randn(num_samples, output_size)

    # Instantiate the model, loss function, and optimizer
    model = SimpleNet(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()

    # Measure performance on CPU
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    cpu_time = measure_performance('cpu', model, criterion, optimizer, input_data, target_data, iterations)
    print(f"Time per iteration on CPU: {cpu_time:.6f} seconds")

    # Check if GPU is available
    if torch.cuda.is_available():
        # Re-create model and optimizer for GPU
        model_gpu = SimpleNet(input_size, hidden_size, output_size).to('cuda')
        optimizer_gpu = optim.Adam(model_gpu.parameters(), lr=learning_rate)

        gpu_time = measure_performance('cuda', model_gpu, criterion, optimizer_gpu, input_data, target_data, iterations)
        print(f"Time per iteration on GPU: {gpu_time:.6f} seconds")
    else:
        print("CUDA is not available. Cannot measure GPU performance.")

if __name__ == "__main__":
    main()
