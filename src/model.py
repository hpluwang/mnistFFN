import torch.nn as nn
import torch.nn.functional as F

class mnistNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Input layer
        self.input = nn.Linear(784, 64)
        
        # Hidden layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)

        # Output layer
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

