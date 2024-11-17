import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=float),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=float),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, dtype=float),
            nn.Dropout(0.5),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.network(x)
        print(x)
        # output = F.log_softmax(x, dim=3)
        # print("output", output)
        return x

# Joint-Action Learning with Deep Agent Modeling
# assume this acts in two-player game
class JAL_AM:
    def __init__(self, observation_dim, action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        self.policy = NeuralNetwork(observation_dim, action_dim)
        self.agent_model = NeuralNetwork(observation_dim, action_dim)

    def policy(self, observation):
        self.policy.forward(observation)
