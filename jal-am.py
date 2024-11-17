import torch
import torch.nn as nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super.__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(0.5),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.network(x)
        output = F.log_softmax(x, dim=1)
        return output

# Joint-Action Learning with Deep Agent Modeling
# assume this acts in two-player game
class JAL_AM:
    def __init__(self, observation_dim, action_dim, learning_rate, exploration_rate, ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim


    def policy()
