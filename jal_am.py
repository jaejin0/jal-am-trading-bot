import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=float),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=float),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, dtype=float),
            # nn.Dropout(0.5),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.network(x)
        print(x)
        # output = F.log_softmax(x, dim=3)
        # print("output", output)
        return x

'''
agents learn models of other agents j conditioned on states s: estimated pi_{-i}(a_{-i} | s)
agents learn value functions conditioned on the joint action: Q_i(s, a)
using the value function and agent models, agent i can compute its expected action values under the current models of other agents: AV_i(s, a_i) = SUM_{a_{-i} in A_{-i}}Q_i(s, <a_i, a_{-i}>) PI_{j in I\{i}} estimated pi_j (a_j | s)
'''

# Joint-Action Learning with Deep Agent Modeling
# assume this acts in two-player game
class JAL_AM:
    def __init__(self, observation_dim, action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        self.policy = NeuralNetwork(observation_dim, action_dim)
        self.agent_model = NeuralNetwork(observation_dim, action_dim)

        # train agent model using the obseration history
        self.observation_history = np.zeros()

        self.loss_fn = nn.CrossEntropyLoss()
        

    def policy(self, observation):
        self.policy.forward(observation)
        
        
    def learn(self, observation, joint_action):
        pass

        # minimize cross-entropy loss between the predicted policy and the observed actions of agent j
    def train_agent_model(self):
        pass

    def train_policy(self):
        pass
