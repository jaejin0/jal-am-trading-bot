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
        output = F.softmax(x, dim=0)
        print(output)
        return output

'''
agents learn models of other agents j conditioned on states s: estimated pi_{-i}(a_{-i} | s)
agents learn value functions conditioned on the joint action: Q_i(s, a)
using the value function and agent models, agent i can compute its expected action values under the current models of other agents: AV_i(s, a_i) = SUM_{a_{-i} in A_{-i}}Q_i(s, <a_i, a_{-i}>) PI_{j in I\{i}} estimated pi_j (a_j | s)
minimize cross-entropy loss between the predicted policy and the observed actions of agent j
'''

'''
NOTE ON ENTROPY
entropy in thermodynamics : degree of disorder OR randomness in a system
interpreted using probability -> probablity of each energy configuration; higher entropy is statistically more likely

entropy in information theory : expected value of surprise, where expected value is x * p(x) + ... + y + p(y), surprise = log_n(1/p(x)), where n is the number of possible output
entropy = SUM(p(x) * log_n(1/p(x))) = -SUM(p(x) * log(p(x)))
by Claude Shannon
'''

# Joint-Action Learning with Deep Agent Modeling
# assume this acts in two-player game
class JAL_AM:
    def __init__(self, observation_dim, action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        self.network = NeuralNetwork(observation_dim + action_dim, action_dim)
        self.other_agent_network = NeuralNetwork(observation_dim, action_dim)

        # train agent model using the obseration history
        # self.observation_history = np.zeros()

        self.loss_fn = nn.CrossEntropyLoss()

    def policy(self, observation):
        # numpy to torch
        observation = torch.from_numpy(observation) 
        
        # predict other agent's action
        other_agent_prob = self.other_agent_network.forward(observation)
        
        # choose action based on observation and predicted other agent's action
        action_prob = self.network.forward(torch.cat(observation, other_agent_prob))
        
        # torch to numpy
        action_prob = action_prob.detach().numpy()
        return action_prob
        
    def learn(self, observation, joint_action):
        pass

    def train_agent_model(self):
        pass

    def train_policy(self):
        pass
