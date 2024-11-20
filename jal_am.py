import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256, dtype=float),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 256, dtype=float),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 32, dtype=float),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, output_dim, dtype=float),
            nn.Dropout(0.5),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.network(x)
        output = F.softmax(x, dim=0)
        return output

'''
Joint Action Learning with Agent Modeling does the following:
    1. based on observation, model predicts other agent's action using neural network
    2. choose optimal action based on the observation and the predicted other agent's action
    3. train agent modeling network with supervised learning
    4. train policy network with reward

details:
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
    def __init__(self, market_observation_dim, action_dim, trader_state_dim):
        self.market_observation_dim = market_observation_dim
        self.action_dim = action_dim
        self.trader_state_dim = trader_state_dim

        self.trader_network = NeuralNetwork(market_observation_dim + action_dim + trader_state_dim, action_dim)
        self.market_network = NeuralNetwork(market_observation_dim, action_dim)

        # train agent model using the obseration history
        # self.observation_history = np.zeros()

        self.loss_fn = nn.CrossEntropyLoss()

    def policy(self, market_observation, trader_state):
        # numpy to torch
        market_observation = torch.from_numpy(market_observation) 
        trader_state = torch.from_numpy(trader_state)

        # predict other agent's action
        market_action_prob = self.market_network.forward(market_observation)
        
        # choose action based on observation and predicted other agent's action
        observation = torch.cat([market_observation, market_action_prob, trader_state], dim=0)
        trader_action_prob = self.trader_network.forward(observation)
        
        # torch to numpy
        trader_action_prob = trader_action_prob.detach().numpy()
        return trader_action_prob
        
    def learn(self, observation, joint_action):
        pass

    def train_agent_model(self):
        pass

    def train_policy(self): 
        pass
