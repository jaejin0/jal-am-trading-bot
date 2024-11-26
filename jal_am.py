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

        self.trader_network = NeuralNetwork(market_observation_dim + trader_state_dim + action_dim, action_dim) # DQN with reinforcement learning
        self.market_network = NeuralNetwork(market_observation_dim, action_dim) # DQN with supervised learning

        # train agent model using the obseration history
        # self.observation_history = np.zeros()
        
        self.trader_optimizer = optim.Adam(self.trader_network.parameters(), lr=learning_rate)
        # self.trader_loss = nn.MSELoss()
        
        self.market_optimizer = optim.Adam(self.market_network.parameters(), lr=learning_rate) 
        # self.market_loss = nn.CrossEntropyLoss()

    def policy(self, market_observation, trader_state):
        # numpy to torch
        market_observation = torch.from_numpy(market_observation).flatten() 
        trader_state = torch.from_numpy(trader_state).flatten()

        # predict other agent's action probability distribution
        market_action_prob = self.market_network.forward(market_observation)

        # Later, changing can be needed to iterate over possible actions of the other agent and multiply the value and probability
        # For now, it is implemented simply as input to the trader network
        # for i in range(self.action_dim):
        #     market_action = np.zeros(self.action_dim)
        #     market_action[i] = 1
        #     market_action = torch.tensor(market_action)
        #     observation = torch.cat([market_observation, market_action_prob, trader_state], dim=0)
        # feed in the observation to the trader_network

        # choose action based on observation and predicted other agent's action
        observation = torch.cat([market_observation, market_action_prob, trader_state], dim=0)
        trader_action_prob = self.trader_network.forward(observation)
        
        # torch to numpy
        trader_action_prob = trader_action_prob.detach().numpy()
        return trader_action_prob
        
    def train_trader_network(self, observation, action, reward):
        # TODO: implement training process for trader and market networks
        '''
        should I make the policy network to compute multiple times over action space multiplied by the possiblity of the action by
        other agent? like how it is written on page 20 of CH9 pt 2
        like we set the input to be (1, 0, 0), (0, 1, 0), (0, 0, 1) and iterate over and get the highest Q value?

        or should I just straight up input the probabilities? (0.2, 0.5, 0.3)? Which is how I thought the architecture will look like

        '''
        pass

    def train_market_network(self):
         
