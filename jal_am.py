import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, output_dim)
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
    def __init__(self, market_observation_dim, action_dim, trader_state_dim, learning_rate, target_update_rate, discount_factor, batch_size):
        self.market_observation_dim = market_observation_dim
        self.action_dim = action_dim
        self.trader_state_dim = trader_state_dim
        self.target_update_rate = target_update_rate
        self.discount_factor = discount_factor
        self.device = torch.device('mps') 
        self.batch_size = batch_size

        self.trader_network = NeuralNetwork(market_observation_dim + trader_state_dim + action_dim, action_dim).to(self.device) # DQN with reinforcement learning
        self.trader_target_network = NeuralNetwork(market_observation_dim + trader_state_dim + action_dim, action_dim).to(self.device)
        self.trader_target_network.load_state_dict(self.trader_network.state_dict())

        self.market_network = NeuralNetwork(market_observation_dim, action_dim).to(self.device) # DQN with supervised learning
        self.market_target_network = NeuralNetwork(market_observation_dim, action_dim).to(self.device)
        self.market_target_network.load_state_dict(self.market_network.state_dict())

        self.trader_optimizer = optim.AdamW(self.trader_network.parameters(), lr=learning_rate)
        self.trader_loss = nn.CrossEntropyLoss()
        
        self.market_optimizer = optim.AdamW(self.market_network.parameters(), lr=learning_rate) 
        self.market_loss = nn.CrossEntropyLoss()

    def policy(self, market_observation, trader_state):
        # predict other agent's action probability distribution
        market_action_prob = self.market_network.forward(market_observation)

        # choose action based on observation and predicted other agent's action
        observation = torch.cat([market_observation, market_action_prob, trader_state], dim=0).to(self.device)
        trader_action_prob = self.trader_network.forward(observation)
        
        return market_action_prob, trader_action_prob
        

    def train_trader_network(self, batch):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.trader_network(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device = self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.trader_target_network(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch
        
        loss = self.trader_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.trader_optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_value_(self.trader_network.parameters(), 100)
        self.trader_optimizer.step()

        # soft update of the target network's weights
        target_net_state_dict = self.trader_target_network.state_dict()
        policy_net_state_dict = self.trader_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.target_update_rate + target_net_state_dict[key] * (1 - self.target_update_rate)
        self.trader_target_network.load_state_dict(target_net_state_dict)

    def train_market_network(self, batch):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.market_network(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device = self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.market_target_network(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch
        
        loss = self.market_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.market_optimizer.zero_grad() 

        loss.backward()
        torch.nn.utils.clip_grad_value_(self.market_network.parameters(), 100)
        self.market_optimizer.step()

        # soft update of the target network's weights
        target_net_state_dict = self.market_target_network.state_dict()
        policy_net_state_dict = self.market_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.target_update_rate + target_net_state_dict[key] * (1 - self.target_update_rate)
        self.market_target_network.load_state_dict(target_net_state_dict) 
    
    def save_model(self, model_dir, iteration):
        torch.save(self.market_network.state_dict(), f"{model_dir}market_network_iteration[{iteration}]")
        torch.save(self.trader_network.state_dict(), f"{model_dir}trader_network_iteration[{iteration}]")

    def load_model(self, model_dir, iteration):
        self.market_network.load_state_dict(torch.load(f"{model_dir}market_network_iteration[{iteration}]", weights_only=True))
        self.trader_network.load_state_dict(torch.load(f"{model_dir}trader_network_iteration[{iteration}]", weights_only=True))
