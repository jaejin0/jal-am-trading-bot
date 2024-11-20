import torch
import numpy as np
from jal_am import JAL_AM 

'''
Trading Bot using Joint Action Learning with Deep Agent Modeling

1. predicts how much market will move
2. predicts how much it can bet
3. bot bets based on budget

Action Space: [
    0: Buy a coin
    1: Sell a coin
    2: No-op
]

'''
class TradingBot:
    def __init__(self, observation_dim, action_dim, status_dim, budget):
        self.model = JAL_AM(observation_dim, action_dim, status_dim) 
        self.budget = budget
        self.coin_num = 0
        
    def action(self, observation):
        trader_status = np.array([self.budget, self.coin_num])
        action_prob = self.model.policy(observation, trader_status)
        action = self.choose_action(action_prob) 

    # modify after test
    def train(self, observation):
        # for timestep in range(len(observation)):
            # current_observation = observation[timestep]
            # action_prob = self.agent.policy(observation)
            # action = self.act(action_prob)
        pass  
        # run over observations and check how much it can earn

    # modify after test
    def eval(self, observation):
        # for timestep in range(len(observation)):
            observation = torch.from_numpy(observation)
            action = self.agent.policy(observation)

        # show how much it earns for every timestep % 100 

    def choose_action(self, action_prob):
        action = np.random.choice(len(action_prob), p=action_prob)
        print(action) 

        return action 
