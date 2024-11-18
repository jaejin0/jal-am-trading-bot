import torch
import numpy as np
from jal_am import JAL_AM 

'''
Trading Bot using Joint Action Learning with Deep Agent Modeling

1. predicts how much market will move
2. predicts how much it can bet
3. bot bets based on budget
'''
class TradingBot:
    def __init__(self, observation_dim, action_dim, budget):
        self.agent = JAL_AM(observation_dim, action_dim) 
        self.budget = budget
        self.hasCoin = False

    def test(self, observation):
        current_observation = observation
        action_prob = self.agent.policy(observation)
        action = self.act(action_prob)

    def train(self, observation):
        # for timestep in range(len(observation)):
            # current_observation = observation[timestep]
            # action_prob = self.agent.policy(observation)
            # action = self.act(action_prob)
        pass  
        # run over observations and check how much it can earn

    # not modified
    def eval(self, observation):
        # for timestep in range(len(observation)):
            observation = torch.from_numpy(observation)
            action = self.agent.policy(observation)

        # show how much it earns for every timestep % 100 

    def act(self, action_prob):
        random_action = np.random.choice(len(action_prob), p=action_prob)
        print(random_action) 
