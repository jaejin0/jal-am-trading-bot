import torch
import numpy as np
from jal_am import JAL_AM 

# call JAL-AM model and call file to train
class TradingBot:
    def __init__(self, observation_dim, action_dim, budget):
        self.agent = JAL_AM(observation_dim, action_dim) 
        self.budget = budget
        self.hasCoin = False

    def train(self, observation):
        for timestep in range(len(observation)):
            current_observation = observation[timestep]
            action_prob = self.agent.policy(observation)
            action = self.act(action_prob)
             
        # run over observations and check how much it can earn

    def test(self, observation):
        for timestep in range(len(observation)):
            observation = torch.from_numpy(observation)
            action = self.agent.policy(observation)
            # act(action)

        # show how much it earns for every timestep % 100 

    def act(self, action_prob):
        random_action = np.random.choice(len(action_prob), p=action_prob)
        print(random_action) 
