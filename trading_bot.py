import torch
from jal_am import JAL_AM 

# call JAL-AM model and call file to train
class TradingBot:
	
    def __init__(self, observation_dim, action_dim):
        self.agent = JAL_AM(observation_dim, action_dim) 
    
    def policy(self, observation):
        observation = torch.from_numpy(observation)
        action = self.agent.policy(observation)
        return action
