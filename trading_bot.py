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
    def __init__(self, observation_dim, action_dim, status_dim, budget, threshold):
        self.model = JAL_AM(observation_dim, action_dim, status_dim) 
        self.budget = budget
        self.coin_num = 0
        self.threshold = threshold
        self.current_coin_price = None

    def action(self, observation):
        self.current_coin_price = (observation[0] + observation[3]) / 2 # average of opening price and closing price
        trader_status = np.array([self.budget, self.coin_num])
        action_prob = self.model.policy(observation, trader_status)
        action = self.choose_action(action_prob)
        reward = self.perform_action(action)

    # modify after test
    def train(self, market_data):
        for t in range(len(market_data)):
            market_observation = market_data[t] # observation at t
            self.current_coin_price = (market_observation[0] + market_observation[3]) / 2 # average of opening price and closing price
            trader_status = np.array([self.budget, self.coin_num]) # observation of itself at t
            action_prob = self.model.policy(market_observation, trader_status) # model's policy outputs a probability distribution of actions
            action = self.choose_action(action_prob) # choose possible action


        

        # run over observations and check how much it can earn

    def eval(self, market_data):
        result = "not modified yet"
        return result

    def choose_action(self, action_prob):
        action = np.random.choice(len(action_prob), p=action_prob)
        match action:
            case 0: # Buy a coin
                chosen_action = 0 if self.current_coin_price < self.budget else 2
            case 1: # Sell a coin
                chosen_action = 1 if self.coin_num > 0 else 2 
            case _: # No-op
                chosen_action = 2
        print(action, chosen_action)
        return chosen_action 
