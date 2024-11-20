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
    def __init__(self, observation_dim, action_dim, status_dim, budget, threshold, transaction_fee):
        # model configuration
        self.model = JAL_AM(observation_dim, action_dim, status_dim) 
        self.initial_budget = budget
        self.threshold = threshold
        self.transaction_fee = transaction_fee

        # status
        self.budget = budget
        self.coin_num = 0
        
        # current data
        self.current_coin_price = None
        self.market_data = None 
        self.timestep = 0

    def train(self, market_data):
        self.market_data = market_data 
         
        market_observation, trader_status = self.reset()
        for t in range(len(market_data)): 
            action = self.action(market_observation, trader_status)
            

            break

    def eval(self, market_data):
        result = "not modified yet"
        return result

    def reset(self):
        # reset to initial setting
        self.budget = self.initial_budget
        self.coin_num = 0
        self.current_coin_price = None
        self.timestep = 0
        
        market_observation, trader_status = self.observation()
        return market_observation, trader_status

    def step(self, action):
        self.timestep += 1
        done = True if self.timestep == len(self.market_data) - 1 else False

        self.action(action)

        market_observation, trader_status = self.observation()
        

        return market_observation, trader_status, reward, done

    def observation(self):
        market_observation = self.market_data[self.timestep] # current market data
        self.current_coin_price = (market_observation[0] + market_observation[3]) / 2 # average of opening price and closing price
        trader_status = np.array([self.budget, self.coin_num]) # observation of itself at t
        return market_observation, trader_status # seperated observation because market model doesn't observe trader_status 

    def action(self, market_observation, trader_status):
        action_prob = self.model.policy(market_observation, trader_status) # call JAL AM model 
        action = np.random.choice(len(action_prob), p=action_prob) # randomly choose an action based on probability distribution
        match action: # verify if the action is possible to perform. If not, replace with No-op
            case 0: # Buy a coin
                chosen_action = 0 if self.current_coin_price < self.budget else 2
            case 1: # Sell a coin
                chosen_action = 1 if self.coin_num > 0 else 2 
            case _: # No-op
                chosen_action = 2

        return action
    
    def reward_function(self, action):
        match action:
            case 0: # Buy a coin
                return -self.current_coin_price - self.transaction_fee
            case 1: # Sell a coin
                return self.current_coin_price - self.transaction_fee
            case _: # No-op
                return 0

    def train(self, market_data):
        self.market_data = market_data
        for t in range(len(market_data)): 
            market_observation, trader_status = self.observation(market_data[t])
            action = self.action(market_observation, trader_status)
            reward = self.reward_function(action)
            print(self.current_coin_price, self.transaction_fee)
            print(action)
            print(reward)
            # self.model.learn()
            break

    def eval(self, market_data):
        result = "not modified yet"
        return result
