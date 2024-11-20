import torch
import numpy as np
from jal_am import JAL_AM 

'''
Trading Bot will take the market data and configuration as inputs.
Based on inputs, the Trading Bot will form a Markov Decision Process.
JAL-AM model will act as a agent.

Action Space: [
    0: Buy a coin
    1: Sell a coin
    2: No-op
]

'''
class TradingBot:
    def __init__(self, market_observation_dim, action_dim, trader_state_dim, budget, threshold, transaction_fee):
        # model configuration
        self.model = JAL_AM(market_observation_dim, action_dim, trader_state_dim) 
        self.initial_budget = budget
        self.threshold = threshold
        self.transaction_fee = transaction_fee

        # trader state
        self.budget = budget
        self.coin_num = 0
        
        # current data
        self.current_coin_price = None
        self.market_data = None 
        self.timestep = 0

    def trade(self, market_data, train=False):
        self.market_data = market_data 
        
        history = []

        market_observation, trader_state = self.reset()
        for t in range(len(market_data)): 
            action = self.action(market_observation, trader_state)
            
            market_observation, trader_state, reward, done = self.step(action) 
            
            if train:
                pass
                # using reward, train trader_network
                # using observation, perform supervised learning on market_network

            if done:
                pass

            
            if t % 100 == 0:
                history.append([self.budget, self.coin_num])

            break

        return history

    def reset(self):
        # reset to initial setting
        self.budget = self.initial_budget
        self.coin_num = 0
        self.current_coin_price = None
        self.timestep = 0
        
        market_observation, trader_state = self.observation()
        return market_observation, trader_state

    def step(self, action):
        # current state
        reward = self.reward_function(action) 

        # state transition
        self.transition_function(action)
       
        # next state
        done = True if self.timestep == len(self.market_data) - 1 else False
        market_observation, trader_state = self.observation()
        
        return market_observation, trader_state, reward, done

    def observation(self):
        market_observation = self.market_data[self.timestep] # current market data
        trader_state = np.array([self.budget, self.coin_num]) # observation of itself at t
        return market_observation, trader_state # seperated observation because market model doesn't observe trader_state 

    def action(self, market_observation, trader_state):
        action_prob = self.model.policy(market_observation, trader_state) # call JAL AM model 
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
    
    def transition_function(self, action): 
        # change trader state
        match action:
            case 0: # Buy a coin
                self.budget -= self.current_coin_price + self.transaction_fee
                self.coin_num += 1
            case 1: # Sell a coin
                self.budget += self.current_coin_price - self.transaction_fee
                self.coin_num -= 1
            case _: # No-op
                pass
        self.timestep += 1
        market_observation = self.market_data[self.timestep] # current market data
        self.current_coin_price = (market_observation[0] + market_observation[3]) / 2 # average of opening price and closing price


