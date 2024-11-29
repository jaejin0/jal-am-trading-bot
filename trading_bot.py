import torch
import numpy as np
from collections import deque
import random

from jal_am import JAL_AM 

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque([], maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, observation, action, reward, next_observation, done):
        self.buffer.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

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
    def __init__(self, market_observation_feature_dim, market_observation_time_range, action_dim, trader_state_dim, budget, threshold, transaction_fee, buffer_size, learning_rate, target_update_rate, discount_factor):
        # model configuration
        self.model = JAL_AM(market_observation_feature_dim * market_observation_time_range, action_dim, trader_state_dim, learning_rate, target_update_rate, discount_factor) 
        self.market_observation_time_range = market_observation_time_range
        self.initial_budget = budget
        self.threshold = threshold
        self.transaction_fee = transaction_fee
        self.device = torch.device('mps')

        # trader state
        self.budget = budget
        self.coin_num = 0
        
        # current data
        self.current_coin_price = None
        self.market_data = None 
        self.timestep = self.market_observation_time_range - 1

        # replay buffer
        self.market_buffer = ReplayBuffer(buffer_size)
        self.trader_buffer = ReplayBuffer(buffer_size)

    def trade(self, market_data, train=False):
        self.market_data = market_data 
        history = []

        market_observation, trader_state = self.reset()
        for t in range(len(market_data) - self.market_observation_time_range):
            action = self.action(market_observation, trader_state)
            
            next_market_observation, next_trader_state, reward, done = self.step(action) 
           
            if train:
                # append data to trader buffer
                self.trader_buffer.push([
                    torch.cat([market_observation, market_action_prob, trader_state], dim=0),
                    action
                    reward,
                    torch.cat([next_market_observation, market_action_prob, trader_state], dim=0)
                    done
                ])
        
                # self.model.train_trader_network()
                # self.model.train_market_network()
                # using reward, train trader_network
                # using observation, perform supervised learning on market_network


            # transition
            market_observation = next_market_observation
            trader_state = next_trader_state

            if t % 100 == 0:
                history.append([self.budget, self.coin_num])

        return history

    def reset(self):
        # reset to initial setting
        self.budget = self.initial_budget
        self.coin_num = 0
        self.current_coin_price = None
        self.timestep = self.market_observation_time_range - 1

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
        # observe data of the market for the last "market_observation_time_range" 
        if self.timestep < self.market_observation_time_range - 1:
            print("ERROR FOUND")
        market_observation = self.market_data[self.timestep - self.market_observation_time_range + 1 : self.timestep + 1] 
        self.current_coin_price = (market_observation[-1][0] + market_observation[-1][3]) / 2 # average of opening price and closing price
        market_observation = torch.from_numpy(market_observation).flatten() 
        trader_state = torch.tensor([self.budget, self.coin_num]) 
        return market_observation, trader_state # seperated observation because market model doesn't observe trader_state 

    def action(self, market_observation, trader_state):
        market_action_prob, trader_action_prob = self.model.policy(market_observation, trader_state)
        trader_action = trader_action_prob.argmax()
         
        match trader_action.item(): # verify if the action is possible to perform. If not, replace with No-op
            case 0: # Buy a coin
                possible_action = 0 if self.current_coin_price < self.budget else 2
            case 1: # Sell a coin
                possible_action = 1 if self.coin_num > 0 else 2 
            case _: # No-op
                possible_action = 2

        possible_action = torch.tensor(possible_action)
        return possible_action

    def reward_function(self, action):
        match action.item():
            case 0: # Buy a coin
                reward = -self.current_coin_price - self.transaction_fee
            case 1: # Sell a coin
                reward = self.current_coin_price - self.transaction_fee
            case _: # No-op
                reward = 0

        return torch.tensor([reward]) #, device=device) 
    
    def transition_function(self, action): 
        match action.item():
            case 0: # Buy a coin
                self.budget -= self.current_coin_price + self.transaction_fee
                self.coin_num += 1
            case 1: # Sell a coin
                self.budget += self.current_coin_price - self.transaction_fee
                self.coin_num -= 1
            case _: # No-op
                pass
        self.timestep += 1

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    def train(self):
        # sample from the replay buffer for each trader and market and train
        # self.model.train_trader_network()
        # self.model.train_market_network()

    def market_reward_function(self, action):
        # gets positive reward?
        # or look for the supervised learning method
        
