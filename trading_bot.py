import torch
import numpy as np
from collections import namedtuple, deque
import random

from jal_am import JAL_AM 

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque([], maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def push(self, transition):
        self.buffer.append(transition)

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
    def __init__(self, market_observation_feature_dim, market_observation_time_range, action_dim, trader_state_dim, budget, threshold, transaction_fee, buffer_size, learning_rate, target_update_rate, discount_factor, batch_size, exploration_parameter, exploration_end, exploration_decay, market_prediction_threshold, model_dir):
        # model configuration
        self.model = JAL_AM(market_observation_feature_dim * market_observation_time_range, action_dim, trader_state_dim, learning_rate, target_update_rate, discount_factor, batch_size) 
        self.market_observation_time_range = market_observation_time_range
        self.initial_budget = budget
        self.threshold = threshold
        self.transaction_fee = transaction_fee
        self.device = torch.device('mps')
        self.batch_size = batch_size
        self.exploration_parameter = exploration_parameter
        self.exploration_end = exploration_end
        self.exploration_decay = exploration_decay
        self.market_prediction_threshold = market_prediction_threshold
        self.model_dir = model_dir

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

        print("Trading Bot is initialized")

    def trade(self, market_data, train=False):
        print(f"Trade Start... train: {train}")
        self.market_data = market_data 
        total_time = len(market_data) # can be deleted
        history = []

        market_observation, trader_state = self.reset()
        for t in range(20): # len(market_data) - self.market_observation_time_range):
            market_action_prob, action = self.action(market_observation, trader_state)
            market_action = market_action_prob.argmax()

            next_market_observation, next_trader_state, reward, done = self.step(action) 
            next_market_action_prob, _ = self.model.policy(next_market_observation, next_trader_state)             
            if train:
                market_reward = self.market_reward_function(market_observation, market_action, next_market_observation)
                # append data to replay buffer
                self.trader_buffer.push([
                    torch.cat((market_observation, market_action_prob, trader_state), dim=0).detach().unsqueeze(0),
                    action.unsqueeze(0),
                    reward,
                    torch.cat((next_market_observation, next_market_action_prob, next_trader_state), dim=0).detach().unsqueeze(0)
                ])
                self.market_buffer.push([
                    market_observation.unsqueeze(0),
                    market_action.unsqueeze(0),
                    market_reward,
                    next_market_observation.unsqueeze(0)
                ])
                
            if train and len(self.market_buffer) >= self.batch_size and len(self.trader_buffer) >= self.batch_size:
                self.train()
              
            # transition
            market_observation = next_market_observation
            trader_state = next_trader_state

            
            # print("[CURRENT STATE]")
            # print(f"timestep: {t}/{total_time}, current budget: {self.budget}, current holding coins: {self.coin_num}, market_price: {market_observation}")
            history.append([t, total_time, self.budget, self.coin_num, (market_observation[0].item() + market_observation[3].item()) / 2, self.budget + self.coin_num * ((market_observation[0].item() + market_observation[3].item()) / 2), self.exploration_parameter, reward.item()])


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
        market_observation = torch.from_numpy(market_observation).flatten().to(self.device) 
        trader_state = torch.tensor([self.budget, self.coin_num]).to(self.device) 
        return market_observation, trader_state # seperated observation because market model doesn't observe trader_state 

    def action(self, market_observation, trader_state):
        market_action_prob, trader_action_prob = self.model.policy(market_observation, trader_state)
        trader_action = trader_action_prob.argmax()
        
        exploit = random.random()
        self.exploration_parameter = max(self.exploration_end, self.exploration_parameter * self.exploration_decay)
        if exploit <= self.exploration_parameter:
            action = np.random.choice(3)

        match trader_action.item(): # verify if the action is possible to perform. If not, replace with No-op
            case 0: # Buy a coin
                possible_action = 0 if self.current_coin_price < self.budget else 2
            case 1: # Sell a coin
                possible_action = 1 if self.coin_num > 0 else 2 
            case _: # No-op
                possible_action = 2

        possible_action = torch.tensor([possible_action]).to(self.device)
        return market_action_prob, possible_action

    def reward_function(self, action):
        match action.item():
            case 0: # Buy a coin
                reward = -self.current_coin_price - self.transaction_fee
            case 1: # Sell a coin
                reward = self.current_coin_price - self.transaction_fee
            case _: # No-op
                reward = 0

        return torch.tensor([reward], device=self.device) 
    
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

    def train(self):
        # train trader
        trader_transitions = self.trader_buffer.sample(self.batch_size)
        trader_batch = Transition(*zip(*trader_transitions))
        self.model.train_trader_network(trader_batch)

        # train market
        market_transitions = self.market_buffer.sample(self.batch_size)
        market_batch = Transition(*zip(*market_transitions))
        self.model.train_market_network(market_batch)


    def market_reward_function(self, market_observation, market_action, next_market_observation):
        market_change = (((next_market_observation[0] + next_market_observation[3]) / 2) - ((market_observation[0] + market_observation[3]) / 2)) / ((market_observation[0] + market_observation[3]) / 2)
        if market_change > self.market_prediction_threshold: # Bull
            match market_action.item():
                case 0:
                    reward = 1
                case 1:
                    reward = -1
                case _:
                    reward = 0
        elif market_change < -self.market_prediction_threshold:
            match market_action.item():
                case 0:
                    reward = -1
                case 1:
                    reward = 1
                case _:
                    reward = 0
        else:
            match market_action.item():
                case 0 | 1:
                    reward = 0
                case _:
                    reward = 1
        return torch.tensor([reward]).to(self.device)

    def save_model(self, iteration):
        self.model.save_model(self.model_dir, iteration)
