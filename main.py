import numpy as np
import csv
from trading_bot import TradingBot

def main(training_file, test_file, observation_dim, action_dim, status_dim, budget, threshold, transaction_fee):
    # read csv file
    with open(training_file, 'r') as f:
        reader = csv.reader(f)
        train_data = list(reader)
        train_data.reverse()
        train_data = np.array(train_data)
    
    with open(test_file, 'r') as f:
        reader = csv.reader(f)
        test_data = list(reader)
        test_data.reverse()
        test_data = np.array(test_data)
    
    # process numpy matrix
    train_data = train_data[:-1, 3:]
    train_data = train_data.astype(float) 
    test_data = test_data[:-1, 3:]
    test_data = test_data.astype(float)
    
    # create JAL-AM model
    agent = TradingBot(observation_dim, action_dim, status_dim, budget, threshold, transaction_fee) 
    
    # train
    result = agent.train(train_data)
    self.display_result(result)

    # evaluate
    result = agent.eval(test_data)
    self.display_result(result)

def display_result(result):
    print(result)
    # print final budget, number of coin holding, and total networth
    # networth = final budget + number of coin * last price

if __name__ == '__main__':
    
    # dataset
    dataset_dir = "./dataset/"
    dataset = [
        "BTC-Hourly.csv",
        "BTC-Daily.csv",
        "BTC-2017min.csv", 
        "BTC-2018min.csv",
        "BTC-2019min.csv",
        "BTC-2020min.csv",
        "BTC-2021min.csv",
    ]
    '''
    format
    unix,date,symbol,open,high,low,close,Volume BTC,Volume USD
    1646092800,2022-03-01 00:00:00,BTC/USD,43221.71,43626.49,43185.48,43185.48,49.00628870,2116360.1005280763
    '''

    # configuration
    training_file = dataset_dir + dataset[1]
    test_file = dataset_dir + dataset[0]
    observation_dim = 6 # open, high, low, close, Volume BTC, Volume USD
    action_dim = 3 # Buy, Sell, No-op
    status_dim = 2 # budget, coin_num
    budget = 1000
    threshold = 2
    transaction_fee = 0.5

    main(training_file, test_file, observation_dim, action_dim, status_dim, budget, threshold, transaction_fee)
