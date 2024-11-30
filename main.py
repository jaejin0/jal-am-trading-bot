import numpy as np
import csv
from trading_bot import TradingBot

def main(training_files, test_file, market_observation_feature_dim, market_observation_time_range, action_dim, trader_state_dim, budget, threshold, transaction_fee, buffer_size, learning_rate, target_update_rate, discount_factor, batch_size, exploration_parameter, exploration_end, exploration_decay, market_prediction_threshold):
    # read csv file
    train_datas = [] 
    for i in range(len(training_files)):
        with open(training_files[i], 'r') as f:
            reader = csv.reader(f)
            train_data = list(reader)
            train_data.reverse()
            train_data = np.array(train_data)
            train_data = train_data[:-1, 3:]
            train_data = train_data.astype(np.float32) 

        train_datas.append(train_data)
    train_datas = np.array(train_datas)

    with open(test_file, 'r') as f:
        reader = csv.reader(f)
        test_data = list(reader)
        test_data.reverse()
        test_data = np.array(test_data)
    
    # process numpy matrix
    test_data = test_data[:-1, 3:]
    test_data = test_data.astype(np.float32)
    
    # create JAL-AM model
    trader = TradingBot(market_observation_feature_dim, market_observation_time_range, action_dim, trader_state_dim, budget, threshold, transaction_fee, buffer_size, learning_rate, target_update_rate, discount_factor, batch_size, exploration_parameter, exploration_end, exploration_decay, market_prediction_threshold) 
    
    # train
    for iteration in range(5):
        for i in range(len(training_files)):
            print("ITERATION: ", iteration)
            result = trader.trade(train_datas[i], train=True)
            # display_result(result) 

    # evaluate
    result = trader.trade(test_data, train=False)
    # display_result(result)

def display_result(result):
    
    print("RESULT")
    for r in result:
        print(r)
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
    # bitcoin price in 2017 and 2021 are extraordinary
    training_files = [dataset_dir + dataset[3], dataset_dir + dataset[4]]
    test_file = dataset_dir + dataset[5]
    market_observation_feature_dim = 6 # open, high, low, close, Volume BTC, Volume USD
    market_observation_time_range = 10 # observe most recent 30 timesteps
    action_dim = 3 # Buy, Sell, No-op
    trader_state_dim = 2 # budget, coin_num
    budget = 100000
    threshold = 10
    transaction_fee = 1

    buffer_size = 100000
    learning_rate = 0.001
    target_update_rate = 0.005
    discount_factor = 0.99
    batch_size = 128
    exploration_parameter = 1.0
    exploration_end = 0.001
    exploration_decay = 0.99999
    market_prediction_threshold = 0.001 # 0.1% up/down in market will be treated as buy/sell
    main(training_files, test_file, market_observation_feature_dim, market_observation_time_range, action_dim, trader_state_dim, budget, threshold, transaction_fee, buffer_size, learning_rate, target_update_rate, discount_factor, batch_size, exploration_parameter, exploration_end, exploration_decay, market_prediction_threshold)
