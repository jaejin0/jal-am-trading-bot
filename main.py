import numpy as np
import csv

def main(training_file, test_file):
    # read csv file
    with open(training_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data_array = np.array(data)
    
    print(data_array)

    # call JAL-AM model, train, and test it


    pass


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

    # configuration
    training_file = dataset_dir + dataset[1]
    test_file = dataset_dir + dataset[0]
    
    main(training_file=training_file, test_file=test_file)
