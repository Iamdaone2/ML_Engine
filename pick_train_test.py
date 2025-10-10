import os
import json
import random
import sys


"""
 randomly pick stocks (50%) as train set, the other will be test set, save stock file names 
"""


def run(data_folder, train_test_file):

    stock_list = [item for item in os.listdir('.\\'+data_folder) if item.endswith(".json")]

    train_test_dict = {'folder':'', 'train':[], 'test':[]}
    train_test_dict['folder'] = os.path.abspath('.\\'+data_folder)

    for stock in stock_list:
        if random.choice([True, False]):
            train_test_dict['train'].append(stock)
            print("Training set : {}".format(stock))
        else:
            train_test_dict['test'].append(stock)
            print("Testing set : {}".format(stock))

    file_path = os.path.join('.', train_test_file)

    print("\n\n Training set number {}; Training set number {}".format(len(train_test_dict['train']), len(train_test_dict['test'])))
    with open(file_path, 'w') as fp:
        json.dump(train_test_dict, fp)

    fp.close()

if __name__ == '__main__':

    data_folder = sys.argv[1]
    train_test_file = sys.argv[2]
    run(data_folder,train_test_file)
