import os
import json
import random
import sys
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread

"""
 test data : how long it takes to get expected percentage 
"""

keep_going = True

def key_capture_thread():
    global keep_going
    input()
    keep_going = False


# t_min and t_max is the range of normalized, usually they are 0 and 1
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i-min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return np.array(norm_arr, dtype=np.float32)

def de_norm(normal_arr, t_min, t_max, min_arr, max_arr):
    arr = []
    diff = t_max - t_min
    diff_arr = max_arr - min_arr
    for i in normal_arr:
        temp = ((i-t_min)*diff_arr)/diff + min_arr
        arr.append(temp)
    return arr

#make output in range 0,1
def norm_output(output_percent):
    if output_percent > 1.0:
        temp = 1.0
    elif output_percent < -1.0:
        temp = -1.0
    else:
        temp = output_percent

    return (temp + 1.0)/2.0



"""
NN input :

input_days = m, unsold_days = n, stock number = s, stock_date_legth = ls(l1, l2, ..., ls)

[[stock1day1value,stock1day2value,...,stock1day(l1-m-n)value, stock2day1value, stock2day2value,...stock2day(l2-m-n)value,...,stock(s)day(ln-m-n)value],
 [stock1day2value,stock1day3value,...,stock1day(l1-m-n+1)value, stock2day2value, stock2day3value,...stock2day(l2-m-n+1)value,...,stock(s)day(ln-m-n+1)value],
...
 [stock1daymvalue,stock1day(m+1)value,...,stock1day(l1-n)value, stock2daymvalue, stock2day(m+1)value,...stock2day(l2-n)value,...,stock(s)day(ln-n)value],
 [stock1day1volume,stock1day2volume,...,stock1day(l1-m-n)volume, stock2day1volume, stock2day2volume,...stock2day(l2-m-n)volume,...,stock(s)day(ln-m-n)volume],
 [stock1day2volume,stock1day3volume,...,stock1day(l1-m-n+1)volume, stock2day2volume, stock2day3volume,...stock2day(l2-m-n+1)volume,...,stock(s)day(ln-m-n+1)volume],
...
 [stock1daymvvolume,stock1day(m+1)volume,...,stock1day(l1-n)volume, stock2daymvolume, stock2day(m+1)volume,...stock2day(l2-n)volume,...,stock(s)day(ln-n)volume],

NN output :
max percent compared to the value of day m in unsold days
[max of stock1day(m to m+n), max of stock1day(m+1 to(m+n+1), ..., max of stock1day(l1-n to l1),max of stock2day(m to m+n), max of stock2day(m+1 to(m+n+1), ..., max of stock2day(l2-n to l2),
...max of stocksday(m to m+n), max of stocksday(m+1 to(m+n+1), ..., max of stocksday(ls-n to ls)]

"""



def run(train_or_test, input_days, unsold_days):

    f_source = open(".\\train_test_list", 'r')
    json_data = json.load(f_source)
    stock_list = json_data[train_or_test]
    data_folder = json_data['folder']
    f_source.close()

    range_to_normalize = (0, 1)
    nn_input_arr = None
    nn_output_list = []
    is_first = True

    Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()

    for stock in stock_list:
        print("Creating " + train_or_test + "data from " + stock + "...")
        stock_file_path = os.path.join(data_folder, stock)
        f_source = open(stock_file_path, 'r')
        json_data = json.load(f_source)
        rows = json_data['data']['tradesTable']['rows']

        date_length = len(rows)

        #ignore stock without enough date data
        if date_length >= input_days+unsold_days:

            close_value_list = [float(rows[i]['close'].replace('$', '').replace(',', '')) for i in range(date_length)]
            close_value_arr = np.array(close_value_list, dtype=np.float32)

            #ignore stock if it has value 0.0, means invalid data

            if 0.0 not in close_value_arr:

                normalized_close_value_arr = normalize(close_value_arr, range_to_normalize[0], range_to_normalize[1])

                volume_list = [float(rows[i]['volume'].replace(',', '')) for i in range(date_length)]
                volume_arr = np.array(volume_list, dtype=np.float32)
                normalized_volume_arr = normalize(volume_arr, range_to_normalize[0], range_to_normalize[1])

                #create arr for first stock
                if is_first:
                    is_first = False
                    nn_input_arr = np.empty([input_days*2, date_length-input_days-unsold_days+1], dtype=np.float32)
                    for m in range(input_days):
                        nn_input_arr[m] = normalized_close_value_arr[m:date_length-input_days-unsold_days+1+m]
                        nn_input_arr[input_days+m] = normalized_volume_arr[m:date_length-input_days-unsold_days+1+m]
                    samples_length = nn_input_arr.shape[1]
                #stack to arr if already exists
                else:
                    stock_input_arr = np.empty([input_days*2, date_length-input_days-unsold_days+1], dtype=np.float32)
                    for m in range(input_days):
                        stock_input_arr[m] = normalized_close_value_arr[m:date_length-input_days-unsold_days+1+m]
                        stock_input_arr[input_days+m] = normalized_volume_arr[m:date_length-input_days-unsold_days+1+m]
                    samples_length = stock_input_arr.shape[1]
                    nn_input_arr = np.hstack((nn_input_arr, stock_input_arr))

                #find max value within unsold_days. Do not use normalized value, since cannot know the max for future
                for i in range(samples_length):
                    #max_unsold = max(normalized_close_value_arr[i+input_days:i+input_days+unsold_days])
                    #max_percent = (max_unsold-normalized_close_value_arr[i+input_days])/normalized_close_value_arr[i+input_days]
                    #max_diff = max_unsold - normalized_close_value_arr[i+input_days]

                    max_unsold = max(close_value_arr[i+input_days+1:i+input_days+unsold_days])
                    max_percent = (max_unsold-close_value_arr[i+input_days])/close_value_arr[i+input_days]
                    max_percent = norm_output(max_percent)
                    nn_output_list.append(max_percent)

        if not keep_going:
            break

    nn_output_arr = np.array(nn_output_list, dtype=np.float32)

    #input_npz_file = '_'.join([train_or_test,'input',str(input_days),'unsold',str(unsold_days),'nn_input'])
    #output_npz_file = '_'.join([train_or_test,'input',str(input_days),'unsold',str(unsold_days),'nn_output'])

    npz_file = '_'.join([train_or_test,'input',str(input_days),'unsold',str(unsold_days)])

    #np.save(input_npz_file, nn_input_arr)
    #np.save(output_npz_file, nn_output_arr)

    print("Input array shape {}".format(nn_input_arr.shape))
    print("Output array shape {}".format(nn_output_arr.shape))

    np.savez_compressed(npz_file, input=nn_input_arr, output=nn_output_arr)

"""
usage: to create numpy array for NN
train_or_test : 'train' or 'test', by which this app will read from train_test_list file to get data
input_days : how many days data as input of NN
unsold_days: search in unsold_days, the highest value as NN output

"""


if __name__ == '__main__':

    train_or_test = sys.argv[1]
    input_days = int(sys.argv[2])
    unsold_days = int(sys.argv[3])
    run(train_or_test, input_days, unsold_days)
