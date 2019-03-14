#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
from functions import nn_train, nn_predict, lr_train, lr_predict
from classes import json2numpy

def main(args):
    lr = args.lr
    n_hu = args.hu
    epoch = args.epoch
    train = args.train
    test = args.test
    test = test.replace('\r','') #Removes the carriage return cuz I use windows

    #Load the data in and convert to numpy
    with open(train,"r") as read_file:
        train = json.load(read_file)
        train = json2numpy(train)
    with open(test,"r") as read_file:
        test = json.load(read_file)
        test = json2numpy(test)

    if n_hu > 0:
        for e in range(1,epoch+1):
            x = 'nn'
            w = nn_train(lr,n_hu,e,train, screen_print = 0)
            po = nn_predict(test,w, screen_print = 0)
            print(x, 'test', e, po[1], sep = ',')
            po = nn_predict(train,w, screen_print = 0)
            print(x, 'train',e, po[1], sep = ',')
    else:
        for e in range(1,epoch+1):
            x = 'lr'
            w = lr_train(lr,e, train, screen_print = 0)
            po = lr_predict(test,w, screen_print = 0)
            print(x, 'test',e, po[1], sep = ',')
            po = lr_predict(train,w, screen_print = 0)
            print(x, 'train',e, po[1], sep = ',')
           
    
if __name__ == "__main__": 
    np.random.seed(0)
    parser = argparse.ArgumentParser(description='Implements one layer neural network')
    parser.add_argument('-lr', type = float, help = 'Learning rate')
    parser.add_argument('-hu', type = int, help = 'Number of hidden units')
    parser.add_argument('-epoch', type = int, help = 'Number of training epochs')
    parser.add_argument('-train', type = str, help='Train data set path')
    parser.add_argument('-test', type = str, help='Test data set path')
    args = parser.parse_args()

    main(args)
