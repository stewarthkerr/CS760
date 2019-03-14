#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
from functions import nn_train
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

    #lr_train trains the weights for the logistic regression
    w = nn_train(lr,n_hu,epoch,train)

    #lr_predict predicts classes for test set
    #predictions = lr_predict(test,w)


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
