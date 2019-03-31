#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
import DecisionTree
from functions import bootstrap
from classes import json2numpy

def main(args):
    trees = args.trees
    depth = args.depth
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

    bootstrap(trees, depth, train, test, display = True)


if __name__ == "__main__": 
    np.random.seed(0)
    parser = argparse.ArgumentParser(description='Implements a bagged decision tree classifier')
    parser.add_argument('-trees', type = int, help = 'Number of trees')
    parser.add_argument('-depth', type = int, help = 'Maximum depth')
    parser.add_argument('-train', type = str, help='Train data set path')
    parser.add_argument('-test', type = str, help='Test data set path')
    args = parser.parse_args()

    main(args)
