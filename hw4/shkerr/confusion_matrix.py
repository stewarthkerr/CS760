#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
import DecisionTree
from functions import adaboost, bootstrap, build_confusion_matrix
from classes import json2numpy

def main(args):
    method = args.method
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

    #Pull the true labels
    truth = test.labels
    #Generate the predictions from the ensemble
    if method == 'bag':
        predictions = bootstrap(trees, depth, train, test, display = False)
    elif method == 'boost':
        predictions = adaboost(trees, depth, train, test, display = False)
    else:
        print('Invalid ensemble method.')

    build_confusion_matrix(test, predictions, verbose = True)


if __name__ == "__main__": 
    np.random.seed(0)
    parser = argparse.ArgumentParser(description='Creates a confusion matrix for given method')
    parser.add_argument('-method', type = str, help = 'Ensemble method to use')
    parser.add_argument('-trees', type = int, help = 'Number of trees')
    parser.add_argument('-depth', type = int, help = 'Maximum depth')
    parser.add_argument('-train', type = str, help='Train data set path')
    parser.add_argument('-test', type = str, help='Test data set path')
    args = parser.parse_args()

    main(args)
