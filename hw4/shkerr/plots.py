#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from functions import bootstrap, adaboost
from classes import json2numpy

def main(args):
    method = args.method
    trees = args.trees
    depth = args.depth
    train = args.train
    test = args.test
    test = test.replace('\r','') #Removes the carriage return cuz I use windows
    data_origin = test.replace('_test.json','')
    data_origin = data_origin.replace('data/','')

    #Load the data in and convert to numpy
    with open(train,"r") as read_file:
        train = json.load(read_file)
        train = json2numpy(train)
    with open(test,"r") as read_file:
        test = json.load(read_file)
        test = json2numpy(test)


    #Pull the true labels
    truth = test.labels
    #Generate the predictions from the ensemble, create list of coordinates
    xlist = []
    ylist = []
    if method == 'bag':
        title = 'Decision Tree Bagging: Accuracy vs. Number of trees'
        filename = 'bagged_tree_plot.pdf'
        caption = ('Dataset: ' + data_origin)
        depth1 = depth - 2
        depth2 = depth 
        depth3 = depth + 2
        for j in [depth1,depth2,depth3]:
            for i in range(1,trees+1):
                predictions = bootstrap(i, j, train, test, display = False)
                accuracy = sum((truth == predictions).astype(float)) / test.length 
                xlist.append(i)
                ylist.append(accuracy)
            plt.plot(xlist,ylist)
            xlist = []
            ylist = []
    elif method == 'boost':
        title = 'Decision Tree AdaBoost: Accuracy vs. Number of trees'
        filename = 'boosted_tree_plot.pdf'
        caption = ('Dataset: ' + data_origin)
        depth1 = depth - 1
        depth2 = depth 
        depth3 = depth + 1
        for j in [depth1,depth2,depth3]:
            for i in range(1,trees+1):
                predictions = adaboost(i, j, train, test, display = False)
                accuracy = sum((truth == predictions).astype(float)) / test.length 
                xlist.append(i)
                ylist.append(accuracy)
            plt.plot(xlist,ylist)
            xlist = []
            ylist = []
    else:
        print('Invalid ensemble method.')
    
    #Plot the data
    plt.legend([depth1,depth2,depth3], loc = "lower right", title = "Maximum tree depth", fancybox = True)
    plt.xlabel('Ensemble size')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1.05])
    plt.xlim([1, trees])
    plt.title(caption, fontsize = 9, wrap = True)
    plt.suptitle(title)
    plt.savefig(filename)

if __name__ == "__main__": 
    np.random.seed(0)
    parser = argparse.ArgumentParser(description='Plots accuracy for the ensemble method')
    parser.add_argument('-method', type = str, help = 'Ensemble method to use')
    parser.add_argument('-trees', type = int, help = 'Number of trees')
    parser.add_argument('-depth', type = int, help = 'Maximum depth')
    parser.add_argument('-train', type = str, help='Train data set path')
    parser.add_argument('-test', type = str, help='Test data set path')
    args = parser.parse_args()

    main(args)
