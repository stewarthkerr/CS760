#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from functions import nn_train, nn_predict, lr_train, lr_predict
from classes import json2numpy

def main(args):
    lr = args.lr
    n_hu = args.hu
    epoch = args.epoch
    train = args.train
    test = args.test
    test = test.replace('\r','') #Removes the carriage return cuz I use windows
    data_origin = test.replace('_test.json','')
    data_origin = data_origin.replace('data/','')

    #Load the data in and convert to numpy
    with open(train,"r") as read_file:
        train = json.load(read_file)
        train = json2numpy(train)
        train = train.train_pp()
    with open(test,"r") as read_file:
        test = json.load(read_file)
        test = json2numpy(test)
        test = test.test_pp(train)

    xlist = []
    trainlist = []
    testlist = []
    if n_hu > 0:
        title = 'Neural Network: F1 vs. Epochs'
        filename = 'nnet_f1_curve.png'
        caption = ('Dataset: ' + data_origin + ', Learning rate: ' + str(lr) + ' , Number of hidden units: ' + str(n_hu) + ', Maximum number of epochs: ' + str(epoch))
        for e in range(1,epoch+1):
            xlist.append(e)
            x = 'nn'
            w = nn_train(lr,n_hu,e,train, screen_print = 0)
            po = nn_predict(test,w, screen_print = 0)
            testlist.append(po[1])
            po = nn_predict(train,w, screen_print = 0)
            trainlist.append(po[1])

    else:
        title = 'Logistic Regression: F1 vs. Epochs'
        filename = 'logistic_f1_curve.png'
        caption = ('Dataset: ' + data_origin + ', Learning rate: ' + str(lr) + ', Maximum number of epochs: ' + str(epoch))
        for e in range(1,epoch+1):
            xlist.append(e)
            x = 'lr'
            w = lr_train(lr,e, train, screen_print = 0)
            po = lr_predict(test,w, screen_print = 0)
            testlist.append(po[1])
            po = lr_predict(train,w, screen_print = 0)
            trainlist.append(po[1])

    #Plot the data
    plt.plot(xlist,trainlist)
    plt.plot(xlist,testlist)
    plt.legend(['Train','Test'], loc = "lower right")
    plt.xlabel('Number of epochs')
    plt.ylabel('F1')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, epoch+5])
    plt.title(caption, fontsize = 9, wrap = True)
    plt.suptitle(title)
    plt.savefig('./figs/'+filename)

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
