#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
from functions import roc_knn, calculate_accuracy, predict_label,calculate_conf, calculate_roc

parser = argparse.ArgumentParser(description='Implement a k-NN algorithm')
parser.add_argument("-k", type=int, help="Number of nearest neighbors to look for in choose(n,k)")
parser.add_argument('-train', type = str, help='Train data set path')
parser.add_argument('-test', type = str, help='Test data set path')
args = parser.parse_args()
k = args.k
train = args.train
test = args.test
test = test.replace('\r','') #Removes the carriage return cuz I use windows

#Load the data in
with open(train,"r") as read_file:
    train = json.load(read_file)
with open(test,"r") as read_file:
    test = json.load(read_file)

#First calculate roc_knn (with weights)
#then calculate confidence
#then calculate roc values
nn, nn_weights = roc_knn(k,train,test)
conf = calculate_conf(nn,nn_weights,train,test)
calculate_roc(test,conf)