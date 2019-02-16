#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
from functions import knn_roc, calculate_accuracy, predict_label

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


#nn = knn_roc(k,train,test)
#print(nn)
"""
let be the test-set instances sorted according to predicted confidence
c (i) that each instance is positive
let num_neg, num_pos be the number of negative/positive instances in the test set
TP = 0, FP = 0
last_TP = 0
for i = 1 to m
// find thresholds where there is a pos instance on high side, neg instance on low side
if (i > 1) and ( c (i) ≠ c (i-1) ) and ( y (i) == neg ) and ( TP > last_TP )
FPR = FP / num_neg, TPR = TP / num_pos
output (FPR, TPR) coordinate
last_TP = TP
if y (i) == pos
++TP
else
++FP
FPR = FP / num_neg, TPR = TP / num_pos
output (FPR, TPR) coordinate
"""
