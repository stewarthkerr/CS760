#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
#See functions.py for code of functions
from functions import knn, predict_label, calculate_accuracy

parser = argparse.ArgumentParser(description='Tune hyperparameters for knn algorithm')
parser.add_argument("-kmax", type = int, help="Highest k to assess")
parser.add_argument('-train', type = str, help='Train data set path')
parser.add_argument('-val', type = str, help='Validation data set path')
parser.add_argument('-test', type = str, help='Test data set path')
args = parser.parse_args()
kmax = args.kmax
train = args.train
val = args.val
test = args.test
test = test.replace('\r','') #Removes the carriage return cuz I use windows

#Load the data in
with open(train,"r") as read_file:
    train = json.load(read_file)
with open(test,"r") as read_file:
    test = json.load(read_file)
with open(val,"r") as read_file:
    val = json.load(read_file)

#Loop through k
accuracy = np.zeros(kmax+1)
for k_index in range(1,kmax+1):
    nn = []
    winners = []
    nn = knn(k_index,train,val)
    winners = predict_label(train,val,nn)
    accuracy[k_index] = calculate_accuracy(val,winners)
    print(k_index, accuracy[k_index], sep = "," )

#Print optimal k
opt_k = np.argmax(accuracy)
print(opt_k)

#Find accuracy on test set
nn = knn(opt_k,train,test)
winners = predict_label(train,test,nn)
test_accuracy = calculate_accuracy(test,winners)
print(test_accuracy)
