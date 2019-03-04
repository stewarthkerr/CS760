#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
from functions import ttest, split_data

parser = argparse.ArgumentParser(description='T-test to see if TAN and naive are different')
parser.add_argument('-data', type = str, help='Data set path')
args = parser.parse_args()
data = args.data
data = data.replace('\r','') #Removes the carriage return cuz I use windows

#Load the data in
with open(data,"r") as read_file:
    data = json.load(read_file)
metadata = np.array(data['metadata']['features'])

#Build a list of list containing accuracy for naive and TAN
accuracy = []
for i in range(0,10):
    split = split_data(data,i,10)
    train = split[0]
    test = split[1]
    accuracy.append(ttest(train,test,metadata))

