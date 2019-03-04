#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
from functions import naive, tan

parser = argparse.ArgumentParser(description='T-test to see if TAN and naive are different')
parser.add_argument('-data', type = str, help='Data set path')
args = parser.parse_args()
data = args.data
data = data.replace('\r','') #Removes the carriage return cuz I use windows

#Load the data in
with open(data,"r") as read_file:
    data = json.load(read_file)
metadata = np.array(data['metadata']['features'])

#This splits the data into k-sections
def split_data(data,pos,k=10):
    assert pos <= k, "3rd argument must be less than or equal to 2nd argument"
    full = np.array(data['data'])
    split = np.array_split(full, k)
    test = split[pos]
    index = []
    for i in range(1,k+1):
        if i != pos:
            index.append(i-1)
    train = np.concatenate([split[i] for i in index])
    split = [train,test]
    return split

