#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
from functions import naive

parser = argparse.ArgumentParser(description='Implements a Bayes net')
parser.add_argument('-train', type = str, help='Train data set path')
parser.add_argument('-test', type = str, help='Test data set path')
parser.add_argument("-bntype", type=str, help="Type of Bayes net to implement (n = naive, t = TAN")
args = parser.parse_args()
train = args.train
test = args.test
bntype = args.bntype
bntype = bntype.replace('\r','') #Removes the carriage return cuz I use windows

#Load the data in
with open(train,"r") as read_file:
    train = json.load(read_file)
with open(test,"r") as read_file:
    test = json.load(read_file)

if bntype == 'n':
    naive(train,test)
    print('naive')
elif bntype == 't':
    #Do TAN
    print('TAN')
else:
    print("3rd argument bntype is not valid. Expect 'n' or 't'.")