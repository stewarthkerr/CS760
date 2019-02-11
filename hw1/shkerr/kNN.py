#!/usr/bin/python3.6
import json
import numpy as np
import sys
test = "./data/digits_test.json"
train = "./data/digits_train.json"
k = 5

#Maybe create a dataset object instead

#Load the data in
with open(train,"r") as read_file:
    train = json.load(read_file)
    metadata = np.array(train['metadata']['features'])
    train = np.array(train['data'])
with open(test,"r") as read_file:
    test = json.load(read_file)
    test = np.array(test['data'])
nTest = len(test)
nTrain = len(train)
nFeat = len(metadata)-1 #-1 to remove label

#test['metadata'] holds the metadata
#test['data'] holds the data
#test['data'][n] holds the data for observation n
#len(test['metadata']['features']) gets # of features + class
#len(test['data']) gets # of observations

#Compute mean for each feature from training set (only for continuous features)
#Compute stddev for each feature from training set (only for numeric)
mean = np.zeros((nFeat))
stddev = np.zeros((nFeat))
for k in range(0,nFeat):
    if test['metadata']['features'][k][1] == 'numeric':
        sum = 0
        for j in range(0,nTrain):
            sum += train['data'][j][k]
        mean[k] = sum/nTrain #Maybe need to do a +1 here?
        sqerror = 0
        for j in range(0,nTrain):
            sqerror += (train['data'][j][k]-mean[k])**2
        stddev[k] = np.sqrt(sqerror/nTrain)
        if stddev[k] == 0: stddev[k] = 1
        print("k ",k," Mean ",mean[k]," Stdev ",stddev[k])
        

#Standardize both training set and test set - maybe can move to loop later


#Initialize distance array 
distance = np.zeros((nTest,nTrain))
#Loop through test set
for i in range(0,nTest):
    #Loop through train set
    for j in range(0,nTrain):
        #Loop through each feature
        for k in range(0,nFeat):
            if test['metadata']['features'][k][1] == 'numeric':
                #sum = np.sum(train['data'], axis = 0)
                #Calculate numeric distance after standardizing
                distance[i][j] += abs(test['data'][i][k]-train['data'][j][k]) 
            elif test['data'][i][k] != train['data'][j][k]:
                #Calculate categorical distance
                distance[i][j] += 1
        print("The distance for",i," ",j,":",distance[i][j])
        

#Calculate Manhattan/Hamming distance for each in test set