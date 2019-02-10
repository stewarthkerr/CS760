#!/usr/bin/python3.6
import json
import numpy as np
test = "./data/digits_test.json"
train = "./data/digits_train.json"
k = 5

#Maybe create a dataset object instead

#Load the data in
with open(test,"r") as read_file:
    test = json.load(read_file)
with open(train,"r") as read_file:
    train = json.load(read_file)
nTest = len(test['data'])
nTrain = len(train['data'])
nFeat = len(test['metadata']['features'])

#Get length of test

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

time.sleep(50)
#Standardize both training set and test set

#Initialize distance array 
distance = np.zeros((nTest,nTrain))
#Loop through test set
for i in range(0,nTest):
    #Loop through train set
    for j in range(0,nTrain):
        #Loop through each feature
        for k in range(0,nFeat):
            if test['metadata']['features'][k][1] == 'numeric':
                #Calculate numeric distance after standardizing
                distance[i][j] += abs(test['data'][i][k]-train['data'][j][k]) 
            elif test['data'][i][k] != train['data'][j][k]:
                #Calculate categorical distance
                distance[i][j] += 1
        print("The distance for",i," ",j,":",distance[i][j])
        

#Calculate Manhattan/Hamming distance for each in test set