#!/usr/bin/python3.6
import json
test = "./data/votes_test.json"
train = "./data/votes_train.json"
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
#Standardize both training set and test set

#Initialize distance array 
distance=[[0 for i in range(0,nTrain)] for j in range(0,nTest)]
#Loop through test set
for i in range(0,nTest):
    #Loop through train set
    for j in range(0,nTrain):
        distance[i][j]=0
        #Loop through each feature
        for k in range(0,nFeat):
            if test['data'][i][k] != train['data'][j][k]:
                distance[i][j] += 1
        print("The distance for",i," ",j,":",distance[i][j])
        

#Calculate Manhattan/Hamming distance for each in test set