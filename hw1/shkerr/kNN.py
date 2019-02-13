#!/usr/bin/python3.6
import json
import numpy as np
import sys
import time
test = "./data/digits_test.json"
train = "./data/digits_train.json"
k = 10

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
#Split numeric and categorical features
mean = np.zeros((nFeat))
stddev = np.zeros((nFeat))
num_feat = []
cat_feat = []
for feat in range(0,nFeat):
    if metadata[feat][1] == 'numeric':
        sum = np.sum(train.T[feat], axis = 0) #sums all observations for a specific feature
        mean[feat] = sum/nTrain #Maybe need to do a +1 here?
        sqerror = np.sum((train.T[feat]-mean[feat])**2)
        stddev[feat] = np.sqrt(sqerror/nTrain)
        if stddev[feat] == 0: 
            stddev[feat] = 1
        train[:][feat] = (train[:][feat] - mean[feat])/stddev[feat] #standardize train set
        test[:][feat] = (test[:][feat] - mean[feat])/stddev[feat]   #standardize test set
        num_feat.append(feat) #Builds a list of the indices of numericFeatures
    else:
        cat_feat.append(feat) #Builds a list of categorical features

#Split array into numeric and categorical
train_num = train[:, num_feat]
train_cat = train[:, cat_feat]
test_num = test[:, num_feat]
test_cat = train[:, cat_feat]

#Initialize distance/nn array 
distance = np.zeros((nTest,nTrain))
smallest = np.ones((nTest,k))*np.inf
nn = np.zeros((nTest,k))
#Loop through test set
for i in range(0,nTest):
    #Loop through train set
    for j in range(0,nTrain):
        #Loop through each feature
        for feat in range(0,nFeat):
            if metadata[feat][1] == 'numeric':
                #Calculate numeric distance
                distance[i][j] += abs(test[i][feat]-train[j][feat]) 
            elif test[i][feat] != train[j][feat]:
                #Calculate categorical distance
                #USE NP.DIFF instead!!!
                distance[i][j] += 1
        if distance[i][j] < smallest[i][k-1]:
            # LOOK INTO USING np.argsort or np.argmin
            smallest[i][k-1] = distance[i][j]
            nn[i][k-1] = j #THIS IS WRONG
            smallest[i] = np.sort(smallest[i]) 
        #print("The distance for",i," ",j,":",distance[i][j])
        time.sleep(0.001)
        print(i, " The smallest distances: ", smallest[i])#,"---- The nearest neighbors: ",nn[i])

""" 
#Loop through feature
for k in range(0,nFeat):
    #Loop through test set
    for i in range(0,nTest):
        if metadata[k][1] == 'numeric':
            #calculate numeric distance
            print('true')
        elif test[i][k] != train.T[k]: distance[i][:][k] = 1
            #calculate categorical distance
            #distance[i][:][k] = int(test[i][k] == train[:][k])
"""