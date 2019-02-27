#!/usr/bin/python3.6
import numpy as np

#Finds the kNN
def naive(train,test):
    #Convert json to array and populate useful variables
    metadata = np.array(train['metadata']['features'])
    train = np.array(train['data'])
    test = np.array(test['data'])
    nTest = len(test)
    nTrain = len(train)
    nFeat = len(metadata)-1 #-1 to remove label
    nLabels = 2 #We are assuming binary classification
    trainLabels = train.T[nFeat][:] #Pulls list of training data labels
    positive = metadata[nFeat][1][0]
    num_pos = np.sum(trainLabels == positive)
    num_neg = np.sum(trainLabels != positive)

    #Subset train to only include positive obs
    train_pos = train[trainLabels == positive]

    #Calculate P(y)
    perc_pos = (num_pos)/(num_pos + num_neg)

    # For each feature and each value of feature
    # calculate P(x|y)
    cond_prob = metadata[0:nFeat]
    for i in range(0,nFeat):
        featValues = metadata[i][1] #Pulls values for this label
        num_featValues = len(featValues)
        for v in range(0,num_featValues):
            trainFeat = train_pos.T[i][:]
            obs_num_pos = train_pos[trainFeat == featValues[v]]
            p = len(obs_num_pos)/num_pos 
            cond_prob[i][1][v] = p

    return num_neg
