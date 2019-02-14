#!/usr/bin/python3.6
import numpy as np

#Finds the kNN
def knn(k,train,test):
    #Convert json to array and populate useful variables
    metadata = np.array(train['metadata']['features'])
    train = np.array(train['data'])
    test = np.array(test['data'])
    nTest = len(test)
    nTrain = len(train)
    nFeat = len(metadata)-1 #-1 to remove label
    nLabels = len(metadata[nFeat][1])

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
            mean[feat] = sum/nTrain 
            sqerror = np.sum((train.T[feat]-mean[feat])**2)
            stddev[feat] = np.sqrt(sqerror/nTrain)
            if stddev[feat] == 0: 
                stddev[feat] = 1
            #train[:][feat] = (train[:][feat] - mean[feat])/stddev[feat] #standardize train set
            #test[:][feat] = (test[:][feat] - mean[feat])/stddev[feat]   #standardize test set
            num_feat.append(feat) #Builds a list of the indices of numericFeatures
        else:
            cat_feat.append(feat) #Builds a list of categorical features

    #Split array into numeric and categorical
    train_num = train[:, num_feat]
    train_cat = train[:, cat_feat]
    test_num = test[:, num_feat]
    test_cat = train[:, cat_feat]

    #Alg to find kNN
    distance = np.zeros((nTest,nTrain))
    smallest = np.zeros(nTest, dtype = int)
    nn = np.zeros((nTest,k), dtype = int)
    #Loop through test set
    for i in range(0,nTest):
        #Loop through features
        for feat in range(0,nFeat):
            if feat in num_feat:
                #Calculate numeric distance
                distance[i] += abs(test_num[i][feat]-train_num.T[feat][:])
            else:
                #Get an array of differences
                cat_diff = np.not_equal(test_cat[i][feat], train_cat.T[feat][:], dtype = object)
                #Add distance back into distance array
                distance[i] += (cat_diff.astype(int))
        #Now, sort by distance and select k nearest neighbors
        smallest = np.argsort(distance[i], axis = 0, kind = "mergesort")
        nn[i] = smallest[0:k]
    return nn

#Writes array of winners to screen
def display_winner(train,test,nn):
    metadata = np.array(train['metadata']['features'])
    train = np.array(train['data'])
    test = np.array(test['data'])
    nTest = len(test)
    nFeat = len(metadata)-1 #-1 to remove label
    nLabels = len(metadata[nFeat][1])

    #Next, I need to let each nn vote:
    for i in range(0,nTest):
        #Reset highest votes and nn_label
        nn_label = []
        highest_votes = 0
        #Loop through nn[i] and build label list
        for neighbor in nn[i]:
            nn_label.append(train[neighbor][nFeat])
        #Loop through labels and build votes
        for l in range(0,nLabels):
            x = metadata[nFeat][1][l] #Pull the l label value, store in x
            num_votes = nn_label.count(x) #Count number of votes for l label
            print(num_votes,end=',') #Print number of votes for l label
            if num_votes > highest_votes:
                highest_votes = num_votes
                winner = x
        print(winner) #After we have counted all votes, print winner and go to new line

#Function to calculate label prediction
def predict_label(train,test,nn):
    metadata = np.array(train['metadata']['features'])
    train = np.array(train['data'])
    test = np.array(test['data'])
    nTest = len(test)
    nFeat = len(metadata)-1 #-1 to remove label
    nLabels = len(metadata[nFeat][1])
    winners = []

    #Next, I need to let each nn vote:
    for i in range(0,nTest):
        #Reset highest votes and nn_label
        nn_label = []
        highest_votes = 0
        #Loop through nn[i] and build label list
        for neighbor in nn[i]:
            nn_label.append(train[neighbor][nFeat])
        #Loop through labels and build votes
        for l in range(0,nLabels):
            x = metadata[nFeat][1][l] #Pull the l label value, store in x
            num_votes = nn_label.count(x) #Count number of votes for l label
            if num_votes > highest_votes:
                highest_votes = num_votes
                winners.append(x)
    return(winners)
