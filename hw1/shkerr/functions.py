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
    num_feat = []
    cat_feat = []
    for feat in range(0,nFeat):
        if metadata[feat][1] == 'numeric':
            num_feat.append(feat) #Builds a list of the indices of numericFeatures
        else:
            cat_feat.append(feat) #Builds a list of categorical features
    
    #Split array into numeric and categorical
    train_num = train[:, num_feat]
    train_num = train_num.astype(float)
    train_cat = train[:, cat_feat]
    test_num = test[:, num_feat]
    test_num = test_num.astype(float)
    test_cat = test[:, cat_feat]

    #Standardize numeric features
    for feat in num_feat:
        sum = np.sum(train_num.T[feat], axis = 0) #sums all observations for a specific feature
        mean = sum/nTrain 
        sqerror = np.sum((train_num.T[feat]-mean)**2)
        stddev = np.sqrt(sqerror/nTrain)
        if stddev == 0: 
            stddev = 1.0
        train_num.T[feat][:] = (train_num.T[feat][:] - mean)/stddev #standardize train set
        test_num.T[feat][:] = (test_num.T[feat][:] - mean)/stddev   #standardize test set

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
                #Calculate categorical distance
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
                leading_label = x
        winners.append(leading_label)
    return(winners)

def calculate_accuracy(val,winners):
    metadata = np.array(val['metadata']['features'])
    nFeat = len(metadata)-1 #-1 to remove label
    val = np.array(val['data'])
    truth = val.T[nFeat][:]
    correct = np.sum(winners == truth)
    accuracy = correct / len(truth)
    return(accuracy)


#Learning curve
def knn_lc(k,train,test, perc = 100):
    #Convert json to array and populate useful variables
    metadata = np.array(train['metadata']['features'])
    train = np.array(train['data'])
    test = np.array(test['data'])
    nTest = len(test)
    nTrain = len(train)
    #If percent is not default (100), we split train set
    if perc != 100:
        frac = perc/100
        divnum = np.int(np.floor(frac*nTrain))
        train = train[0:divnum]
        nTrain = len(train)
    nFeat = len(metadata)-1 #-1 to remove label
    nLabels = len(metadata[nFeat][1])

    #Compute mean for each feature from training set (only for continuous features)
    #Compute stddev for each feature from training set (only for numeric)
    #Split numeric and categorical features
    num_feat = []
    cat_feat = []
    for feat in range(0,nFeat):
        if metadata[feat][1] == 'numeric':
            num_feat.append(feat) #Builds a list of the indices of numericFeatures
        else:
            cat_feat.append(feat) #Builds a list of categorical features
    
    #Split array into numeric and categorical
    train_num = train[:, num_feat]
    train_num = train_num.astype(float)
    train_cat = train[:, cat_feat]
    test_num = test[:, num_feat]
    test_num = test_num.astype(float)
    test_cat = test[:, cat_feat]

    #Standardize numeric features
    for feat in num_feat:
        sum = np.sum(train_num.T[feat], axis = 0) #sums all observations for a specific feature
        mean = sum/nTrain 
        sqerror = np.sum((train_num.T[feat]-mean)**2)
        stddev = np.sqrt(sqerror/nTrain)
        if stddev == 0: 
            stddev = 1.0
        train_num.T[feat][:] = (train_num.T[feat][:] - mean)/stddev #standardize train set
        test_num.T[feat][:] = (test_num.T[feat][:] - mean)/stddev   #standardize test set

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
                #Calculate categorical distance
                #Get an array of differences
                cat_diff = np.not_equal(test_cat[i][feat], train_cat.T[feat][:], dtype = object)
                #Add distance back into distance array
                distance[i] += (cat_diff.astype(int))
        #Now, sort by distance and select k nearest neighbors
        smallest = np.argsort(distance[i], axis = 0, kind = "mergesort")
        nn[i] = smallest[0:k]
    return(nn,nTrain)

#Finds the kNN
def roc_knn(k,train,test):
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
    num_feat = []
    cat_feat = []
    for feat in range(0,nFeat):
        if metadata[feat][1] == 'numeric':
            num_feat.append(feat) #Builds a list of the indices of numericFeatures
        else:
            cat_feat.append(feat) #Builds a list of categorical features
    
    #Split array into numeric and categorical
    train_num = train[:, num_feat]
    train_num = train_num.astype(float)
    train_cat = train[:, cat_feat]
    test_num = test[:, num_feat]
    test_num = test_num.astype(float)
    test_cat = test[:, cat_feat]

    #Standardize numeric features
    for feat in num_feat:
        sum = np.sum(train_num.T[feat], axis = 0) #sums all observations for a specific feature
        mean = sum/nTrain 
        sqerror = np.sum((train_num.T[feat]-mean)**2)
        stddev = np.sqrt(sqerror/nTrain)
        if stddev == 0: 
            stddev = 1.0
        train_num.T[feat][:] = (train_num.T[feat][:] - mean)/stddev #standardize train set
        test_num.T[feat][:] = (test_num.T[feat][:] - mean)/stddev   #standardize test set

    #Alg to find kNN with weights
    distance = np.zeros((nTest,nTrain))
    smallest = np.zeros(nTest, dtype = int)
    weight = np.zeros(nTest,dtype = float)
    nn = np.zeros((nTest,k), dtype = int)
    nn_weights = np.zeros((nTest,k), dtype = float)
    epsilon = 1*(10**-5)
    #Loop through test set
    for i in range(0,nTest):
        #Loop through features
        for feat in range(0,nFeat):
            if feat in num_feat:
                #Calculate numeric distance
                distance[i] += abs(test_num[i][feat]-train_num.T[feat][:])
            else:
                #Calculate categorical distance
                #Get an array of differences
                cat_diff = np.not_equal(test_cat[i][feat], train_cat.T[feat][:], dtype = object)
                #Add distance back into distance array
                distance[i] += (cat_diff.astype(int))
        #Now, sort by distance and select k nearest neighbors
        smallest = np.argsort(distance[i], axis = 0, kind = "mergesort")
        weight = 1/(epsilon + distance[i]**2)
        nn_weights[i] = (-weight).sort()
        nn[i] = smallest[0:k]
    return(nn,weight)

