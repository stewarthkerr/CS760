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
    negative = metadata[nFeat][1][1]
    num_pos = np.sum(trainLabels == positive)
    num_neg = np.sum(trainLabels != positive)

    #Subset train to only include positive obs
    train_pos = train[trainLabels == positive]
    train_neg = train[trainLabels != positive]

    #Calculate P(y) with Laplace estimate
    perc_pos = (num_pos+1)/((num_pos+1) + (num_neg+1)) 
    perc_neg = (num_neg+1)/((num_pos+1) + (num_neg+1))

    # For each feature and each value of feature
    # calculate P(x|y)
    ycond_prob = []
    notycond_prob = []

    for i in range(0,nFeat):
        print(metadata[i][0], 'class') #this is okay because naive
        featValues = metadata[i][1] #Pulls values for this label
        num_featValues = len(featValues)
        posFeat = train_pos.T[i][:]
        negFeat = train_neg.T[i][:]
        ycondprob_temp = []
        notycondprob_temp = []
        for v in range(0,num_featValues):           
            pos_obs_num = len(train_pos[posFeat == featValues[v]])
            neg_obs_num = len(train_neg[negFeat == featValues[v]])
            pxgiveny = (pos_obs_num+1)/(num_pos+num_featValues) 
            pxgivennoty = (neg_obs_num+1)/(num_neg+num_featValues)
            ycondprob_temp.append(pxgiveny)
            notycondprob_temp.append(pxgivennoty)
        ycond_prob.append(ycondprob_temp)
        notycond_prob.append(notycondprob_temp)
    
    print() #print a new line to match expected output

    #We now have conditional probabilities, next step is to calculate
    #expected class probabilities
    num_correct = 0
    for i in range(0,nTest):
        obs = test[i]
        pypxgy = perc_pos
        pnypxgny = perc_neg
        for j in range(0,nFeat):
            testfeat = obs[j]
            index = metadata[j][1].index(testfeat)
            pypxgy = pypxgy * ycond_prob[j][index]
            pnypxgny = pnypxgny * notycond_prob[j][index]
        pygx = (pypxgy)/(pypxgy+pnypxgny)
        if pygx >= 0.5:
            print(positive, obs[nFeat], "{:10.12f}".format(pygx))
            if positive == obs[nFeat]: num_correct += 1
        else:
            print(negative, obs[nFeat], "{:10.12f}".format(1-pygx))
            if negative == obs[nFeat]: num_correct += 1
        
    print('\n' , num_correct, '\n')

    return 
 