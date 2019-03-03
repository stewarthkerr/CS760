#!/usr/bin/python3.6
import numpy as np
import scipy as sp
import pandas

#Finds the kNN
def naive(train,test):
    #Convert json to array and populate useful variables
    metadata = np.array(train['metadata']['features'])
    train = np.array(train['data'])
    test = np.array(test['data'])
    nTest = len(test)
    nFeat = len(metadata)-1 #-1 to remove label
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

    print()    
    print(num_correct,'\n')

    return 

def tan(train,test):
    #Convert json to array and populate useful variables
    metadata = np.array(train['metadata']['features'])
    train = np.array(train['data'])
    test = np.array(test['data'])
    nTest = len(test)
    nFeat = len(metadata)-1 #-1 to remove label
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
    y2cond_prob = []
    noty2cond_prob = []

    #This loop calculates probabilities of 1 and 2 x's
    for i in range(0,nFeat):
        featValues = metadata[i][1] #Pulls values for this label
        num_featValues = len(featValues)
        posFeat = train_pos.T[i][:]
        negFeat = train_neg.T[i][:]
        ycondprob_temp = []
        notycondprob_temp = []
        y2cond_prob_temp3 = []
        noty2cond_prob_temp3 = []
        for v in range(0,num_featValues):           
            pos_obs = train_pos[posFeat == featValues[v]]
            neg_obs = train_neg[negFeat == featValues[v]]
            pxgiveny = (len(pos_obs)+1)/(num_pos+num_featValues) 
            pxgivennoty = (len(neg_obs)+1)/(num_neg+num_featValues)
            ycondprob_temp.append(pxgiveny)
            notycondprob_temp.append(pxgivennoty)
            y2cond_prob_temp2 = []
            noty2cond_prob_temp2 = []
            for j in range(0,nFeat):
                jfeatValues = metadata[j][1]
                jnum_featValues = len(jfeatValues)
                jposFeat = pos_obs.T[j][:]
                jnegFeat = neg_obs.T[j][:]
                y2cond_prob_temp = []
                noty2cond_prob_temp = []
                for w in range(0,jnum_featValues):
                    jpos_obs = pos_obs[jposFeat == jfeatValues[w]]
                    jneg_obs = neg_obs[jnegFeat == jfeatValues[w]]
                    # Maybe need to multiply by 2 in the denominator
                    jpos_obs_prob = (len(jpos_obs)+1)/(num_pos+(jnum_featValues*num_featValues))
                    jneg_obs_prob = (len(jneg_obs)+1)/(num_neg+(jnum_featValues*num_featValues))
                    y2cond_prob_temp.append(jpos_obs_prob)
                    noty2cond_prob_temp.append(jneg_obs_prob)
                y2cond_prob_temp2.append(y2cond_prob_temp)
                noty2cond_prob_temp2.append(noty2cond_prob_temp)
            y2cond_prob_temp3.append(y2cond_prob_temp2)
            noty2cond_prob_temp3.append(noty2cond_prob_temp2)
        ycond_prob.append(ycondprob_temp)
        notycond_prob.append(notycondprob_temp)
        y2cond_prob.append(y2cond_prob_temp3)
        noty2cond_prob.append(noty2cond_prob_temp3)

    #Now, to calculate mutual info
    mutual_info = np.zeros((nFeat,nFeat))
    for i in range(0,nFeat):
        ifeatValues = metadata[i][1] #Pulls values for this label
        inum_featValues = len(ifeatValues)
        for j in range(0,nFeat):
            jfeatValues = metadata[j][1]
            jnum_featValues = len(jfeatValues)
            total_mutual_info = 0
            for v in range(0,inum_featValues):
                icond_prob = ycond_prob[i][v]
                inotcond_prob = notycond_prob[i][v]
                for w in range(0,jnum_featValues):
                    jcond_prob = ycond_prob[j][w]
                    jnotcond_prob = notycond_prob[j][w]
                    ijcond_prob = y2cond_prob[i][v][j][w]
                    ijnotcond_prob = noty2cond_prob[i][v][j][w]
                    ij_prob = ijcond_prob*perc_pos
                    ijnot_prob = ijnotcond_prob*perc_neg
                    ymutual_info = ij_prob*np.log2(ijcond_prob/(icond_prob*jcond_prob))
                    notymutual_info = ijnot_prob*np.log2(ijnotcond_prob/(inotcond_prob*jnotcond_prob))
                    total_mutual_info += ymutual_info+notymutual_info
            mutual_info[i][j] = total_mutual_info

    # Prim's algorithm  
    # initialize empty edges array and empty MST
    vertex = 0   # initial vertex is first feature
    MST = []
    edges = []
    visited = []
    maxEdge = [None,None,0]
  
    # run prims algorithm until we create an MST
    # that contains every vertex from the graph
    while len(MST) != nFeat-1:
        # mark this vertex as visited
        visited.append(vertex)
        # add each edge to list of potential edges
        for r in range(0, nFeat):
            if mutual_info[vertex][r] != 0:
                edges.append([vertex,r,mutual_info[vertex][r]])
        # find edge with the largest mutual info to a vertex
        # that has not yet been visited
        for e in range(0, len(edges)):
            if edges[e][2] > maxEdge[2] and edges[e][1] not in visited:
                maxEdge = edges[e]
        # remove max weight edge from list of edges
        edges.remove(maxEdge)
        # push max edge to MST
        MST.append(maxEdge)
        # start at new vertex and reset max edge
        vertex = maxEdge[1]
        maxEdge = [None,None,0]
    
    #Add direction to graph and add class
    directed_MST = dict()
    for i in range(0,len(MST)):
        vertex = MST[i][0]
        direction = MST[i][1]
        if direction in directed_MST:
            directed_MST[direction].append(vertex)
        else:
            directed_MST[direction] = [vertex]

    #Add an edge from class to each feature
    for i in range(0,nFeat):
        if i in directed_MST:
            directed_MST[i].append(nFeat)
        else:
            directed_MST[i] = [nFeat]
    
    #Now, I want to output the tree structure
    for i in range(0,nFeat):
        feature = metadata[i][0]
        parents = directed_MST[i]
        if len(parents) == 2:
            parent1 = metadata[parents[0]][0]
            parent2 = metadata[parents[1]][0]
            print(feature, parent1, parent2)
        else:
            parent1 = metadata[parents][0][0]
            print(feature, parent1)
    print()

    #Classification
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
     
    print()
    print(num_correct,'\n')

    return

 