#!/usr/bin/python3.6
import numpy as np
import sys

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

def pr_plot_naive(train,test,threshold):
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
    
    #We now have conditional probabilities, next step is to calculate
    #expected class probabilities
    true_positive = 0
    false_positive = 0
    false_negative = 0
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
        if pygx >= threshold:
            if positive == obs[nFeat]: 
                true_positive += 1
            else:
                false_positive += 1        
        else:
            if negative != obs[nFeat]: false_negative += 1
    
    #Now, calculate precision and recall
    if (true_positive+false_positive==0) or (true_positive+false_negative==0):
        PR = [1,0]
    else:
        precision = true_positive/(true_positive+false_positive)
        recall = true_positive/(true_positive+false_negative)
        PR = [precision,recall]

    return PR

def pr_plot_tan(train,test,threshold):
    #Convert json to array and populate useful variables
    metadata = np.array(train['metadata']['features'])
    train = np.array(train['data'])
    test = np.array(test['data'])
    nTrain = len(train)
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
    xi_poscount = []
    xi_negcount = []
    xixj_poscount = []
    xixj_negcount = []

    #This loop calculates probabilities of 1 and 2 x's
    for i in range(0,nFeat):
        featValues = metadata[i][1] #Pulls values for this label
        num_featValues = len(featValues)
        posFeat = train_pos.T[i][:]
        negFeat = train_neg.T[i][:]
        xi_poscount_temp = []
        xi_negcount_temp = []
        xixj_poscount_temp3 = []
        xixj_negcount_temp3 = []
        for v in range(0,num_featValues):           
            pos_obs = train_pos[posFeat == featValues[v]]
            neg_obs = train_neg[negFeat == featValues[v]]
            xi_poscount_temp.append(len(pos_obs))
            xi_negcount_temp.append(len(neg_obs))
            xixj_poscount_temp2 = []
            xixj_negcount_temp2 = []
            for j in range(0,nFeat):
                jfeatValues = metadata[j][1]
                jnum_featValues = len(jfeatValues)
                jposFeat = pos_obs.T[j][:]
                jnegFeat = neg_obs.T[j][:]
                xixj_poscount_temp = []
                xixj_negcount_temp = []
                for w in range(0,jnum_featValues):
                    jpos_obs = pos_obs[jposFeat == jfeatValues[w]]
                    jneg_obs = neg_obs[jnegFeat == jfeatValues[w]]
                    xixj_poscount_temp.append(len(jpos_obs))
                    xixj_negcount_temp.append(len(jneg_obs))
                xixj_poscount_temp2.append(xixj_poscount_temp)
                xixj_negcount_temp2.append(xixj_negcount_temp)
            xixj_poscount_temp3.append(xixj_poscount_temp2)
            xixj_negcount_temp3.append(xixj_negcount_temp2)
        xi_poscount.append(xi_poscount_temp)
        xi_negcount.append(xi_negcount_temp)
        xixj_poscount.append(xixj_poscount_temp3)
        xixj_negcount.append(xixj_negcount_temp3)

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
                i_poscount = xi_poscount[i][v]
                i_negcount = xi_negcount[i][v]
                for w in range(0,jnum_featValues):
                    j_poscount = xi_poscount[j][w]
                    j_negcount = xi_negcount[j][w]
                    ij_poscount = xixj_poscount[i][v][j][w]
                    ij_negcount = xixj_negcount[i][v][j][w]
                    pxixjy = (ij_poscount+1)/((nTrain)+(2*inum_featValues*jnum_featValues))
                    pxixjny = (ij_negcount+1)/((nTrain)+(2*inum_featValues*jnum_featValues))
                    pxixjgy = (ij_poscount+1)/((num_pos)+(inum_featValues*jnum_featValues))
                    pxixjgny = (ij_negcount+1)/((num_neg)+(inum_featValues*jnum_featValues))
                    pxigy = (i_poscount+1)/((num_pos)+(inum_featValues))
                    pxigny = (i_negcount+1)/((num_neg)+(inum_featValues))
                    pxjgy = (j_poscount+1)/((num_pos)+(jnum_featValues))
                    pxjgny = (j_negcount+1)/((num_neg)+(jnum_featValues))
                    ymutual_info = pxixjy*np.log2(pxixjgy/(pxigy*pxjgy))
                    notymutual_info = pxixjny*np.log2(pxixjgny/(pxigny*pxjgny))
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

    #Classification
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(0,nTest):
        obs = test[i]
        pypxgy = perc_pos
        pnypxgny = perc_neg
        for j in range(0,nFeat):
            parents = directed_MST[j] #From directed_MST, get parents
            testfeat = obs[j]
            index = metadata[j][1].index(testfeat)
            featValues = metadata[j][1] #Pulls values for this label
            num_featValues = len(featValues)
            if len(parents) == 2:
                parent1 = parents[0] 
                parentfeat = obs[parent1]
                parentindex = metadata[parent1][1].index(parentfeat)
                ij_poscount = xixj_poscount[j][index][parent1][parentindex]
                ij_negcount = xixj_negcount[j][index][parent1][parentindex]
                parent_poscount = xi_poscount[parent1][parentindex]
                parent_negcount = xi_negcount[parent1][parentindex]
                pxigxjy = (ij_poscount+1)/(parent_poscount+num_featValues)
                pxigxjny = (ij_negcount+1)/(parent_negcount+num_featValues)
                pypxgy = pypxgy * pxigxjy
                pnypxgny = pnypxgny * pxigxjny
            else:
                i_poscount = xi_poscount[j][index]
                i_negcount = xi_negcount[j][index]
                pxigy = (i_poscount+1)/((num_pos)+(num_featValues))
                pxigny = (i_negcount+1)/((num_neg)+(num_featValues))
                pypxgy = pypxgy * pxigy
                pnypxgny = pnypxgny * pxigny
        pygx = (pypxgy)/(pypxgy+pnypxgny)
        if pygx >= threshold:
            if positive == obs[nFeat]: 
                true_positive += 1
            else:
                false_positive += 1        
        else:
            if negative != obs[nFeat]: false_negative += 1
    
    #Now, calculate precision and recall
    if (true_positive+false_positive==0) or (true_positive+false_negative==0):
        PR = [1,0]
    else:
        precision = true_positive/(true_positive+false_positive)
        recall = true_positive/(true_positive+false_negative)
        PR = [precision,recall]

    return PR

def split_data(data,pos,k=10):
    assert pos <= k, "3rd argument must be less than or equal to 2nd argument"
    full = np.array(data['data'])
    split = np.array_split(full, k)
    test = split[pos]
    index = []
    for i in range(1,k+1):
        if i != pos:
            index.append(i-1)
    train = np.concatenate([split[i] for i in index])
    split = [train,test]
    return split

def ttest(train,test,metadata):
    #Convert json to array and populate useful variables
    nTest = len(test)
    nTrain = len(train)
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
    xi_poscount = []
    xi_negcount = []
    xixj_poscount = []
    xixj_negcount = []

    #This loop calculates probabilities of 1 and 2 x's
    for i in range(0,nFeat):
        featValues = metadata[i][1] #Pulls values for this label
        num_featValues = len(featValues)
        posFeat = train_pos.T[i][:]
        negFeat = train_neg.T[i][:]
        xi_poscount_temp = []
        xi_negcount_temp = []
        xixj_poscount_temp3 = []
        xixj_negcount_temp3 = []
        for v in range(0,num_featValues):           
            pos_obs = train_pos[posFeat == featValues[v]]
            neg_obs = train_neg[negFeat == featValues[v]]
            xi_poscount_temp.append(len(pos_obs))
            xi_negcount_temp.append(len(neg_obs))
            xixj_poscount_temp2 = []
            xixj_negcount_temp2 = []
            for j in range(0,nFeat):
                jfeatValues = metadata[j][1]
                jnum_featValues = len(jfeatValues)
                jposFeat = pos_obs.T[j][:]
                jnegFeat = neg_obs.T[j][:]
                xixj_poscount_temp = []
                xixj_negcount_temp = []
                for w in range(0,jnum_featValues):
                    jpos_obs = pos_obs[jposFeat == jfeatValues[w]]
                    jneg_obs = neg_obs[jnegFeat == jfeatValues[w]]
                    xixj_poscount_temp.append(len(jpos_obs))
                    xixj_negcount_temp.append(len(jneg_obs))
                xixj_poscount_temp2.append(xixj_poscount_temp)
                xixj_negcount_temp2.append(xixj_negcount_temp)
            xixj_poscount_temp3.append(xixj_poscount_temp2)
            xixj_negcount_temp3.append(xixj_negcount_temp2)
        xi_poscount.append(xi_poscount_temp)
        xi_negcount.append(xi_negcount_temp)
        xixj_poscount.append(xixj_poscount_temp3)
        xixj_negcount.append(xixj_negcount_temp3)

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
                i_poscount = xi_poscount[i][v]
                i_negcount = xi_negcount[i][v]
                for w in range(0,jnum_featValues):
                    j_poscount = xi_poscount[j][w]
                    j_negcount = xi_negcount[j][w]
                    ij_poscount = xixj_poscount[i][v][j][w]
                    ij_negcount = xixj_negcount[i][v][j][w]
                    pxixjy = (ij_poscount+1)/((nTrain)+(2*inum_featValues*jnum_featValues))
                    pxixjny = (ij_negcount+1)/((nTrain)+(2*inum_featValues*jnum_featValues))
                    pxixjgy = (ij_poscount+1)/((num_pos)+(inum_featValues*jnum_featValues))
                    pxixjgny = (ij_negcount+1)/((num_neg)+(inum_featValues*jnum_featValues))
                    pxigy = (i_poscount+1)/((num_pos)+(inum_featValues))
                    pxigny = (i_negcount+1)/((num_neg)+(inum_featValues))
                    pxjgy = (j_poscount+1)/((num_pos)+(jnum_featValues))
                    pxjgny = (j_negcount+1)/((num_neg)+(jnum_featValues))
                    ymutual_info = pxixjy*np.log2(pxixjgy/(pxigy*pxjgy))
                    notymutual_info = pxixjny*np.log2(pxixjgny/(pxigny*pxjgny))
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
    
    #Classification
    num_correct = 0
    for i in range(0,nTest):
        obs = test[i]
        pypxgy = perc_pos
        pnypxgny = perc_neg
        for j in range(0,nFeat):
            parents = directed_MST[j] #From directed_MST, get parents
            testfeat = obs[j]
            index = metadata[j][1].index(testfeat)
            featValues = metadata[j][1] #Pulls values for this label
            num_featValues = len(featValues)
            if len(parents) == 2:
                parent1 = parents[0] 
                parentfeat = obs[parent1]
                parentindex = metadata[parent1][1].index(parentfeat)
                ij_poscount = xixj_poscount[j][index][parent1][parentindex]
                ij_negcount = xixj_negcount[j][index][parent1][parentindex]
                parent_poscount = xi_poscount[parent1][parentindex]
                parent_negcount = xi_negcount[parent1][parentindex]
                pxigxjy = (ij_poscount+1)/(parent_poscount+num_featValues)
                pxigxjny = (ij_negcount+1)/(parent_negcount+num_featValues)
                pypxgy = pypxgy * pxigxjy
                pnypxgny = pnypxgny * pxigxjny
            else:
                i_poscount = xi_poscount[j][index]
                i_negcount = xi_negcount[j][index]
                pxigy = (i_poscount+1)/((num_pos)+(num_featValues))
                pxigny = (i_negcount+1)/((num_neg)+(num_featValues))
                pypxgy = pypxgy * pxigy
                pnypxgny = pnypxgny * pxigny
        pygx = (pypxgy)/(pypxgy+pnypxgny)
        if pygx >= 0.5:
            if positive == obs[nFeat]: num_correct += 1
        else:
            if negative == obs[nFeat]: num_correct += 1
    tan_correct = num_correct

    #Classification - Naive
    num_correct = 0
    for i in range(0,nTest):
        obs = test[i]
        pypxgy = perc_pos
        pnypxgny = perc_neg
        for j in range(0,nFeat):
            featValues = metadata[j][1] #Pulls values for this label
            num_featValues = len(featValues)
            testfeat = obs[j]
            index = metadata[j][1].index(testfeat)
            i_poscount = xi_poscount[j][index]
            i_negcount = xi_negcount[j][index]
            pxigy = (i_poscount+1)/((num_pos)+(num_featValues))
            pxigny = (i_negcount+1)/((num_neg)+(num_featValues))
            pypxgy = pypxgy * pxigy
            pnypxgny = pnypxgny * pxigny
        pygx = (pypxgy)/(pypxgy+pnypxgny)
        if pygx >= 0.5:
            if positive == obs[nFeat]: num_correct += 1
        else:
            if negative == obs[nFeat]: num_correct += 1
    naive_correct = num_correct
    correct = [naive_correct, tan_correct]
 
    return correct

def tan(train,test):
    #Convert json to array and populate useful variables
    metadata = np.array(train['metadata']['features'])
    train = np.array(train['data'])
    test = np.array(test['data'])
    nTrain = len(train)
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
    xi_poscount = []
    xi_negcount = []
    xixj_poscount = []
    xixj_negcount = []

    #This loop calculates probabilities of 1 and 2 x's
    for i in range(0,nFeat):
        featValues = metadata[i][1] #Pulls values for this label
        num_featValues = len(featValues)
        posFeat = train_pos.T[i][:]
        negFeat = train_neg.T[i][:]
        xi_poscount_temp = []
        xi_negcount_temp = []
        xixj_poscount_temp3 = []
        xixj_negcount_temp3 = []
        for v in range(0,num_featValues):           
            pos_obs = train_pos[posFeat == featValues[v]]
            neg_obs = train_neg[negFeat == featValues[v]]
            xi_poscount_temp.append(len(pos_obs))
            xi_negcount_temp.append(len(neg_obs))
            xixj_poscount_temp2 = []
            xixj_negcount_temp2 = []
            for j in range(0,nFeat):
                jfeatValues = metadata[j][1]
                jnum_featValues = len(jfeatValues)
                jposFeat = pos_obs.T[j][:]
                jnegFeat = neg_obs.T[j][:]
                xixj_poscount_temp = []
                xixj_negcount_temp = []
                for w in range(0,jnum_featValues):
                    jpos_obs = pos_obs[jposFeat == jfeatValues[w]]
                    jneg_obs = neg_obs[jnegFeat == jfeatValues[w]]
                    xixj_poscount_temp.append(len(jpos_obs))
                    xixj_negcount_temp.append(len(jneg_obs))
                xixj_poscount_temp2.append(xixj_poscount_temp)
                xixj_negcount_temp2.append(xixj_negcount_temp)
            xixj_poscount_temp3.append(xixj_poscount_temp2)
            xixj_negcount_temp3.append(xixj_negcount_temp2)
        xi_poscount.append(xi_poscount_temp)
        xi_negcount.append(xi_negcount_temp)
        xixj_poscount.append(xixj_poscount_temp3)
        xixj_negcount.append(xixj_negcount_temp3)

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
                i_poscount = xi_poscount[i][v]
                i_negcount = xi_negcount[i][v]
                for w in range(0,jnum_featValues):
                    j_poscount = xi_poscount[j][w]
                    j_negcount = xi_negcount[j][w]
                    ij_poscount = xixj_poscount[i][v][j][w]
                    ij_negcount = xixj_negcount[i][v][j][w]
                    pxixjy = (ij_poscount+1)/((nTrain)+(2*inum_featValues*jnum_featValues))
                    pxixjny = (ij_negcount+1)/((nTrain)+(2*inum_featValues*jnum_featValues))
                    pxixjgy = (ij_poscount+1)/((num_pos)+(inum_featValues*jnum_featValues))
                    pxixjgny = (ij_negcount+1)/((num_neg)+(inum_featValues*jnum_featValues))
                    pxigy = (i_poscount+1)/((num_pos)+(inum_featValues))
                    pxigny = (i_negcount+1)/((num_neg)+(inum_featValues))
                    pxjgy = (j_poscount+1)/((num_pos)+(jnum_featValues))
                    pxjgny = (j_negcount+1)/((num_neg)+(jnum_featValues))
                    ymutual_info = pxixjy*np.log2(pxixjgy/(pxigy*pxjgy))
                    notymutual_info = pxixjny*np.log2(pxixjgny/(pxigny*pxjgny))
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
            parents = directed_MST[j] #From directed_MST, get parents
            testfeat = obs[j]
            index = metadata[j][1].index(testfeat)
            featValues = metadata[j][1] #Pulls values for this label
            num_featValues = len(featValues)
            if len(parents) == 2:
                parent1 = parents[0] 
                parentfeat = obs[parent1]
                parentindex = metadata[parent1][1].index(parentfeat)
                ij_poscount = xixj_poscount[j][index][parent1][parentindex]
                ij_negcount = xixj_negcount[j][index][parent1][parentindex]
                parent_poscount = xi_poscount[parent1][parentindex]
                parent_negcount = xi_negcount[parent1][parentindex]
                pxigxjy = (ij_poscount+1)/(parent_poscount+num_featValues)
                pxigxjny = (ij_negcount+1)/(parent_negcount+num_featValues)
                pypxgy = pypxgy * pxigxjy
                pnypxgny = pnypxgny * pxigxjny
            else:
                i_poscount = xi_poscount[j][index]
                i_negcount = xi_negcount[j][index]
                pxigy = (i_poscount+1)/((num_pos)+(num_featValues))
                pxigny = (i_negcount+1)/((num_neg)+(num_featValues))
                pypxgy = pypxgy * pxigy
                pnypxgny = pnypxgny * pxigny
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



