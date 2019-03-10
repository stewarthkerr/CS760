#!/usr/bin/python3.6
import numpy as np
import sys

class json2numpy:

    def __init__(self, set):
        self.metadata = np.array(set['metadata']['features'])
        self.data = np.array(set['data'])
        self.length = len(self.data)
        self.feat_length = len(self.metadata)-1

        #Build list of numeric and categorical features
        num_feat = []
        cat_feat = []
        for feat in range(0,self.feat_length):
            if self.metadata[feat][1] == 'numeric':
                num_feat.append(feat) #Builds a list of the indices of numericFeatures
            else:
                cat_feat.append(feat) #Builds a list of categorical features
        self.numeric = num_feat
        self.categorical = cat_feat
        
        #Split dataset into numeric and categorical
        self.data_numeric = (self.data[:, self.numeric]).astype(float)
        self.data_categorical = self.data[:, self.categorical]

    #Method to standardize numeric features
    def standardize_num(self):
        for feat in range(0,len(self.numeric)):
            sum = np.sum(self.data_numeric.T[feat], axis = 0) #sums all observations for a specific feature
            mean = sum/self.length
            sqerror = np.sum((self.data_numeric.T[feat]-mean)**2)
            stddev = np.sqrt(sqerror/self.length)
            if stddev == 0: 
                stddev = 1.0
            #Standardize the set
            self.data_numeric.T[feat][:] = (self.data_numeric.T[feat][:] - mean)/stddev 
        return()







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

