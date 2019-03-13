#!/usr/bin/python3.6
import numpy as np
import pandas as pd
import sys

#This takes the json input and converts to numpy
#Note - it automatically will standardize numeric features and do one-hot encoding for categorical features
class json2numpy:

    def __init__(self, set):
        self.metadata = np.array(set['metadata']['features'])
        self.data = np.array(set['data'])
        self.length = len(self.data)
        self.feat_length = len(self.metadata)-1
        self.labels = (self.data.T[self.feat_length][:]).T

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

        # One hot encoding & normalizing
        # This replaces each categorical feature with a set of columns
        # (one for each label in feature) where 1 = that observation
        # has that particular label for that particular feature
        feature_vectors = [np.ones(self.length, dtype = float)] #ones handle the intercept
        for feat in range(0,self.feat_length):

            #If feature is numeric, we want to normalize
            if feat in self.numeric:
                    cat_values = (self.data.T[feat]).astype(float)
                    sum = np.sum(cat_values, axis = 0)
                    mean = sum/self.length
                    sqerror = np.sum((cat_values-mean)**2)
                    stddev = np.sqrt(sqerror/self.length)
                    if stddev == 0: 
                        stddev = 1.0
                    #Standardize the set
                    feature_vectors.append((cat_values - mean)/stddev)

            #Else, the feature is categorical and we want one-hot encoding
            else:
                cat_values = (self.data.T[feat])
                num_cat_values = len(self.metadata[feat][1])
                #Split the one-hot encoding so we have one row per feat per value
                for i in range(0,num_cat_values):
                    value_label = self.metadata[feat][1][i]
                    split_encoding = np.equal(cat_values, value_label, dtype = object).astype(float)
                    feature_vectors.append(split_encoding)

        #Pre-processed data
        self.pp_data = np.array(feature_vectors).T

        #Number of units
        self.n_units = len(self.pp_data.T)

        #Want numeric labels
        one_lab = self.metadata[self.feat_length][1][1]
        self.num_labels = (self.labels == one_lab).astype(float)



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

