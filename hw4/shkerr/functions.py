#!/usr/bin/python3.6
import numpy as np
import sys
import math
import DecisionTree as dt

def bootstrap(trees,depth,train,test,display = False):
    
    #Build an array for indices/predictions for output
    indices = np.zeros((train.length,trees), dtype = int)
    prediction_labels = np.zeros((test.length,trees), dtype = str)
    prediction_probs = np.zeros((test.length,test.label_length), dtype = float)
    for i in range(0,trees):
        #Randomly sample data from train to use
        bs_sample = np.random.choice(range(0,train.length), size = train.length)
        indices.T[i] = bs_sample
        bs_features = train.features[bs_sample]
        bs_labels = train.labels[bs_sample]

        #Create and train boostrap decision tree
        tree = dt.DecisionTree()
        tree.fit(bs_features, bs_labels, train.metadata, max_depth = depth)

        #Using this tree, do prediction on test
        prediction_probs += tree.predict(test.features, prob = True)
        prediction_labels.T[i] = tree.predict(test.features, prob = False)

    #Now, vote for predicted class using prediction_probs matrix
    predictions = []
    truth = []
    correct = 0
    for i in range(0,test.length):
        #Finds the class that received the most probability
        prediction_index = np.argmax(prediction_probs[i])
        yhat = test.metadata[-1][1][prediction_index]
        y = test.labels[i]
        predictions.append(yhat)
        truth.append(y)

        #Increment number of correct predictions
        if yhat == y:
            correct += 1
    #calculate accuracy
    accuracy = correct/test.length

    if display:
        #Print the tree training indices
        for i in range(0,train.length):
            print(','.join(map(str,indices[i])))
        
        #Print the predictions
        print()
        for i in range(0,test.length):
            print(','.join(prediction_labels[i]), predictions[i], truth[i], sep = ',')  

        #Print accuracy
        print()
        print(accuracy)   

