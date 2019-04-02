#!/usr/bin/python3.6
import numpy as np
import sys
import math
import DecisionTree as dt

def bootstrap(trees,depth,train,test,display = False):
    """Performs bootstrap aggregation with a decision tree learner for k-class classification"""
    
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

def adaboost(trees,depth,train,test,display = False): 
    """Implements the adaboost algorithm for k-class classification using a decision tree learner"""

    #initialize weights
    weights = np.ones(train.length)/train.length

    train_truth = train.labels
    test_predictions_list = np.zeros((test.length, trees), dtype = str)
    num_classes = train.label_length
    alpha_list = []
    weights_list = np.zeros((train.length, trees))
    test_possible_labels = [test.metadata[-1][1],]*test.length
    cx = 0
    #Create decision trees
    for i in range(0,trees):
        weights_list.T[i] = weights #Creates a list of weights for output

        #Create and train decision tree
        tree = dt.DecisionTree()
        tree.fit(train.features, train.labels, train.metadata, max_depth = depth, instance_weights = weights)

        #Using this tree, do prediction on train
        train_predictions = tree.predict(train.features, prob = False)
        incorrect = (train_predictions != train_truth).astype(float)

        #Calculate error
        error = np.dot(weights,incorrect) / sum(weights)

        #If error is large, break out of the loop
        if error >= 1 - 1/num_classes:
            trees = i - 1
            break

        #Calculate alpha
        alpha = np.log((1-error)/error) + np.log(num_classes - 1)
        alpha_list.append(alpha)

        #Update weights
        weights = weights * np.exp(alpha*incorrect) 
        weights = weights/sum(weights)

        #Now, to do prediction on test
        test_predictions = tree.predict(test.features, prob = False)
        test_predictions_list.T[i] = test_predictions
        #This is a bit of an ugly workaround
        x = np.vstack((test_predictions,test_predictions)).T
        kclass = (test_possible_labels == x).astype(float)

        #Creates a sum which we will use for overall classification
        cx += alpha * kclass 

    #Classification is the class that maximizes cx
    classifications_index = np.argmax(cx, axis = 1).astype(int)

    #Build the classifications output array and calculate accuracy
    test_truth = test.labels
    classifications = np.take(test.metadata[-1][1], classifications_index)
    accuracy = sum((test_truth == classifications).astype(float)) / test.length 

    if display:
        #Print the tree training weights
        for i in range(0,train.length):
            print(','.join(map(str,weights_list[i])))

        #Print the alphas
        print()
        print(','.join(map(str,alpha_list[:])))
        
        #Print the predictions
        print()
        for i in range(0,test.length):
            print(','.join(test_predictions_list[i]), classifications[i], test_truth[i], sep = ',')  

        #Print accuracy
        print()
        print(accuracy) 
        





        

