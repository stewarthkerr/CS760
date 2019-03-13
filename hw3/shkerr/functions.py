#!/usr/bin/python3.6
import numpy as np
import sys

def lr_train(lr,epoch,train,screen_print = 1):
    #Initialize all weights to match hw3.pdf
    w = np.random.uniform(low=-0.01, high=0.01, size=(1, train.n_units))
    
    #Until we run through number of epochs, do the following look
    i = 1
    while i <= epoch:
        total_error = 0 #Total error for this epoch
        num_correct = 0 #Total number correct for this epoch
        #For each training instance
        for j in range(0,train.length):
            #input x into the network and calculate expected class
            net = w*train.pp_data[j]
            net_sum = np.sum(net)
            sigmoid = 1/(1+np.exp(-net_sum))
            true_class = train.num_labels[j]
            if sigmoid >= 0.5:
                pred_class = 1
            else:
                pred_class = 0
            if pred_class == true_class:
                    num_correct += 1


            #calculate the cross-entropy error
            error = -1.0*true_class*np.log(sigmoid) - (1.0-true_class)*np.log(1.0-sigmoid)
            total_error += error

            #calculate the gradient wrt weights
            gradient = train.pp_data[j]*(pred_class-true_class)

            #update the weights (i.e. the network)
            gradient *= lr
            w -= gradient
        # If we want to print to screen, do it    
        if screen_print:
            print(i, total_error, num_correct, train.length - num_correct)
        #Increment epoch
        i += 1

    #Return the trained weights    
    return(w)

def lr_predict(test,w, screen_print = 1):
    #loop through test
    num_correct = 0
    TP = FP = TN = FN = 0
    predictions = np.zeros(test.length)
    for i in range(0,test.length):
        net = w*test.pp_data[i]
        net_sum = np.sum(net)
        sigmoid = 1/(1+np.exp(-net_sum))
        true_class = test.num_labels[i].astype(int)
        if sigmoid >= 0.5:
            pred_class = 1
            predictions[i] = 1
            if true_class == 1:
                TP += 1
            else:
                FP += 1
        else:
            pred_class = 0
            predictions[i] = 0
            if true_class == 0:
                TN += 1
            else:
                FN += 1

        #If we want to print to screen
        if screen_print:
            print(sigmoid, pred_class, true_class)
    
    #Now that we've looped through all test instances, print total num correct
    if screen_print:
        print(TP+TN, test.length-(TP+TN))
    
    #Calculate/print F1 score
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*precision*recall/(precision+recall)
    if screen_print:
        print(F1)

    return(predictions)
            






