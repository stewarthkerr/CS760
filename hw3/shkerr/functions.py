#!/usr/bin/python3.6
import numpy as np
import sys
import math

def lr_train(lr,epoch,train,screen_print = 1):
    #Initialize all weights to match hw3.pdf
    w = np.random.uniform(low=-0.01, high=0.01, size=(1, train.n_units))
    
    #Until we run through number of epochs, do the following 
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
            gradient = train.pp_data[j]*(sigmoid-true_class)

            #update the weights (i.e. the network)
            gradient *= lr
            w -= gradient

        # If we want to print to screen, do it    
        if screen_print:
            print(i, "{:10.12f}".format(total_error), num_correct, train.length - num_correct)
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
        sigmoid = 1.0/(1.0+np.exp(-net_sum))
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
            print("{:10.12f}".format(sigmoid), pred_class, true_class)
    
    #Now that we've looped through all test instances, print total num correct
    if screen_print:
        print(TP+TN, test.length-(TP+TN))
    
    #Calculate/print F1 score
    if (TP+FP) != 0:
        precision = TP/(TP+FP)
    else:
        precision = 0

    if (TP+FN) != 0:
        recall = TP/(TP+FN)
    else:
        recall = 0
    if (precision+recall) != 0:
        F1 = 2*precision*recall/(precision+recall)
    else:
        F1 = 0.0
    if screen_print:
        print("{:10.12f}".format(F1))

    return(predictions, F1)

def nn_train(lr,n_hu,epoch,train,screen_print = 1):
    #Initialize weights and hidden units vector
    w_i_h = np.random.uniform(low=-0.01, high=0.01, size=(n_hu, train.n_units))
    w_h_o = np.random.uniform(low=-0.01, high=0.01, size=(1, n_hu+1))

    i = 1
    #Until we run through number of epochs, do the following
    while i <= epoch:
        total_error = 0 #Total error for this epoch
        num_correct = 0 #Total number correct for this epoch
        #For each training instance
        for j in range(0,train.length):
            #using inputs, calculate hidden units according to initial weights
            a1 = np.dot(w_i_h,train.pp_data[j])
            a1 = 1/(1+np.exp(-a1)) #Activation of hidden units
            hu = np.insert(a1,0,1.0) #prepend bias unit

            #using hidden units, calculate expected class
            a2 = np.dot(w_h_o,hu)
            a2 = 1/(1+np.exp(-a2[0]))
            true_class = train.num_labels[j]
            if a2 >= 0.5:
                pred_class = 1
            else:
                pred_class = 0
            if pred_class == true_class:
                    num_correct += 1

            #calculate the cross-entropy error
            error = -1.0*true_class*np.log(a2) - (1.0-true_class)*np.log(1.0-a2)
            total_error += error

            #calculate errors for back propagation
            output_error = a2 - true_class
            hu_error = a1*(1-a1) * (w_h_o[0][1:] * output_error)

            #calculate the gradient wrt weights 
            g2 = hu*output_error
            g1 = np.multiply.outer(hu_error, train.pp_data[j])

            #update the weights (i.e. the network)
            g2 *= lr
            g1 *= lr
            w_h_o -= g2
            w_i_h -= g1

        # If we want to print to screen, do it    
        if screen_print:
            print(i, "{:10.12f}".format(total_error), num_correct, train.length - num_correct)
        #Increment epoch
        i += 1

    #Return the trained weights    
    return(w_i_h, w_h_o)

def nn_predict(test,w, screen_print = 1):
    w_i_h = w[0]
    w_h_o = w[1]
    #loop through test
    num_correct = 0
    TP = FP = TN = FN = 0
    predictions = np.zeros(test.length)
    for i in range(0,test.length):
        a1 = np.sum(w_i_h*test.pp_data[i], axis = 1)
        a1 = 1/(1+np.exp(-a1)) #Activation of hidden units
        hu = np.insert(a1,0,1.0) #prepend bias unit

        a2 = np.sum(w_h_o*hu)
        a2 = 1/(1+np.exp(-a2))
        true_class = test.num_labels[i].astype(int)
        if a2 >= 0.5:
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
            print("{:10.12f}".format(a2), pred_class, true_class)
    
    #Now that we've looped through all test instances, print total num correct
    if screen_print:
        print(TP+TN, test.length-(TP+TN))
    
    #Calculate/print F1 score
    if (TP+FP) != 0:
        precision = TP/(TP+FP)
    else:
        precision = 0

    if (TP+FN) != 0:
        recall = TP/(TP+FN)
    else:
        recall = 0
    if (precision+recall) != 0:
        F1 = 2*precision*recall/(precision+recall)
    else:
        F1 = 0.0
    if screen_print:
        print("{:10.12f}".format(F1))

    return(predictions, F1)


    