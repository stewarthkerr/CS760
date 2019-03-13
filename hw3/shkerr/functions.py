#!/usr/bin/python3.6
import numpy as np
import sys

def lr_train(lr,epoch,train):
    #Initialize all weights to match hw3.pdf
    w = np.random.uniform(low=-0.01, high=0.01, size=(1, train.n_units))
    
    #Until we run through number of epochs, do the following look
    i = 0
    while i <= epoch:
        #For each training instance
        for j in range(0,train.length):
            #input x into the network and calculate expected class
            net = w*train.pp_data[j]
            net_sum = np.sum(net)
            sigmoid = 1/(1+np.exp(-net_sum))
            if sigmoid >= 0.5:
                pred_class = 1
            else:
                pred_class = 0
            real_class = train.num_labels[j]

            #calculate the cross-entropy error
            error = -1.0*real_class*np.log(sigmoid) - (1.0-real_class)*np.log(1.0-sigmoid)

            #calculate the gradient wrt weights
            #I think this is derror/dweight?

            #update the weights (i.e. the network)
        i += 1

