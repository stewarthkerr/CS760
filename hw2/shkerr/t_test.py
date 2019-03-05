#!/usr/bin/python3.6
import json
import numpy as np
from scipy.stats import t
import sys
import argparse
from functions import ttest, split_data

parser = argparse.ArgumentParser(description='T-test to see if TAN and naive are different')
parser.add_argument('-data', type = str, help='Data set path')
args = parser.parse_args()
data = args.data
data = data.replace('\r','') #Removes the carriage return cuz I use windows

#Load the data in
with open(data,"r") as read_file:
    data = json.load(read_file)
metadata = np.array(data['metadata']['features'])

#Build a list of list containing accuracy for naive and TAN
accuracy = []
for i in range(0,10):
    split = split_data(data,i,10)
    train = split[0]
    test = split[1]
    accuracy.append(ttest(train,test,metadata))
accuracy = np.array(accuracy)
differences = accuracy.T[1] - accuracy.T[0]
print("Accuracy (NB, TAN): \n", accuracy)
print("Differences (TAN - NB): \n", differences)

#Calculate average accuracy difference:
diff_avg = np.mean(differences)

#Calculate standard deviation of differences
diff_sd = np.std(differences)

#Calculate standard error from SD
diff_se = diff_sd/np.sqrt(len(differences))

#Degrees of freedom defined as n-1
diff_df = len(differences)-1

#Calculate test statistic t
diff_t = diff_avg/diff_se

#Find the probability of a result as extreme or more extreme
diff_pvalue = 2*(1-t.cdf(diff_t,diff_df))

print("Average: ",diff_avg,
    "\nStandard Deviation: ", diff_sd,
    "\nStandard Error: ", diff_se,
    "\nDegrees of freedom: ", diff_df, 
    "\nT-statistic: ", diff_t,
    "\np-value: ", diff_pvalue)
