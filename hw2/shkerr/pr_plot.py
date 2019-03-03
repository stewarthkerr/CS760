#!/usr/bin/python3.6
import json
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from functions import pr_plot_naive, pr_plot_tan

parser = argparse.ArgumentParser(description='Implements a Bayes net')
parser.add_argument('-train', type = str, help='Train data set path')
parser.add_argument('-test', type = str, help='Test data set path')
parser.add_argument("-bntype", type=str, help="Type of Bayes net to implement (n = naive, t = TAN")
args = parser.parse_args()
train = args.train
test = args.test
bntype = args.bntype
bntype = bntype.replace('\r','') #Removes the carriage return cuz I use windows

#Load the data in
with open(train,"r") as read_file:
    train = json.load(read_file)
with open(test,"r") as read_file:
    test = json.load(read_file)

PR_coord = []
if bntype == 'n':
    title = "Naive Bayes Tic-tac-toe Precision-Recall Curve"
    filename = "pr_curve_naive.png"
    for i in range(0,101):
        threshold = i/100
        PR_coord.append(pr_plot_naive(train,test,threshold))
    PR_coord = np.array(PR_coord)
    precision = PR_coord.T[0]
    recall = PR_coord.T[1]
    plt.plot(recall,precision)
elif bntype == 't':
    title = "TAN Tic-tac-toe Precision-Recall Curve"
    filename = "pr_curve_tan.png"
    for i in range(0,101):
        threshold = i/100
        PR_coord.append(pr_plot_tan(train,test,threshold))
    PR_coord = np.array(PR_coord)
    precision = PR_coord.T[0]
    recall = PR_coord.T[1]
    plt.plot(recall,precision)
elif bntype == 'b':
    title = "PR-Curve Comparison of Naive Bayes and TAN"
    filename = "pr_curve_comparison.png"
    for i in range(0,101):
        threshold = i/100
        PR_coord.append(pr_plot_naive(train,test,threshold))
    PR_coord = np.array(PR_coord)
    precision = PR_coord.T[0]
    recall = PR_coord.T[1]
    plt.plot(recall,precision)
    PR_coord = []
    for i in range(0,101):
        threshold = i/100
        PR_coord.append(pr_plot_tan(train,test,threshold))
    PR_coord = np.array(PR_coord)
    precision = PR_coord.T[0]
    recall = PR_coord.T[1]
    plt.plot(recall,precision)
    plt.legend(['Naive Bayes','TAN'], loc = "lower right")
else:
    print("3rd argument bntype is not valid. Expect 'n', 't', or 'b'.")

#Construct the plots
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(title)
plt.savefig(filename)
