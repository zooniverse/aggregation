#!/usr/bin/env python
import cPickle as pickle

results = pickle.load(open("/home/greg/Databases/milkyway.pickle","rb"))

true_positive = {0:[],1:[],2:[]}
false_positive = {0:[],1:[],2:[]}

#split on t1

for probabilities,t0,t1,t2 in results:
    if t0:
        true_positive[0].append(probabilities[0])


alpha_list = X_false_positive[:]
alpha_list.extend(Y_true_positive)
alpha_list.sort()

for alpha in alpha_list:
    positive_count = sum([1 for x in Y_true_positive if x >= alpha])
    positive_rate = positive_count/float(len(Y_true_positive))

    negative_count = sum([1 for x in X_false_positive if x >= alpha])
    negative_rate = negative_count/float(len(X_false_positive))

    roc_X.append(negative_rate)
    roc_Y.append(positive_rate)