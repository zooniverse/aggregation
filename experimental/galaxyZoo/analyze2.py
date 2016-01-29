#!/usr/bin/env python
import cPickle as pickle
import os

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

results = pickle.load(open(base_directory+"/Databases/milkyway.pickle","rb"))

Y_true_positive = []
X_false_positive = []

true_positive = 0
false_positive = 0

positive_total = 0.
negative_total = 0.

total = 0.

for r in results:
    t1 = r[0][0]
    t2 = max(r[0][1],r[0][2])

    p1 = t1/(t1+t2)
    p2 = t2/(t1+t2)

    if r[1]:
        Y_true_positive.append(p1)

        if p1 > p2:
            true_positive += 1

        positive_total += 1
    else:
        X_false_positive.append(p1)

        if p1 > p2:
            false_positive += 1

        negative_total += 1

    total += 1


roc_X = []
roc_Y = []

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

#print roc_X
#true_positive = len(Y_true_positive)/float(len(Y_true_positive)+len(X_false_positive))
#false_postive = len(X_false_positive)/float(len(Y_true_positive)+len(X_false_positive))

true_positive = true_positive / positive_total
false_positive = false_positive / negative_total

import matplotlib.pyplot as plt
plt.plot(roc_X,roc_Y)
#plt.xlim((0,1.05))
plt.plot((0,1),(0,1),'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot([false_positive],[true_positive],'o')
plt.show()