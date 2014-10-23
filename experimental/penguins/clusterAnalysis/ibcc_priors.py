#!/usr/bin/env python
__author__ = 'greghines'
import numpy as np
import matplotlib.pyplot as plt
import pymongo
import cPickle as pickle
import os
import math
import sys

if os.path.exists("/home/ggdhines"):
    sys.path.append("/home/ggdhines/PycharmProjects/reduction/experimental/clusteringAlg")
else:
    sys.path.append("/home/greg/github/reduction/experimental/clusteringAlg")

from clusterCompare import cluster_compare

if os.path.exists("/home/ggdhines"):
    base_directory = "/home/ggdhines"
else:
    base_directory = "/home/greg"

penguins,temp = pickle.load(open(base_directory+"/Databases/penguins_vote_.pickle","rb"))

#does this cluster have a corresponding cluster in the gold standard data?
#ie. does this cluster represent an actual penguin?

# #user penguins for first image - with 5 images
# print len(penguins[5][0])
# #user data
# print penguins[5][0][0]
# #gold standard data
# #print penguins[5][0][1]
#
# #users who annotated the first "penguin" in the first image
# print penguins[5][0][0][0][1]
# #and their corresponds points
# print penguins[5][0][0][0][0]

#have as a list not a tuple since we need the index
user_set = []

max_users = 20

#first - create a list of ALL users - so we can figure out who has annotated a "penguin" or hasn't
for image_index in range(len(penguins[max_users])):
    for penguin_index in range(len(penguins[max_users][image_index][0])):
        users = penguins[max_users][image_index][0][penguin_index][1]
        for u in users:
            if not(u in user_set):
                user_set.append(u)

confusion_matrix = {u:[[0,0],[0,0]] for u in user_set}
overall_confusion_matrix = [[0,0],[0,0]]
print len(user_set)
#now actually figure out how has annotated a penguin or hasn't
for image_index in range(len(penguins[max_users])):
    for penguin_index in range(len(penguins[max_users][image_index][0])):
        users = penguins[max_users][image_index][0][penguin_index][1]
        if len(users) >= 6:
            penguin = 1
        else:
            penguin = 0
        for user_index,u_ in enumerate(user_set):
            if u_ in users:
                confusion_matrix[u_][1][penguin] += 1
                overall_confusion_matrix[penguin][1] += 1
            else:
                confusion_matrix[u_][0][penguin] += 1
                overall_confusion_matrix[penguin][0] += 1
true_negative = []
true_positive = []
for cm in confusion_matrix.values():
    true_negative.append(cm[0][0]/float(sum(cm[0])))
    true_positive.append(cm[1][1]/float(sum(cm[1])))

print np.mean(true_negative)
print np.mean(true_positive)

